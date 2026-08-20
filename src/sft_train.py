"""SFT training loop for chat/instruction tuning (TASK 005 / 005.2).

Trains the accepted from-scratch BASE checkpoint into a chat assistant using
human-only SFT data. Key properties:

- init-from "base": loads ONLY the base model weights (checkpoints/pretrain_v1/
  best.pt) and starts a NEW optimizer / scaler / scheduler (never reuses
  pretraining AdamW moments). Full-model fine-tuning (no LoRA). The failed
  TASK 005.1 diagnostic checkpoint (checkpoints/chat_v1/) is NEVER used as an
  initialization source.
- CAUSAL assistant-only loss (TASK 005.2 root-cause fix): the dataset labels
  carry the next-token target: labels[i] = ids[i + 1] over the final assistant
  target + EOS; -100 everywhere else (BOS / user / context / role markers /
  padding / the EOS input position). GPTModel computes cross entropy at the
  SAME tensor positions (logits[i] vs targets[i]), so with these labels the
  model predicts the NEXT token — never the already-present input token.
- variable-length batches padded to batch max length (<= context 256).
- deterministic validation (Part H): a FIXED seed samples a fixed set of
  example indices ACROSS the FULL SFT validation set (not just the first
  batches); the same indices are used at every evaluation.
- three losses at each evaluation:
    1) SFT validation causal assistant-only loss
    2) base-language validation loss on the original validation.bin
       (catastrophic-forgetting metric vs the pre-SFT baseline)
    3) anchor replay loss (base-language next-token objective, weighted)
- base-language replay anchor (Part K): ONE batch from OUR OWN
  data/processed/train.bin per optimizer step, loss scaled by
  --anchor-loss-weight (default 0.10), backpropagated into the same step.
  Anchor tokens are counted separately (base_replay_tokens_seen) and are NOT
  supervised SFT target tokens. Objective:
      L = (1/G) * sum_{mb=1..G} SFT_mb + w * ANCHOR
- retention guards (Part E fix): best.pt eligible only if
  base_loss <= baseline * eligibility_factor (default 1.10); HARD STOP if
  base_loss > baseline * hard_stop_factor (default 1.15);
  eligibility_factor < hard_stop_factor is validated at startup;
  early stop if the SFT val loss fails to improve for `patience` consecutive
  evaluations (after warmup).
- pilot gate (Part M/N): the first `pilot_steps` optimizer steps are a pilot
  evaluated at fixed steps (25/50/100/200); if the SFT val loss collapses
  suspiciously (< 0.5 within the pilot), if gradients/loss are non-finite,
  or retention hard-stops, training stops and reports.
- token accounting (Part F fix): per optimizer step the supervised/total
  tokens of ALL grad-accum microbatches are summed (never just the last
  microbatch); skipped non-finite steps do not add to the trained-token
  counters.
- full resume: model, optimizer, scaler, step, tokens/supervised-tokens seen,
  base replay tokens seen, best SFT val loss, base baseline/current loss,
  dataset RNG/order (SFT + anchor), RNG states, source-data revision metadata;
  atomic writes; graceful KeyboardInterrupt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset import BinaryDataset, verify_bin_integrity
from .model import GPTModel, ModelConfig
from .sft_dataset import SFTDataset
from .train import (  # noqa: F401  (reused pretraining-tested helpers)
    CHECKPOINT_BEST,
    CHECKPOINT_LATEST,
    collect_rng_state,
    compute_grad_norm,
    configure_optimizer,
    estimate_loss,
    lr_at,
    restore_rng_state,
    skip_nonfinite_grad_step,
    write_metrics,
)

DEFAULT_OUT_DIR = os.path.join("checkpoints", "chat_v1_corrected")

# fixed pilot evaluation steps (Part M): step 0 baseline + 25/50/100/200
PILOT_EVAL_STEPS = (25, 50, 100, 200)
# suspicious-loss sanity floor (Part N): a val loss below this inside the pilot
# means the corrected objective is still collapsing (identity-copy warning)
SUSPICIOUS_SFT_VAL_LOSS = 0.5
# deterministic SFT validation sampling seed offset (Part H)
SFT_VAL_SEED_OFFSET = 42


def parse_args():
    p = argparse.ArgumentParser(description="SFT the from-scratch base model into a chat assistant")
    p.add_argument("--config", default="configs/sft_chat_v1.json")
    p.add_argument("--base-checkpoint", default="checkpoints/pretrain_v1/best.pt")
    p.add_argument("--tokenizer-model", default="data/tokenizer/tokenizer_v1.model")
    p.add_argument("--tokenizer-meta", default="data/tokenizer/tokenizer_v1_meta.json")
    p.add_argument("--data-sft-train", default="data/sft/processed/sft_train.jsonl")
    p.add_argument("--data-sft-val", default="data/sft/processed/sft_val.jsonl")
    p.add_argument("--data-val", default="data/processed/validation.bin",
                   help="original base-language validation.bin (retention metric)")
    p.add_argument("--dataset-meta", default="data/processed/dataset_meta.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--init-from", choices=["base", "resume"], default="base")
    p.add_argument("--resume-path", default=None)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--pilot-steps", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--retention-baseline", type=float, default=None,
                   help="pre-SFT base-language validation loss on validation.bin (measured if omitted)")
    p.add_argument("--eligibility-factor", type=float, default=1.10)
    p.add_argument("--hard-stop-factor", type=float, default=1.15)
    p.add_argument("--patience-evals", type=int, default=4)
    p.add_argument("--anchor-loss-weight", type=float, default=0.10,
                   help="base-language replay anchor loss weight per optimizer step (Part K)")
    p.add_argument("--anchor-data", default="data/processed/train.bin",
                   help="OUR OWN pretraining corpus used as language-retention replay (never test.bin)")
    p.add_argument("--no-anchor", action="store_true",
                   help="disable the base-language replay anchor entirely")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--run-config", default=None)
    return p.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_sft_loss(model: GPTModel, ds: SFTDataset, iters: int,
                  device: torch.device, use_amp: bool, seed: int = 0) -> float:
    """Deterministic causal assistant-only validation loss over the SFT val set.

    Part H: evaluates a FIXED set of example indices sampled deterministically
    across the FULL validation set (val_eval_indices), so every evaluation
    sees the same representative sample and never just the first batches of
    the JSONL. Does not mutate the dataset's RNG/order state.
    """
    model.eval()
    idx = val_eval_indices(len(ds.ids), ds.batch_size, iters, seed)
    losses: List[float] = []
    for start in range(0, len(idx), ds.batch_size):
        batch_idx = idx[start:start + ds.batch_size]
        T = max(len(ds.ids[i]) for i in batch_idx)
        x = torch.full((len(batch_idx), T), ds.pad_id, dtype=torch.long)
        y = torch.full((len(batch_idx), T), -100, dtype=torch.long)
        for r, i in enumerate(batch_idx):
            x[r, :len(ds.ids[i])] = torch.tensor(ds.ids[i], dtype=torch.long)
            y[r, :len(ds.labels[i])] = torch.tensor(ds.labels[i], dtype=torch.long)
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y, ignore_index=-100)
        else:
            _, loss = model(x, y, ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def validate_retention_factors(eligibility_factor: float, hard_stop_factor: float) -> None:
    """Startup validation: the eligibility line must sit BELOW the hard-stop
    line, otherwise the gate is degenerate."""
    if eligibility_factor <= 0 or hard_stop_factor <= 0:
        raise SystemExit("[FATAL] retention factors must be positive")
    if not (eligibility_factor < hard_stop_factor):
        raise SystemExit(
            f"[FATAL] eligibility factor {eligibility_factor} must be < "
            f"hard stop factor {hard_stop_factor}")


def retention_guard(base_loss: float, baseline: float,
                    eligibility_factor: float, hard_stop_factor: float) -> Tuple[bool, bool]:
    """Part E: base-language retention gate with SEPARATE factors.

    eligible  = base_loss <= baseline * eligibility_factor
    hard_stop = base_loss >  baseline * hard_stop_factor
    (The old single-factor code returned eligible=True for the whole window up
    to the hard-stop line, wrongly accepting the middle band.)
    """
    eligible = base_loss <= baseline * eligibility_factor
    hard_stop = base_loss > baseline * hard_stop_factor
    return hard_stop, eligible


def val_eval_indices(n: int, batch_size: int, iters: int, seed: int) -> torch.Tensor:
    """Part H: deterministic, representative SFT-val indices.

    Samples WITHOUT replacement across the FULL validation set (never only the
    first JSONL batches); covers everything when iters*batch_size >= n.
    """
    rng = torch.Generator().manual_seed(seed)
    want = min(n, iters * batch_size)
    perm = torch.randperm(n, generator=rng)
    return perm[:want]


def validate_anchor_data(anchor_data: str) -> None:
    """The replay corpus must be OUR OWN pretraining data; test.bin is sealed."""
    if "test.bin" in anchor_data:
        raise SystemExit("[FATAL] test.bin is SEALED and may never be used as replay data")


def assert_out_dir_free_for_base(out_dir: str) -> None:
    """Refuses --init-from base over an out-dir that already contains checkpoints
    (a failed run or a live run must never be silently overwritten)."""
    for fname in (CHECKPOINT_LATEST, CHECKPOINT_BEST):
        if os.path.exists(os.path.join(out_dir, fname)):
            raise SystemExit(
                f"[FATAL] {os.path.join(out_dir, fname)} exists. Use --init-from resume "
                f"or move the directory aside (never overwrite an existing chat run).")


def step_token_counts(batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, int]:
    """Part F: sums supervised/total tokens over ALL grad-accum microbatches of
    one optimizer step (the old code counted only the LAST microbatch)."""
    supervised = sum(int(y[y != -100].numel()) for _x, y in batches)
    total = sum(int(x.numel()) for x, _y in batches)
    return supervised, total


def anchor_replay_tokens(anchor_x: torch.Tensor) -> int:
    """Part K: base-language replay tokens for one anchor batch. These are
    counted SEPARATELY (base_replay_tokens_seen) and are never supervised SFT
    target tokens."""
    return int(anchor_x.numel())


def save_checkpoint(out_dir: str, kind: str, model: GPTModel, optimizer,
                    scaler, step: int, supervised_tokens_seen: int,
                    total_tokens_seen: int, best_sft_val_loss: float,
                    base_baseline_loss: float, base_current_loss: float,
                    cfg_dict: dict, args, train_ds: SFTDataset, val_ds: SFTDataset,
                    rng_state: dict, source_meta: dict,
                    base_replay_tokens_seen: int = 0,
                    anchor_ds: Optional[BinaryDataset] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{kind}.pt")
    payload = {
        "step": step,
        "supervised_tokens_seen": supervised_tokens_seen,
        "total_tokens_seen": total_tokens_seen,
        "base_replay_tokens_seen": base_replay_tokens_seen,
        "best_sft_val_loss": best_sft_val_loss,
        "base_baseline_loss": base_baseline_loss,
        "base_current_loss": base_current_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": cfg_dict,
        "args": vars(args),
        "train_ds": train_ds.state_dict(),
        "val_ds": val_ds.state_dict(),
        "anchor_ds": anchor_ds.state_dict() if anchor_ds is not None else None,
        "rng": rng_state,
        "source_meta": source_meta,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_chat_checkpoint(path: str, model: GPTModel, optimizer, scaler,
                         device: torch.device, expected_config: dict):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("config") != expected_config:
        raise SystemExit(
            f"[FATAL] checkpoint config does not match current config:\n"
            f"  checkpoint: {ckpt.get('config')}\n  current:    {expected_config}")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def write_run_config(args, cfg: ModelConfig, device: torch.device,
                     n_params: dict, steps_per_epoch: int,
                     train_ds: SFTDataset) -> str:
    path = args.run_config or os.path.join(args.out_dir, "run_config.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "task": "TASK 005 SFT (chat/instruction tuning v1)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "device": str(device),
        "mixed_precision": device.type == "cuda" and not args.no_amp,
        "base_checkpoint": args.base_checkpoint,
        "hyperparameters": {
            "init_from": args.init_from,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "context_length": cfg.block_size,
            "peak_lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "eval_interval": args.eval_interval,
            "pilot_steps": args.pilot_steps,
            "retention_eligibility_factor": args.eligibility_factor,
            "retention_hard_stop_factor": args.hard_stop_factor,
            "early_stop_patience_evals": args.patience_evals,
            "anchor_loss_weight": args.anchor_loss_weight if not args.no_anchor else 0.0,
            "anchor_data": args.anchor_data if not args.no_anchor else None,
            "sft_val_eval_seed": args.seed + SFT_VAL_SEED_OFFSET,
            "label_convention": "causal_next_token (labels[i]=ids[i+1] over the final "
                                "assistant target + EOS; never same-position labels)",
        },
        "model_config": cfg.to_dict(),
        "expected": {
            "params_total": n_params["total"],
            "params_trainable": n_params["trainable"],
            "steps_per_epoch": steps_per_epoch,
            "unique_train_supervised_tokens": train_ds.unique_supervised_tokens(),
            "effective_train_supervised_tokens": train_ds.effective_supervised_tokens(),
            "effective_train_examples": len(train_ds),
            "unique_train_examples": train_ds.n_examples,
            "target_steps": steps_per_epoch * args.max_epochs,
            "eval_interval": args.eval_interval,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"device: {device} | mixed precision: {use_amp} (float16)")
    validate_retention_factors(args.eligibility_factor, args.hard_stop_factor)

    cfg = ModelConfig.from_json(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- tokenizer special ids / metadata validation ----
    tok_meta = load_json(args.tokenizer_meta)
    for key, val in [("pad_id", cfg.pad_id), ("bos_id", cfg.bos_id),
                     ("eos_id", cfg.eos_id), ("unk_id", cfg.unk_id)]:
        if tok_meta.get(key) != val:
            raise SystemExit(f"[FATAL] tokenizer meta {key}={tok_meta.get(key)} != config {val}")

    # ---- base-language validation (retention metric) ----
    dataset_meta = load_json(args.dataset_meta)
    val_tokens_meta = (dataset_meta.get("splits", {}).get("val", {}) or {}).get("tokens")
    val_tokens = verify_bin_integrity(args.data_val, val_tokens_meta, name="validation.bin")
    print(f"validation.bin integrity OK: {val_tokens:,} tokens")
    base_val_ds = BinaryDataset(args.data_val, cfg.block_size, args.batch_size,
                                seed=args.seed + 1, start_frac=0.0, end_frac=1.0)

    # ---- base-language replay anchor (Part K): OUR OWN pretraining corpus ----
    anchor_ds = None
    if not args.no_anchor:
        validate_anchor_data(args.anchor_data)
        anchor_ds = BinaryDataset(args.anchor_data, cfg.block_size, args.batch_size,
                                  seed=args.seed + 2)
        print(f"base-language replay anchor: {args.anchor_data} "
              f"(weight {args.anchor_loss_weight}, one batch per optimizer step)")

    model = GPTModel(cfg).to(device)
    n_params = model.count_parameters()
    print(f"model params: total={n_params['total']:,} trainable={n_params['trainable']:,}")

    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = SFTDataset(args.data_sft_train, args.batch_size, cfg.block_size,
                          cfg.pad_id, seed=args.seed, shuffle=True)
    val_ds = SFTDataset(args.data_sft_val, args.batch_size, cfg.block_size,
                        cfg.pad_id, seed=args.seed, shuffle=False)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    max_iters = steps_per_epoch * args.max_epochs
    warmup = max(args.warmup_steps, int(0.03 * max_iters))
    # ensure several evaluations per epoch when the epoch is short
    eval_interval = args.eval_interval
    if steps_per_epoch < eval_interval * 3:
        eval_interval = max(50, steps_per_epoch // 3)
    args.eval_interval = eval_interval
    print(f"train examples (effective): {len(train_ds)} | unique: {train_ds.n_examples} | "
          f"val examples: {len(val_ds)} | steps/epoch: {steps_per_epoch} | "
          f"max_iters: {max_iters} | warmup: {warmup} | eval every {eval_interval}")

    source_meta = {
        "sft_train": args.data_sft_train,
        "sft_val": args.data_sft_val,
        "sft_stats": os.path.join("data", "sft", "stats", "sft_stats.json"),
        "revision_notes": "see data/sft/manifests/sources.jsonl for exact dataset revisions",
    }

    step = 0
    supervised_tokens_seen = 0
    total_tokens_seen = 0
    base_replay_tokens_seen = 0
    best_sft_val_loss = float("inf")
    base_baseline_loss = args.retention_baseline
    base_current_loss = None
    sft_val_baseline = None

    if args.init_from == "resume":
        ckpt_path = args.resume_path or os.path.join(args.out_dir, CHECKPOINT_LATEST)
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"[FATAL] resume checkpoint not found: {ckpt_path}")
        ckpt = load_chat_checkpoint(ckpt_path, model, optimizer, scaler, device, cfg.to_dict())
        step = ckpt["step"]
        supervised_tokens_seen = ckpt["supervised_tokens_seen"]
        total_tokens_seen = ckpt["total_tokens_seen"]
        base_replay_tokens_seen = ckpt.get("base_replay_tokens_seen", 0)
        best_sft_val_loss = ckpt["best_sft_val_loss"]
        base_baseline_loss = ckpt.get("base_baseline_loss")
        base_current_loss = ckpt.get("base_current_loss")
        sft_val_baseline = ckpt.get("sft_val_baseline")
        if "rng" in ckpt:
            restore_rng_state(ckpt["rng"], device)
        if "train_ds" in ckpt:
            train_ds.load_state_dict(ckpt["train_ds"])
        if "val_ds" in ckpt:
            val_ds.load_state_dict(ckpt["val_ds"])
        if anchor_ds is not None and ckpt.get("anchor_ds") is not None:
            anchor_ds.load_state_dict(ckpt["anchor_ds"])
        if "source_meta" in ckpt:
            source_meta.update(ckpt["source_meta"])
        print(f"resumed from {ckpt_path} at step {step}")
    else:
        assert_out_dir_free_for_base(args.out_dir)
        # Load ONLY the base model weights; new optimizer/scaler/scheduler state.
        base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(base_ckpt["model_state"])
        print(f"base model weights loaded from {args.base_checkpoint} "
              f"(base step {base_ckpt.get('step')}, best val {base_ckpt.get('best_val_loss')})")
        # chat-format baselines BEFORE any weight modification
        if base_baseline_loss is None:
            base_baseline_loss = estimate_loss(model, base_val_ds, 50, device, use_amp, args.seed + 1000)
            print(f"[baseline] base-language validation loss: {base_baseline_loss:.4f} (retention baseline)")
        sft_val_baseline = eval_sft_loss(model, val_ds, 50, device, use_amp,
                                         args.seed + SFT_VAL_SEED_OFFSET)
        print(f"[baseline] SFT val causal assistant-only loss (untrained chat format): "
              f"{sft_val_baseline:.4f}")
        write_metrics(args.metrics_file or os.path.join(args.out_dir, "metrics.jsonl"), {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 0,
            "train_sft_loss": None,
            "sft_val_loss": round(float(sft_val_baseline), 6),
            "base_val_loss": round(float(base_baseline_loss), 6) if base_baseline_loss else None,
            "base_baseline_loss": round(float(base_baseline_loss), 6) if base_baseline_loss else None,
            "learning_rate": 0.0,
            "supervised_tokens_seen": 0,
            "total_tokens_seen": 0,
            "base_replay_tokens_seen": 0,
            "anchor_base_loss": None,
            "note": "step-0 baseline: untrained chat format on the SFT val set + base-language baseline",
        })
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_baseline_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta,
                        base_replay_tokens_seen, anchor_ds)
        write_run_config(args, cfg, device, n_params, steps_per_epoch, train_ds)
        print(f"initial latest checkpoint written to {os.path.join(args.out_dir, CHECKPOINT_LATEST)}")

    model.train()
    t0 = time.time()
    run_start_supervised = supervised_tokens_seen
    running_loss = None
    val_seed = args.seed + 1000
    metrics_path = args.metrics_file or os.path.join(args.out_dir, "metrics.jsonl")
    no_improve_evals = 0
    stopped = False

    def log_metrics(sft_val: Optional[float] = None, base_val: Optional[float] = None,
                    anchor_loss: Optional[float] = None):
        dt = time.time() - t0
        sps = (supervised_tokens_seen - run_start_supervised) / max(dt, 1e-9)
        peak = 0.0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"step {step}/{max_iters}  sft_loss {running_loss:.4f}  lr {lr:.2e}  "
              f"sup_tokens {supervised_tokens_seen:,}  ({sps:,.0f} sup_tok/s)"
              + (f"  anchor {anchor_loss:.4f}" if anchor_loss is not None else "")
              + (f"  peak VRAM {peak:.2f} GB" if device.type == "cuda" else ""))
        write_metrics(metrics_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "train_sft_loss": round(float(running_loss), 6) if running_loss is not None else None,
            "sft_val_loss": round(float(sft_val), 6) if sft_val is not None else None,
            "base_val_loss": round(float(base_val), 6) if base_val is not None else None,
            "base_baseline_loss": round(float(base_baseline_loss), 6) if base_baseline_loss else None,
            "learning_rate": float(lr),
            "supervised_tokens_seen": int(supervised_tokens_seen),
            "total_tokens_seen": int(total_tokens_seen),
            "base_replay_tokens_seen": int(base_replay_tokens_seen),
            "anchor_base_loss": round(float(anchor_loss), 6) if anchor_loss is not None else None,
            "peak_vram_gb": round(float(peak), 4),
            "best_sft_val_loss": float(best_sft_val_loss) if best_sft_val_loss != float("inf") else None,
        })
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def evaluate(save: bool = True, eval_iters: Optional[int] = None) -> Tuple[bool, float, float, bool]:
        nonlocal best_sft_val_loss, base_current_loss, no_improve_evals
        iters = eval_iters or args.eval_iters
        sft_val = eval_sft_loss(model, val_ds, iters, device, use_amp, args.seed + SFT_VAL_SEED_OFFSET)
        base_val = estimate_loss(model, base_val_ds, 50, device, use_amp, val_seed)
        base_current_loss = base_val
        hard_stop, eligible = retention_guard(base_val, base_baseline_loss,
                                              args.eligibility_factor, args.hard_stop_factor)
        suspicious = step <= args.pilot_steps and sft_val < SUSPICIOUS_SFT_VAL_LOSS
        improved = sft_val < best_sft_val_loss
        print(f"  [eval] step {step}: sft_val {sft_val:.4f} | base_val {base_val:.4f} "
              f"(baseline {base_baseline_loss:.4f}; eligible <= {base_baseline_loss * args.eligibility_factor:.4f}; "
              f"hard stop > {base_baseline_loss * args.hard_stop_factor:.4f}) | eligible {eligible}"
              + (" | SUSPICIOUS-LOW SFT VAL (identity-copy warning)" if suspicious else ""))
        if improved:
            no_improve_evals = 0
            best_sft_val_loss = sft_val
            if eligible and save:
                save_checkpoint(args.out_dir, "best", model, optimizer, scaler, step,
                                supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                                base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                                train_ds, val_ds, collect_rng_state(device), source_meta,
                                base_replay_tokens_seen, anchor_ds)
                print(f"    [eval] new best SFT val loss {sft_val:.4f} -> best.pt (retention OK)")
        else:
            no_improve_evals += 1
        if save:
            save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                            supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                            base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                            train_ds, val_ds, collect_rng_state(device), source_meta,
                            base_replay_tokens_seen, anchor_ds)
        return hard_stop, sft_val, base_val, suspicious

    try:
        while step < max_iters:
            if step % steps_per_epoch == 0:
                train_ds.reset_epoch()
            optimizer.zero_grad(set_to_none=True)
            micro_loss_sum = 0.0
            step_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            # Part F fix: accumulate supervised/total tokens over ALL microbatches
            for mb in range(args.grad_accum):
                x, y = train_ds.get_batch(device)
                step_batches.append((x, y))
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, loss = model(x, y, ignore_index=-100)
                else:
                    _, loss = model(x, y, ignore_index=-100)
                if not torch.isfinite(loss):
                    save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                                    supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                                    base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                                    train_ds, val_ds, collect_rng_state(device), source_meta,
                                    base_replay_tokens_seen, anchor_ds)
                    raise SystemExit(
                        f"[FATAL] non-finite SFT loss ({loss.item()}) at step {step} micro-batch {mb}. "
                        f"Latest healthy checkpoint preserved. Stopping.")
                micro_loss_sum += loss.item()
                scaled = scaler.scale(loss / args.grad_accum)
                scaled.backward()

            step_supervised_tokens, step_total_tokens = step_token_counts(step_batches)
            micro_loss = micro_loss_sum / args.grad_accum

            # Part K: ONE base-language replay batch per optimizer step.
            # Objective: L = (1/G) * sum SFT_mb + anchor_loss_weight * ANCHOR.
            # The anchor coefficient is applied exactly ONCE (not divided by
            # grad_accum, not doubled) and participates in the same AMP scaler.
            anchor_loss = None
            step_replay_tokens = 0
            if anchor_ds is not None:
                ax, ay = anchor_ds.get_batch(device)
                step_replay_tokens = anchor_replay_tokens(ax)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, anchor_loss_t = model(ax, ay, ignore_index=-100)
                else:
                    _, anchor_loss_t = model(ax, ay, ignore_index=-100)
                anchor_loss = float(anchor_loss_t.item())
                if not torch.isfinite(anchor_loss_t):
                    save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                                    supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                                    base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                                    train_ds, val_ds, collect_rng_state(device), source_meta,
                                    base_replay_tokens_seen, anchor_ds)
                    raise SystemExit(
                        f"[FATAL] non-finite anchor loss ({anchor_loss}) at step {step}. "
                        f"Latest healthy checkpoint preserved. Stopping.")
                scaler.scale(anchor_loss_t * args.anchor_loss_weight).backward()

            if skip_nonfinite_grad_step(model, optimizer, scaler, use_amp, step,
                                        total_tokens_seen, metrics_path):
                # skipped step: the consumed batch tokens are NOT counted as
                # optimized/trained tokens (Part F)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            lr = lr_at(step, max_iters, warmup, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            supervised_tokens_seen += step_supervised_tokens
            total_tokens_seen += step_total_tokens
            base_replay_tokens_seen += step_replay_tokens
            step += 1
            running_loss = micro_loss if running_loss is None else 0.9 * running_loss + 0.1 * micro_loss

            # Part M: fixed pilot evaluation schedule (25/50/100/200) with at
            # least 50 validation batches; regular evals after the pilot.
            do_eval = step in PILOT_EVAL_STEPS and step <= args.pilot_steps
            do_eval = do_eval or (step % args.eval_interval == 0 and step > args.pilot_steps)
            if do_eval:
                eval_iters = max(args.eval_iters, 50) if step <= args.pilot_steps else None
                hard_stop, sft_val, base_val, suspicious = evaluate(eval_iters=eval_iters)
                log_metrics(sft_val, base_val, anchor_loss)
                if suspicious:
                    print(f"  [STOP] SFT val {sft_val:.4f} < {SUSPICIOUS_SFT_VAL_LOSS} inside the pilot: "
                          f"the corrected objective is still collapsing (identity-copy warning). "
                          f"Run diagnostics before continuing.")
                    stopped = True
                    break
                if hard_stop:
                    print(f"  [STOP] base-language validation loss {base_val:.4f} "
                          f"> baseline * {args.hard_stop_factor} ({base_baseline_loss * args.hard_stop_factor:.4f}); "
                          f"hard forgetting stop.")
                    stopped = True
                    break
                if no_improve_evals >= args.patience_evals and step >= warmup:
                    print(f"  [STOP] SFT val loss not improved for {no_improve_evals} "
                          f"consecutive evaluations; early stop.")
                    stopped = True
                    break
                # pilot gate: after the pilot window, verify health
                if step == args.pilot_steps:
                    if running_loss is None or not math.isfinite(running_loss):
                        print("  [PILOT] non-finite running loss; STOP and report.")
                        stopped = True
                        break
                    if sft_val is not None and sft_val_baseline is not None and sft_val > sft_val_baseline * 1.5:
                        print(f"  [PILOT] SFT val {sft_val:.4f} is severely worse than the chat-format "
                              f"baseline {sft_val_baseline:.4f}; STOP and report.")
                        stopped = True
                        break
                    print(f"  [PILOT] healthy at step {args.pilot_steps}: causal objective, loss finite, "
                          f"base-language retention OK. Continuing to full SFT.")

            if step % args.log_interval == 0 and not do_eval:
                log_metrics(anchor_loss=anchor_loss)

        if step > 0 and not stopped:
            hard_stop, sft_val, base_val, suspicious = evaluate()
            log_metrics(sft_val, base_val, anchor_loss)
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta,
                        base_replay_tokens_seen, anchor_ds)
        print(f"training finished at step {step}; final latest checkpoint saved")
        if stopped:
            print("NOTE: run stopped early (guard triggered); see report.")
    except KeyboardInterrupt:
        print("\n[interrupt] KeyboardInterrupt received; preserving resumable state...")
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta,
                        base_replay_tokens_seen, anchor_ds)
        print(f"[interrupt] latest checkpoint saved to {os.path.join(args.out_dir, CHECKPOINT_LATEST)} "
              f"(step={step}); resume with --init-from resume")


if __name__ == "__main__":
    main()