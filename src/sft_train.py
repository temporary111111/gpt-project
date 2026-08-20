"""SFT training loop for chat/instruction tuning (TASK 005).

Trains the accepted from-scratch BASE checkpoint into a chat assistant using
human-only SFT data. Key properties:

- init-from "base": loads ONLY the base model weights (checkpoints/pretrain_v1/
  best.pt) and starts a NEW optimizer / scaler / scheduler (never reuses
  pretraining AdamW moments). Full-model fine-tuning (no LoRA).
- assistant-only loss: labels = -100 for BOS / user / context / role markers /
  padding; cross-entropy only over the final assistant target + EOS.
- variable-length batches padded to batch max length (<= context 256).
- deterministic validation (fixed seed / fixed order, no shuffle for val).
- two losses at each evaluation:
    1) SFT validation assistant-only loss
    2) base-language validation loss on the original validation.bin
       (catastrophic-forgetting metric vs the pre-SFT baseline)
- retention guards: best.pt eligible only if base loss <= baseline * 1.15;
  HARD STOP if base loss > baseline * 1.20; early stop if the SFT val loss
  fails to improve for `patience` consecutive evaluations (after warmup).
- pilot gate: the first `pilot_steps` optimizer steps are a pilot; if the
  loss/gradients are non-finite, CUDA is unhealthy, or there is a severe
  regression vs the chat-format baseline, training stops and reports.
- full resume: model, optimizer, scaler, step, tokens/supervised-tokens seen,
  best SFT val loss, base baseline/current loss, dataset RNG/order, RNG states,
  source-data revision metadata; atomic writes; graceful KeyboardInterrupt.
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

DEFAULT_OUT_DIR = os.path.join("checkpoints", "chat_v1")


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
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
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
    p.add_argument("--eligibility-factor", type=float, default=1.15)
    p.add_argument("--hard-stop-factor", type=float, default=1.20)
    p.add_argument("--patience-evals", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--run-config", default=None)
    return p.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_sft_loss(model: GPTModel, ds: SFTDataset, iters: int,
                  device: torch.device, use_amp: bool) -> float:
    """Deterministic assistant-only validation loss over the SFT val set.

    Uses a fixed evaluation order (no shuffle) and evaluates `iters` batches
    (or the whole set, whichever is smaller).
    """
    model.eval()
    ds.rng = np.random.default_rng(0)  # deterministic
    ds._order = np.arange(len(ds.ids))  # fixed order for validation
    ds._pos = 0
    losses: List[float] = []
    n = min(iters, max(1, math.ceil(len(ds) / ds.batch_size)))
    for _ in range(n):
        x, y = ds.get_batch(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y, ignore_index=-100)
        else:
            _, loss = model(x, y, ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def retention_guard(base_loss: float, baseline: float,
                    hard_stop_factor: float) -> Tuple[bool, bool]:
    """Returns (hard_stop, best_eligible)."""
    hard_stop = base_loss > baseline * hard_stop_factor
    best_eligible = base_loss <= baseline * hard_stop_factor
    return hard_stop, best_eligible


def save_checkpoint(out_dir: str, kind: str, model: GPTModel, optimizer,
                    scaler, step: int, supervised_tokens_seen: int,
                    total_tokens_seen: int, best_sft_val_loss: float,
                    base_baseline_loss: float, base_current_loss: float,
                    cfg_dict: dict, args, train_ds: SFTDataset, val_ds: SFTDataset,
                    rng_state: dict, source_meta: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{kind}.pt")
    payload = {
        "step": step,
        "supervised_tokens_seen": supervised_tokens_seen,
        "total_tokens_seen": total_tokens_seen,
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
        if "source_meta" in ckpt:
            source_meta.update(ckpt["source_meta"])
        print(f"resumed from {ckpt_path} at step {step}")
    else:
        for fname in (CHECKPOINT_LATEST, CHECKPOINT_BEST):
            if os.path.exists(os.path.join(args.out_dir, fname)):
                raise SystemExit(
                    f"[FATAL] {os.path.join(args.out_dir, fname)} exists. Use --init-from resume "
                    f"or move the directory aside (never overwrite an existing chat run).")
        # Load ONLY the base model weights; new optimizer/scaler/scheduler state.
        base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(base_ckpt["model_state"])
        print(f"base model weights loaded from {args.base_checkpoint} "
              f"(base step {base_ckpt.get('step')}, best val {base_ckpt.get('best_val_loss')})")
        # chat-format baselines BEFORE any weight modification
        if base_baseline_loss is None:
            base_baseline_loss = estimate_loss(model, base_val_ds, 50, device, use_amp, args.seed + 1000)
            print(f"[baseline] base-language validation loss: {base_baseline_loss:.4f} (retention baseline)")
        sft_val_baseline = eval_sft_loss(model, val_ds, 50, device, use_amp)
        print(f"[baseline] SFT val assistant-only loss (untrained chat format): {sft_val_baseline:.4f}")
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_baseline_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta)
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

    def log_metrics(sft_val: Optional[float] = None, base_val: Optional[float] = None):
        dt = time.time() - t0
        sps = (supervised_tokens_seen - run_start_supervised) / max(dt, 1e-9)
        peak = 0.0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"step {step}/{max_iters}  sft_loss {running_loss:.4f}  lr {lr:.2e}  "
              f"sup_tokens {supervised_tokens_seen:,}  ({sps:,.0f} sup_tok/s)"
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
            "peak_vram_gb": round(float(peak), 4),
            "best_sft_val_loss": float(best_sft_val_loss) if best_sft_val_loss != float("inf") else None,
        })
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def evaluate(save: bool = True) -> None:
        nonlocal best_sft_val_loss, base_current_loss, no_improve_evals
        sft_val = eval_sft_loss(model, val_ds, args.eval_iters, device, use_amp)
        base_val = estimate_loss(model, base_val_ds, 50, device, use_amp, val_seed)
        base_current_loss = base_val
        hard_stop, eligible = retention_guard(base_val, base_baseline_loss, args.hard_stop_factor)
        improved = sft_val < best_sft_val_loss
        print(f"  [eval] step {step}: sft_val {sft_val:.4f} | base_val {base_val:.4f} "
              f"(baseline {base_baseline_loss:.4f}) | eligible {eligible}")
        if improved:
            no_improve_evals = 0
            best_sft_val_loss = sft_val
            if eligible and save:
                save_checkpoint(args.out_dir, "best", model, optimizer, scaler, step,
                                supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                                base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                                train_ds, val_ds, collect_rng_state(device), source_meta)
                print(f"    [eval] new best SFT val loss {sft_val:.4f} -> best.pt (retention OK)")
        else:
            no_improve_evals += 1
        if save:
            save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                            supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                            base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                            train_ds, val_ds, collect_rng_state(device), source_meta)
        return hard_stop, sft_val, base_val

    try:
        while step < max_iters:
            if step % steps_per_epoch == 0:
                train_ds.reset_epoch()
            optimizer.zero_grad(set_to_none=True)
            micro_loss_sum = 0.0
            for mb in range(args.grad_accum):
                x, y = train_ds.get_batch(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, loss = model(x, y, ignore_index=-100)
                else:
                    _, loss = model(x, y, ignore_index=-100)
                if not torch.isfinite(loss):
                    save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                                    supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                                    base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                                    train_ds, val_ds, collect_rng_state(device), source_meta)
                    raise SystemExit(
                        f"[FATAL] non-finite SFT loss ({loss.item()}) at step {step} micro-batch {mb}. "
                        f"Latest healthy checkpoint preserved. Stopping.")
                micro_loss_sum += loss.item()
                scaled = scaler.scale(loss / args.grad_accum)
                scaled.backward()

            micro_loss = micro_loss_sum / args.grad_accum

            if skip_nonfinite_grad_step(model, optimizer, scaler, use_amp, step,
                                        total_tokens_seen, metrics_path):
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

            supervised_tokens_seen += int(y[y != -100].numel())
            total_tokens_seen += x.numel()
            step += 1
            running_loss = micro_loss if running_loss is None else 0.9 * running_loss + 0.1 * micro_loss

            if step % args.eval_interval == 0:
                hard_stop, sft_val, base_val = evaluate()
                log_metrics(sft_val, base_val)
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
                    print(f"  [PILOT] healthy at step {args.pilot_steps}: loss finite, CUDA healthy. Continuing.")

            if step % args.log_interval == 0 and step % args.eval_interval != 0:
                log_metrics()

        if step > 0 and not stopped:
            hard_stop, sft_val, base_val = evaluate()
            log_metrics(sft_val, base_val)
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta)
        print(f"training finished at step {step}; final latest checkpoint saved")
        if stopped:
            print("NOTE: run stopped early (guard triggered); see report.")
    except KeyboardInterrupt:
        print("\n[interrupt] KeyboardInterrupt received; preserving resumable state...")
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        supervised_tokens_seen, total_tokens_seen, best_sft_val_loss,
                        base_baseline_loss, base_current_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device), source_meta)
        print(f"[interrupt] latest checkpoint saved to {os.path.join(args.out_dir, CHECKPOINT_LATEST)} "
              f"(step={step}); resume with --init-from resume")


if __name__ == "__main__":
    main()