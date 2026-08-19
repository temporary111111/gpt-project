"""Training loop for the from-scratch GPT model (TASK 003.5 hardened).

Features:
- cross-entropy next-token prediction over every target (packed corpus has no
  padding, so no ignore_index is used during pretraining; <pad> stays reserved
  with the correct id and is not used in the packed bins)
- separate --data (train.bin, used 100%) and --val-data (validation.bin)
- startup validation of special-token ids / vocab against dataset + tokenizer
  metadata (fails loudly on mismatch)
- binary size integrity checks against metadata (uint16, tokens * 2 bytes)
- AdamW with weight decay on matrices only, tied-embedding de-duplication,
  single-group membership assertion
- correct AMP flow: scale -> backward -> unscale_ -> clip -> step -> update
- non-finite LOSS guards stop safely and preserve the latest healthy checkpoint;
  rare non-finite GRADIENT events self-heal the standard AMP way: the optimizer
  step is skipped, the scale is halved, the grads dropped, and training continues
- deterministic validation batches (fixed seed reset per evaluation)
- full resume state (model, optimizer, scaler, step, tokens_seen, best val loss,
  Python/NumPy/torch CPU + CUDA RNG, train/val dataset samplers)
- atomic checkpoint writes (tmp + os.replace) to latest.pt / best.pt
- rich metrics: step, loss, val loss, lr, grad norm, tokens_seen, corpus passes,
  tokens/sec, peak VRAM
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
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .dataset import BinaryDataset, verify_bin_integrity
from .model import GPTModel, ModelConfig

DEFAULT_OUT_DIR = os.path.join("checkpoints", "pretrain")
DEFAULT_DATASET_META = os.path.join("data", "processed", "dataset_meta.json")
DEFAULT_TOKENIZER_META = os.path.join("data", "tokenizer", "tokenizer_v1_meta.json")
CHECKPOINT_LATEST = "latest.pt"
CHECKPOINT_BEST = "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Train the from-scratch GPT model")
    p.add_argument("--config", default="configs/model_small.json")
    p.add_argument("--data", required=True, help="tokenized train .bin file (used 100%)")
    p.add_argument("--val-data", required=True, help="tokenized validation .bin file (used 100%)")
    p.add_argument("--dataset-meta", default=DEFAULT_DATASET_META)
    p.add_argument("--tokenizer-meta", default=DEFAULT_TOKENIZER_META)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--init-from", choices=["scratch", "resume"], default="scratch")
    p.add_argument("--resume-path", default=None,
                   help="explicit resume checkpoint (default: <out-dir>/latest.pt)")
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--metrics-file", default=None,
                   help="append-only metrics log (default: <out-dir>/metrics.jsonl)")
    p.add_argument("--run-config", default=None,
                   help="JSON file with the exact run settings (default: <out-dir>/run_config.json)")
    return p.parse_args()


def load_metadata(path: str, required: bool) -> dict:
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"[FATAL] required metadata file missing: {path}")
        print(f"[warn] metadata file not found (skipping validation): {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config_against_metadata(cfg: ModelConfig,
                                    dataset_meta: dict,
                                    tokenizer_meta: dict) -> None:
    """Fails loudly if config special-token ids / vocab disagree with metadata."""
    errors: list = []

    def check(key: str, got, expected, source: str):
        if expected is None:
            return
        if got != expected:
            errors.append(
                f"  {key}: config={got}  vs  {source}={expected}"
            )

    if dataset_meta:
        dst = dataset_meta.get("special_token_ids", {})
        check("vocab_size", cfg.vocab_size,
              dataset_meta.get("tokenizer_vocab_size"), "dataset_meta.json")
        check("pad_id", cfg.pad_id, dst.get("pad"), "dataset_meta.json(special_token_ids.pad)")
        check("bos_id", cfg.bos_id, dst.get("bos"), "dataset_meta.json(special_token_ids.bos)")
        check("eos_id", cfg.eos_id, dst.get("eos"), "dataset_meta.json(special_token_ids.eos)")
        check("unk_id", cfg.unk_id, dst.get("unk"), "dataset_meta.json(special_token_ids.unk)")

    if tokenizer_meta:
        check("vocab_size", cfg.vocab_size,
              tokenizer_meta.get("vocab_size"), "tokenizer_v1_meta.json")
        check("pad_id", cfg.pad_id, tokenizer_meta.get("pad_id"), "tokenizer_v1_meta.json")
        check("bos_id", cfg.bos_id, tokenizer_meta.get("bos_id"), "tokenizer_v1_meta.json")
        check("eos_id", cfg.eos_id, tokenizer_meta.get("eos_id"), "tokenizer_v1_meta.json")
        check("unk_id", cfg.unk_id, tokenizer_meta.get("unk_id"), "tokenizer_v1_meta.json")

    if errors:
        raise SystemExit(
            "[FATAL] model config disagrees with corpus/tokenizer metadata:\n"
            + "\n".join(errors)
        )


def configure_optimizer(model: GPTModel, lr: float, weight_decay: float) -> AdamW:
    """AdamW with weight decay on matrix weights only.

    Tied lm_head/tok_emb share a single Parameter object: de-duplicate by id so
    the shared tensor is added to exactly one optimizer group.
    """
    decay: list = []
    no_decay: list = []
    seen: set = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue  # tied weights (e.g. lm_head.weight is tok_emb.weight)
        seen.add(id(param))
        if param.ndim < 2 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # Every unique trainable parameter must belong to exactly one group.
    all_params = set(id(p) for p in model.parameters() if p.requires_grad)
    grouped = set()
    for g in groups:
        for p in g["params"]:
            if id(p) in grouped:
                raise AssertionError(f"parameter {p.shape} appears in multiple optimizer groups")
            grouped.add(id(p))
    if grouped != all_params:
        missing = all_params - grouped
        raise AssertionError(
            f"optimizer groups do not cover all trainable parameters: missing {len(missing)}"
        )
    return AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step: int, max_iters: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step > max_iters:
        return min_lr
    ratio = (step - warmup) / max(1, max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPTModel, dataset: BinaryDataset, iters: int,
                  device: torch.device, use_amp: bool, seed: int) -> float:
    """Deterministic validation: the dataset RNG is reset to a fixed seed so
    every evaluation sees the same batches."""
    model.eval()
    dataset.reset_rng(seed)
    losses = []
    for _ in range(iters):
        x, y = dataset.get_batch(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def collect_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location moves tensors to `device`; the CPU RNG state must be a CPU
    # ByteTensor, so pin it back to CPU explicitly.
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu"))
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(s).to("cpu") for s in state["torch_cuda"]]
        )


def save_checkpoint(out_dir: str, kind: str, model: GPTModel, optimizer: AdamW,
                    scaler, step: int, tokens_seen: int, best_val_loss: float,
                    cfg_dict: dict, args, train_ds: BinaryDataset, val_ds: BinaryDataset,
                    rng_state: dict) -> str:
    """Atomically writes <out_dir>/<kind>.pt (kind in {latest, best})."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{kind}.pt")
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": cfg_dict,
        "args": vars(args),
        "train_ds": train_ds.state_dict(),
        "val_ds": val_ds.state_dict(),
        "rng": rng_state,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, model: GPTModel, optimizer: AdamW, scaler,
                    device: torch.device, expected_config: dict):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("config") != expected_config:
        raise SystemExit(
            f"[FATAL] checkpoint config does not match current config:\n"
            f"  checkpoint: {ckpt.get('config')}\n"
            f"  current:    {expected_config}"
        )
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def run_eval_and_save(model: GPTModel, optimizer: AdamW, scaler, step: int,
                      tokens_seen: int, best_val_loss: float, val_loss: float,
                      out_dir: str, cfg_dict: dict, args, train_ds: BinaryDataset,
                      val_ds: BinaryDataset, rng_state: dict) -> float:
    """Evaluates an already-computed val_loss against best_val_loss and saves
    checkpoints in the correct order.

    Correct logical flow (architect-mandated):
        1. if val_loss < best_val_loss: update best_val_loss and save best.pt
           using the UPDATED best_val_loss
        2. save latest.pt using the UPDATED best_val_loss

    latest.pt must ALWAYS know the true best validation loss observed up to
    that step, so a crash/resume can never regress best-model bookkeeping.

    Returns the (possibly updated) best_val_loss.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(out_dir, "best", model, optimizer, scaler, step,
                        tokens_seen, best_val_loss, cfg_dict, args,
                        train_ds, val_ds, rng_state)
        print(f"  [eval] new best val_loss {val_loss:.4f} -> {CHECKPOINT_BEST}")
    save_checkpoint(out_dir, "latest", model, optimizer, scaler, step,
                    tokens_seen, best_val_loss, cfg_dict, args,
                    train_ds, val_ds, rng_state)
    print(f"  [eval] latest checkpoint saved at step {step}")
    return best_val_loss


def write_metrics(path: str, record: dict) -> None:
    """Appends one JSON record to the metrics log and flushes it so progress
    survives an interrupted process."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def assert_scratch_dir_empty(out_dir: str) -> None:
    """Refuses to let --init-from scratch overwrite an existing production run."""
    for fname in (CHECKPOINT_LATEST, CHECKPOINT_BEST):
        existing = os.path.join(out_dir, fname)
        if os.path.exists(existing):
            raise SystemExit(
                f"[FATAL] {existing} already exists. Refusing to overwrite an "
                f"existing run with --init-from scratch. Inspect it; resume with "
                f"--init-from resume or move the directory aside."
            )


def compute_grad_norm(model: GPTModel) -> torch.Tensor:
    """Raw gradient L2 norm WITHOUT mutating grads (clip_grad_norm_ with an
    infinite max_norm would corrupt inf grads to NaN/0 and hide them from the
    GradScaler's inf detection, so the norm is computed by hand)."""
    norms = [p.grad.detach().norm() for p in model.parameters()
             if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.sqrt(sum(n * n for n in norms))


def skip_nonfinite_grad_step(model: GPTModel, optimizer: AdamW, scaler,
                             use_amp: bool, step: int, tokens_seen: int,
                             metrics_path: str) -> bool:
    """Standard AMP self-healing for rare non-finite gradient events.

    Unscales, computes the raw grad norm, and if it is non-finite: lets the
    GradScaler see the infs (it skips the optimizer step and marks found_inf),
    halves the scale via update(), drops the grads, logs a warning + metrics
    record, and returns True. The caller then continues training with the
    previous healthy weights. A non-finite LOSS remains a hard stop (the loss
    guard is separate); this path never corrupts weights.

    Returns True if the step was skipped, False otherwise.
    """
    if use_amp:
        scaler.unscale_(optimizer)
    grad_norm = compute_grad_norm(model)
    if not torch.isfinite(grad_norm):
        if use_amp:
            scaler.step(optimizer)  # sees infs -> skips optimizer.step
            scaler.update()         # halves the scale
        optimizer.zero_grad(set_to_none=True)
        print(f"  [warn] step {step}: non-finite grad norm ({grad_norm}); "
              f"skipping optimizer step, AMP scale halved, training continues")
        write_metrics(metrics_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "train_loss": None,
            "val_loss": None,
            "learning_rate": None,
            "grad_norm": "inf",
            "tokens_seen": int(tokens_seen),
            "corpus_passes": None,
            "tokens_per_second": None,
            "peak_vram_gb": None,
            "best_val_loss": None,
            "note": "non-finite gradient; optimizer step skipped, AMP scale halved",
        })
        return True
    return False


def write_run_config(args, cfg: ModelConfig, device: torch.device,
                     train_tokens: int, val_tokens: int, n_params: int) -> str:
    """Writes the exact production settings + model config to run_config.json."""
    path = args.run_config or os.path.join(args.out_dir, "run_config.json")
    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum
    record = {
        "task": "TASK 004 production pretraining",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "device": str(device),
        "mixed_precision": device.type == "cuda" and not args.no_amp,
        "hyperparameters": {
            "init_from": args.init_from,
            "data": args.data,
            "val_data": args.val_data,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "context_length": cfg.block_size,
            "max_iters": args.max_iters,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "seed": args.seed,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "save_interval": args.save_interval,
        },
        "model_config": cfg.to_dict(),
        "data": {
            "train_bin": args.data,
            "train_tokens": train_tokens,
            "val_bin": args.val_data,
            "val_tokens": val_tokens,
            "dataset_meta": args.dataset_meta,
            "tokenizer_meta": args.tokenizer_meta,
        },
        "expected": {
            "params_total": n_params["total"],
            "params_trainable": n_params["trainable"],
            "tokens_per_step": tokens_per_step,
            "target_tokens": args.max_iters * tokens_per_step,
            "target_corpus_passes": args.max_iters * tokens_per_step / max(1, train_tokens),
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"device: {device}  |  mixed precision: {use_amp} (float16)")

    cfg = ModelConfig.from_json(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- startup metadata validation (fails loudly on disagreement) ---
    dataset_meta = load_metadata(args.dataset_meta, required=True)
    tokenizer_meta = load_metadata(args.tokenizer_meta, required=True)
    validate_config_against_metadata(cfg, dataset_meta, tokenizer_meta)
    print(f"config validated against dataset_meta.json + tokenizer_v1_meta.json "
          f"(vocab={cfg.vocab_size} pad={cfg.pad_id} bos={cfg.bos_id} "
          f"eos={cfg.eos_id} unk={cfg.unk_id})")

    # --- binary integrity checks against metadata (tokens * 2 bytes = uint16) ---
    train_tokens_meta = (dataset_meta.get("splits", {}).get("train", {}) or {}).get("tokens")
    val_tokens_meta = (dataset_meta.get("splits", {}).get("val", {}) or {}).get("tokens")
    train_tokens = verify_bin_integrity(args.data, train_tokens_meta, name="train.bin")
    val_tokens = verify_bin_integrity(args.val_data, val_tokens_meta, name="validation.bin")
    print(f"train.bin integrity OK: {train_tokens:,} tokens ({train_tokens * 2:,} bytes)")
    print(f"validation.bin integrity OK: {val_tokens:,} tokens ({val_tokens * 2:,} bytes)")

    model = GPTModel(cfg).to(device)
    n_params = model.count_parameters()
    print(f"model params: total={n_params['total']:,}  trainable={n_params['trainable']:,}")

    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = BinaryDataset(args.data, cfg.block_size, args.batch_size,
                             seed=args.seed, start_frac=0.0, end_frac=1.0)
    val_ds = BinaryDataset(args.val_data, cfg.block_size, args.batch_size,
                           seed=args.seed + 1, start_frac=0.0, end_frac=1.0)

    step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    run_start_tokens = 0

    if args.init_from == "resume":
        ckpt_path = args.resume_path or os.path.join(args.out_dir, CHECKPOINT_LATEST)
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"[FATAL] resume checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scaler, device, cfg.to_dict())
        step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val_loss = ckpt["best_val_loss"]
        if "rng" in ckpt:
            restore_rng_state(ckpt["rng"], device)
        if "train_ds" in ckpt:
            train_ds.load_state_dict(ckpt["train_ds"])
        if "val_ds" in ckpt:
            val_ds.load_state_dict(ckpt["val_ds"])
        run_start_tokens = tokens_seen
        print(f"resumed from {ckpt_path} at step {step} (tokens_seen={tokens_seen:,})")
    else:
        # Never silently overwrite an existing production run.
        assert_scratch_dir_empty(args.out_dir)
        # Save an initial healthy latest checkpoint so a crash can always be
        # resumed from a valid state.
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        tokens_seen, best_val_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device))
        print(f"initial latest checkpoint written to {os.path.join(args.out_dir, CHECKPOINT_LATEST)}")
        write_run_config(args, cfg, device, train_tokens, val_tokens, n_params)

    model.train()
    t0 = time.time()
    running_loss = None
    val_seed = args.seed + 1000  # fixed seed so every eval is identical
    metrics_path = args.metrics_file or os.path.join(args.out_dir, "metrics.jsonl")

    def log_metrics(val_loss: Optional[float] = None):
        dt = time.time() - t0
        tps = (tokens_seen - run_start_tokens) / max(dt, 1e-9)
        passes = tokens_seen / max(1, train_tokens)
        peak = 0.0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  |  peak VRAM {peak:.2f} GB", end="")
        print(
            f"step {step}/{args.max_iters}  loss {running_loss:.4f}  "
            f"lr {lr:.2e}  grad_norm {grad_norm:.3f}  "
            f"tokens {tokens_seen:,} ({passes:.3f} passes)  "
            f"{tps:,.0f} tok/s"
        )
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "train_loss": round(float(running_loss), 6) if running_loss is not None else None,
            "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
            "learning_rate": float(lr),
            "grad_norm": float(grad_norm),
            "tokens_seen": int(tokens_seen),
            "corpus_passes": round(float(passes), 6),
            "tokens_per_second": round(float(tps), 2),
            "peak_vram_gb": round(float(peak), 4),
            "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        }
        write_metrics(metrics_path, record)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    try:
        while step < args.max_iters:
            optimizer.zero_grad(set_to_none=True)
            micro_loss_sum = 0.0
            for mb in range(args.grad_accum):
                x, y = train_ds.get_batch(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    # Preserve the latest healthy checkpoint (pre-step weights) and stop.
                    save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                                    tokens_seen, best_val_loss, cfg.to_dict(), args,
                                    train_ds, val_ds, collect_rng_state(device))
                    raise SystemExit(
                        f"[FATAL] non-finite loss ({loss.item()}) at optimizer step {step} "
                        f"micro-batch {mb}. Latest healthy checkpoint preserved. Stopping."
                    )
                micro_loss_sum += loss.item()
                scaled = scaler.scale(loss / args.grad_accum)
                scaled.backward()

            micro_loss = micro_loss_sum / args.grad_accum

            if skip_nonfinite_grad_step(model, optimizer, scaler, use_amp, step,
                                        tokens_seen, metrics_path):
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            lr = lr_at(step, args.max_iters, args.warmup_iters, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            tokens_seen += args.batch_size * cfg.block_size * args.grad_accum
            step += 1

            running_loss = micro_loss if running_loss is None else 0.9 * running_loss + 0.1 * micro_loss

            # Evaluation first so best_val_loss is updated BEFORE latest is
            # written; latest.pt must always carry the true best val loss.
            val_loss = None
            if step % args.eval_interval == 0:
                val_loss = estimate_loss(model, val_ds, args.eval_iters, device,
                                         use_amp, val_seed)
                print(f"  [eval] step {step} val_loss {val_loss:.4f}")
                best_val_loss = run_eval_and_save(
                    model, optimizer, scaler, step, tokens_seen, best_val_loss,
                    val_loss, args.out_dir, cfg.to_dict(), args,
                    train_ds, val_ds, collect_rng_state(device))

            if step % args.save_interval == 0 and step % args.eval_interval != 0:
                save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                                tokens_seen, best_val_loss, cfg.to_dict(), args,
                                train_ds, val_ds, collect_rng_state(device))
                print(f"  [save] latest checkpoint saved at step {step}")

            if step % args.log_interval == 0:
                log_metrics(val_loss=val_loss)

        # Final evaluation + final checkpoint at the end of training.
        if step > 0:
            val_loss = estimate_loss(model, val_ds, args.eval_iters, device,
                                     use_amp, val_seed)
            print(f"  [final eval] step {step} val_loss {val_loss:.4f}")
            best_val_loss = run_eval_and_save(
                model, optimizer, scaler, step, tokens_seen, best_val_loss,
                val_loss, args.out_dir, cfg.to_dict(), args,
                train_ds, val_ds, collect_rng_state(device))
            log_metrics(val_loss=val_loss)
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        tokens_seen, best_val_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device))
        print(f"training finished at step {step}; final latest checkpoint saved")
    except KeyboardInterrupt:
        # Graceful stop: preserve a fully resumable latest.pt and current
        # bookkeeping. Never touch best.pt (its writes are atomic anyway).
        print("\n[interrupt] KeyboardInterrupt received; preserving resumable state...")
        save_checkpoint(args.out_dir, "latest", model, optimizer, scaler, step,
                        tokens_seen, best_val_loss, cfg.to_dict(), args,
                        train_ds, val_ds, collect_rng_state(device))
        print(f"[interrupt] latest checkpoint saved to "
              f"{os.path.join(args.out_dir, CHECKPOINT_LATEST)} (step={step}, "
              f"tokens_seen={tokens_seen:,})")
        print("[interrupt] training can be resumed with: "
              f"--init-from resume --resume-path {os.path.join(args.out_dir, CHECKPOINT_LATEST)}")
        raise SystemExit(0)


if __name__ == "__main__":
    main()