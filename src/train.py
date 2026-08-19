"""Training skeleton for the from-scratch GPT model.

Features:
- cross-entropy next-token prediction
- AdamW + linear warmup + cosine decay
- gradient clipping, gradient accumulation
- mixed precision (fp16 autocast + GradScaler) on CUDA, fp32 fallback on CPU
- checkpoint save/load with full trainer state
- streaming memmap dataset (no full corpus in RAM)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .dataset import BinaryDataset
from .model import GPTModel, ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train the from-scratch GPT model")
    p.add_argument("--config", default="configs/model_small.json")
    p.add_argument("--data", required=True, help="tokenized .bin training file")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--init-from", choices=["scratch", "resume"], default="scratch")
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
    return p.parse_args()


def configure_optimizer(model: GPTModel, lr: float, weight_decay: float) -> AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
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
                  device: torch.device, use_amp: bool, ignore_index: Optional[int],
                  scaler) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = dataset.get_batch(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y, ignore_index=ignore_index)
        else:
            _, loss = model(x, y, ignore_index=ignore_index)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def save_checkpoint(path: str, model, optimizer, scaler, step: int,
                    best_val_loss: float, cfg_dict: dict, args, rng_state: dict):
    torch.save({
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": cfg_dict,
        "args": vars(args),
        "rng": rng_state,
    }, path)


def load_checkpoint(path: str, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np_random = __import__("numpy").random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    dtype = torch.float16 if use_amp else torch.float32
    print(f"device: {device}  |  mixed precision: {use_amp} ({dtype})")

    cfg = ModelConfig.from_json(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    model = GPTModel(cfg).to(device)
    n_params = model.count_parameters()
    print(f"model params: total={n_params['total']:,}  trainable={n_params['trainable']:,}")

    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    best_val_loss = float("inf")
    if args.init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {ckpt_path} at step {step}")

    train_ds = BinaryDataset(args.data, cfg.block_size, args.batch_size,
                             seed=args.seed, start_frac=0.0, end_frac=0.9)
    val_ds = BinaryDataset(args.data, cfg.block_size, args.batch_size,
                           seed=args.seed + 1, start_frac=0.9, end_frac=1.0)
    print(f"train tokens: {len(train_ds):,}  |  val tokens: {len(val_ds):,}")

    model.train()
    tokens_seen = 0
    t0 = time.time()

    while step < args.max_iters:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = train_ds.get_batch(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y, ignore_index=cfg.pad_id)
            else:
                _, loss = model(x, y, ignore_index=cfg.pad_id)
            micro_loss += loss.item() / args.grad_accum
            scaled = scaler.scale(loss / args.grad_accum)
            scaled.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
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

        if step % args.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-9)
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f"  |  VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            print(f"step {step}/{args.max_iters}  loss {micro_loss:.4f}  lr {lr:.2e}  "
                  f"{tps:,.0f} tok/s{gpu_mem}")

        if step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_ds, args.eval_iters, device, use_amp,
                                     cfg.pad_id, scaler)
            print(f"  [eval] val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(args.out_dir, "checkpoint.pt"), model,
                                optimizer, scaler, step, best_val_loss, cfg.to_dict(),
                                args, {"torch": torch.get_rng_state()})
                print(f"  [eval] saved best checkpoint at step {step}")

        if step % args.save_interval == 0:
            save_checkpoint(os.path.join(args.out_dir, "checkpoint.pt"), model,
                            optimizer, scaler, step, best_val_loss, cfg.to_dict(),
                            args, {"torch": torch.get_rng_state()})
            print(f"  [save] checkpoint saved at step {step}")

    save_checkpoint(os.path.join(args.out_dir, "checkpoint.pt"), model,
                    optimizer, scaler, step, best_val_loss, cfg.to_dict(),
                    args, {"torch": torch.get_rng_state()})
    print(f"training finished at step {step}; final checkpoint saved")


if __name__ == "__main__":
    main()