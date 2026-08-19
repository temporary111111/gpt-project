"""Text generation: greedy, temperature + top-k (and optional top-p) sampling."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import GPTModel, ModelConfig


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None) -> int:
    """Samples one token id from a (vocab,) logits vector.

    temperature <= 0 or top_k == 1 => greedy.
    """
    logits = logits.float()
    if temperature > 0.0 and temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.numel())
        top_vals, _ = torch.topk(logits, k)
        logits[logits < top_vals[-1]] = float("-inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum - sorted_probs <= top_p
        keep = torch.cat((keep[:1], keep[1:]))
        sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, 1).item()
        return int(sorted_idx[sampled].item())

    if temperature <= 0.0 or top_k == 1:
        return int(logits.argmax().item())

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(model: GPTModel, tokenizer, prompt_ids: List[int],
             max_new_tokens: int = 64, temperature: float = 0.8,
             top_k: Optional[int] = 40, top_p: Optional[float] = None,
             seed: Optional[int] = None, stop_at_eos: bool = True) -> List[int]:
    """Generates a continuation of prompt_ids. Returns the full sequence ids."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    ctx = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = ctx[-cfg.block_size:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_id = sample_next(logits[0, -1, :], temperature=temperature,
                              top_k=top_k, top_p=top_p)
        ctx.append(next_id)
        if stop_at_eos and next_id == cfg.eos_id:
            break
    return ctx


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained checkpoint")
    p.add_argument("--config", default="configs/model_small.json")
    p.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    p.add_argument("--tokenizer-model", default=None,
                   help="optional SentencePiece .model path for decode")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig.from_json(args.config)
    model = GPTModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    tokenizer = None
    if args.tokenizer_model:
        from .tokenizer import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer(args.tokenizer_model)
        prompt_ids = tokenizer.encode(args.prompt)
        special = tokenizer.special_token_ids
        cfg.bos_id = special["bos"]
        cfg.eos_id = special["eos"]
        cfg.pad_id = special["pad"]
    else:
        prompt_ids = [int(t) for t in args.prompt.split(",")] if args.prompt else []

    temperature = 0.0 if args.greedy else args.temperature
    top_k = 1 if args.greedy else args.top_k

    ids = generate(model, tokenizer, prompt_ids, args.max_new_tokens,
                   temperature=temperature, top_k=top_k, top_p=args.top_p,
                   seed=args.seed)
    text = tokenizer.decode(ids) if tokenizer else str(ids)
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()