"""Multi-head causal self-attention implemented from scratch with PyTorch primitives.

Supports rotary positional embeddings (RoPE) applied to queries and keys.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension (RoPE helper)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies precomputed rotary embeddings to x: (B, T, H, D)."""
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Scaled dot-product causal self-attention with optional RoPE."""

    def __init__(self, d_model: int, n_heads: int, block_size: int,
                 dropout: float = 0.0, use_rope: bool = True, rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if use_rope:
            self._build_rope_cache(rope_theta)

        # Causal mask: (block_size, block_size), upper triangle -> True (masked)
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _build_rope_cache(self, theta: float):
        head_dim = self.head_dim
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(self.block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)            # (block_size, head_dim/2)
        # rotate_half() uses the "two halves" pairing convention
        # (first half is the real part, second half the imaginary part), so the
        # per-dimension angle must be duplicated as (freq_0, freq_1, ..., freq_0,
        # freq_1, ...). repeat_interleave would produce adjacent-pair angles and
        # silently rotate the wrong coordinates.
        freqs = torch.cat((freqs, freqs), dim=-1)   # (block_size, head_dim)
        freqs = freqs.unsqueeze(1)                  # (block_size, 1, head_dim)
        cos = freqs.cos()  # (T, 1, D) broadcasts over (B, T, H, D)
        sin = freqs.sin()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        if self.use_rope:
            cos = self.rope_cos[:T].unsqueeze(0)
            sin = self.rope_sin[:T].unsqueeze(0)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v.transpose(1, 2)  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))