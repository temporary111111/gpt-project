"""GPT-style decoder-only Transformer built from scratch with PyTorch primitives.

No Hugging Face model implementations are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


@dataclass
class ModelConfig:
    """Configuration for the decoder-only GPT-style model."""
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 2048
    block_size: int = 256
    norm: str = "rms"                 # "rms" | "layer"
    activation: str = "gelu"          # "gelu" | "relu" | "silu"
    pos_encoding: str = "rope"        # "rope" | "learned"
    tie_weights: bool = True
    dropout: float = 0.0
    init_std: float = 0.02
    rope_theta: float = 10000.0
    bos_id: int = 2
    eos_id: int = 3
    pad_id: int = 1
    unk_id: int = 0

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Root mean square layer normalization (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class FeedForward(nn.Module):
    """Position-wise feed-forward network with configurable activation."""

    def __init__(self, d_model: int, ffn_dim: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.act = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[activation]
        self.down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.gate(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm -> attention -> residual, norm -> ffn -> residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.norm == "rms":
            self.norm1 = RMSNorm(cfg.d_model)
            self.norm2 = RMSNorm(cfg.d_model)
        elif cfg.norm == "layer":
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
        else:
            raise ValueError(f"unknown norm: {cfg.norm}")

        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            use_rope=(cfg.pos_encoding == "rope"),
            rope_theta=cfg.rope_theta,
        )
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.activation, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """Decoder-only causal language model with random initialization."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        elif cfg.pos_encoding == "rope":
            self.pos_emb = None
        else:
            raise ValueError(f"unknown pos_encoding: {cfg.pos_encoding}")

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        if cfg.norm == "rms":
            self.ln_f = RMSNorm(cfg.d_model)
        else:
            self.ln_f = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # Scale residual projections (GPT-2 style) for stable training at depth
        residual_layers = []
        for block in self.blocks:
            residual_layers.append(block.attn.out_proj)
            residual_layers.append(block.ffn.down)
        scale = 1.0 / (2.0 * self.cfg.n_layers) ** 0.5
        for layer in residual_layers:
            with torch.no_grad():
                layer.weight.mul_(scale)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                ignore_index: int | None = None):
        """idx/targets: (B, T) long tensors. Returns (logits, loss_or_None)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(positions)

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=ignore_index if ignore_index is not None else -100,
            )
        return logits, loss

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def estimate_vram_bytes(self, batch_size: int = 8) -> float:
        """Rough fp32-training VRAM estimate in MiB (weights + grads + AdamW states)."""
        params = self.count_parameters()["total"]
        # fp32 weights (4B) + fp32 grads (4B) + AdamW fp32 moments (8B)
        per_param = 16.0
        # rough activation estimate: per-token per-layer, dominated by attention scores
        cfg = self.cfg
        attn_acts = batch_size * cfg.n_heads * cfg.block_size * cfg.block_size * 4.0
        acts = (attn_acts + batch_size * cfg.block_size * cfg.d_model * 8.0) * cfg.n_layers
        total = params * per_param + acts
        return total / (1024.0 ** 2)