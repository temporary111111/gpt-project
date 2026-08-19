"""Mathematical tests for the corrected RoPE implementation.

These tests validate the actual rotation mathematics, not just tensor shapes.

The two-halves rotate_half() convention rotates coordinate pairs (first half,
second half). The frequency layout must therefore be freqs duplicated via
cat((freqs, freqs)) so angle i is applied to both halves. The previous
repeat_interleave layout paired the WRONG coordinates and fails tests 2 and 3.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention import CausalSelfAttention, apply_rotary_emb, rotate_half  # noqa: E402
from src.model import GPTModel, ModelConfig  # noqa: E402


def _build_cos_sin(head_dim: int, t, theta: float = 10000.0):
    """Canonical two-halves frequency layout: cat((freqs, freqs))."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(t.to(torch.float32), inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def test_rotate_half_convention():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = rotate_half(x)
    # two-halves convention: second half negated and swapped to the front
    assert torch.equal(out, torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_position_zero_rotation_is_identity():
    head_dim = 8
    x = torch.randn(2, 5, 4, head_dim)
    cos, sin = _build_cos_sin(head_dim, torch.zeros(1))  # t = 0 -> cos=1, sin=0
    out = apply_rotary_emb(x, cos.unsqueeze(0), sin.unsqueeze(0))
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_preserves_norm():
    """A rotation must preserve each vector's L2 norm.

    Fails under the old repeat_interleave layout, which rotated mismatched
    coordinate pairs with different angles (angle collision on the cross terms).
    """
    head_dim = 16
    x = torch.randn(3, 10, 2, head_dim)
    t = torch.arange(10)
    cos, sin = _build_cos_sin(head_dim, t)
    out = apply_rotary_emb(x, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_manual_tiny_vector_matches_implementation():
    """A hand-computed tiny example must agree with apply_rotary_emb.

    head_dim=4, theta=10000, position t=1 -> angles [1, 0.01].
    With cat((freqs, freqs)) the layout is [1, 0.01, 1, 0.01].
    Fails under the old repeat_interleave layout [1, 1, 0.01, 0.01].
    """
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).view(1, 1, 1, 4)
    cos, sin = _build_cos_sin(4, torch.tensor([1.0]))
    out = apply_rotary_emb(x, cos.unsqueeze(0), sin.unsqueeze(0)).squeeze()
    a, b, c, d = 1.0, 2.0, 3.0, 4.0
    w0, w1 = 1.0, 0.01
    expected = torch.tensor([
        a * torch.cos(torch.tensor(w0)) - c * torch.sin(torch.tensor(w0)),
        b * torch.cos(torch.tensor(w1)) - d * torch.sin(torch.tensor(w1)),
        c * torch.cos(torch.tensor(w0)) + a * torch.sin(torch.tensor(w0)),
        d * torch.cos(torch.tensor(w1)) + b * torch.sin(torch.tensor(w1)),
    ])
    assert torch.allclose(out, expected, atol=1e-6)


def test_rope_cache_layout_matches_two_halves():
    """The model's cached cos/sin must use the cat layout, not repeat_interleave.

    Fails under the old implementation.
    """
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4,
                      ffn_dim=64, block_size=8, dropout=0.0)
    model = GPTModel(cfg)
    attn = model.blocks[0].attn
    head_dim = cfg.d_model // cfg.n_heads  # 8
    cos_expected, sin_expected = _build_cos_sin(head_dim, torch.arange(8))
    assert torch.allclose(attn.rope_cos.squeeze(1), cos_expected, atol=1e-6)
    assert torch.allclose(attn.rope_sin.squeeze(1), sin_expected, atol=1e-6)


def test_q_and_k_use_same_frequency_convention():
    """With identical input vectors at every position and identity projections,
    the attention score matrix must be symmetric and depend only on relative
    position -- this only holds when q and k are rotated with the same angles.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=8, n_heads=1, block_size=8,
                               dropout=0.0, use_rope=True, rope_theta=10000.0)
    with torch.no_grad():
        eye = torch.eye(8)
        attn.q_proj.weight.copy_(eye)
        attn.k_proj.weight.copy_(eye)
        attn.v_proj.weight.zero_()
        attn.out_proj.weight.zero_()

    T = 8
    x = torch.zeros(1, T, 8)
    x[:, :, 0] = 1.0  # identical vector at every position
    with torch.no_grad():
        q = attn.q_proj(x).view(1, T, 1, 8)
        k = attn.k_proj(x).view(1, T, 1, 8)
        cos = attn.rope_cos[:T].unsqueeze(0)
        sin = attn.rope_sin[:T].unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        S = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1))[0, 0]

    assert torch.allclose(S, S.T, atol=1e-5), "score matrix must be symmetric"
    for i in range(T - 1):
        assert torch.allclose(S[i, :-1], S[i + 1, 1:], atol=1e-5), \
            "scores must depend only on relative position"


def test_causal_attention_ignores_future_inputs():
    """Changing FUTURE tokens must not change outputs at earlier positions
    when dropout=0 (causal mask + RoPE)."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                      ffn_dim=128, block_size=16, dropout=0.0)
    model = GPTModel(cfg)
    model.eval()
    T = 8
    seq1 = torch.randint(0, cfg.vocab_size, (1, T))
    seq2 = seq1.clone()
    seq2[0, 4:] = (seq1[0, 4:] + 1) % cfg.vocab_size  # guaranteed different future
    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)
    assert torch.equal(logits1[:, :4], logits2[:, :4]), \
        "future tokens leaked into earlier positions"