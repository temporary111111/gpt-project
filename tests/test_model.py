"""Unit tests for the from-scratch model."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import GPTModel, ModelConfig  # noqa: E402


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=512, d_model=128, n_layers=2, n_heads=4,
        ffn_dim=256, block_size=64, norm="rms", activation="gelu",
        pos_encoding="rope", tie_weights=True, dropout=0.0,
    )


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
def test_forward_shape(cfg, pos_encoding):
    cfg.pos_encoding = pos_encoding
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 32))
    logits, loss = model(x)
    assert logits.shape == (3, 32, cfg.vocab_size)
    assert loss is None


def test_loss_is_finite(cfg):
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 64))
    y = torch.randint(0, cfg.vocab_size, (4, 64))
    logits, loss = model(x, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_and_step(cfg):
    model = GPTModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    _, loss = model(x, y)
    loss.backward()
    opt.step()


def test_causal_attention_masks_future(cfg):
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    # position 3 must not depend on position 0 information flows: hard to test directly;
    # instead verify the attention buffer is strictly upper-triangular
    mask = model.blocks[0].attn.causal_mask
    assert mask[0, 1:].all() and not mask[0, 0] and not mask[2, 0]


def test_tied_weights(cfg):
    model = GPTModel(cfg)
    if cfg.tie_weights:
        assert model.lm_head.weight is model.tok_emb.weight


def test_param_count(cfg):
    model = GPTModel(cfg)
    counts = model.count_parameters()
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]


def test_generation_greedy_deterministic(cfg):
    model = GPTModel(cfg)
    from src.generate import generate

    prompt = [1, 2, 3]
    a = generate(model, None, prompt, max_new_tokens=8, temperature=0.0, top_k=1, seed=0)
    b = generate(model, None, prompt, max_new_tokens=8, temperature=0.0, top_k=1, seed=0)
    assert a == b


def test_generation_respects_block_size(cfg):
    model = GPTModel(cfg)
    from src.generate import generate

    prompt = list(range(40))
    ids = generate(model, None, prompt, max_new_tokens=10, temperature=0.0, top_k=1)
    assert len(ids) == 50


def test_checkpoint_roundtrip(cfg, tmp_path):
    model = GPTModel(cfg)
    path = tmp_path / "ck.pt"
    torch.save({"config": cfg.to_dict(), "model_state": model.state_dict()}, path)
    cfg2 = ModelConfig.from_dict(torch.load(path, weights_only=False)["config"])
    model2 = GPTModel(cfg2)
    model2.load_state_dict(torch.load(path, weights_only=False)["model_state"])
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.equal(model(x)[0], model2(x)[0])


def test_sampling_functions(cfg):
    from src.generate import sample_next

    torch.manual_seed(0)
    logits = torch.randn(100)
    g = sample_next(logits.clone(), temperature=0.0, top_k=1)
    assert g == int(logits.argmax())
    for _ in range(20):
        s = sample_next(logits.clone(), temperature=1.0, top_k=10)
        assert 0 <= s < 100
    for _ in range(20):
        s = sample_next(logits.clone(), temperature=0.9, top_p=0.9)
        assert 0 <= s < 100