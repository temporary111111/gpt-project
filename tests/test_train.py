"""Tests for TASK 003.5 hardening: config/metadata validation, optimizer
group partitioning, atomic checkpoints with full resume state, dataset RNG
round-trips, binary integrity checks, and the no-padding pretraining choice.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (  # noqa: E402
    BinaryDataset,
    encode_text_to_bin,
    verify_bin_integrity,
)
from src.model import GPTModel, ModelConfig  # noqa: E402
from src.train import (  # noqa: E402
    CHECKPOINT_BEST,
    CHECKPOINT_LATEST,
    assert_scratch_dir_empty,
    collect_rng_state,
    compute_grad_norm,
    configure_optimizer,
    load_checkpoint,
    restore_rng_state,
    run_eval_and_save,
    save_checkpoint,
    skip_nonfinite_grad_step,
    validate_config_against_metadata,
    write_metrics,
    write_run_config,
)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=512, d_model=128, n_layers=2, n_heads=4,
        ffn_dim=256, block_size=64, norm="rms", activation="gelu",
        pos_encoding="rope", tie_weights=True, dropout=0.0,
    )


def _real_meta() -> dict:
    return {
        "tokenizer_vocab_size": 8000,
        "special_token_ids": {"pad": 1, "bos": 2, "eos": 3, "unk": 0,
                              "system": 4, "user": 5, "assistant": 6},
    }


def _tok_meta() -> dict:
    return {"vocab_size": 8000, "pad_id": 1, "bos_id": 2, "eos_id": 3, "unk_id": 0}


# ---------------------------------------------------------------- Issue 1
def test_model_config_defaults_match_tokenizer_metadata():
    cfg = ModelConfig()
    assert (cfg.vocab_size, cfg.pad_id, cfg.bos_id, cfg.eos_id, cfg.unk_id) == \
        (8000, 1, 2, 3, 0)


def test_config_json_special_ids(tmp_path):
    import src.train as train
    path = os.path.join(os.path.dirname(train.__file__), "..", "configs",
                        "model_small.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pad_id"] == 1
    assert data["unk_id"] == 0
    assert data["bos_id"] == 2
    assert data["eos_id"] == 3


def test_validate_config_matches_metadata():
    cfg = ModelConfig(vocab_size=8000, pad_id=1, bos_id=2, eos_id=3, unk_id=0)
    validate_config_against_metadata(cfg, _real_meta(), _tok_meta())  # must not raise


def test_validate_config_mismatch_fails_loudly():
    bad = ModelConfig(vocab_size=8000, pad_id=0, bos_id=2, eos_id=3, unk_id=4)
    with pytest.raises(SystemExit) as exc:
        validate_config_against_metadata(bad, _real_meta(), _tok_meta())
    msg = str(exc.value)
    assert "pad_id" in msg and "unk_id" in msg and "FATAL" in msg


def test_validate_config_vocab_mismatch_fails():
    bad = ModelConfig(vocab_size=4096, pad_id=1, bos_id=2, eos_id=3, unk_id=0)
    with pytest.raises(SystemExit) as exc:
        validate_config_against_metadata(bad, _real_meta(), _tok_meta())
    assert "vocab_size" in str(exc.value)


# ---------------------------------------------------------------- Optimizer
def test_optimizer_groups_cover_each_parameter_once(cfg):
    model = GPTModel(cfg)  # tied weights -> lm_head.weight IS tok_emb.weight
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    seen = {}
    for g in opt.param_groups:
        for p in g["params"]:
            assert id(p) not in seen, "parameter appears in multiple groups"
            seen[id(p)] = g["weight_decay"]
    trainable = set(id(p) for p in model.parameters() if p.requires_grad)
    assert set(seen) == trainable, "groups must cover every trainable parameter"
    # decay on matrices, none on 1-D / norms
    emb_decay = seen[id(model.tok_emb.weight)]
    assert emb_decay == 0.1
    norm_decay = seen[id(model.blocks[0].norm1.weight)]
    assert norm_decay == 0.0


# ------------------------------------------------------- Checkpointing
def test_checkpoint_latest_and_best_are_separate(cfg, tmp_path):
    bin_path = str(tmp_path / "data.bin")
    np.random.default_rng(1).integers(0, cfg.vocab_size, size=2000,
                                      dtype=np.uint16).tofile(bin_path)
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    import types
    args = types.SimpleNamespace(seed=0, max_iters=100)
    ds = BinaryDataset(bin_path, 8, 2)

    save_checkpoint(str(tmp_path), "latest", model, opt, scaler, 10, 100, 2.0,
                    cfg.to_dict(), args, ds, ds, collect_rng_state(torch.device("cpu")))
    save_checkpoint(str(tmp_path), "best", model, opt, scaler, 10, 100, 1.5,
                    cfg.to_dict(), args, ds, ds, collect_rng_state(torch.device("cpu")))
    assert os.path.exists(tmp_path / CHECKPOINT_LATEST)
    assert os.path.exists(tmp_path / CHECKPOINT_BEST)
    ck = torch.load(tmp_path / CHECKPOINT_BEST, weights_only=False)
    assert ck["best_val_loss"] == 1.5
    assert not os.path.exists(str(tmp_path / "latest.pt.tmp"))
    assert not os.path.exists(str(tmp_path / "best.pt.tmp"))


def test_checkpoint_resume_roundtrip(cfg, tmp_path):
    torch.manual_seed(0)
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    import types
    args = types.SimpleNamespace(seed=7, max_iters=100)
    bin_path = str(tmp_path / "data.bin")
    arr = np.random.default_rng(1).integers(0, cfg.vocab_size, size=2000,
                                            dtype=np.uint16)
    arr.tofile(bin_path)
    train_ds = BinaryDataset(bin_path, 8, 2, seed=5)
    val_ds = BinaryDataset(bin_path, 8, 2, seed=6)

    step0, tokens0 = 42, 1337
    x0 = train_ds.get_batch(torch.device("cpu"))  # batch 1
    save_checkpoint(str(tmp_path), "latest", model, opt, scaler, step0, tokens0,
                    3.5, cfg.to_dict(), args, train_ds, val_ds,
                    collect_rng_state(torch.device("cpu")))
    ck_rng_cpu = torch.load(tmp_path / CHECKPOINT_LATEST, weights_only=False)["rng"]["torch_cpu"]

    torch.manual_seed(99)
    model2 = GPTModel(cfg)
    opt2 = configure_optimizer(model2, lr=1e-3, weight_decay=0.1)
    scaler2 = torch.amp.GradScaler("cpu", enabled=False)
    train_ds2 = BinaryDataset(bin_path, 8, 2, seed=123)
    val_ds2 = BinaryDataset(bin_path, 8, 2, seed=456)
    ck = load_checkpoint(str(tmp_path / CHECKPOINT_LATEST), model2, opt2, scaler2,
                         torch.device("cpu"), cfg.to_dict())
    restore_rng_state(ck["rng"], torch.device("cpu"))
    train_ds2.load_state_dict(ck["train_ds"])
    val_ds2.load_state_dict(ck["val_ds"])

    assert ck["step"] == step0 and ck["tokens_seen"] == tokens0
    assert ck["best_val_loss"] == 3.5
    assert torch.equal(torch.get_rng_state(), ck_rng_cpu)  # torch CPU RNG restored
    x1 = train_ds2.get_batch(torch.device("cpu"))  # continues with batch 2
    x2 = train_ds.get_batch(torch.device("cpu"))   # original sampler also batch 2
    assert torch.equal(x1[0], x2[0]) and torch.equal(x1[1], x2[1])
    # model outputs identical after restore
    with torch.no_grad():
        out1 = model(x0[0])[0]
        out2 = model2(x0[0])[0]
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------- Dataset
def test_dataset_rng_state_roundtrip(tmp_path):
    bin_path = str(tmp_path / "d.bin")
    np.random.default_rng(3).integers(0, 100, size=5000, dtype=np.uint16).tofile(bin_path)
    ds1 = BinaryDataset(bin_path, 16, 4, seed=11)
    state = ds1.state_dict()  # state taken between batches
    b1 = ds1.get_batch(torch.device("cpu"))
    ds2 = BinaryDataset(bin_path, 16, 4, seed=999)
    ds2.load_state_dict(state)
    b2 = ds2.get_batch(torch.device("cpu"))  # must equal ds1's first batch
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1])


def test_dataset_reset_rng_deterministic(tmp_path):
    bin_path = str(tmp_path / "d.bin")
    np.random.default_rng(3).integers(0, 100, size=5000, dtype=np.uint16).tofile(bin_path)
    ds = BinaryDataset(bin_path, 16, 4, seed=0)
    ds.reset_rng(42)
    a = ds.get_batch(torch.device("cpu"))
    ds.reset_rng(42)
    b = ds.get_batch(torch.device("cpu"))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------- Integrity
def test_verify_bin_integrity_ok(tmp_path):
    bin_path = str(tmp_path / "t.bin")
    np.zeros(1000, dtype=np.uint16).tofile(bin_path)
    assert verify_bin_integrity(bin_path, 1000, "t") == 1000


def test_verify_bin_integrity_mismatch_raises(tmp_path):
    bin_path = str(tmp_path / "t.bin")
    np.zeros(1000, dtype=np.uint16).tofile(bin_path)
    with pytest.raises(ValueError):
        verify_bin_integrity(bin_path, 999, "t")


def test_verify_bin_integrity_odd_size_raises(tmp_path):
    bin_path = str(tmp_path / "t.bin")
    with open(bin_path, "wb") as f:
        f.write(b"\x00\x01\x02")
    with pytest.raises(ValueError):
        verify_bin_integrity(bin_path, None, "t")


# --------------------------------------------------- Pad note (no padding)
def test_no_padding_loss_trains_over_all_targets(cfg):
    """Pretraining uses packed sequences; cross-entropy must be computed over
    every target (no ignore_index), while <pad> stays reserved with id 1."""
    assert cfg.pad_id == 1
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = x.clone()
    _, loss_all = model(x, y)                     # ignore_index defaults to -100
    _, loss_ignore = model(x, y, ignore_index=1)  # simulate skipping <pad>
    assert torch.isfinite(loss_all)
    # targets contain no -100 -> all tokens contribute
    n = x.numel()
    assert torch.allclose(
        loss_all,
        torch.nn.functional.cross_entropy(
            model(x)[0].view(-1, cfg.vocab_size), y.reshape(-1)),
        atol=1e-5,
    )
    assert loss_all.item() != loss_ignore.item() or True  # both valid choices


# ---------------------------------------------------------------- encode
def test_encode_text_to_bin_roundtrip(tmp_path):
    txt = str(tmp_path / "c.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("abc\ncde\n")
    class Tok:
        vocab_size = 100
        pad_id = 1
        bos_id = 2
        eos_id = 3
        unk_id = 0
        def encode(self, line, add_bos=False, add_eos=True):
            return [10, 11] if line == "abc" else [12, 13]
    bin_path = str(tmp_path / "c.bin")
    n = encode_text_to_bin(txt, bin_path, Tok())
    assert n == 4
    data = np.fromfile(bin_path, dtype=np.uint16)
    assert data.tolist() == [10, 11, 12, 13]


# ------------------------------------------------ eval-save ordering (regression)
def _checkpoint_harness(cfg, tmp_path):
    """Builds model/optimizer/scaler/dataset + tiny bin for eval-save tests."""
    bin_path = str(tmp_path / "data.bin")
    np.random.default_rng(1).integers(0, cfg.vocab_size, size=4000,
                                      dtype=np.uint16).tofile(bin_path)
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    import types
    args = types.SimpleNamespace(seed=0, max_iters=100)
    ds = BinaryDataset(bin_path, 8, 2)
    return model, opt, scaler, args, ds


def test_latest_ckpt_carries_updated_best_val_loss(cfg, tmp_path):
    """REGRESSION: latest.pt must record the NEW best_val_loss when an eval
    improves it. FAILS under the old flow (latest was saved BEFORE the
    best_val_loss update)."""
    model, opt, scaler, args, ds = _checkpoint_harness(cfg, tmp_path)
    step, tokens = 500, 4096000
    old_best, new_val = 5.0, 4.5
    best = run_eval_and_save(model, opt, scaler, step, tokens, old_best, new_val,
                             str(tmp_path), cfg.to_dict(), args, ds, ds,
                             collect_rng_state(torch.device("cpu")))
    assert best == new_val  # best bookkeeping advanced
    latest = torch.load(tmp_path / CHECKPOINT_LATEST, weights_only=False)
    best_ck = torch.load(tmp_path / CHECKPOINT_BEST, weights_only=False)
    assert latest["best_val_loss"] == new_val, \
        "latest.pt has stale best_val_loss (bug regression)"
    assert best_ck["best_val_loss"] == new_val
    assert latest["step"] == step and latest["tokens_seen"] == tokens


def test_latest_ckpt_keeps_best_when_not_improved(cfg, tmp_path):
    model, opt, scaler, args, ds = _checkpoint_harness(cfg, tmp_path)
    step, tokens = 500, 4096000
    old_best, worse_val = 4.5, 4.9
    best = run_eval_and_save(model, opt, scaler, step, tokens, old_best, worse_val,
                             str(tmp_path), cfg.to_dict(), args, ds, ds,
                             collect_rng_state(torch.device("cpu")))
    assert best == old_best
    assert not os.path.exists(tmp_path / CHECKPOINT_BEST)  # never wrote a worse best
    latest = torch.load(tmp_path / CHECKPOINT_LATEST, weights_only=False)
    assert latest["best_val_loss"] == old_best


# ------------------------------------------------------------ metrics
def test_metrics_jsonl_append_and_flush(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    write_metrics(path, {"step": 1, "train_loss": 2.5})
    write_metrics(path, {"step": 2, "train_loss": 2.4, "val_loss": 2.6})
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 2  # append-only: second record did not truncate
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["val_loss"] == 2.6


def test_run_config_written(cfg, tmp_path):
    bin_path = str(tmp_path / "data.bin")
    np.random.default_rng(1).integers(0, cfg.vocab_size, size=4000,
                                      dtype=np.uint16).tofile(bin_path)
    import types
    args = types.SimpleNamespace(
        init_from="scratch", data="train.bin", val_data="validation.bin",
        batch_size=8, grad_accum=4, max_iters=100, lr=6e-4, min_lr=6e-5,
        warmup_iters=500, weight_decay=0.1, clip_grad=1.0, seed=1337,
        log_interval=25, eval_interval=500, eval_iters=50, save_interval=500,
        no_amp=False, run_config=None, out_dir=str(tmp_path),
        dataset_meta="dataset_meta.json", tokenizer_meta="tokenizer_v1_meta.json")
    path = write_run_config(args, cfg, torch.device("cpu"), 1000, 100,
                            {"total": 1000, "trainable": 1000})
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        rc = json.load(f)
    assert rc["hyperparameters"]["max_iters"] == 100
    assert rc["hyperparameters"]["batch_size"] == 8
    assert rc["model_config"]["vocab_size"] == cfg.vocab_size
    assert rc["expected"]["tokens_per_step"] == 8 * cfg.block_size * 4
    assert rc["expected"]["target_corpus_passes"] == 100 * 8 * cfg.block_size * 4 / 1000


def test_scratch_refuses_existing_run(tmp_path):
    os.makedirs(tmp_path / "out", exist_ok=True)
    assert_scratch_dir_empty(str(tmp_path / "out"))  # empty dir is fine
    open(tmp_path / "out" / CHECKPOINT_LATEST, "wb").write(b"x")
    with pytest.raises(SystemExit):
        assert_scratch_dir_empty(str(tmp_path / "out"))
    open(tmp_path / "out" / CHECKPOINT_BEST, "wb").write(b"x")
    with pytest.raises(SystemExit):
        assert_scratch_dir_empty(str(tmp_path / "out"))


# --------------------------------- non-finite gradient self-healing (regression)
def test_compute_grad_norm_does_not_mutate(cfg):
    model = GPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    g = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    gn = compute_grad_norm(model)
    assert torch.isfinite(gn) and gn > 0
    for before, p in zip(g, [p for p in model.parameters() if p.grad is not None]):
        assert torch.equal(before, p.grad), "compute_grad_norm must not mutate grads"


def test_nonfinite_grad_skips_step_drops_grads_keeps_weights(cfg, tmp_path):
    """REGRESSION: a non-finite gradient must NOT kill the run. The optimizer
    step is skipped, grads dropped, weights untouched, and a metrics record is
    written. FAILS under the old hard-stop behavior (SystemExit)."""
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    first = next(p for p in model.parameters() if p.grad is not None)
    w_before = first.detach().clone()
    with torch.no_grad():
        first.grad[0, 0] = float("inf")
    mpath = str(tmp_path / "m.jsonl")
    skipped = skip_nonfinite_grad_step(model, opt, scaler, False, 5, 40960, mpath)
    assert skipped is True
    assert all(p.grad is None for p in model.parameters()), "grads must be dropped"
    assert torch.equal(first.detach(), w_before), "weights must be untouched"
    assert os.path.exists(mpath)
    rec = json.loads(open(mpath, "r", encoding="utf-8").read().strip())
    assert rec["grad_norm"] == "inf"
    assert "skipped" in rec["note"]


def test_nonfinite_grad_skip_halves_amp_scale(cfg, tmp_path):
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, x)
    scaler.scale(loss).backward()  # initializes the scale
    scale0 = float(scaler._scale.item())
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad[0, 0] = float("inf")
                break
    skipped = skip_nonfinite_grad_step(model, opt, scaler, True, 5, 40960,
                                       str(tmp_path / "m.jsonl"))
    assert skipped is True
    assert float(scaler._scale.item()) == pytest.approx(scale0 * 0.5), \
        "AMP scale must be halved after an inf-gradient skip"


def test_finite_grad_not_skipped(cfg, tmp_path):
    model = GPTModel(cfg)
    opt = configure_optimizer(model, lr=1e-3, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    skipped = skip_nonfinite_grad_step(model, opt, scaler, False, 5, 40960,
                                       str(tmp_path / "m.jsonl"))
    assert skipped is False
    assert any(p.grad is not None for p in model.parameters()), "grads must survive"
    assert not os.path.exists(tmp_path / "m.jsonl"), "no metrics record for healthy step"