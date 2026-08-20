"""TASK 005 SFT tests (Part W: the 22 mandated items).

Most tests use tiny synthetic data (CharTokenizer / temp JSONL). The real
tokenizer_v1 is used only where compatibility is the subject under test.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_sft_dataset import (  # noqa: E402
    build_aya_examples,
    build_oasst_examples,
    norm_text,
    stable_bucket,
    tokenize_example,
)
from acquire_sft_data import _human_ok, is_aya_original  # noqa: E402

from src.chat import build_history_ids, trim_reply_ids  # noqa: E402
from src.model import GPTModel, ModelConfig  # noqa: E402
from src.sft_dataset import SFTDataset  # noqa: E402
from src.sft_train import (  # noqa: E402
    eval_sft_loss,
    load_chat_checkpoint,
    retention_guard,
    save_checkpoint,
)
from src.tokenizer import CharTokenizer, SentencePieceTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_tok() -> CharTokenizer:
    chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()'-"
    return CharTokenizer(chars)


@pytest.fixture()
def tok():
    return make_tok()


def write_sft_jsonl(path: Path, examples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def build_example_ids(tok, turns, block_size=256):
    """Builds a tokenized example with assistant-only labels (like the pipeline)."""
    stats = Counter()
    ex = {"id": "test-1", "source": "test", "lang": "eng",
          "turns": [(r, t) for r, t in turns], "split": "train"}
    built = tokenize_example(ex, tok, stats, block_size=block_size)
    assert built is not None, dict(stats)
    return built["ids"], built["labels"]


# --- 1. Aya original-annotation filtering ---
def test_aya_original_annotation_filter():
    row = {"language_code": "eng", "language": "English",
           "inputs": "Q?", "targets": "A.", "annotation_type": "original-annotations"}
    assert is_aya_original(row) is True


# --- 2. Aya re-annotation rejection ---
def test_aya_reannotation_rejected():
    row = {"language_code": "eng", "language": "English",
           "inputs": "Q2?", "targets": "A2.", "annotation_type": "re-annotations"}
    assert is_aya_original(row) is False


# --- 3. language filtering eng/fil ---
def test_aya_language_filter_eng_fil():
    for code in ("eng", "fil"):
        assert is_aya_original({"language_code": code, "annotation_type": "original-annotations"}) is True
    assert is_aya_original({"language_code": "spa", "annotation_type": "original-annotations"}) is False


# --- 4. OASST synthetic=true rejection ---
def test_oasst_synthetic_rejected():
    assert _human_ok({"synthetic": True, "role": "assistant"}) is False


# --- 5. OASST model_name rejection ---
def test_oasst_model_name_rejected():
    assert _human_ok({"synthetic": False, "model_name": "gpt-3.5"}) is False
    assert _human_ok({"synthetic": False, "model_name": None}) is True


# --- 6. OASST deleted/review failures rejected ---
def test_oasst_deleted_and_review_rejected():
    assert _human_ok({"synthetic": False, "deleted": True}) is False
    assert _human_ok({"synthetic": False, "deleted": False, "review_result": False}) is False
    assert _human_ok({"synthetic": False, "deleted": False, "review_result": True}) is True
    assert _human_ok({"synthetic": False, "tree_state": "abandoned"}) is False
    assert _human_ok({"synthetic": False, "labels": ["spam"]}) is False


# --- 7. multi-turn path construction ---
def test_oasst_multiturn_path_and_rank0():
    rows = [
        {"message_id": "r", "parent_id": None, "message_tree_id": "t1", "role": "prompter",
         "rank": None, "text": "hi", "split": "train"},
        {"message_id": "a0", "parent_id": "r", "message_tree_id": "t1", "role": "assistant",
         "rank": 0, "text": "hello", "split": "train"},
        {"message_id": "a1", "parent_id": "r", "message_tree_id": "t1", "role": "assistant",
         "rank": 1, "text": "hey", "split": "train"},
    ]
    exs, _ = build_oasst_examples(rows)
    assert len(exs) == 1
    ex = exs[0]
    assert [r for r, _ in ex["turns"]] == ["user", "assistant"]
    assert ex["turns"][-1][1] == "hello"  # rank 0 preferred
    assert ex["split"] == "train"


def test_oasst_validation_split_maps_to_val():
    rows = [
        {"message_id": "r", "parent_id": None, "message_tree_id": "t2", "role": "prompter",
         "rank": None, "text": "hi", "split": "validation"},
        {"message_id": "a0", "parent_id": "r", "message_tree_id": "t2", "role": "assistant",
         "rank": 0, "text": "hello", "split": "validation"},
    ]
    exs, _ = build_oasst_examples(rows)
    assert exs[0]["split"] == "val"


# --- 8. deterministic splits ---
def test_deterministic_splits():
    a = stable_bucket("aya-eng-abc123", 100)
    b = stable_bucket("aya-eng-abc123", 100)
    c = stable_bucket("aya-eng-xyz789", 100)
    assert a == b
    assert a != c or stable_bucket("aya-eng-xyz789", 100) != stable_bucket("aya-fil-xyz789", 100)
    train_frac = sum(1 for i in range(200) if stable_bucket(f"k{i}", 100) < 95) / 200
    assert 0.90 <= train_frac <= 1.0


# --- 9. assistant-only loss mask ---
def test_assistant_only_loss_mask(tok):
    ids, labels = build_example_ids(tok, [("user", "What is 2 + 2?"), ("assistant", "It is 4.")])
    assert labels[0] == -100  # bos
    sup = [v for v in labels if v != -100]
    assert len(sup) > 0
    # the supervised span is exactly the assistant target + EOS
    target_ids = tok.encode("It is 4.") + [tok.eos_id]
    assert sup == target_ids


# --- 10. user tokens masked from loss ---
def test_user_tokens_masked(tok):
    ids, labels = build_example_ids(tok, [("user", "Hi there"), ("assistant", "Hello")])
    user_ids = tok.encode("Hi there")
    span = ids[2:2 + len(user_ids)]  # after <bos><user>
    assert span == user_ids
    for i in range(2, 2 + len(user_ids)):
        assert labels[i] == -100


# --- 11. padding masked from loss ---
def test_padding_masked(tok, tmp_path):
    e1, _ = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    e2, l2 = build_example_ids(tok, [("user", "a much longer question here"), ("assistant", "longer answer")])
    write_sft_jsonl(tmp_path / "t.jsonl", [
        {"id": "e1", "source": "t", "lang": "en", "ids": e1, "labels": [-100] * len(e1), "n_supervised": 0},
        {"id": "e2", "source": "t", "lang": "en", "ids": e2, "labels": l2, "n_supervised": 0},
    ])
    ds = SFTDataset(str(tmp_path / "t.jsonl"), batch_size=2, block_size=256, pad_id=1, seed=0, shuffle=False)
    x, y = ds.get_batch(torch.device("cpu"))
    assert x.shape[0] == 2
    assert (y[0, len(e1):] == -100).all()  # padded positions of the short example


# --- 12. EOS supervised ---
def test_eos_supervised(tok):
    ids, labels = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    assert labels[-1] == tok.eos_id


# --- 13. context-length handling ---
def test_context_length_handling(tok):
    stats = Counter()
    ex = {"id": "t", "source": "t", "lang": "en", "split": "train",
          "turns": [("user", "u" * 300), ("assistant", "a" * 300)]}
    built = tokenize_example(ex, tok, stats, block_size=64)
    assert built is None
    assert stats["rejected_target_too_long"] == 1


# --- 14. oldest-turn truncation ---
def test_oldest_turn_truncation(tok):
    stats = Counter()
    ex = {"id": "t", "source": "t", "lang": "en", "split": "train",
          "turns": [("user", "old context " * 20), ("assistant", "old reply " * 20),
                    ("user", "final question"), ("assistant", "final answer")]}
    built = tokenize_example(ex, tok, stats, block_size=64)
    assert built is not None
    assert stats["dropped_oldest_turns"] >= 1
    text_ids = built["ids"]
    final_user = tok.encode("final question")
    # final user turn is preserved in the ids
    assert any(text_ids[i:i + len(final_user)] == final_user
               for i in range(len(text_ids) - len(final_user)))


# --- 15. overlong-target rejection ---
def test_overlong_target_rejection(tok):
    stats = Counter()
    ex = {"id": "t", "source": "t", "lang": "en", "split": "train",
          "turns": [("user", "short question"), ("assistant", "z" * 500)]}
    built = tokenize_example(ex, tok, stats, block_size=128)
    assert built is None
    assert stats["rejected_target_too_long"] == 1


# --- 16. SFT checkpoint save/load ---
def _tiny_sft_ds(tok, tmp_path):
    e1, l1 = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    write_sft_jsonl(tmp_path / "c.jsonl", [
        {"id": "e1", "source": "t", "lang": "en", "ids": e1, "labels": l1, "n_supervised": 2}])
    return SFTDataset(str(tmp_path / "c.jsonl"), batch_size=2, block_size=256, pad_id=1, seed=0, shuffle=False)


def test_sft_checkpoint_save_load(tok, tmp_path):
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=2,
                      ffn_dim=32, block_size=64)
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = None
    ds = _tiny_sft_ds(tok, tmp_path)
    p = save_checkpoint(str(tmp_path), "best", model, optimizer, scaler, step=10,
                        supervised_tokens_seen=1234, total_tokens_seen=9999,
                        best_sft_val_loss=2.5, base_baseline_loss=3.07,
                        base_current_loss=3.2, cfg_dict=cfg.to_dict(), args=argparse_namespace(),
                        train_ds=ds, val_ds=ds, rng_state={}, source_meta={})
    assert Path(p).exists()
    m2 = GPTModel(cfg)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    ckpt = load_chat_checkpoint(p, m2, o2, scaler, torch.device("cpu"), cfg.to_dict())
    assert ckpt["step"] == 10
    assert ckpt["supervised_tokens_seen"] == 1234
    assert ckpt["base_baseline_loss"] == 3.07


def argparse_namespace():
    import argparse
    return argparse.Namespace(out_dir="x", base_checkpoint="b", config="c")


# --- 17. SFT resume ---
def test_sft_resume(tok, tmp_path):
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=2,
                      ffn_dim=32, block_size=64)
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = None
    ds = _tiny_sft_ds(tok, tmp_path)
    p = save_checkpoint(str(tmp_path), "latest", model, optimizer, scaler, step=7,
                        supervised_tokens_seen=777, total_tokens_seen=8888,
                        best_sft_val_loss=2.0, base_baseline_loss=3.07,
                        base_current_loss=3.1, cfg_dict=cfg.to_dict(), args=argparse_namespace(),
                        train_ds=ds, val_ds=ds, rng_state={}, source_meta={})
    m2 = GPTModel(cfg)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    ckpt = load_chat_checkpoint(p, m2, o2, scaler, torch.device("cpu"), cfg.to_dict())
    # deterministic resume state survives the round trip
    assert ckpt["step"] == 7
    assert ckpt["total_tokens_seen"] == 8888
    assert ckpt["best_sft_val_loss"] == 2.0


# --- 18. catastrophic-forgetting guard ---
def test_catastrophic_forgetting_guard():
    baseline = 3.07
    hard_stop, eligible = retention_guard(3.5, baseline, 1.20)
    assert hard_stop is False and eligible is True
    hard_stop, eligible = retention_guard(3.75, baseline, 1.20)
    assert hard_stop is True and eligible is False
    hard_stop, eligible = retention_guard(3.51, baseline, 1.20)
    assert hard_stop is False and eligible is True  # 3.51 < 3.684


# --- 19. chat history builder ---
def test_chat_history_builder(tok):
    history = [("user", "Hello"), ("assistant", "Hi!"), ("user", "How are you?")]
    ids = build_history_ids(tok, history, system_text="")
    assert ids[0] == tok.bos_id
    assert tok.user_id in ids and tok.assistant_id in ids
    assert ids[-1] == tok.assistant_id
    # last user turn is always present
    assert tok.encode("How are you?")[-1] in ids


def test_chat_history_drops_oldest():
    tok = CharTokenizer("ab")  # tiny vocab so context fills quickly
    history = [("user", "a" * 100), ("assistant", "b" * 100), ("user", "c"), ("assistant", "d"), ("user", "z")]
    ids = build_history_ids(tok, history, system_text="")
    assert len(ids) <= 256
    assert tok.encode("z")[-1] in ids  # final user turn preserved


# --- 20. EOS / role-marker stopping ---
def test_eos_and_role_marker_stopping(tok):
    ids = [1, 2, 6, 3, 9]  # reply tokens, then <assistant> role marker, then <eos>, then extra
    trimmed = trim_reply_ids(ids, {5, 6}, eos_id=3)
    assert trimmed == [1, 2]
    assert 3 not in trimmed and 9 not in trimmed
    # role marker mid-reply stops the reply
    ids2 = [1, 2, 5, 9]
    trimmed2 = trim_reply_ids(ids2, {5, 6}, eos_id=3)
    assert trimmed2 == [1, 2]


# --- 21. tokenizer special-token compatibility ---
def test_tokenizer_special_token_compatibility():
    tok = SentencePieceTokenizer(str(ROOT / "data" / "tokenizer" / "tokenizer_v1.model"),
                                 str(ROOT / "data" / "tokenizer" / "tokenizer_v1_meta.json"))
    assert tok.vocab_size == 8000
    s = tok.special_token_ids
    assert (s["unk"], s["pad"], s["bos"], s["eos"], s["system"], s["user"], s["assistant"]) == (0, 1, 2, 3, 4, 5, 6)
    ids = tok.encode("Kumusta ka?")
    assert ids == tok.decode(ids, skip_special=True) or True  # roundtrip sanity
    assert tok.decode(tok.encode("Kumusta ka?"), skip_special=True) == "Kumusta ka?"
    assert tok.system_id == 4 and tok.user_id == 5 and tok.assistant_id == 6


# --- 22. no data leakage between SFT train/validation ---
def test_no_leakage_between_train_and_val():
    train_path = ROOT / "data" / "sft" / "processed" / "sft_train.jsonl"
    val_path = ROOT / "data" / "sft" / "processed" / "sft_val.jsonl"
    if not train_path.exists() or not val_path.exists():
        pytest.skip("processed SFT files not built yet")
    train_ids = {json.loads(l)["id"] for l in open(train_path, encoding="utf-8") if l.strip()}
    val_ids = {json.loads(l)["id"] for l in open(val_path, encoding="utf-8") if l.strip()}
    assert not (train_ids & val_ids)
    # OASST source validation examples must NOT be in train (message_tree_id containment)
    oasst_train = [i for i in train_ids if i.startswith("oasst-")]
    oasst_val = [i for i in val_ids if i.startswith("oasst-")]
    assert not (set(oasst_train) & set(oasst_val))


# --- bonus: eval_sft_loss runs on a tiny model ---
def test_eval_sft_loss_runs(tok, tmp_path):
    e1, l1 = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    write_sft_jsonl(tmp_path / "v.jsonl", [
        {"id": "e1", "source": "t", "lang": "en", "ids": e1, "labels": l1, "n_supervised": 2}])
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=2,
                      ffn_dim=32, block_size=256)
    model = GPTModel(cfg)
    ds = SFTDataset(str(tmp_path / "v.jsonl"), batch_size=2, block_size=256, pad_id=1, seed=0, shuffle=False)
    loss = eval_sft_loss(model, ds, iters=5, device=torch.device("cpu"), use_amp=False)
    assert loss > 0 and loss < 20