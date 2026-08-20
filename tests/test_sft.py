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
    aya_eng_looks_english,
    build_aya_examples,
    build_dolly_examples,
    build_oasst_examples,
    build_taskmaster_examples,
    compute_sampling_weights,
    cross_source_dedup,
    exclude_eval_probes,
    norm_text,
    stable_bucket,
    tokenize_example,
)
from acquire_sft_data import _human_ok, is_aya_original  # noqa: E402

from src.chat import build_history_ids, trim_reply_ids  # noqa: E402
from src.model import GPTModel, ModelConfig  # noqa: E402
from src.sft_dataset import SFTDataset  # noqa: E402
from src.sft_train import (  # noqa: E402
    DEFAULT_OUT_DIR,
    anchor_replay_tokens,
    assert_out_dir_free_for_base,
    eval_sft_loss,
    load_chat_checkpoint,
    retention_guard,
    save_checkpoint,
    step_token_counts,
    val_eval_indices,
    validate_anchor_data,
    validate_retention_factors,
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


# --- 12. EOS supervised (causal convention, TASK 005.2) ---
def test_eos_supervised(tok):
    ids, labels = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    # EOS is PREDICTED from the last assistant content token; the EOS input
    # position itself is masked (-100). The OLD same-position labels made
    # labels[-1] == eos_id which was the identity-copy bug.
    assert labels[-2] == tok.eos_id
    assert ids[-1] == tok.eos_id
    assert labels[-1] == -100


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


# --- 18. catastrophic-forgetting guard (Part E: SEPARATE factors) ---
def test_catastrophic_forgetting_guard():
    baseline = 3.07
    hard_stop, eligible = retention_guard(3.30, baseline, 1.10, 1.15)
    assert hard_stop is False and eligible is True   # 3.30 <= 3.07*1.10
    hard_stop, eligible = retention_guard(3.42, baseline, 1.10, 1.15)
    assert hard_stop is False and eligible is False  # middle window: 3.377 < 3.42 <= 3.5305
    hard_stop, eligible = retention_guard(3.60, baseline, 1.10, 1.15)
    assert hard_stop is True and eligible is False   # 3.60 > 3.07*1.15


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


# --------------------------------------------------------------------------
# TASK 005.1 additions (Dolly, Taskmaster, dedup, gate, sampling, handoff)
# --------------------------------------------------------------------------

# --- Dolly: deterministic formatting ---
def test_dolly_formatting_and_empty_reject():
    rows = [
        {"id": "0", "instruction": "  What  is  2+2? ", "context": "", "response": "4.", "category": "general"},
        {"id": "1", "instruction": "Q", "context": "ctx", "response": "A"},
        {"id": "2", "instruction": "", "context": "", "response": "A"},
        {"id": "3", "instruction": "Q", "context": "", "response": ""},
    ]
    exs = build_dolly_examples(rows)
    assert len(exs) == 2
    assert exs[0]["turns"][0][1] == "What is 2+2?"  # normalized, no context
    assert exs[1]["turns"][0][1] == "Q\n\nctx"  # deterministic separator
    assert exs[0]["lang"] == "en" and exs[0]["source"] == "dolly"
    assert exs[0]["split"] in ("train", "val")
    assert (stable_bucket(exs[0]["id"], 100) < 95) == (exs[0]["split"] == "train")


# --- Dolly: overlong target rejected ---
def test_dolly_overlong_rejected(tok):
    stats = Counter()
    ex = build_dolly_examples([{"id": "9", "instruction": "Q", "context": "",
                                "response": "z" * 500, "category": "g"}])[0]
    built = tokenize_example(ex, tok, stats, block_size=128)
    assert built is None
    assert stats["rejected_target_too_long"] == 1


# --- Taskmaster: speaker parsing + assistant target extraction ---
def test_taskmaster_assistant_extraction():
    conv = {"id": "c1", "domain": "flights", "split": "train",
            "utterances": [
                {"speaker": "user", "text": "hi"},
                {"speaker": "assistant", "text": "hello"},
                {"speaker": "user", "text": "book a flight"},
                {"speaker": "assistant", "text": "where to?"},
                {"speaker": "user", "text": "manila"},
                {"speaker": "assistant", "text": "done"},
            ]}
    exs, _ = build_taskmaster_examples([conv])
    assert len(exs) == 3  # 3 assistant turns <= cap
    for ex in exs:
        assert ex["turns"][-1][0] == "assistant"
        assert ex["split"] == "train"
        assert ex["source"] == "taskmaster1"
        assert ex["turns"][0][0] == "user"


# --- Taskmaster: max 4 targets per conversation, deterministic coverage ---
def test_taskmaster_max4_cap_deterministic():
    utts = []
    for i in range(20):
        utts.append({"speaker": "user", "text": f"u{i}"})
        utts.append({"speaker": "assistant", "text": f"a{i}"})
    conv = {"id": "c2", "domain": "x", "split": "val", "utterances": utts}
    exs1, s1 = build_taskmaster_examples([conv])
    exs2, _ = build_taskmaster_examples([conv])
    assert len(exs1) == len(exs2) == 4
    assert s1["capped_candidate_turns"] == 16
    assert [ex["turns"][-1][1] for ex in exs1] == [ex["turns"][-1][1] for ex in exs2]
    # deterministic even coverage: first and last assistant turns included
    targets = [ex["turns"][-1][1] for ex in exs1]
    assert targets[0] == "a0" and targets[-1] == "a19"


# --- Taskmaster: conversation never crosses splits ---
def test_taskmaster_conv_split_isolation():
    conv = {"id": "c3", "domain": "x", "split": "val",
            "utterances": [{"speaker": "user", "text": "u"},
                           {"speaker": "assistant", "text": "a"},
                           {"speaker": "user", "text": "u2"},
                           {"speaker": "assistant", "text": "a2"}]}
    exs, _ = build_taskmaster_examples([conv])
    assert len(exs) == 2
    assert all(ex["split"] == "val" for ex in exs)  # single source of truth per conv


# --- Taskmaster: root-not-user rejected ---
def test_taskmaster_root_not_user():
    conv = {"id": "c4", "domain": "x", "split": "train",
            "utterances": [{"speaker": "assistant", "text": "a"},
                           {"speaker": "user", "text": "u"}]}
    exs, skipped = build_taskmaster_examples([conv])
    assert len(exs) == 0
    assert skipped["root_not_user"] == 1


# --- Aya mislabeled-language rows rejected (English-check heuristic) ---
def test_aya_eng_mislabel_rejection():
    assert aya_eng_looks_english("What is light reflection?", "Light reflection is the process by which light waves bounce.") is True
    assert aya_eng_looks_english("Who wrote the classic novel The Mayor of Casterbridge?", "Thomas Hardy.") is True
    # Somali prompt + Somali target: no English function words in either
    assert aya_eng_looks_english("Sheeg magacyada gobolada Somalia?",
                                 "Magacyada gobolada Somalia waa sida hoos ku qoran.") is False
    # short non-English prompt with long non-English target
    assert aya_eng_looks_english("Waa maxay AI?",
                                 "AI waxaa laga soo gaabiyey Artificial Intelligence, waxaana loola jeedaa in caqliga bani-aadam ka la baro mashiino.") is False
    # English prompt with a code target is NOT rejected
    assert aya_eng_looks_english("Write a Python program to find the second largest number in a list.",
                                 "def second_largest(lst): return sorted(lst)[-2]") is True
    # non-Latin script content
    assert aya_eng_looks_english("Что такое вода?", "Вода — это жидкость.") is False
    # short English answer must NOT be rejected
    assert aya_eng_looks_english("Which animal is the fastest on land?", "cheetah") is True


# --- cross-source dedup: pair removed, source pair reported, prompt-only kept ---
def test_cross_source_dedup():
    aya = {"source": "aya", "turns": [("user", "Q1"), ("assistant", "A1")]}
    oasst = {"source": "oasst1", "turns": [("user", "Q1"), ("assistant", "A1")]}  # exact pair dup
    dolly = {"source": "dolly", "turns": [("user", "Q1"), ("assistant", "A2")]}   # prompt-only dup
    kept, stats = cross_source_dedup([aya, oasst, dolly])
    assert len(kept) == 2
    assert stats["rejected_duplicates"] == 1
    assert stats["dup_pair_aya|oasst1"] == 1
    assert stats["prompt_only_duplicates"] == 1
    assert kept[0]["source"] == "aya"  # first occurrence wins


# --- probe exclusion checks ANY user turn (multi-turn) ---
def test_probe_exclusion_any_user_turn():
    ex = {"turns": [("user", "Ano ang 2 + 2?"), ("assistant", "A"),
                    ("user", "next question"), ("assistant", "B")]}
    kept, rejected = exclude_eval_probes([ex], probes=["Ano ang 2 + 2?"])
    assert rejected == 1 and kept == []


# --- gate: unique tokens only, oversampling never counts ---
def test_gate_ignores_oversampling():
    rows = [
        {"source": "aya", "lang": "fil", "n_supervised": 100},
        {"source": "aya", "lang": "fil", "n_supervised": 100},
    ] + [{"source": "dolly", "lang": "en", "n_supervised": 10} for _ in range(180)]
    sampling, copies = compute_sampling_weights(rows)
    unique = sum(r["n_supervised"] for r in rows)
    effective = sum(r["n_supervised"] * copies[r["source"]] for r in rows)
    assert unique == 2000  # 200 fil + 1800 en
    assert effective > unique  # sampling inflates effective, not unique
    assert sampling["effective_fil_share"] >= 0.15  # fil up-weighted to target


# --- Filipino sampling weight: up to 4x, effective share -> ~15% ---
def test_filipino_sampling_weight_cap():
    rows = [{"source": "aya", "lang": "fil", "n_supervised": 10} for _ in range(1000)]
    rows += [{"source": "dolly", "lang": "en", "n_supervised": 10} for _ in range(100000)]
    sampling, copies = compute_sampling_weights(rows)
    assert copies["aya"] == 4  # capped at FIL_MAX_WEIGHT
    assert sampling["fil_weight"] == 4.0


def test_filipino_sampling_effective_share():
    # en = 17 * fil tokens -> w = 3 exactly -> effective share = 15%
    rows = [{"source": "aya", "lang": "fil", "n_supervised": 10} for _ in range(5000)]
    rows += [{"source": "dolly", "lang": "en", "n_supervised": 10} for _ in range(85000)]
    sampling, copies = compute_sampling_weights(rows)
    assert copies["aya"] == 3
    assert abs(sampling["effective_fil_share"] - 0.15) < 0.001


# --- source balance: English source > 50% down-weighted ---
def test_source_balance_downweight():
    rows = [{"source": "taskmaster1", "lang": "en", "n_supervised": 10} for _ in range(1000)]
    rows += [{"source": "dolly", "lang": "en", "n_supervised": 10} for _ in range(100)]
    sampling, copies = compute_sampling_weights(rows)
    # dominant source lands at 50% by up-weighting the other English source
    assert copies["dolly"] == 10
    eff = sampling["effective_english_source_tokens"]
    assert abs(eff["taskmaster1"] / (eff["taskmaster1"] + eff["dolly"]) - 0.5) < 0.01


# --- SFTDataset: copies -> effective length, deterministic order, resume ---
def test_sft_dataset_copies_effective_and_resume(tok, tmp_path):
    e1, l1 = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    e2, l2 = build_example_ids(tok, [("user", "Q2?"), ("assistant", "A2.")])
    write_sft_jsonl(tmp_path / "c.jsonl", [
        {"id": "e1", "source": "aya", "lang": "fil", "ids": e1, "labels": l1,
         "n_supervised": 2, "copies": 3},
        {"id": "e2", "source": "dolly", "lang": "en", "ids": e2, "labels": l2,
         "n_supervised": 2, "copies": 1},
    ])
    ds = SFTDataset(str(tmp_path / "c.jsonl"), batch_size=2, block_size=256,
                    pad_id=1, seed=42, shuffle=True)
    assert len(ds) == 4  # 3 copies of e1 + 1 of e2 (effective)
    assert ds.n_examples == 2  # unique
    assert ds.unique_supervised_tokens() == 4
    assert ds.effective_supervised_tokens() == 8
    st = ds.state_dict()
    ds.load_state_dict(st)
    order_after_resume = list(ds._order)
    assert order_after_resume == st["order"]
    # deterministic: same seed + same state -> same order
    ds2 = SFTDataset(str(tmp_path / "c.jsonl"), batch_size=2, block_size=256,
                     pad_id=1, seed=42, shuffle=True)
    assert list(ds2._order) == order_after_resume


# --- SFTDataset: validation does not expand copies ---
def test_sft_dataset_val_no_copies(tok, tmp_path):
    e1, l1 = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    write_sft_jsonl(tmp_path / "v.jsonl", [
        {"id": "e1", "source": "aya", "lang": "fil", "ids": e1, "labels": l1,
         "n_supervised": 2, "copies": 3}])
    ds = SFTDataset(str(tmp_path / "v.jsonl"), batch_size=2, block_size=256,
                    pad_id=1, seed=0, shuffle=False)
    assert len(ds) == 1
    assert ds.n_supervised_tokens() == 2  # unique on the val path


# --------------------------------------------------------------------------
# TASK 005.2 additions: causal label alignment, guards, accounting, sampling
# --------------------------------------------------------------------------

# --- exact positional mapping of the CORRECTED causal labels ---
def test_causal_next_token_positional_labels(tok):
    ids, labels = build_example_ids(tok, [("user", "Q"), ("assistant", "ABC")])
    # ids = [<bos> <user> Q <assistant> A B C <eos>]
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.user_id
    assert ids[3] == tok.assistant_id
    a, b, c = tok.encode("ABC")
    assert ids[4] == a and ids[5] == b and ids[6] == c
    assert ids[7] == tok.eos_id
    # causal next-token labels: labels[i] = ids[i+1] over the target span
    assert labels[0] == -100          # <bos> masked
    assert labels[1] == -100          # <user> masked
    assert labels[2] == -100          # user content masked
    assert labels[3] == a             # <assistant> predicts 'A'
    assert labels[4] == b             # 'A' predicts 'B'
    assert labels[5] == c             # 'B' predicts 'C'
    assert labels[6] == tok.eos_id    # 'C' predicts <eos>
    assert labels[7] == -100          # EOS input position masked
    assert len(labels) == len(ids)


# --- previous assistant turns are CONTEXT (masked), only the final answer learns ---
def test_previous_assistant_context_masked(tok):
    turns = [("user", "Hi"), ("assistant", "Hello"), ("user", "Q?"), ("assistant", "A.")]
    ids, labels = build_example_ids(tok, turns)
    start = 1 + 1 + len(tok.encode("Hi")) + 1  # after <bos><user>Hi<assistant>
    first_asst = tok.encode("Hello")
    for i in range(start, start + len(first_asst)):
        assert labels[i] == -100      # first assistant turn = context only
    assert labels[start - 1] == -100  # first <assistant> marker masked
    sup = [v for v in labels if v != -100]
    assert sup == tok.encode("A.") + [tok.eos_id]  # ONLY the final answer supervised


# --- identity-copy trap: same-position predictors must NOT yield a low loss ---
def test_identity_copy_trap_loss_high():
    import torch.nn.functional as F
    ids = torch.tensor([[1, 2, 5, 6, 10, 11, 12, 3]])   # bos user u asst A B C eos
    labels = torch.tensor([[-100, -100, -100, 10, 11, 12, 3, -100]])
    V, T = 20, 8
    logits = torch.zeros(1, T, V)
    logits[0, torch.arange(T), ids[0]] = 10.0  # predicts the token AT the same position
    loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    assert loss.item() > 1.0
    # (under the OLD same-position labels this exact setup approached 0)


# --- perfect next-token predictors DO yield ~0 loss on the corrected labels ---
def test_perfect_next_token_logits_low():
    import torch.nn.functional as F
    ids = torch.tensor([[1, 2, 5, 6, 10, 11, 12, 3]])
    labels = torch.tensor([[-100, -100, -100, 10, 11, 12, 3, -100]])
    V, T = 20, 8
    logits = torch.zeros(1, T, V)
    logits[0, torch.arange(T - 1), ids[0, 1:]] = 10.0  # predicts ids[i+1]
    loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    assert loss.item() < 0.01


# --- Part G: truncation keeps the MOST RECENT turns, drops the OLDEST ---
def test_recent_turn_truncation_keeps_recent(tok):
    stats = Counter()
    ex = {"id": "t", "source": "t", "lang": "en", "split": "train",
          "turns": [("user", "OLD" * 40), ("assistant", "oldreply"),
                    ("user", "mid" * 5), ("assistant", "midreply"),
                    ("user", "final question"), ("assistant", "final answer")]}
    built = tokenize_example(ex, tok, stats, block_size=64)
    assert built is not None
    assert stats["dropped_oldest_turns"] >= 1
    text_ids = built["ids"]

    def contains(seq):
        return any(text_ids[i:i + len(seq)] == seq
                   for i in range(len(text_ids) - len(seq) + 1))

    assert contains(tok.encode("final question"))  # final user turn kept
    assert contains(tok.encode("mid"))             # RECENT history kept
    assert contains(tok.encode("oldreply"))        # mid-recent kept
    assert not contains(tok.encode("OLD"))         # OLDEST history dropped


# --- Part H: deterministic, representative val sampling ---
def test_val_eval_indices_representative_and_deterministic():
    n, bs, iters = 1000, 8, 50
    i1 = val_eval_indices(n, bs, iters, seed=1379)
    i2 = val_eval_indices(n, bs, iters, seed=1379)
    assert torch.equal(i1, i2)                  # deterministic
    assert i1.numel() == 400                    # exactly iters*batch_size
    assert i1.max().item() >= 400               # reaches beyond the first rows
    assert len(set(i1.tolist())) == 400         # sampled WITHOUT replacement
    i3 = val_eval_indices(n, bs, iters, seed=42)
    assert not torch.equal(i1, i3)              # different seed -> different set
    i4 = val_eval_indices(10, bs, 50, seed=0)   # full coverage when iters*bs >= n
    assert i4.numel() == 10 and set(i4.tolist()) == set(range(10))


def test_eval_sft_loss_deterministic_untouched(tok, tmp_path):
    examples = []
    for i in range(60):
        ids, labels = build_example_ids(tok, [("user", f"q{i}"), ("assistant", f"a{i}")])
        examples.append({"id": f"e{i}", "source": "t", "lang": "en",
                         "ids": ids, "labels": labels, "n_supervised": 3})
    write_sft_jsonl(tmp_path / "v.jsonl", examples)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=2,
                      ffn_dim=32, block_size=256)
    model = GPTModel(cfg)
    ds = SFTDataset(str(tmp_path / "v.jsonl"), batch_size=8, block_size=256,
                    pad_id=1, seed=0, shuffle=False)
    l1 = eval_sft_loss(model, ds, iters=5, device=torch.device("cpu"), use_amp=False, seed=42)
    l2 = eval_sft_loss(model, ds, iters=5, device=torch.device("cpu"), use_amp=False, seed=42)
    assert l1 == l2                            # deterministic
    assert ds._pos == 0 and len(ds._order) == 60  # dataset state untouched


# --- Part E: eligibility must be below hard-stop ---
def test_retention_guard_requires_eligibility_lt_hard_stop():
    validate_retention_factors(1.10, 1.15)  # OK
    with pytest.raises(SystemExit):
        validate_retention_factors(1.15, 1.15)
    with pytest.raises(SystemExit):
        validate_retention_factors(1.20, 1.15)
    with pytest.raises(SystemExit):
        validate_retention_factors(-1.0, 1.15)


# --- Part F: token accounting sums ALL grad-accum microbatches ---
def test_grad_accum_token_accounting_all_microbatches():
    b1 = (torch.full((2, 4), 7), torch.tensor([[-100, 1, 2, -100], [-100, 3, 4, 5]]))
    b2 = (torch.full((2, 4), 7), torch.tensor([[6, -100, 7, 8], [-100, -100, 9, -100]]))
    b3 = (torch.full((2, 4), 7), torch.tensor([[10, 11, 12, -100], [13, 14, 15, 16]]))
    sup, tot = step_token_counts([b1, b2, b3])
    assert sup == 16   # 5 + 4 + 7 over ALL microbatches
    assert tot == 24   # 3 batches * 8 tokens
    assert sup != 7    # the OLD buggy code counted only the LAST microbatch


# --- Part K: base replay tokens are counted SEPARATELY, never supervised ---
def test_base_replay_tokens_counted_separately():
    anchor_x = torch.full((2, 4), 7)
    sup, _tot = step_token_counts([(anchor_x, torch.full((2, 4), -100))])
    assert sup == 0                               # replay has no supervised targets
    assert anchor_replay_tokens(anchor_x) == 8    # counted by its OWN counter


# --- Part L: anchor coefficient applied exactly once (L = (1/G) sum SFT + w*ANCHOR) ---
def test_anchor_loss_coefficient_scaling(tok):
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=8, n_layers=1, n_heads=2,
                      ffn_dim=16, block_size=64)
    m1 = GPTModel(cfg)
    m2 = GPTModel(cfg)
    m2.load_state_dict(m1.state_dict())
    ids, labels = build_example_ids(tok, [("user", "Q?"), ("assistant", "A.")])
    x = torch.tensor([ids])
    y = torch.tensor([labels])
    anchor_full = [tok.bos_id] + tok.encode("the cat sat") + [tok.eos_id]
    ax = torch.tensor([anchor_full[:-1]])  # pretraining semantics: x[:-1] -> y[1:]
    ay = torch.tensor([anchor_full[1:]])
    G, w = 2, 0.10

    def ce(m, xx, yy):
        return m(xx, yy, ignore_index=-100)[1]

    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
    opt1.zero_grad()
    (ce(m1, x, y) / G + w * ce(m1, ax, ay)).backward()
    g1 = [p.grad.clone() for p in m1.parameters()]

    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    opt2.zero_grad()
    (ce(m2, x, y) / G).backward()       # SFT part, single 1/G
    (w * ce(m2, ax, ay)).backward()     # anchor part, single w
    g2 = [p.grad.clone() for p in m2.parameters()]

    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-6)


# --- failed chat_v1 checkpoint is never silently reused as init ---
def test_failed_chat_v1_checkpoint_not_used_as_init(tmp_path):
    assert DEFAULT_OUT_DIR.endswith("chat_v1_corrected")
    free = tmp_path / "free"
    free.mkdir()
    assert_out_dir_free_for_base(str(free))  # empty dir OK
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "latest.pt").write_text("x")
    with pytest.raises(SystemExit):
        assert_out_dir_free_for_base(str(dirty))


# --- base checkpoint integrity: pretrain_v1/best.pt SHA unchanged ---
def test_base_checkpoint_sha_unchanged():
    import hashlib
    p = ROOT / "checkpoints" / "pretrain_v1" / "best.pt"
    if not p.exists():
        pytest.skip("base checkpoint not present")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == "ba40ad8ce0644720d243209049f91791d1163113e55a9fd4e3738326db8b8350"


# --- test.bin is sealed: never usable as replay/anchor data ---
def test_anchor_data_never_test_bin():
    with pytest.raises(SystemExit):
        validate_anchor_data("data/processed/test.bin")
    validate_anchor_data("data/processed/train.bin")  # own pretraining corpus OK