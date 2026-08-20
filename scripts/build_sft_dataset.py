"""Build the TASK 005 SFT dataset (chat format, assistant-only loss labels).

Inputs (raw, git-ignored):
  data/sft/raw/aya_eng_fil_original.jsonl
  data/sft/raw/oasst1_en_human_messages.jsonl

Processing:
  - normalization: UTF-8, Unicode NFC, whitespace cleanup (no lowercasing,
    no punctuation removal)
  - quality rejects: empty prompt/response, pathological repeated characters,
    corrupted Unicode, exact (prompt, target) duplicates across sources
  - tokenizer <unk> analysis: prompt unk rate / target unk rate; targets
    containing <unk> are rejected when practical
  - Aya: single-turn examples, deterministic 95/5 train/val split by stable
    hash of the record id (Aya test split is never used)
  - OASST: multi-turn paths (human-only context + final assistant target,
    rank==0 preferred), source-provided train/validation split; no
    message_tree_id crosses the split
  - chat format: <bos><user>U<assistant>A<eos> (multi-turn: earlier turns are
    context; ONLY the final assistant target + EOS is supervised, labels=-100
    elsewhere)
  - context 256: keep the final user turn + complete target; drop oldest
    turns first; reject examples whose target itself cannot fit
  - evaluation probes from TASK 005 Part K/S are excluded from training

Outputs:
  data/sft/processed/sft_train.jsonl  (git-ignored)
  data/sft/processed/sft_val.jsonl    (git-ignored)
  data/sft/stats/sft_stats.json       (tracked)
  data/sft/stats/SFT_DATA_REPORT.md   (tracked)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\build_sft_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import SentencePieceTokenizer  # noqa: E402

RAW_DIR = ROOT / "data" / "sft" / "raw"
OUT_DIR = ROOT / "data" / "sft" / "processed"
STATS_DIR = ROOT / "data" / "sft" / "stats"
TOKENIZER_MODEL = ROOT / "data" / "tokenizer" / "tokenizer_v1.model"
TOKENIZER_META = ROOT / "data" / "tokenizer" / "tokenizer_v1_meta.json"

BLOCK_SIZE = 256
TRAIN_FRAC = 0.95

# TASK 005 Part K / Part S evaluation prompts + probes (NEVER in training data)
EVAL_PROBES = [
    "Kumusta ka?",
    "Bakit mahalaga ang tubig?",
    "Ano ang araw?",
    "Ipaliwanag nang simple kung ano ang gravity.",
    "Pagod ako ngayon. Ano ang pwede kong gawin?",
    "Ano ang pagkakaiba ng ilog at dagat?",
    "Gumawa ng maikling pangungusap tungkol sa mangga.",
    "Ano ang ibig sabihin ng kalayaan?",
    "Mahilig ako sa aso. Ano ang magandang pag-usapan natin?",
    "Ano ang 2 + 2?",
    "Hello, how are you?",
    "Why is water important?",
    "What is the Sun?",
    "Explain gravity simply.",
    "I'm tired today. What can I do?",
    "What is the difference between a river and an ocean?",
    "Write a short sentence about mangoes.",
    "What does freedom mean?",
    "I like dogs. What could we talk about?",
    "What is 2 + 2?",
    "Pangalan ng aso ko ay Bruno.",
    "Ano nga ang pangalan ng aso ko?",
    "My dog's name is Bruno.",
    "What is my dog's name?",
]

PATHOLOGICAL_RUN = 15          # max run of the same character
PATHOLOGICAL_RATIO = 0.50      # max fraction of single repeated char


def norm_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_corrupted(text: str) -> bool:
    return "\ufffd" in text


def is_pathological(text: str) -> bool:
    if len(text) < 2:
        return False
    for ch in set(text):
        if ch.isspace():
            continue
        run = max(len(m.group(0)) for m in re.finditer(re.escape(ch) + "+", text))
        if run >= PATHOLOGICAL_RUN:
            return True
        if text.count(ch) / len(text) >= PATHOLOGICAL_RATIO and len(text) >= 10:
            return True
    return False


def stable_bucket(key: str, divisor: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % divisor


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_aya_examples(rows: List[dict]) -> List[dict]:
    examples = []
    for r in rows:
        prompt = norm_text(r["inputs"])
        target = norm_text(r["targets"])
        examples.append({
            "id": r["id"],
            "source": "aya",
            "lang": r["language_code"],
            "turns": [("user", prompt), ("assistant", target)],
            "split": "train" if stable_bucket(r["id"], 100) < TRAIN_FRAC * 100 else "val",
        })
    return examples


def build_oasst_examples(rows: List[dict]) -> Tuple[List[dict], dict]:
    by_id = {r["message_id"]: r for r in rows}
    # choose best assistant candidate per parent (rank 0 preferred, deterministic)
    best_assistant: Dict[str, dict] = {}
    for r in rows:
        if r["role"] != "assistant":
            continue
        parent = r["parent_id"]
        cur = best_assistant.get(parent)
        if cur is None:
            best_assistant[parent] = r
            continue
        rk, ck = r.get("rank"), cur.get("rank")
        if rk is not None and (ck is None or rk < ck):
            best_assistant[parent] = r
        elif rk is None and ck is not None:
            continue
        elif rk == ck and r["message_id"] < cur["message_id"]:
            best_assistant[parent] = r

    def norm_split(s: str) -> str:
        return "val" if s == "validation" else "train"

    examples = []
    skipped = Counter()
    for parent, tgt in best_assistant.items():
        path: List[dict] = []
        node = tgt
        while node is not None:
            path.append(node)
            node = by_id.get(node.get("parent_id"))
        path.reverse()
        if not path or path[0]["role"] != "prompter":
            skipped["root_not_prompter"] += 1
            continue
        turns = [(("user" if m["role"] == "prompter" else "assistant"), norm_text(m["text"]))
                 for m in path]
        if not turns or turns[-1][0] != "assistant":
            skipped["target_not_assistant"] += 1
            continue
        examples.append({
            "id": f"oasst-{tgt['message_id']}",
            "source": "oasst1",
            "lang": "en",
            "turns": turns,
            "split": norm_split(tgt["split"]),
        })
    return examples, skipped


def tokenize_example(ex: dict, tok: SentencePieceTokenizer,
                     stats: Counter, block_size: int = BLOCK_SIZE) -> Optional[dict]:
    """Builds (ids, labels) in chat format with assistant-only supervision."""
    ids: List[int] = [tok.bos_id]
    labels: List[int] = [-100]

    # 1) final user turn + final assistant target must fit (always kept)
    core_turns = ex["turns"]
    final_user, final_assistant = core_turns[-2], core_turns[-1]
    core_ids = ([tok.user_id] + tok.encode(final_user[1])
                + [tok.assistant_id] + tok.encode(final_assistant[1]) + [tok.eos_id])
    if len(ids) + len(core_ids) > block_size:
        stats["rejected_target_too_long"] += 1
        return None

    # 2) prepend older complete turns while they fit
    prefix: List[int] = []
    for role, text in core_turns[:-2]:
        turn_ids = ([tok.user_id] + tok.encode(text)) if role == "user" else \
                   ([tok.assistant_id] + tok.encode(text))
        if len(prefix) + len(turn_ids) + len(ids) + len(core_ids) > block_size:
            stats["dropped_oldest_turns"] += 1
            break
        prefix.extend(turn_ids)

    full_ids = ids + prefix + core_ids
    full_labels = [-100] * len(full_ids)
    target_start = len(full_ids) - len(tok.encode(final_assistant[1])) - 1  # -1 = eos
    for i in range(target_start, len(full_ids)):
        full_labels[i] = full_ids[i]

    return {"ids": full_ids, "labels": full_labels, "n_supervised": len(full_labels) - target_start}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the TASK 005 SFT dataset")
    ap.add_argument("--max-target-unk-rate", type=float, default=0.0,
                    help="reject examples whose target contains <unk> unless rate is below this threshold")
    args = ap.parse_args()

    tok = SentencePieceTokenizer(str(TOKENIZER_MODEL), str(TOKENIZER_META))
    special = tok.special_token_ids
    print(f"tokenizer: vocab={tok.vocab_size} unk={special['unk']} bos={special['bos']} "
          f"eos={special['eos']} user={special['user']} assistant={special['assistant']}")

    stats: Counter = Counter()
    aya_rows = load_jsonl(RAW_DIR / "aya_eng_fil_original.jsonl")
    oasst_rows = load_jsonl(RAW_DIR / "oasst1_en_human_messages.jsonl")
    stats["aya_raw"] = len(aya_rows)
    stats["oasst_raw"] = len(oasst_rows)

    aya_examples = build_aya_examples(aya_rows)
    oasst_examples, oasst_skipped = build_oasst_examples(oasst_rows)
    stats.update({f"oasst_skipped_{k}": v for k, v in oasst_skipped.items()})
    stats["aya_examples"] = len(aya_examples)
    stats["oasst_examples"] = len(oasst_examples)

    # normalization + quality rejects
    kept: List[dict] = []
    for ex in aya_examples + oasst_examples:
        turns = [(r, norm_text(t)) for r, t in ex["turns"]]
        if any(not t for _, t in turns):
            stats["rejected_empty"] += 1
            continue
        if any(is_corrupted(t) for _, t in turns):
            stats["rejected_corrupted_unicode"] += 1
            continue
        if any(is_pathological(t) for _, t in turns):
            stats["rejected_pathological_repeats"] += 1
            continue
        ex["turns"] = turns
        kept.append(ex)
    stats["after_quality"] = len(kept)

    # exact (prompt, target) duplicate rejection across sources (keep first)
    seen_pairs: set = set()
    deduped: List[dict] = []
    for ex in kept:
        key = hash(tuple((r, t) for r, t in ex["turns"]))
        pair_key = tuple(t for _, t in ex["turns"][-2:])
        if pair_key in seen_pairs:
            stats["rejected_duplicates"] += 1
            continue
        seen_pairs.add(pair_key)
        deduped.append(ex)
    stats["after_dedup"] = len(deduped)

    # eval-probe exclusion (never train on the Part K/S prompts)
    probe_set = {norm_text(p).lower() for p in EVAL_PROBES}
    no_probe: List[dict] = []
    for ex in deduped:
        if ex["turns"][-2][1].lower() in probe_set:
            stats["rejected_eval_probes"] += 1
            continue
        no_probe.append(ex)
    stats["after_probe_exclusion"] = len(no_probe)

    # tokenization + <unk> analysis + target-unk rejection
    tokenized: List[dict] = []
    unk_stats = {"prompt_tokens": 0, "prompt_unk": 0, "target_tokens": 0, "target_unk": 0}
    for ex in no_probe:
        prompt_ids = tok.encode(ex["turns"][-2][1])
        target_ids = tok.encode(ex["turns"][-1][1])
        unk_stats["prompt_tokens"] += len(prompt_ids)
        unk_stats["prompt_unk"] += prompt_ids.count(special["unk"])
        unk_stats["target_tokens"] += len(target_ids)
        unk_stats["target_unk"] += target_ids.count(special["unk"])
        if special["unk"] in target_ids:
            stats["rejected_target_unk"] += 1
            continue
        built = tokenize_example(ex, tok, stats)
        if built is None:
            continue
        tokenized.append({**ex, "ids": built["ids"], "labels": built["labels"],
                          "n_supervised": built["n_supervised"]})
    stats["final_examples"] = len(tokenized)

    # splits
    train_rows, val_rows = [], []
    for ex in tokenized:
        if ex["split"] == "val":
            val_rows.append(ex)
        else:
            train_rows.append(ex)
    # OASST source validation split; Aya deterministic 95/5 already applied.

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "sft_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def supervised_tokens(rows) -> int:
        return sum(r["n_supervised"] for r in rows)

    def lang_counts(rows) -> Counter:
        return Counter(r["lang"] for r in rows)

    sft_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": str(TOKENIZER_MODEL),
        "block_size": BLOCK_SIZE,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_supervised_target_tokens": supervised_tokens(train_rows),
        "val_supervised_target_tokens": supervised_tokens(val_rows),
        "total_supervised_target_tokens": supervised_tokens(train_rows) + supervised_tokens(val_rows),
        "train_lang_examples": dict(lang_counts(train_rows)),
        "val_lang_examples": dict(lang_counts(val_rows)),
        "train_lang_target_tokens": {
            k: sum(r["n_supervised"] for r in train_rows if r["lang"] == k)
            for k in sorted(lang_counts(train_rows))
        },
        "val_lang_target_tokens": {
            k: sum(r["n_supervised"] for r in val_rows if r["lang"] == k)
            for k in sorted(lang_counts(val_rows))
        },
        "tokenizer_unk": {
            "prompt_token_count": unk_stats["prompt_tokens"],
            "prompt_unk_count": unk_stats["prompt_unk"],
            "prompt_unk_rate": round(unk_stats["prompt_unk"] / max(1, unk_stats["prompt_tokens"]), 6),
            "target_token_count": unk_stats["target_tokens"],
            "target_unk_count": unk_stats["target_unk"],
            "target_unk_rate": round(unk_stats["target_unk"] / max(1, unk_stats["target_tokens"]), 6),
        },
        "example_percent_with_target_unk": round(
            stats["rejected_target_unk"] / max(1, len(no_probe)) * 100, 3),
        "counters": dict(stats),
        "outputs": {
            "train": str(OUT_DIR / "sft_train.jsonl"),
            "val": str(OUT_DIR / "sft_val.jsonl"),
        },
    }
    with open(STATS_DIR / "sft_stats.json", "w", encoding="utf-8") as f:
        json.dump(sft_stats, f, indent=2)
        f.write("\n")

    print(json.dumps(sft_stats, indent=2))
    print("DONE.")


if __name__ == "__main__":
    main()