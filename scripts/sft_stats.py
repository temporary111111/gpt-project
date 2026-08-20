"""SFT dataset statistics (TASK 005 Part D/E + TASK 005.1 Part L/S).

Prints and optionally rewrites data/sft/stats/sft_stats.json from the current
processed SFT files, including UNIQUE vs EFFECTIVE (sampling-weighted) metrics
and the unique supervised-token gate. Deterministic.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\sft_stats.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import SentencePieceTokenizer  # noqa: E402

STATS_DIR = ROOT / "data" / "sft" / "stats"
TOKENIZER_META = ROOT / "data" / "tokenizer" / "tokenizer_v1_meta.json"

GATE_FLOOR = 1_000_000


def load(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    train = load(ROOT / "data" / "sft" / "processed" / "sft_train.jsonl")
    val = load(ROOT / "data" / "sft" / "processed" / "sft_val.jsonl")
    tok = SentencePieceTokenizer(str(ROOT / "data" / "tokenizer" / "tokenizer_v1.model"),
                                 str(TOKENIZER_META))
    unk = tok.unk_id

    def sup_tokens(rows):
        return sum(r["n_supervised"] for r in rows)

    def langs(rows):
        return Counter(r["lang"] for r in rows)

    def sources(rows):
        return Counter(r["source"] for r in rows)

    def lang_tokens(rows):
        return {k: sum(r["n_supervised"] for r in rows if r["lang"] == k)
                for k in sorted(langs(rows))}

    def unk_rate(rows):
        pt = pt_unk = tt = t_unk = 0
        for r in rows:
            i = r["ids"]
            l = r["labels"]
            start = next((n for n, v in enumerate(l) if v != -100), len(i))
            tt += len(i) - start
            t_unk += sum(1 for v in i[start:] if v == unk)
            pt += start
            pt_unk += sum(1 for v in i[:start] if v == unk)
        return {"prompt_tokens": pt, "prompt_unk": pt_unk,
                "prompt_unk_rate": round(pt_unk / max(1, pt), 6),
                "target_tokens": tt, "target_unk": t_unk,
                "target_unk_rate": round(t_unk / max(1, tt), 6)}

    unique_total = sup_tokens(train) + sup_tokens(val)
    unique_fil = sum(r["n_supervised"] for r in train + val if r["lang"] == "fil")
    eff_examples = sum(r.get("copies", 1) for r in train)
    eff_tokens = sum(r["n_supervised"] * r.get("copies", 1) for r in train)
    eff_fil_tokens = sum(r["n_supervised"] * r.get("copies", 1)
                         for r in train if r["lang"] == "fil")
    eff_src_examples = {s: sum(r.get("copies", 1) for r in train if r["source"] == s)
                        for s in sorted(sources(train))}

    stats = {
        "unique_examples": len(train) + len(val),
        "train_examples": len(train),
        "val_examples": len(val),
        "unique_supervised_target_tokens": unique_total,
        "train_supervised_target_tokens": sup_tokens(train),
        "val_supervised_target_tokens": sup_tokens(val),
        "unique_fil_target_tokens": unique_fil,
        "unique_en_target_tokens": unique_total - unique_fil,
        "train_lang_examples": dict(langs(train)),
        "val_lang_examples": dict(langs(val)),
        "train_lang_target_tokens": lang_tokens(train),
        "val_lang_target_tokens": lang_tokens(val),
        "unique_source_examples": dict(sources(train + val)),
        "unique_source_target_tokens": {
            s: sum(r["n_supervised"] for r in train + val if r["source"] == s)
            for s in sorted(sources(train + val))
        },
        "sampling": {
            "effective_train_examples": eff_examples,
            "unique_train_examples": len(train),
            "effective_train_supervised_tokens": eff_tokens,
            "unique_train_supervised_tokens": sup_tokens(train),
            "effective_fil_supervised_tokens": eff_fil_tokens,
            "effective_fil_share": round(eff_fil_tokens / max(1, eff_tokens), 4),
            "effective_source_examples": eff_src_examples,
        },
        "gate": {
            "floor": GATE_FLOOR,
            "UNIQUE_SUPERVISED_TARGET_TOKENS": unique_total,
            "gate_passed": unique_total >= GATE_FLOOR,
            "note": "unique accepted human tokens BEFORE oversampling; "
                    "weighted repeats never count toward this gate",
        },
        "unk": {"train": unk_rate(train), "val": unk_rate(val)},
        "max_length": max(len(r["ids"]) for r in train + val),
    }
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_DIR / "sft_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()