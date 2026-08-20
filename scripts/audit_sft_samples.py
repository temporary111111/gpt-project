"""TASK 005.1 Part I manual quality audit helper.

Samples 30 examples per source (Aya fil, Aya eng, OASST, Dolly, Taskmaster)
deterministically from data/sft/processed and prints a compact review table
plus a full-text dump file. Reviews are MANUAL (the engineer reads the output
and records per-source verdicts in the task report); nothing is rewritten.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\audit_sft_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PER_SOURCE = 30
SEED = 1337


def load(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_deterministic(rows, n, seed):
    """Deterministic pseudo-random sample (stable across runs)."""
    out = []
    for i in range(n):
        idx = (seed * 2654435761 + i * 40503) % 2**31
        out.append(rows[idx % len(rows)])
    return out


def main() -> None:
    train = load(ROOT / "data" / "sft" / "processed" / "sft_train.jsonl")
    groups = {
        "aya_fil": [r for r in train if r["source"] == "aya" and r["lang"] == "fil"],
        "aya_eng": [r for r in train if r["source"] == "aya" and r["lang"] == "eng"],
        "oasst": [r for r in train if r["source"] == "oasst1"],
        "dolly": [r for r in train if r["source"] == "dolly"],
        "taskmaster": [r for r in train if r["source"] == "taskmaster1"],
    }
    report_lines = []
    total = 0
    for group, rows in groups.items():
        if len(rows) < PER_SOURCE:
            print(f"WARNING: {group} has only {len(rows)} rows (< {PER_SOURCE})")
        sampled = sample_deterministic(rows, PER_SOURCE, SEED + total)
        total += len(sampled)
        report_lines.append(f"=== {group} ({len(sampled)} samples) ===")
        for r in sampled:
            turns = r["turns"]
            n_turns = len(turns)
            prompt = turns[-2][1]
            target = turns[-1][1]
            report_lines.append(
                f"[{r['id']}] n_turns={n_turns} prompt={prompt[:100]!r} "
                f"target={target[:100]!r}")
        report_lines.append("")

    out = ROOT / "data" / "sft" / "stats" / "quality_audit_samples.txt"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines[:60]))
    print(f"... (full dump: {out})")
    print(f"TOTAL SAMPLES: {total}")


if __name__ == "__main__":
    main()