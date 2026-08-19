"""Corpus statistics report: raw, cleaned, tokenized, and disk usage.

Usage:
    .venv\\Scripts\\python.exe scripts\\corpus_stats.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.corpus_builder import read_docs  # noqa: E402
from src.data.manifest import load_source_records  # noqa: E402


def dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def count_docs(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    print("=" * 66)
    print("RAW CORPUS")
    print("=" * 66)
    records = load_source_records(os.path.join("data", "manifests", "sources.jsonl"))
    by_source = {}
    for r in records:
        by_source.setdefault(r.source_name, r)
    print(f"manifest records: {len(records)}")
    for lang in ("en", "tl"):
        raw_dir = os.path.join("data", "raw", lang)
        for fname in sorted(os.listdir(raw_dir)):
            path = os.path.join(raw_dir, fname)
            n = count_docs(path)
            print(f"  raw/{lang}/{fname}: {n:,} docs, {os.path.getsize(path) / 1024 ** 2:.1f} MiB")

    print()
    print("=" * 66)
    print("CLEANED CORPUS")
    print("=" * 66)
    stats_path = os.path.join("data", "manifests", "cleaned_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    for lang in ("en", "tl"):
        path = os.path.join("data", "cleaned", lang, "all.jsonl")
        if os.path.exists(path):
            docs = 0
            chars = 0
            words = 0
            split_counts = {"train": 0, "val": 0, "test": 0}
            for doc in read_docs(path):
                docs += 1
                chars += len(doc["text"])
                words += len(doc["text"].split())
                split_counts[doc["split"]] += 1
            print(f"  cleaned/{lang}: {docs:,} docs, {chars:,} chars, {words:,} words, "
                  f"splits {split_counts}")

    print()
    print("=" * 66)
    print("TOKENIZED DATASET")
    print("=" * 66)
    meta_path = os.path.join("data", "processed", "dataset_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))
    else:
        print("  not built yet")

    print()
    print("=" * 66)
    print("DISK USAGE")
    print("=" * 66)
    for d in ("data/sources", "data/raw", "data/cleaned", "data/processed", "data/tokenizer"):
        b = dir_bytes(d)
        print(f"  {d}: {b / 1024 ** 2:.1f} MiB")
    free = shutil.disk_usage(os.getcwd()).free / 1024 ** 3
    print(f"  free disk: {free:.1f} GB")


if __name__ == "__main__":
    main()