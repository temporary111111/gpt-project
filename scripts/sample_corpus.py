"""Random sample of cleaned documents for manual quality inspection.

Prints short excerpts of 20 English and 20 Filipino documents (seeded).

Usage:
    .venv\\Scripts\\python.exe scripts\\sample_corpus.py [--per-lang 20] [--excerpt-chars 300]
"""

from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.corpus_builder import read_docs  # noqa: E402


def sample(path: str, per_lang: int, seed: int, excerpt: int) -> list:
    doc_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc_ids.append(i)
    rng = random.Random(seed)
    chosen = set(rng.sample(doc_ids, min(per_lang, len(doc_ids))))
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in chosen:
                import json
                out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=20)
    p.add_argument("--excerpt-chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    for lang, path in (("EN", os.path.join("data", "cleaned", "en", "all.jsonl")),
                       ("TL", os.path.join("data", "cleaned", "tl", "all.jsonl"))):
        print("=" * 70)
        print(f"SAMPLE {lang} DOCUMENTS ({args.per_lang})")
        print("=" * 70)
        for doc in sample(path, args.per_lang, args.seed, args.excerpt_chars):
            text = doc["text"].replace("\n", " ")[:args.excerpt_chars]
            print(f"[{doc['source']} | {doc['split']}] {doc['doc_id']}")
            print(f"  {text}...")
            print()


if __name__ == "__main__":
    main()