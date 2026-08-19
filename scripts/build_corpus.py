"""Build the binary pretraining dataset from cleaned documents.

Steps:
1. --corpus-text : write a single streaming plain-text file (data/processed/corpus_text.txt)
   for SentencePiece training, with streaming SHA-256 checksum.
2. --build-bins  : tokenize cleaned docs with tokenizer_v1 and write
   train.bin / validation.bin / test.bin (uint16, <bos>...<eos> document
   boundaries) plus dataset_meta.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data.corpus_builder import read_docs  # noqa: E402
from src.tokenizer import SentencePieceTokenizer  # noqa: E402

CORPUS_TEXT = os.path.join("data", "processed", "corpus_text.txt")
TOKENIZER_PREFIX = os.path.join("data", "tokenizer", "tokenizer_v1")
BIN_DIR = os.path.join("data", "processed")
CLEANED = {
    "en": os.path.join("data", "cleaned", "en", "all.jsonl"),
    "tl": os.path.join("data", "cleaned", "tl", "all.jsonl"),
}


def make_corpus_text() -> dict:
    n_docs = 0
    chars = 0
    h = hashlib.sha256()
    os.makedirs(os.path.dirname(CORPUS_TEXT), exist_ok=True)
    with open(CORPUS_TEXT, "w", encoding="utf-8") as out:
        for lang in ("en", "tl"):
            for doc in read_docs(CLEANED[lang]):
                out.write(doc["text"] + "\n\n")
                n_docs += 1
                chars += len(doc["text"])
    with open(CORPUS_TEXT, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    checksum = h.hexdigest()
    print(f"corpus text: {n_docs:,} docs, {chars:,} chars -> {CORPUS_TEXT}")
    print(f"sha256: {checksum}")
    return {"docs": n_docs, "chars": chars, "sha256": checksum}


def build_bins(en_token_budget: int = 24_000_000,
               tl_token_budget: int = 16_000_000) -> dict:
    """Tokenize cleaned docs and write uint16 bins.

    Documents are consumed in deterministic on-disk order per language until
    the language token budget is reached (budgets keep the corpus at the
    25-50M token target with a 60/40 EN/TL mix).
    """
    tok = SentencePieceTokenizer(TOKENIZER_PREFIX + ".model",
                                 TOKENIZER_PREFIX + "_meta.json")
    os.makedirs(BIN_DIR, exist_ok=True)
    split_files = {
        "train": os.path.join(BIN_DIR, "train.bin"),
        "val": os.path.join(BIN_DIR, "validation.bin"),
        "test": os.path.join(BIN_DIR, "test.bin"),
    }
    counts = {s: {"tokens": 0, "docs": 0, "en_tokens": 0, "tl_tokens": 0} for s in split_files}
    handles = {s: open(p, "wb") for s, p in split_files.items()}
    t0 = time.time()
    try:
        for lang, budget in (("en", en_token_budget), ("tl", tl_token_budget)):
            used = 0
            for doc in read_docs(CLEANED[lang]):
                split = doc["split"]
                ids = tok.encode(doc["text"])
                arr = np.asarray([tok.bos_id] + ids + [tok.eos_id], dtype=np.uint16)
                n = len(arr)
                if used + n > budget and used > 0:
                    break
                used += n
                handles[split].write(arr.tobytes())
                counts[split]["tokens"] += n
                counts[split]["docs"] += 1
                counts[split][f"{lang}_tokens"] += n
                if counts[split]["docs"] % 5000 == 0:
                    print(f"  {split}: {counts[split]['docs']:,} docs ...")
    finally:
        for f in handles.values():
            f.close()
    elapsed = time.time() - t0
    print(f"tokenized in {elapsed:.0f}s")

    meta = {
        "tokenizer_version": "tokenizer_v1",
        "tokenizer_vocab_size": tok.vocab_size,
        "special_token_ids": tok.special_token_ids,
        "dtype": "uint16",
        "document_boundary": "<bos> text <eos>",
        "sampling": {
            "method": "deterministic on-disk order, first-N docs per language",
            "en_token_budget": en_token_budget,
            "tl_token_budget": tl_token_budget,
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    total_tokens = 0
    for split, f in split_files.items():
        size = os.path.getsize(f)
        sha = hashlib.sha256()
        with open(f, "rb") as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                sha.update(block)
        counts[split]["bytes"] = size
        counts[split]["sha256"] = sha.hexdigest()
        total_tokens += counts[split]["tokens"]
    meta["splits"] = counts
    meta["total_tokens"] = total_tokens
    meta["total_documents"] = sum(c["docs"] for c in counts.values())
    with open(os.path.join(BIN_DIR, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-text", action="store_true")
    p.add_argument("--build-bins", action="store_true")
    p.add_argument("--en-tokens", type=int, default=24_000_000)
    p.add_argument("--tl-tokens", type=int, default=16_000_000)
    args = p.parse_args()
    if args.corpus_text or args.build_bins or not (args.corpus_text or args.build_bins):
        if args.corpus_text or not args.build_bins:
            make_corpus_text()
        if args.build_bins or not args.corpus_text:
            build_bins(args.en_tokens, args.tl_tokens)


if __name__ == "__main__":
    main()