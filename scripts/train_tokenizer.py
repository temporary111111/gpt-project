"""Train a SentencePiece tokenizer from scratch on a local raw text corpus.

Usage:
    python scripts/train_tokenizer.py --input-dir data/raw --output-prefix data/tokenizer/sp --vocab-size 8000
    python scripts/train_tokenizer.py --input-files data/processed/corpus_text.txt --output-prefix data/tokenizer/tokenizer_v1 --vocab-size 8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import USER_DEFINED_SYMBOLS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer from scratch")
    p.add_argument("--input-dir", default=None, help="directory of .txt corpus files")
    p.add_argument("--input-files", default=None, help="comma-separated list of text files")
    p.add_argument("--output-prefix", required=True, help="output prefix, e.g. data/tokenizer/sp")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "word", "char"])
    p.add_argument("--languages", default="en,tl")
    return p.parse_args()


def sha256_files(paths: list) -> str:
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    if args.input_files:
        corpus_files = [p.strip() for p in args.input_files.split(",") if p.strip()]
    elif args.input_dir:
        corpus_files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".txt")
        )
    else:
        print("ERROR: provide --input-dir or --input-files")
        sys.exit(1)

    if not corpus_files:
        print(f"ERROR: no input files found")
        sys.exit(1)

    import sentencepiece as spm

    checksum = sha256_files(corpus_files)
    print(f"training SentencePiece ({args.model_type}, vocab={args.vocab_size}) "
          f"on {len(corpus_files)} file(s), corpus sha256={checksum[:16]}...")
    sp_args = {
        "input": corpus_files,
        "model_prefix": args.output_prefix,
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "user_defined_symbols": USER_DEFINED_SYMBOLS,
        "unk_piece": "<unk>",
        "pad_id": -1,
        "bos_id": -1,
        "eos_id": -1,
        "character_coverage": 0.9995,
        "split_digits": True,
    }
    t0 = time.time()
    spm.SentencePieceTrainer.train(**sp_args)
    print(f"tokenizer trained in {time.time() - t0:.0f}s")

    sp = spm.SentencePieceProcessor(model_file=args.output_prefix + ".model")
    special_ids = {tok: sp.piece_to_id(tok) for tok in USER_DEFINED_SYMBOLS}
    special_ids["<unk>"] = sp.unk_id()
    meta = {
        "vocab_size": sp.get_piece_size(),
        "model_type": args.model_type,
        "training_files": corpus_files,
        "corpus_sha256": checksum,
        "languages": [l for l in args.languages.split(",") if l],
        "sentencepiece_args": {k: (list(v) if isinstance(v, list) else v)
                               for k, v in sp_args.items()},
        "pad_id": special_ids["<pad>"],
        "bos_id": special_ids["<bos>"],
        "eos_id": special_ids["<eos>"],
        "unk_id": special_ids["<unk>"],
        "system_id": special_ids["<system>"],
        "user_id": special_ids["<user>"],
        "assistant_id": special_ids["<assistant>"],
        "special_tokens": list(special_ids.keys()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"done. model saved to {args.output_prefix}.model")
    print(f"vocab size: {sp.get_piece_size()}")
    print(f"special token ids: {special_ids}")


if __name__ == "__main__":
    main()