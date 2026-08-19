"""Tokenizer V1 quality test on manually written unseen sentences.

Usage:
    .venv\\Scripts\\python.exe scripts\\tokenizer_quality_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import SentencePieceTokenizer  # noqa: E402

TEST_SENTENCES = {
    "en": [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Learning a language model from scratch takes patience and data.",
        "She reads books at the library every Saturday morning.",
    ],
    "tl": [
        "Kumusta ka? Ayos ka lang ba?",
        "Masarap kumain ng mangga kapag tag-init.",
        "Gusto kong matutong gumawa ng sarili kong language model.",
        "Ang bata ay naglalaro sa labas habang umuulan.",
    ],
    "mixed": [
        "Hello! Kumusta ang training ng model natin?",
    ],
}


def stats(tok: SentencePieceTokenizer, sentence: str) -> dict:
    ids = tok.encode(sentence)
    decoded = tok.decode(ids)
    words = len(sentence.split())
    return {
        "tokens": ids,
        "decoded": decoded,
        "tokens_per_char": round(len(ids) / max(1, len(sentence)), 3),
        "tokens_per_word": round(len(ids) / max(1, words), 3),
        "unk_count": sum(1 for i in ids if i == tok.unk_id),
        "matches": decoded == sentence,
    }


def main():
    tok = SentencePieceTokenizer(
        os.path.join("data", "tokenizer", "tokenizer_v1.model"),
        os.path.join("data", "tokenizer", "tokenizer_v1_meta.json"),
    )
    print(f"tokenizer: tokenizer_v1 | vocab_size={tok.vocab_size}")
    print(f"special ids: {tok.special_token_ids}")
    print()
    total_unk = 0
    for lang, sentences in TEST_SENTENCES.items():
        print(f"--- {lang.upper()} ---")
        for s in sentences:
            r = stats(tok, s)
            total_unk += r["unk_count"]
            print(f"  in : {s}")
            print(f"  ids: {r['tokens']}")
            print(f"  out: {r['decoded']}")
            print(f"  tok/char={r['tokens_per_char']} tok/word={r['tokens_per_word']} "
                  f"unk={r['unk_count']} exact_roundtrip={r['matches']}")
        print()
    print(f"total <unk> tokens across test set: {total_unk}")


if __name__ == "__main__":
    main()