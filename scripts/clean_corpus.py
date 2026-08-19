"""Clean the raw corpus: normalize, filter, language-verify, deduplicate,
assign deterministic splits, and write cleaned JSONL per language.

Usage:
    .venv\\Scripts\\python.exe scripts\\clean_corpus.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.corpus_builder import doc_split, read_docs  # noqa: E402
from src.data.deduplicate import ExactDedup, MinHashNearDup  # noqa: E402
from src.data.filters import LanguageScorer, check_document_quality  # noqa: E402
from src.data.normalize import normalize_text, strip_wiki_markup  # noqa: E402

CLEANED_OUT = {
    "en": os.path.join("data", "cleaned", "en", "all.jsonl"),
    "tl": os.path.join("data", "cleaned", "tl", "all.jsonl"),
}
STATS_PATH = os.path.join("data", "manifests", "cleaned_stats.json")
SAMPLE_DOCS = 400


def fit_language_scorer() -> LanguageScorer:
    scorer = LanguageScorer(n=3)
    for lang in ("en", "tl"):
        samples = []
        for raw_file in sorted(os.listdir(os.path.join("data", "raw", lang))):
            path = os.path.join("data", "raw", lang, raw_file)
            for i, doc in enumerate(read_docs(path)):
                if i >= SAMPLE_DOCS // max(1, len(os.listdir(os.path.join("data", "raw", lang)))):
                    break
                samples.append(doc["text"])
        scorer.fit(lang, samples)
    return scorer


def clean_file(raw_path: str, lang: str, scorer: LanguageScorer,
               exact: ExactDedup, near: MinHashNearDup, stats: dict):
    accepted = 0
    for doc in read_docs(raw_path):
        text = normalize_text(doc.get("text", ""))
        if doc.get("source", "").endswith("wiki") or doc.get("source", "").endswith("wikisource"):
            text = strip_wiki_markup(text)
            text = normalize_text(text)
        if not text:
            stats["rejected"]["empty"] += 1
            continue
        ok, reason = check_document_quality(text)
        if not ok:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        pred, margin = scorer.predict_margin(text)
        if pred != lang or margin < 0.5:
            stats["rejected"]["wrong_language"] = stats["rejected"].get("wrong_language", 0) + 1
            continue
        if exact.is_duplicate(text):
            stats["dedup_exact"] += 1
            continue
        if near.is_near_duplicate(text):
            stats["dedup_near"] += 1
            continue
        doc_id = f"{doc['source']}:{doc['doc_id']}"
        clean_doc = {
            "source": doc["source"],
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "lang": lang,
            "split": doc_split(doc_id),
            "text": text,
        }
        yield clean_doc
        accepted += 1
        stats["chars"] += len(text)
        stats["words"] += len(text.split())
    print(f"  {raw_path}: {accepted} accepted")


def main():
    print("fitting language scorer on corpus samples ...")
    scorer = fit_language_scorer()

    stats = {
        "input_docs": {"en": 0, "tl": 0},
        "accepted": {"en": 0, "tl": 0},
        "rejected": {"empty": 0, "wrong_language": 0},
        "dedup_exact": 0,
        "dedup_near": 0,
        "chars": 0,
        "words": 0,
    }

    exact = ExactDedup()
    near = MinHashNearDup(seed=2024)

    for lang in ("en", "tl"):
        raw_dir = os.path.join("data", "raw", lang)
        os.makedirs(os.path.dirname(CLEANED_OUT[lang]), exist_ok=True)
        with open(CLEANED_OUT[lang], "w", encoding="utf-8") as out:
            for fname in sorted(os.listdir(raw_dir)):
                if not fname.endswith(".jsonl"):
                    continue
                raw_path = os.path.join(raw_dir, fname)
                stats["input_docs"][lang] += sum(1 for _ in read_docs(raw_path))
                print(f"cleaning {raw_path} ...")
                n = 0
                for clean_doc in clean_file(raw_path, lang, scorer, exact, near, stats):
                    out.write(json.dumps(clean_doc, ensure_ascii=False) + "\n")
                    n += 1
                stats["accepted"][lang] += n

    total_in = stats["input_docs"]["en"] + stats["input_docs"]["tl"]
    total_out = stats["accepted"]["en"] + stats["accepted"]["tl"]
    print(f"\ninput docs: {total_in}  accepted: {total_out}  "
          f"exact-dup removed: {stats['dedup_exact']}  near-dup removed: {stats['dedup_near']}")

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"stats written to {STATS_PATH}")


if __name__ == "__main__":
    main()