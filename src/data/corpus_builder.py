"""Corpus building orchestration: streaming doc pipeline, deterministic
document-level train/validation/test splits, JSONL document I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterator, List, Optional

TRAIN_FRAC = 0.98
VAL_FRAC = 0.01
TEST_FRAC = 0.01


def doc_split(doc_id: str) -> str:
    """Deterministic document-level split: 98% train / 1% val / 1% test.

    A document (and any exact duplicate with the same id hash) always lands
    in the same split.
    """
    bucket = int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:4], "big") % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "val"
    return "test"


def read_docs(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_docs(path: str, docs: Iterator[Dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_doc_texts(path: str) -> Iterator[str]:
    for doc in read_docs(path):
        yield doc.get("text", "")


def hash_streaming(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()