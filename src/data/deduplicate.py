"""Exact and near-duplicate document detection.

- Exact: SHA-1 of normalized text.
- Near: MinHash-style fingerprints over word 5-gram shingles. A document is
  a near-duplicate when it shares at least NEAR_DUP_REQUIRED of its
  signature hashes with an earlier document. Memory use is bounded by
  (number of documents x number of hashes) - conservative on 8 GB RAM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def gram_hash(gram: str) -> int:
    """Deterministic 64-bit hash with strong avalanche (BLAKE2b).

    FNV-1a (and crc32) mix short ASCII strings poorly: the low bits of the
    output are dominated by the low bits of the input bytes, so common
    space-crossing character windows hash to consistently small values and
    dominate top-K signatures (false near-duplicates across unrelated docs).
    BLAKE2b is stdlib and hashes tiny inputs uniformly.
    """
    return int.from_bytes(
        hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"
    )


class ExactDedup:
    """Streaming exact-document deduplication by normalized-text SHA-1."""

    def __init__(self):
        self._seen: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        key = sha1_hex(text)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    @property
    def count(self) -> int:
        return len(self._seen)


class MinHashNearDup:
    """Near-duplicate detection via top-K signature hashes.

    For each document, hash its word 5-gram shingles and keep the K smallest
    hash values as a signature (uniform hashing makes those correspond to
    uniformly random shingles of the document). A new document is a
    near-duplicate when at least REQUIRED_SHARED of its signature hashes
    were already seen in earlier documents' signatures. Two unrelated
    documents share a signature hash with probability ~ (grams/2**64), so
    false positives are negligible, while edited copies of the same document
    share most of their signature.

    Word shingles (not character windows) keep the signature doc-specific:
    character windows cross word boundaries and repeat across any English
    text ("the q", "he qu"), which dominated the top-K signature and made
    unrelated documents look near-duplicate.

    Deterministic (pure hashing) and memory-bounded: one Python set holding
    K * N distinct integers.
    """

    def __init__(self, k: int = 12, gram_k: int = 5, required_shared: int = 8,
                 seed: int = 1337):
        self.k = k
        self.gram_k = gram_k
        self.required_shared = required_shared
        self.seed = seed
        self._seen: set = set()
        self._doc_count = 0
        self.removed = 0

    def _grams(self, text: str, max_chars: int = 250_000) -> list:
        words = _WORD_RE.findall(text.lower())
        kept: list = []
        used = 0
        for w in words:
            if used + len(w) + 1 > max_chars:
                break
            kept.append(w)
            used += len(w) + 1
        if len(kept) < self.gram_k:
            return []
        return [
            " ".join(kept[i:i + self.gram_k])
            for i in range(len(kept) - self.gram_k + 1)
        ]

    def _signature(self, text: str) -> Tuple[int, ...]:
        grams = self._grams(text)
        if not grams:
            return tuple()
        sig = sorted(gram_hash(g) for g in grams)[:self.k]
        return tuple(sig)

    def is_near_duplicate(self, text: str) -> bool:
        sig = self._signature(text)
        if not sig:
            return False
        shared = sum(1 for h in sig if h in self._seen)
        self._seen.update(sig)
        self._doc_count += 1
        if shared >= self.required_shared:
            self.removed += 1
            return True
        return False