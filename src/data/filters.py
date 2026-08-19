"""Document and line quality filters, plus a from-scratch character n-gram
language classifier trained only on our own corpus samples.

No pretrained models are used anywhere in this module.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Tuple

MIN_DOC_CHARS = 300
MAX_DOC_CHARS = 1_500_000
MAX_REPEATED_CHAR_RATIO = 0.35
MIN_MEAN_LINE_LEN = 10.0
MIN_PROSE_LINE_RATIO = 0.55
MAX_DUPLICATE_LINE_RATIO = 0.30

WIKI_BOILERPLATE = re.compile(
    r"Jump to navigation|^\s*Categories:|Wikimedia|Hidden categories|"
    r"Retrieved from|CS1 errors|stub\b|^\{\{Infobox|For other uses|"
    r"Coordinates:|Help:|This article does not cite|^\s*\[\s*edit\s*\]\s*$",
    re.I,
)


def repeated_char_ratio(text: str) -> float:
    """Fraction of characters that appear in runs of 5+ identical chars."""
    runs = re.findall(r"((.)\2{4,})", text)
    if not runs:
        return 0.0
    return min(1.0, sum(len(r) for r, _ in runs) / max(1, len(text)))


def duplicate_line_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    seen = set()
    dup = 0
    for ln in lines:
        if ln in seen:
            dup += 1
        seen.add(ln)
    return dup / len(lines)


def prose_line_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    long = sum(1 for ln in lines if len(ln) >= 20)
    return long / len(lines)


def is_boilerplate(text: str) -> bool:
    return bool(WIKI_BOILERPLATE.search(text))


def check_document_quality(text: str) -> Tuple[bool, str]:
    """Returns (ok, reason). Reason empty when ok."""
    n = len(text)
    if n < MIN_DOC_CHARS:
        return False, "too_short"
    if n > MAX_DOC_CHARS:
        return False, "too_long"
    if repeated_char_ratio(text) > MAX_REPEATED_CHAR_RATIO:
        return False, "repetitive_chars"
    if duplicate_line_ratio(text) > MAX_DUPLICATE_LINE_RATIO:
        return False, "duplicate_lines"
    if prose_line_ratio(text) < MIN_PROSE_LINE_RATIO:
        return False, "non_prose"
    if is_boilerplate(text):
        return False, "boilerplate"
    return True, ""


class LanguageScorer:
    """Character n-gram language classifier fit on our own corpus samples."""

    def __init__(self, n: int = 3, k: int = 1):
        self.n = n
        self.k = k
        self._profiles: Dict[str, Dict[str, float]] = {}
        self._vocab: set = set()

    @staticmethod
    def _grams(text: str) -> list:
        s = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        s = re.sub(r"\s+", " ", s).strip()
        return [s[i:i + 3] for i in range(max(0, len(s) - 2))]

    def fit(self, lang: str, texts) -> None:
        counts: Dict[str, int] = {}
        total = 0
        for t in texts:
            for g in self._grams(t):
                counts[g] = counts.get(g, 0) + 1
                total += 1
        vocab = set(counts)
        self._vocab |= vocab
        prof = {}
        for g, c in counts.items():
            prof[g] = math.log((c + self.k) / (total + self.k * (len(vocab) + 1)))
        self._profiles[lang] = prof

    def score(self, lang: str, text: str) -> float:
        prof = self._profiles[lang]
        grams = self._grams(text)
        if not grams:
            return float("-inf")
        unk = math.log(self.k / 1e9)
        total = 0.0
        for g in grams:
            total += prof.get(g, unk)
        return total / len(grams)

    def predict(self, text: str) -> Tuple[str, float]:
        best_lang, best_score = None, float("-inf")
        for lang in self._profiles:
            s = self.score(lang, text)
            if s > best_score:
                best_lang, best_score = lang, s
        return best_lang, best_score

    def predict_margin(self, text: str) -> Tuple[str, float]:
        """Returns (lang, margin) where margin = best score - second best."""
        scores = {lang: self.score(lang, text) for lang in self._profiles}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if len(ordered) < 2:
            return ordered[0][0], 0.0
        return ordered[0][0], ordered[0][1] - ordered[1][1]


def word_count(text: str) -> int:
    return len(text.split())