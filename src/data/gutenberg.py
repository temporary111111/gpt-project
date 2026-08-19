"""Project Gutenberg plain-text extraction.

Strips the boilerplate header/footer and the full-license block, keeping
only the book body. Public-domain texts only.
"""

from __future__ import annotations

import re

_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK[^*]*\*{3}", re.I)
_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK[^*]*\*{3}", re.I)
_LICENSE_RE = re.compile(r"\*{3}\s*START: FULL LICENSE\s*\*{3}", re.I)
_STARS_LINE = re.compile(r"^\s*\*{3,}\s*$")
_ETEXT_FOOTER_RE = re.compile(r"End of (?:the )?Project Gutenberg Etext", re.I)


def decode_text(data: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_gutenberg_body(text: str) -> str:
    has_markers = bool(_START_RE.search(text))
    m = _START_RE.search(text)
    body = text[m.end():] if m else text
    m = _END_RE.search(body)
    if m:
        body = body[:m.start()]
    m = _LICENSE_RE.search(body)
    if m:
        body = body[:m.start()]
    if not has_markers:
        # Legacy 1970s E-texts have no markers: the editorial preamble ends
        # at the last standalone "***" separator before the work itself.
        lines = body.splitlines()
        last_sep = -1
        for i, ln in enumerate(lines):
            if _STARS_LINE.match(ln):
                last_sep = i
        if last_sep >= 0:
            body = "\n".join(lines[last_sep + 1:])
    m = _ETEXT_FOOTER_RE.search(body)
    if m:
        body = body[:m.start()]
    body = body.replace("_", "")
    body = body.replace("\f", "")
    return body.strip()