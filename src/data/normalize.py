"""Text normalization utilities.

Deliberately conservative: we normalize Unicode and whitespace but do NOT
aggressively strip punctuation or capitalization - the model should learn
real language.
"""

from __future__ import annotations

import html
import re
import unicodedata

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RUN = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
HTML_TAG = re.compile(r"<[^>]+>")


def normalize_text(text: str) -> str:
    """NFC-normalize, decode entities, remove control chars, normalize whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    text = CONTROL_CHARS.sub("", text)
    text = text.replace("\ufeff", "")
    lines = text.splitlines()
    out = []
    for line in lines:
        line = WHITESPACE_RUN.sub(" ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def strip_html_tags(text: str) -> str:
    return HTML_TAG.sub("", text)


def strip_underscores(text: str) -> str:
    """Gutenberg marks italics with underscores; underscores are not language."""
    return text.replace("_", "")


def collapse_blank_lines(text: str) -> str:
    return "\n\n".join(line for line in text.split("\n") if line.strip())


_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_FILE_LINK_RE = re.compile(r"\[\[\s*(?:File|Image|Talaksan)\s*:[^\[\]]*\|([^\[\]]*)\]\]", re.I)
_PIPED_LINK_RE = re.compile(r"\[\[[^\[\]]*\|([^\[\]]*)\]\]")
_PLAIN_LINK_RE = re.compile(r"\[\[([^\[\]]*)\]\]")
_THUMB_RE = re.compile(r"\bthumb\s*\|?\b", re.I)
_PX_RE = re.compile(r"\b\d{1,4}px\s*\|?", re.I)
_IMG_PARAM_RE = re.compile(
    r"(?m)^\s*(?:left|right|center|frame|thumb|none|\d{1,4}px|\d{1,4}x\d{1,4}px)\s*\|", re.I
)
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")


def strip_wiki_markup(text: str) -> str:
    """Remove MediaWiki markup artifacts (templates, links, bold/italic).

    Only applied to wiki-derived documents. Keeps link display text and
    file captions where available; drops the rest.
    """
    text = _TEMPLATE_RE.sub(" ", text)
    text = _FILE_LINK_RE.sub(lambda m: m.group(1).split("|")[-1], text)
    text = _PIPED_LINK_RE.sub(r"\1", text)
    text = _PLAIN_LINK_RE.sub(r"\1", text)
    text = text.replace("[[", "").replace("]]", "")
    text = text.replace("{{", " ").replace("}}", " ")
    text = text.replace("'''''", "").replace("''''", "")
    text = text.replace("'''", "").replace("''", "")
    text = _THUMB_RE.sub("", text)
    text = _PX_RE.sub("", text)
    text = _IMG_PARAM_RE.sub("", text)
    text = _EMPTY_PAREN_RE.sub("", text)
    return text