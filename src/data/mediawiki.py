"""MediaWiki wikitext -> plain text extraction (streaming-friendly).

Implements balanced removal of templates/ tables, stripping of refs, files,
categories, links, headings and tags. Keeps article prose.
"""

from __future__ import annotations

import html
import re


def _strip_balanced(text: str, open_delim: str, close_delim: str) -> str:
    """Removes regions between balanced open/close delimiters (e.g. {{ }})."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(open_delim, i):
            depth = 0
            j = i
            while j < n:
                if text.startswith(open_delim, j):
                    depth += 1
                    j += len(open_delim)
                elif text.startswith(close_delim, j):
                    depth -= 1
                    j += len(close_delim)
                    if depth == 0:
                        break
                else:
                    j += 1
            if depth == 0:
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _strip_tables(text: str) -> str:
    return _strip_balanced(text, "{|", "|}")


def extract_mw_text(wikitext: str) -> str:
    s = wikitext
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<ref[^>/]*>.*?</ref>", "", s, flags=re.S | re.I)
    s = re.sub(r"<ref[^>]*/>", "", s, flags=re.I)
    s = re.sub(r"<gallery[^>]*>.*?</gallery>", "", s, flags=re.S | re.I)
    s = re.sub(r"<math[^>]*>.*?</math>", "", s, flags=re.S | re.I)
    s = _strip_tables(s)
    s = _strip_balanced(s, "{{", "}}")
    s = re.sub(r"<nowiki[^>]*>(.*?)</nowiki>", r"\1", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(
        r"\[\[(?:File|Image|Category|Template|Media|Wikipedia|Help|Portal|"
        r"Special|Module):[^\]]*\]\]",
        "", s, flags=re.I,
    )
    def repl_link(m):
        inner = m.group(1)
        return inner.split("|", 1)[1] if "|" in inner else inner
    s = re.sub(r"\[\[([^\]]+)\]\]", repl_link, s)
    s = re.sub(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]+)\]", r"\1", s)
    s = re.sub(r"\[(?:https?|ftp)://[^\s\]]+\]", "", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", s, flags=re.M)
    s = re.sub(r"^[*#;:]+", "", s, flags=re.M)
    s = re.sub(r"^\s*\{\|\s*$", "", s, flags=re.M)
    s = html.unescape(s)
    lines = []
    for ln in s.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)