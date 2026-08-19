"""Unit tests for the Task 003 data pipeline modules."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.corpus_builder import doc_split, read_docs, write_docs  # noqa: E402
from src.data.deduplicate import ExactDedup, MinHashNearDup, gram_hash  # noqa: E402
from src.data.filters import LanguageScorer, check_document_quality  # noqa: E402
from src.data.gutenberg import decode_text, extract_gutenberg_body  # noqa: E402
from src.data.normalize import (  # noqa: E402
    normalize_text,
    strip_html_tags,
    strip_wiki_markup,
)


def test_normalize_whitespace_and_control_chars():
    assert normalize_text("  a\t\u00a0b  \n \n  c \r\n") == "a b\nc"
    assert normalize_text("bad\x00ctrl\x1fchars") == "badctrlchars"
    assert normalize_text("cafe\u0301") == "caf\u00e9"  # NFC composed


def test_strip_html_tags():
    assert strip_html_tags("a <b>bold</b> c") == "a bold c"


def test_strip_wiki_markup():
    text = (
        "'''Paris''' is a [[city|large city]] of [[France]]. "
        "{{infobox|pop=1}}\n"
        "[[File:map.png|thumb|right|250px|A map of the city]]\n"
        "''Italic'' text."
    )
    out = strip_wiki_markup(text)
    assert "'''" not in out and "''" not in out
    assert "{{" not in out and "}}" not in out
    assert "[[" not in out and "]]" not in out
    assert "large city" in out and "France" in out
    assert "A map of the city" in out
    assert "250px" not in out and "thumb" not in out and "right|" not in out


def test_quality_filters_accept_prose():
    ok, reason = check_document_quality(
        "The quick brown fox jumps over the lazy dog. " * 30
    )
    assert ok
    assert not reason


def test_quality_filters_reject_garbage():
    ok, reason = check_document_quality("a" * 1000)
    assert not ok and reason == "repetitive_chars"
    ok, reason = check_document_quality("a " * 3)
    assert not ok and reason == "too_short"


def test_language_scorer_en_tl():
    scorer = LanguageScorer(n=3)
    scorer.fit("en", [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "This is a simple English sentence about everyday things. " * 10,
    ])
    scorer.fit("tl", [
        "Ang mabilis na aso ay tumatakbo sa labas ng bahay. " * 20,
        "Ang mga bata ay naglalaro sa kalsada tuwing umaga. " * 10,
    ])
    pred, margin = scorer.predict_margin(
        "English words appear in ordinary English documents all the time. " * 20
    )
    assert pred == "en" and margin >= 0.5
    pred, margin = scorer.predict_margin(
        "Ang wikang Tagalog ay ginagamit ng maraming tao sa Pilipinas. " * 20
    )
    assert pred == "tl" and margin >= 0.5


def test_doc_split_deterministic():
    assert doc_split("simplewiki:Paris") == doc_split("simplewiki:Paris")
    splits = {doc_split(f"src:{i}") for i in range(5000)}
    assert splits <= {"train", "val", "test"}


def test_gram_hash_deterministic_and_mixed():
    assert gram_hash("hello") == gram_hash("hello")
    a = set(gram_hash(g) for g in "the quick brown fox jumps".split())
    b = set(gram_hash(g) for g in "ang mabilis na aso ay".split())
    assert a.isdisjoint(b) or len(a & b) < 2


def test_exact_dedup():
    d = ExactDedup()
    assert not d.is_duplicate("same text")
    assert d.is_duplicate("same text")
    assert not d.is_duplicate("different text")


def test_near_dedup_finds_edited_copy_but_not_unrelated():
    near = MinHashNearDup()
    base = " ".join(
        f"The quick brown fox jumps over the lazy dog number {i}." for i in range(40)
    )
    edit = base.replace("number 17", "number seventeen")
    other = " ".join(
        f"Ang mga bata ay naglalaro sa kalsada tuwing umaga {i}." for i in range(40)
    )
    assert not near.is_near_duplicate(base)
    assert near.is_near_duplicate(edit)      # edited copy of same doc
    assert not near.is_near_duplicate(other)  # unrelated document


def test_gutenberg_extract_and_decode():
    body = extract_gutenberg_body(
        "TITLE\r\n\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "Hello world.\r\n\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "license text"
    )
    assert "Hello world." in body
    assert "PROJECT GUTENBERG" not in body
    # cp1252 encoded book must decode (e.g. smart quotes, em dashes)
    cp = "caf\xe9 \u2013 here".encode("cp1252")
    assert "\u2013" in decode_text(cp)
    # utf-8 first
    assert decode_text(b"caf\xc3\xa9") == "caf\xe9"


def test_write_read_docs_roundtrip(tmp_path):
    path = str(tmp_path / "docs.jsonl")
    docs = [
        {"source": "test", "doc_id": "a", "lang": "en", "text": "one"},
        {"source": "test", "doc_id": "b", "lang": "tl", "text": "dalawa"},
    ]
    assert write_docs(path, iter(docs)) == 2
    got = list(read_docs(path))
    assert [d["doc_id"] for d in got] == ["a", "b"]
    assert got[0]["text"] == "one"
