"""Unit tests for the tokenizer wrapper."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import (  # noqa: E402
    SPECIAL_TOKENS,
    CharTokenizer,
    SentencePieceTokenizer,
)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    assert tok.decode([]) == ""


def test_char_tokenizer_special_ids():
    tok = CharTokenizer("abc")
    assert tok.bos_id != tok.eos_id
    assert tok.pad_id != tok.eos_id
    assert set(tok.special_token_ids.keys()) == {
        "pad", "bos", "eos", "unk", "system", "user", "assistant"
    }


def test_char_tokenizer_add_bos_eos():
    tok = CharTokenizer("abc")
    ids = tok.encode("abc", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_char_tokenizer_unknown():
    tok = CharTokenizer("abc")
    ids = tok.encode("xyz")
    assert all(i == tok.unk_id for i in ids)


def test_sentencepiece_wrapper(tmp_path):
    import sentencepiece as spm
    from src.tokenizer import USER_DEFINED_SYMBOLS

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(
        ["the quick brown fox", "hello world", "tiny corpus test",
         "attention is all you need"] * 50
    ), encoding="utf-8")
    prefix = str(tmp_path / "sp")
    spm.SentencePieceTrainer.train(
        input=[str(corpus)],
        model_prefix=prefix,
        vocab_size=128,
        model_type="bpe",
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        pad_id=-1, bos_id=-1, eos_id=-1,
    )
    tok = SentencePieceTokenizer(prefix + ".model")
    assert tok.vocab_size > 0
    ids = tok.encode("the quick brown fox")
    assert tok.decode(ids) == "the quick brown fox"
    assert tok.system_id != tok.user_id != tok.assistant_id
    assert tok.unk_id == tok.sp.unk_id()
    # special tokens must be preserved as single pieces
    for sym in USER_DEFINED_SYMBOLS:
        assert tok.sp.piece_to_id(sym) < tok.vocab_size