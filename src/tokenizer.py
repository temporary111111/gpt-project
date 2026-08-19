"""Tokenizer infrastructure trained from scratch.

Wraps a locally trained SentencePiece model. A char-level fallback tokenizer
is provided for tests and for use before a SentencePiece model exists.
No pretrained tokenizers are loaded.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<system>", "<user>", "<assistant>"]
# <unk> is SentencePiece's built-in unknown piece; the rest are user-defined
USER_DEFINED_SYMBOLS = ["<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>"]

_PAD, _BOS, _EOS, _UNK, _SYSTEM, _USER, _ASSISTANT = SPECIAL_TOKENS


class SentencePieceTokenizer:
    """Wrapper around a locally trained SentencePiece model."""

    def __init__(self, model_path: str, meta_path: Optional[str] = None):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.meta: Dict = {}
        if meta_path:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self.pad_id = self.sp.piece_to_id(_PAD)
        self.bos_id = self.sp.piece_to_id(_BOS)
        self.eos_id = self.sp.piece_to_id(_EOS)
        self.unk_id = self.sp.unk_id()
        self.system_id = self.sp.piece_to_id(_SYSTEM)
        self.user_id = self.sp.piece_to_id(_USER)
        self.assistant_id = self.sp.piece_to_id(_ASSISTANT)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def special_token_ids(self) -> Dict[str, int]:
        return {
            "pad": self.pad_id, "bos": self.bos_id, "eos": self.eos_id,
            "unk": self.unk_id, "system": self.system_id,
            "user": self.user_id, "assistant": self.assistant_id,
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in self.special_token_ids.values()]
        return self.sp.decode(ids)


class CharTokenizer:
    """Char-level fallback tokenizer with the same interface (for tests/dev)."""

    def __init__(self, chars: str):
        uniq = list(dict.fromkeys(chars))
        self._chars = SPECIAL_TOKENS + uniq
        self._stoi = {c: i for i, c in enumerate(self._chars)}
        self._itos = self._chars
        self.pad_id = self._stoi[_PAD]
        self.bos_id = self._stoi[_BOS]
        self.eos_id = self._stoi[_EOS]
        self.unk_id = self._stoi[_UNK]
        self.system_id = self._stoi[_SYSTEM]
        self.user_id = self._stoi[_USER]
        self.assistant_id = self._stoi[_ASSISTANT]

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def special_token_ids(self) -> Dict[str, int]:
        return {
            "pad": self.pad_id, "bos": self.bos_id, "eos": self.eos_id,
            "unk": self.unk_id, "system": self.system_id,
            "user": self.user_id, "assistant": self.assistant_id,
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self._stoi.get(c, self.unk_id) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in self.special_token_ids.values()]
        return "".join(self._itos[i] for i in ids)


def tokenizer_special_ids_from_meta(meta: Dict) -> Dict[str, int]:
    return {
        "pad": meta.get("pad_id", 0),
        "bos": meta.get("bos_id", 1),
        "eos": meta.get("eos_id", 2),
        "unk": meta.get("unk_id", 3),
        "system": meta.get("system_id", 4),
        "user": meta.get("user_id", 5),
        "assistant": meta.get("assistant_id", 6),
    }