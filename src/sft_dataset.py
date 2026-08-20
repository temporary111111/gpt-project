"""SFT dataset: chat-formatted examples with assistant-only loss labels.

Loads the JSONL produced by scripts/build_sft_dataset.py. Each example has
`ids` (full token sequence, <= context length), `labels` (-100 for BOS/user/
context/role-marker/padding, token id for the final assistant target + EOS),
plus metadata (source, lang, id, n_supervised, weight, copies).

Sampling (TASK 005.1 Part G/H): each train example carries an integer
`copies` multiplier (Filipino up to 4x; English sources above 50% of effective
English tokens deterministically down-weighted). The training epoch order is
built by repeating each example `copies` times, then shuffling deterministically.
This is SAMPLING ONLY: it never counts toward the unique-token gate, and
validation is never duplicated (val uses plain order, copies ignored).

Batches are padded to the longest sequence in the batch (pad token id 1,
labels -100) so every batch can have a different length (<= context limit).

Supports deterministic shuffling, seed control, and state_dict/load_state_dict
so training can resume deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch


class SFTDataset:
    """In-memory chat-formatted SFT dataset with deterministic batching."""

    def __init__(self, path: str, batch_size: int, block_size: int,
                 pad_id: int, seed: int = 0, shuffle: bool = True):
        self.path = str(path)
        self.batch_size = batch_size
        self.block_size = block_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self._load(path)
        self.rng = np.random.default_rng(seed)
        self._order = np.arange(self._effective_len)
        self._pos = 0

    def _load(self, path: str) -> None:
        self.ids: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.meta: List[Dict] = []
        self.copies: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                ids = ex["ids"]
                if len(ids) > self.block_size:
                    raise ValueError(
                        f"example {ex.get('id')} length {len(ids)} > block_size {self.block_size}")
                self.ids.append(ids)
                self.labels.append(ex["labels"])
                self.meta.append({
                    "id": ex.get("id", ""),
                    "source": ex.get("source", ""),
                    "lang": ex.get("lang", ""),
                    "n_supervised": ex.get("n_supervised", 0),
                })
                self.copies.append(max(1, int(ex.get("copies", 1))))
        self._effective_len = sum(self.copies) if self.shuffle else len(self.ids)
        if self.shuffle:
            self._expanded: List[int] = []
            for i, c in enumerate(self.copies):
                self._expanded.extend([i] * c)
        else:
            self._expanded = list(range(len(self.ids)))

    def __len__(self) -> int:
        return self._effective_len

    @property
    def n_examples(self) -> int:
        return len(self.ids)

    def unique_supervised_tokens(self) -> int:
        return sum(m["n_supervised"] for m in self.meta)

    def effective_supervised_tokens(self) -> int:
        return sum(m["n_supervised"] * c for m, c in zip(self.meta, self.copies))

    def n_supervised_tokens(self) -> int:
        return self.effective_supervised_tokens() if self.shuffle \
            else self.unique_supervised_tokens()

    def reset_epoch(self) -> None:
        self._pos = 0
        if self.shuffle:
            self.rng.shuffle(self._order)

    def state_dict(self) -> dict:
        return {
            "path": self.path,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "pad_id": self.pad_id,
            "rng_state": self.rng.bit_generator.state,
            "order": self._order.tolist(),
            "pos": self._pos,
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        self._order = np.asarray(state["order"], dtype=np.int64)
        self._pos = state["pos"]

    def _next_batch(self) -> Tuple[List[List[int]], List[List[int]], List[Dict]]:
        """Returns one batch of raw (ids, labels, meta) in deterministic order."""
        batch_pos = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        batch_idx = [self._expanded[i] for i in batch_pos]
        return ([self.ids[i] for i in batch_idx],
                [self.labels[i] for i in batch_idx],
                [self.meta[i] for i in batch_idx])

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) padded to the batch max length; y is -100 masked."""
        raw_ids, raw_labels, _ = self._next_batch()
        T = max(len(s) for s in raw_ids)
        B = len(raw_ids)
        x = torch.full((B, T), self.pad_id, dtype=torch.long)
        y = torch.full((B, T), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(zip(raw_ids, raw_labels)):
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
        return x.to(device), y.to(device)

    def epoch_batches(self, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yields all batches of the current epoch (shuffled deterministically)."""
        self.reset_epoch()
        while self._pos < len(self):
            yield self.get_batch(device)