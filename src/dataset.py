"""Streaming dataset utilities.

Training data is pre-tokenized into a binary uint16 memmap file so the whole
corpus is never loaded into RAM at once.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch


def encode_text_to_bin(text_path: str, bin_path: str, tokenizer,
                       add_bos: bool = False, add_eos: bool = True) -> int:
    """Tokenizes a text file and writes ids to a uint16 binary file.

    Returns the total number of tokens written.
    """
    ids: List[int] = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.extend(tokenizer.encode(line, add_bos=add_bos, add_eos=add_eos))
    arr = np.asarray(ids, dtype=np.uint16)
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())
    return len(ids)


class BinaryDataset:
    """Random-window access over a uint16 memmap of token ids."""

    def __init__(self, bin_path: str, block_size: int, batch_size: int,
                 seed: int = 0, start_frac: float = 0.0, end_frac: float = 1.0):
        self.bin_path = bin_path
        self.block_size = block_size
        self.batch_size = batch_size
        self._mmap = np.memmap(bin_path, dtype=np.uint16, mode="r")
        total = len(self._mmap)
        lo = int(total * start_frac)
        hi = int(total * end_frac)
        self.offset = lo
        self.length = max(0, hi - lo)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.length < self.block_size + 1:
            raise ValueError("dataset too small for block_size + 1 tokens")
        ix = self.rng.integers(0, self.length - self.block_size, size=self.batch_size)
        x = np.empty((self.batch_size, self.block_size), dtype=np.uint16)
        y = np.empty((self.batch_size, self.block_size), dtype=np.uint16)
        for i, start in enumerate(ix):
            seg = self._mmap[self.offset + start: self.offset + start + self.block_size + 1]
            x[i] = seg[:-1]
            y[i] = seg[1:]
        return (torch.from_numpy(x).long().to(device),
                torch.from_numpy(y).long().to(device))

    def stream_batches(self, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.get_batch(device)

    def close(self):
        self._mmap._mmap.close()
        del self._mmap