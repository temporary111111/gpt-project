# Local ChatGPT-Like Assistant (from scratch)

A GPT-style decoder-only transformer built **strictly from scratch** with PyTorch
primitives. No pretrained weights, no pretrained tokenizers, no external LLM APIs.

## Constraints

- Model architecture: our own `torch.nn` implementation (no Hugging Face models).
- Tokenizer: SentencePiece trained locally on our own corpus.
- Weights: random initialization, trained locally.
- Target hardware: RTX 3050 Laptop GPU (4 GB VRAM), 8 GB system RAM.

## Setup

```powershell
# Python 3.11 required (NOT 3.14)
& 'C:\Users\dev\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Project structure

```
configs/model_small.json   # main model config (~30M params)
data/sources/              # downloaded dumps (wikipedia xml.bz2, gutenberg .txt)
data/manifests/            # source provenance records + cleaning stats
data/raw/                  # extracted raw documents (jsonl, one doc per line)
data/cleaned/              # cleaned documents (jsonl per language)
data/processed/            # corpus_text.txt, tokenized .bin files (uint16 memmap)
data/tokenizer/            # trained SentencePiece model + meta
src/model.py               # GPTModel, ModelConfig, RMSNorm
src/attention.py           # multi-head causal self-attention + RoPE
src/tokenizer.py           # SentencePiece wrapper + char fallback
src/dataset.py             # streaming memmap dataset
src/train.py               # training loop (AMP, grad accumulation, checkpoints)
src/generate.py            # greedy / temperature / top-k / top-p generation
src/chat.py                # terminal chat interface
src/data/                  # data pipeline: normalize, filters, dedup, corpus_builder
scripts/acquire_corpus.py  # download wiki dumps + Gutenberg books, record provenance
scripts/clean_corpus.py    # normalize, filter, language-verify, dedupe, split
scripts/build_corpus.py    # corpus text + uint16 train/val/test .bin files
scripts/train_tokenizer.py # train a SentencePiece tokenizer on the cleaned corpus
scripts/corpus_stats.py    # raw/cleaned/tokenized/dataset statistics
scripts/sample_corpus.py   # random quality-sample review (20 EN + 20 TL)
scripts/tokenizer_quality_test.py  # mandated EN/TL/mixed tokenizer test sentences
scripts/smoke_test.py      # end-to-end pipeline smoke test
tests/                     # pytest unit tests
checkpoints/               # training checkpoints
```

## Usage

```powershell
# 1. Acquire sources (wiki dumps + Gutenberg books; writes data/manifests/sources.jsonl)
.\.venv\Scripts\python.exe scripts\acquire_corpus.py

# 2. Clean (normalize, filters, language check, dedup, deterministic splits)
.\.venv\Scripts\python.exe scripts\clean_corpus.py

# 3. Build corpus text + train tokenizer
.\.venv\Scripts\python.exe scripts\build_corpus.py --corpus-text
.\.venv\Scripts\python.exe scripts\train_tokenizer.py `
    --input-files data\processed\corpus_text.txt `
    --output-prefix data\tokenizer\tokenizer_v1 --vocab-size 8000

# 4. Tokenize to binary (en 24M / tl 16M token budgets -> ~40M tokens total)
.\.venv\Scripts\python.exe scripts\build_corpus.py --build-bins

# 5. Train
.\.venv\Scripts\python.exe -m src.train --data data\processed\train.bin `
    --config configs\model_small.json --batch-size 8 --grad-accum 4 --max-iters 5000

# 6. Generate / chat
.\.venv\Scripts\python.exe -m src.generate --checkpoint checkpoints\checkpoint.pt `
    --tokenizer-model data\tokenizer\tokenizer_v1.model --prompt "Hello"
.\.venv\Scripts\python.exe -m src.chat --checkpoint checkpoints\checkpoint.pt `
    --tokenizer-model data\tokenizer\tokenizer_v1.model

# Statistics + quality review
.\.venv\Scripts\python.exe scripts\corpus_stats.py
.\.venv\Scripts\python.exe scripts\sample_corpus.py
.\.venv\Scripts\python.exe scripts\tokenizer_quality_test.py

# Smoke test + unit tests
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Design notes

- Pre-norm transformer blocks, RMSNorm, RoPE (or learned positional embeddings).
- Tied input/output embeddings, GPT-2 style residual scaling init.
- fp16 autocast + GradScaler on CUDA; fp32 CPU fallback; gradient accumulation;
  AdamW with weight decay on matrices only; linear warmup + cosine LR decay.
- Training data is streamed from a uint16 memmap — never loaded fully into RAM.