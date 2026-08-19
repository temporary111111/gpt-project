# CONTINUITY — Living Project Document

This is the permanent living memory of the project. Keep it structured and
up-to-date. It is NOT a chat log: store decisions, verified results,
constraints, important failures, fixes, current state, and the next action.

## Project goal

Build a ChatGPT-LIKE language model from scratch — a local chat/assistant
experience powered by our own model, tokenizer, and corpus. No pretrained
weights, no pretrained tokenizer, no distillation, no external LLM API as the
model brain.

IMPORTANT distinction:
- ChatGPT-LIKE PRODUCT EXPERIENCE (chat UI, roles, streaming UX) is different
  from ChatGPT-LEVEL FOUNDATION MODEL INTELLIGENCE (frontier-scale pretraining).
- The current 29.27M-parameter model is a PROOF-OF-CONCEPT foundation model.
  It demonstrates the full from-scratch stack; it is not frontier intelligence.

## Strict from-scratch philosophy

- no pretrained LLM weights (no Ollama/Qwen/Llama/DeepSeek/GPT weights as model intelligence)
- no pretrained tokenizer
- no pretrained embeddings
- no LLM distillation
- no external LLM API as final model brain
- DeepSeek/OpenCode is a coding engineer only, never the model brain

## Role separation

| Role | Who | Responsibilities |
|---|---|---|
| USER | human operator | relay/operator only; does not maintain files, run git manually, or remember recovery prompts |
| BROWSER AI | ChatGPT etc. | lead architect / reviewer; no shell access; sends copy-paste-ready tasks |
| DEEPSEEK / OPENCODE | implementation engineer | editing, testing, training, debugging, continuity updates, git, handoffs |

Git repository + docs/ai = AUTHORITATIVE project memory. Chat history is not.

## Hardware & environment

- Windows 11, AMD Ryzen 5 5600H, 8 GB RAM
- RTX 3050 Laptop GPU, 4 GB VRAM (CUDA available)
- Python 3.11, venv: `.\.venv\Scripts\python.exe` (NEVER global Python)
- Project root: C:\Users\dev\Desktop\chatgpt-like
- torch 2.13.0+cu126, numpy 2.4.6, sentencepiece 0.2.2, pytest 9.1.1

## Model architecture

GPT-style decoder-only Transformer:
- 29,270,528 params (all trainable), vocab 8000, d_model 512
- 8 layers, 8 heads, FFN 2048, context 256
- RMSNorm, RoPE (cat-based layout), tied input/output embeddings, dropout 0
- Pre-norm blocks, GPT-2 style residual scaling init
- Files: src/model.py (GPTModel, ModelConfig), src/attention.py (causal MHA + RoPE)

## Tokenizer

- SentencePiece BPE, vocab 8000, trained from scratch on our own corpus
- data/tokenizer/tokenizer_v1.model (+ .vocab + _meta.json)
- Special tokens: <unk>=0 <pad>=1 <bos>=2 <eos>=3 <system>=4 <user>=5 <assistant>=6

## Corpus

- Human-written English (~60%) + Filipino (~40%), legally sourced:
  Simple English Wikipedia, Tagalog Wikipedia, Tagalog Wikisource, Project Gutenberg (public domain)
- train = 39,299,287 tokens; validation = 415,832; test = 284,270 (SEALED)
- Pipeline: scripts/acquire_corpus.py → clean_corpus.py → build_corpus.py → train_tokenizer.py
- data/processed/*.bin are uint16 memmaps (never loaded fully into RAM)

## Completed tasks

- TASK 001: machine/environment audit
- TASK 002: GPT architecture, tokenizer, training pipeline, generation, checkpointing, CUDA, tests
- TASK 003: legal/provenance-aware corpus pipeline + tokenizer_v1 + ~40M-token corpus
- TASK 003.5: correctness hardening (special IDs, val split, AMP clip, RoPE, RNG resume, checkpoint semantics, param groups, deterministic val)
- TASK 004: FIRST real from-scratch pretraining (ACCEPTED) — see below
- TASK 004.5: self-maintaining memory, git, handoff infrastructure (this)

## Important bugs/fixes (verified)

- Special-token IDs were wrong → fixed (pad=1 bos=2 eos=3 unk=0)
- Validation was sourced from train.bin → fixed (separate validation.bin)
- AMP gradient clipping scaled the gradients → fixed (clip unscaled)
- RoPE layout mathematically inconsistent → fixed (cat convention, verified vs old)
- CUDA RNG resume: torch.load(map_location=cuda) moves ByteTensors; fixed by pinning CPU RNG and passing CPU tensors to torch.cuda.set_rng_state_all
- Checkpoint best/latest save order: latest.pt was saved with the OLD best_val_loss → fixed (update best → save best → save latest, all with updated value)
- Optimizer param grouping/dedup (weight decay on matrices only)
- Deterministic validation
- AMP non-finite GRADIENT norm (12 events in TASK 004): standard self-heal — skip optimizer step, halve GradScaler scale, drop grads, log warning + metrics, continue. Root cause: scale doubles every 2000 clean steps → unbounded growth → rare fp16 overflow. Non-finite LOSS remains a hard stop. This supersedes the TASK 003.5 hard-stop-on-inf-grad behavior (architect-approved).

## Production pretraining result (TASK 004)

- 38,379 optimizer steps; 314,400,768 tokens; 8.000 corpus passes
- best val loss 3.073326 at step 36,500 (perplexity ~21.61)
- final val loss 3.076171; final train loss 2.480394
- peak VRAM 1.38 GB; median ~17,139 tok/s
- base model produces grammatical English + fluent Filipino continuations (generation_samples.txt)

## Current checkpoint locations

- BEST: checkpoints/pretrain_v1/best.pt (step 36,500, val 3.073326)
- LATEST: checkpoints/pretrain_v1/latest.pt (step 38,379, tokens 314,400,768)
- Do NOT modify/delete/move them. Never commit *.pt.

## Current limitations

- base model NOT chat/instruction tuned (no <system>/<user>/<assistant> formatting training yet)
- repetition, especially greedy decoding
- fluent factual hallucination
- small-model reasoning limitations
- rare replacement-character output for OOV symbols (tokenizer coverage)

## Next planned stage

TASK 005 — chat/instruction tuning (SFT with chat roles on top of best.pt).
Await the lead architect's review of AI_LEAD_HANDOFF.zip and explicit task text.

## Long-term roadmap

base pretraining (DONE) → chat/instruction tuning (TASK 005) → better
evaluation → reasoning curriculum → tool-use training → memory → agent/tool
integration → files/shell/calculator/search → possible larger model scaling

## Git state

- Repository: yes (origin https://github.com/temporary111111/gpt-project.git, branch main)
- Milestone tags: task-004-pretraining-complete
- Checkpoints/corpus binaries are NOT in git (see RECOVERY_PROTOCOL MODE 3 limitation)