# MODEL STATUS — checkpoints/pretrain_v1

## Base model (TASK 004, accepted)

| Item | Value |
|---|---|
| Architecture | GPT-style decoder-only Transformer (src/model.py) |
| Parameters | 29,270,528 |
| Vocab | 8000 (SentencePiece BPE, tokenizer_v1) |
| d_model / layers / heads / FFN | 512 / 8 / 8 / 2048 |
| Context | 256 |
| Norm / position | RMSNorm / RoPE (cat convention) |
| Tied embeddings / dropout | yes / 0 |
| Training | 38,379 steps, 314,400,768 tokens, 8.000 passes, AMP fp16, seed 1337 |

## Checkpoints

| File | Step | Tokens | Val loss | Notes |
|---|---|---|---|---|
| checkpoints/pretrain_v1/best.pt | 36,500 | 299,008,000 | 3.073326 | lowest val; ppl ~21.61 |
| checkpoints/pretrain_v1/latest.pt | 38,379 | 314,400,768 | 3.076171 | final; ppl ~21.68 |

Both are fully resumable (model + optimizer + scaler + RNG + dataset positions).
Never modify, delete, move, or commit them.

## Behavior (verified via generation_samples.txt)

- English continuations: grammatical, encyclopedic tone; repetitive in greedy mode.
- Filipino continuations: fluent Tagalog with correct particles; strong code-switching.
- Weaknesses: repetition (greedy), fluent hallucination, small-model reasoning
  limits, rare "�" for OOV symbols.

## Status

- Chat/instruction tuning: NOT YET (TASK 005 pipeline delivered; training
  STOPPED at Part E gate — 624,057 supervised tokens < 1M floor).
- Fine-tuning on best.pt is the expected TASK 005 base.

## TASK 005 baselines (checkpoints/chat_v1, eval-only — no weight modification)

| Item | Value |
|---|---|
| BASELINE_PRETRAIN_VAL_LOSS | 3.093633 (validation.bin, 50 iters, fp16, 2026-08-20) |
| Baseline chat greedy | EOS term 0.0, role leak 0.0, unk 0.0, rep 0.6805, fil-in-fil 0.4, eng-in-eng 0.5 |
| Baseline chat sampled (t0.8/k40) | EOS term 0.1, role leak 0.0, unk 0.0, rep 0.1321, fil-in-fil 0.4, eng-in-eng 0.5 |
| Files | baseline_chat_samples.txt, eval_metrics.json (NO .pt files in chat_v1) |

The retention guard in src/sft_train.py uses BASELINE_PRETRAIN_VAL_LOSS as
its baseline; best.pt eligible only while base loss <= 3.093633 * 1.15.