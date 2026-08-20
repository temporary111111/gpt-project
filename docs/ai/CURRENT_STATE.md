# CURRENT STATE — ~30 second summary

**PROJECT**: ChatGPT-like language model built strictly FROM SCRATCH.

**CURRENT STATUS**: TASK 005 (Chat/Instruction Tuning V1) STOPPED BEFORE FULL
TRAINING per Part E gate: usable supervised target tokens (624,057) < 1,000,000
floor. The complete SFT pipeline, provenance, baseline evals, chat.py upgrade,
and 108/108 tests were delivered; full training NOT started. Awaiting lead
architect decision on the corpus path (additional human-only data vs.
authorization to train on the reduced corpus).

**LAST COMPLETED TASK**: TASK 005 (pipeline phase) — delivered: Aya + OASST1
human-only SFT corpus (6,911 examples, 624,057 assistant-supervised tokens,
Filipino 8.2%), fixed 2 real bugs found by the mandated tests (assistant-label
shift onto user tokens; OASST val→train split leak), 25 new tests (22 mandated
+ 3), baseline chat eval (Part K) + baseline base-LM validation loss 3.093633
(Part L), upgraded src/chat.py (real multi-turn terminal chat), SOURCES.md +
SFT_DATA_REPORT.md + TASK_005_REPORT.txt. Training, pilot, milestone tag: NOT
done (STOP gate).

**MODEL**: GPT-style decoder-only Transformer, 29,270,528 params, vocab 8000,
d_model 512, 8 layers, 8 heads, FFN 2048, ctx 256, RMSNorm, RoPE, tied
embeddings, dropout 0. Architecture in src/model.py, src/attention.py.

**BEST CHECKPOINT**: checkpoints/pretrain_v1/best.pt (step 36,500, val 3.073326)
— UNCHANGED (no SFT weights exist; chat_v1 has no .pt files).

**KEY RESULT (PRETRAINING)**: 38,379 steps, 314,400,768 tokens, 8.000 passes;
best val loss 3.073326 (~21.61 ppl); final train loss 2.480394. Peak VRAM 1.38 GB.
Produces grammatical English + fluent Filipino continuations.

**SFT DATA**: Aya (CohereLabs/aya_dataset rev f9ea0458…, eng+fil
original-annotations only, 2,934 accepted) + OASST1 (OpenAssistant/oasst1 rev
fdf72ae0…, en human-only, 39,751 accepted) → 6,911 final examples; supervised
target tokens 624,057 (train 594,179 / val 29,878); fil 51,126 (8.2%),
Aya-eng 133,318 (21.4%), OASST-en 439,613 (70.4%); target unk rate 0.0.

**BASELINE EVALS (pretrain_v1/best.pt)**: base-LM validation.bin loss
3.093633 (Part L baseline); chat-format baseline (Part K, checkpoints/chat_v1/
baseline_chat_samples.txt + eval_metrics.json): greedy EOS 0.0%, repetition
0.68, fil-in-fil 40% / eng-in-eng 50%.

**CURRENT LIMITATION**: base model only (not chat-tuned); SFT corpus below the
1M supervised-token floor → training blocked until the lead architect decides.

**NEXT ACTION**: Lead architect reviews TASK_005_REPORT.txt + AI_LEAD_HANDOFF.zip
(STOP-gate report) and decides: (a) expand human-only SFT corpus and retarget,
(b) authorize reduced-corpus SFT, or (c) other direction. Do NOT start training
or TASK 006 until ordered.
