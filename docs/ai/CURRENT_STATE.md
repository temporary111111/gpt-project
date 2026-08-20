# CURRENT STATE — ~30 second summary

**PROJECT**: ChatGPT-like language model built strictly FROM SCRATCH.

**CURRENT STATUS**: TASK 005.1 (Expand Human SFT Corpus + Complete Chat V1
Training) IN PROGRESS — pipeline phase DONE. Expanded corpus built from 4
human-only sources: **UNIQUE_SUPERVISED_TARGET_TOKENS = 1,863,853 ≥ 1,000,000
floor → TOKEN GATE PASSED** (Part M: continue automatically). Code + corpus +
docs committed; SFT pilot + full training NOT yet run (training starts next).

**LAST COMPLETED TASK**: TASK 005 (pipeline; training STOPPED at Part E gate
624,057 < 1M). TASK 005.1 continues it: acquired Databricks Dolly 15K (rev
bdd27f4d…, cc-by-sa-3.0, 15,010 accepted) + Taskmaster-1 (official repo rev
d92cb6af…, CC BY 4.0, 13,170 conversations, conv-level splits); rebuilt corpus
50,776 examples / 1,863,853 unique supervised tokens (train 43,660 / val
7,116); fixed Aya eng mislabeling (104 non-English rows rejected deterministically);
cross-source dedup 1,160 exact pairs; fil 4x sampling cap (effective fil share
10.5%); English source balance auto-checked (no source >50% → weights 1.0);
127/127 tests; 150-sample quality audit passed; handoff allowlist fixed (Parts
J/K); new held-out multi-turn probes added (Part U).

**MODEL**: GPT-style decoder-only Transformer, 29,270,528 params, vocab 8000,
d_model 512, 8 layers, 8 heads, FFN 2048, ctx 256, RMSNorm, RoPE, tied
embeddings, dropout 0. Architecture in src/model.py, src/attention.py.

**BEST CHECKPOINT**: checkpoints/pretrain_v1/best.pt (step 36,500, val
3.073326; SHA-256 ba40ad8c… re-verified 2026-08-20) — UNCHANGED. No SFT
weights yet (chat_v1 has only baseline eval outputs).

**KEY RESULT (PRETRAINING)**: 38,379 steps, 314,400,768 tokens, 8.000 passes;
best val loss 3.073326 (~21.61 ppl); final train loss 2.480394. Peak VRAM
1.38 GB. Produces grammatical English + fluent Filipino continuations.

**SFT CORPUS (TASK 005.1)**: 4 human-only sources — Aya (2,568 examples,
180,896 tokens), OASST1 (4,266, 439,354), Dolly (10,926, 715,354),
Taskmaster-1 (33,016, 528,249). UNIQUE 1,863,853 supervised target tokens;
fil 51,126 (2.7% unique; effective 10.5% at 4x cap); target unk rate 0.0;
leakage checks all 0 (incl. Taskmaster conversation-level isolation).

**BASELINES (unchanged)**: BASELINE_PRETRAIN_VAL_LOSS = 3.093633 (retention:
eligible ≤ 3.557678; hard stop > 3.712360). Part K chat baseline: greedy EOS
0.0, repetition 0.68, fil-in-fil 0.4 / eng-in-eng 0.5.

**NEXT ACTION**: Run SFT pilot (~200 optimizer steps, `src/sft_train.py` run 1
`--max-epochs 1`, resume-proof) → retention eligibility check → full training
(run 2 resume `--max-epochs 3`, early stop 4 evals) → post-SFT eval
(`evaluate_chat.py --mode compare`, multi-turn incl. 6 new held-out probes) →
interactive chat verification → TASK_005_1_REPORT.txt + finalize + milestone
tag `task-005-chat-v1-complete` on success. Do NOT start TASK 006.