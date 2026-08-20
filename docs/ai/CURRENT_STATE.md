# CURRENT STATE — ~30 second summary

**PROJECT**: ChatGPT-like language model built strictly FROM SCRATCH.

**CURRENT STATUS**: TASK 005.1 IN PROGRESS — corpus pipeline DONE + gate PASSED
(UNIQUE 1,863,853 tokens), but the **SFT PILOT HARD-STOPPED at step 200**
(retention guard fired: base-val 15.8796 > 3.7124 hard-stop threshold).
Full training run 2 NOT started. Awaiting lead architect decision on
mitigation (LR/regularization/data-mix) before any retraining.

**LAST COMPLETED TASK**: TASK 005 (pipeline; training STOPPED at Part E gate
624,057 < 1M). TASK 005.1 pipeline phase: Dolly 15K + Taskmaster-1 added,
corpus 50,776 examples / UNIQUE 1,863,853 supervised tokens (gate PASSED),
127/127 tests, quality audit passed — all committed+pushed (5a51aa6).

**MODEL**: GPT-style decoder-only Transformer, 29,270,528 params, vocab 8000,
d_model 512, 8 layers, 8 heads, FFN 2048, ctx 256, RMSNorm, RoPE, tied
embeddings, dropout 0. Architecture in src/model.py, src/attention.py.

**BEST CHECKPOINT**: checkpoints/pretrain_v1/best.pt (step 36,500, val
3.073326; SHA-256 ba40ad8c… verified unchanged) — UNCHANGED. chat_v1 has NO
best.pt; chat_v1/latest.pt is the FAILED pilot step-200 checkpoint (kept for
diagnosis only).

**KEY RESULT (PRETRAINING)**: 38,379 steps, 314,400,768 tokens, 8.000 passes;
best val loss 3.073326 (~21.61 ppl). Produces grammatical English + fluent
Filipino continuations.

**SFT PILOT RESULT (FAILED — retention HARD STOP)**: run 1
(`--init-from base --max-epochs 1 --out-dir checkpoints/chat_v1
--retention-baseline 3.093633`) stopped at step 200/1429: sft_val 0.0021
(memorization of held-out chat targets), base_val 15.8796 (baseline 3.093633;
eligible ≤ 3.557678; hard stop > 3.712360). Train loss 0.0020. 2 AMP
non-finite grad events self-healed (steps 5, 32). Peak VRAM 2.20 GB.
Generation probe: base "The quick brown fox had been seen by the ladder…" vs
step-200 "foxxxxxx…" — genuine catastrophic forgetting (eval path validated:
base best.pt re-eval = 3.0733).

**SFT CORPUS (FINAL, gate PASSED)**: 4 human-only sources — Aya 2,568 ex
(180,896 tok), OASST1 4,266 (439,354), Dolly 10,926 (715,354), Taskmaster-1
33,016 (528,249). UNIQUE 1,863,853 supervised target tokens (train
1,697,262 / val 166,591); fil 51,126 unique (effective 10.5% at 4x cap);
English weights 1.0; target unk 0.0; leakage all 0.

**BASELINES (unchanged)**: BASELINE_PRETRAIN_VAL_LOSS = 3.093633 (retention:
eligible ≤ 3.557678; hard stop > 3.712360). Part K chat baseline: greedy EOS
0.0, repetition 0.68, fil-in-fil 0.4 / eng-in-eng 0.5.

**NEXT ACTION**: Architect decision REQUIRED: retrain mitigation — options:
(a) lower peak LR (e.g. 1e-5), (b) stronger regularization (wd up / dropout),
(c) fewer/filtered epochs, (d) mix base-language pretraining data into SFT
batches as an anchor, (e) layer-wise freeze/lower LR. Any mitigation is
re-validated with the same pilot gate before full training. Do NOT start
TASK 006.