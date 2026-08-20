# CURRENT TASK

**TASK 005.1 — EXPAND HUMAN SFT CORPUS AND COMPLETE CHAT V1 TRAINING**
**(status: PILOT HARD STOPPED — awaiting architect mitigation decision)**

## Objective
Expand the TASK 005 SFT corpus with Databricks Dolly 15K + Taskmaster-1
(human-only), re-audit everything, and IF unique supervised target tokens
>= 1,000,000, automatically run pilot → full SFT (up to 3 epochs, retention
guards) → post-SFT evaluation → milestone tag `task-005-chat-v1-complete`.
Do NOT start TASK 006.

## Status
- DONE (pipeline phase): corpus 50,776 examples / UNIQUE 1,863,853 tokens →
  **GATE PASSED**; 127/127 tests; quality audit; base SHA re-verified;
  committed+pushed (5a51aa6).
- **FAILED (Part Q pilot, run 1)**: hard stop at step 200/1429 — base-val
  15.8796 > 3.7124 (baseline 3.093633 × 1.20). Model memorized SFT targets
  (sft_val 0.0021, train 0.0020) and lost base language ability (probe:
  "foxxxxxx…"). Eval path validated (base best.pt re-eval 3.0733) — genuine
  forgetting, not a measurement artifact. chat_v1/latest.pt (step 200) kept
  for diagnosis; chat_v1/best.pt does NOT exist.
- BLOCKED: run 2 (full SFT) must NOT start until the lead architect approves
  a mitigation (LR reduction, regularization, epoch filtering, base-data
  mixing, layer freezing, or another approach). Same pilot gate applies to
  any retry: eligible base-val ≤ 3.557678; hard stop > 3.712360.

## Config (Part P, unchanged)
batch 8, grad accum 4, ctx 256, peak LR 5e-5 → min 5e-6, wd 0.01, clip 1.0,
AMP fp16, seed 1337, max 3 epochs, warmup max(100, 3%), eval ~200 steps with
several evals per epoch (auto-tightened), early stop 4 evals. Full fine-tune
(no LoRA). NOTE: the pilot failure suggests 5e-5 is too aggressive for this
29.27M model on this corpus; mitigation pending.

## Outputs so far
- data/sft/manifests + stats (tracked); raw/processed (git-ignored).
- checkpoints/chat_v1/: baseline eval outputs + failed-pilot latest.pt
  (diagnosis only) + metrics.jsonl + run_config.json.
- docs/ai/reports/TASK_005_1_REPORT.txt (full failure report).

## Rules while doing this task
- Preserve the TASK 005 fixes (label alignment; no OASST leak) + TASK 005.1
  checks (TM conv isolation, mislabel filter, dedup).
- test.bin stays sealed; never modify best.pt / latest.pt / *.bin.
- Only .\.venv\Scripts\python.exe; strict from-scratch still applies.
- Do NOT start TASK 006. Do NOT retrain without architect approval.