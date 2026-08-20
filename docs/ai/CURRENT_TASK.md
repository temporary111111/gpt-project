# CURRENT TASK

**TASK 005 — CHAT / INSTRUCTION TUNING V1** (STOPPED BEFORE FULL TRAINING — Part E gate)

## Objective
SFT the accepted base model (checkpoints/pretrain_v1/best.pt) into a basic
English + Filipino chat assistant using ONLY human-written data (Aya eng/fil
original-annotations + English OASST1 human-only), with full provenance,
deterministic splits, assistant-only loss masking, retention guards, pilot
gate, 22 mandated tests, chat.py upgrade, report, and finalization.

## Part E STOP gate (HIT)
**624,057 usable supervised target tokens < 1,000,000 floor** →
STOP BEFORE FULL TRAINING and report to the lead architect. No training,
no pilot, no milestone tag.

## Status
- DONE: acquisition + build pipeline (scripts/acquire_sft_data.py,
  scripts/build_sft_dataset.py, scripts/sft_stats.py, src/sft_dataset.py,
  src/sft_train.py, configs/*, scripts/evaluate_chat.py), provenance
  (sources.jsonl, SOURCES.md), SFT_DATA_REPORT.md, Part K baseline chat eval,
  Part L baseline base-LM validation loss (3.093633), src/chat.py upgrade
  (real multi-turn terminal chat), 25 new tests (22 mandated + 3),
  108/108 full suite, Part U review (90/90 samples passed), docs/ai updated,
  TASK_005_REPORT.txt written.
- 2 real bugs found by mandated tests and FIXED: (1) assistant-label shift
  onto user tokens (labels shorter than ids), (2) OASST validation→train
  split leak (217 examples). Dataset rebuilt after both fixes.
- Awaiting lead architect decision: expand corpus vs. authorize reduced-corpus
  training vs. other direction.

## Outputs
- data/sft/: raw/ (git-ignored), processed/ (git-ignored), manifests/
  (sources.jsonl, SOURCES.md — tracked), stats/ (sft_stats.json,
  SFT_DATA_REPORT.md — tracked).
- checkpoints/chat_v1/: baseline_chat_samples.txt + eval_metrics.json
  (Part K; no .pt files — no training).
- docs/ai/reports/TASK_005_REPORT.txt (exact mandated headings).

## Rules while doing this task
- Do NOT start training / pilot without an architect order (Part E gate).
- Do NOT modify best.pt / latest.pt / any .bin; test.bin stays sealed.
- Only .\.venv\Scripts\python.exe; strict from-scratch still applies.
