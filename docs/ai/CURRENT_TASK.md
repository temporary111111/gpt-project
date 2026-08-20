# CURRENT TASK

**TASK 005.1 — EXPAND HUMAN SFT CORPUS AND COMPLETE CHAT V1 TRAINING**

## Objective
Expand the TASK 005 SFT corpus with Databricks Dolly 15K + Taskmaster-1
(human-only), re-audit everything, and IF unique supervised target tokens
>= 1,000,000, automatically run pilot → full SFT (up to 3 epochs, retention
guards) → post-SFT evaluation → milestone tag `task-005-chat-v1-complete`.
Do NOT start TASK 006.

## Status
- DONE (pipeline phase): Dolly acquisition (15,010) + Taskmaster-1 acquisition
  (13,170 convs, official dialog-ID splits); build_sft_dataset.py extended
  (Dolly/TM builders, cross-source dedup with source-pair report, probe
  exclusion on ANY user turn, fil sampling weight ≤4x + English source
  balance, unique/effective gate stats, aya-eng mislabel filter — 104 rows);
  src/sft_dataset.py (copies/effective epochs, unique vs effective token
  methods); src/sft_train.py (dynamic eval interval, unique/effective
  accounting); sft_stats.py; build_handoff.py Parts J/K (manifests/stats +
  chat_v1 small files included; chat_v1 best/latest hashed-excluded;
  data/sft/raw + processed denied); 12 new held-out multi-turn eval probes;
  127/127 tests green; 150-sample quality audit passed; corpus rebuilt →
  UNIQUE_SUPERVISED_TARGET_TOKENS = 1,863,853 → **GATE PASSED** (Part M:
  continue automatically); SOURCES.md + SFT_DATA_REPORT.md updated; base
  checkpoint SHA re-verified (ba40ad8c…).
- NEXT (training phase): Part Y commit+push of tested code (this commit) →
  Part Q pilot run 1 (`--max-epochs 1`, ~200 optimizer steps; retention
  eligibility base-val ≤ 3.557678, hard stop > 3.712360) → Part R run 2
  (`--init-from resume --max-epochs 3`, early stop 4 evals) → Parts T/U/V/W
  post-SFT eval + interactive chat → docs/report/finalize/tag.

## Config (Part P)
batch 8, grad accum 4, ctx 256, peak LR 5e-5 → min 5e-6, wd 0.01, clip 1.0,
AMP fp16, seed 1337, max 3 epochs, warmup max(100, 3%), eval ~200 steps with
several evals per epoch (auto-tightened), early stop 4 evals. New
optimizer/scaler/scheduler state (NOT a pretraining resume); full fine-tune
(no LoRA).

## Outputs so far
- data/sft/raw/ + processed/ (git-ignored); manifests/sources.jsonl +
  SOURCES.md, stats/sft_stats.json + SFT_DATA_REPORT.md +
  quality_audit_samples.txt (tracked).
- checkpoints/chat_v1/: baseline outputs only (no .pt yet).

## Rules while doing this task
- Preserve the 2 TASK 005 fixes (label alignment; no OASST leak).
- test.bin stays sealed; never modify best.pt / latest.pt / *.bin.
- Only .\.venv\Scripts\python.exe; strict from-scratch still applies (no
  synthetic data, no auto-MT, no external LLM answers; DeepSeek writes code
  only, never training examples).