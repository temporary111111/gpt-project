# TASK HISTORY — Chronological Records

Future tasks MUST append here automatically (see OPERATING_PROTOCOL,
definition of DONE). Each entry: objective, major outputs, validation/tests,
issues found, final status, artifact paths.

---

## TASK 001 — Machine/environment audit

- **Objective**: Audit the machine and environment for from-scratch LLM development.
- **Major outputs**: Hardware/env facts — Windows 11, AMD Ryzen 5 5600H, 8 GB RAM, RTX 3050 Laptop 4 GB VRAM, CUDA available, Python 3.11 .venv, torch 2.13.0+cu126.
- **Validation**: Verified Python/CUDA availability.
- **Issues found**: None blocking.
- **Final status**: COMPLETE.
- **Artifacts**: (documented in project memory).

## TASK 002 — Architecture, tokenizer, training pipeline

- **Objective**: Build our own GPT architecture, tokenizer infrastructure, training pipeline, generation, checkpointing, CUDA support, and tests.
- **Major outputs**: src/model.py, src/attention.py (RoPE), src/tokenizer.py, src/dataset.py, src/train.py, src/generate.py, src/chat.py, configs/model_small.json.
- **Validation**: Unit tests pass.
- **Issues found**: (later hardening in TASK 003.5).
- **Final status**: COMPLETE.

## TASK 003 — Legal/provenance-aware corpus pipeline

- **Objective**: Build human-written English/Tagalog corpus pipeline.
- **Major outputs**: scripts/acquire_corpus.py, clean_corpus.py, build_corpus.py, train_tokenizer.py, corpus_stats.py, sample_corpus.py, tokenizer_quality_test.py, src/data/*; tokenizer_v1 (SentencePiece BPE, vocab 8000); ~40M-token binary corpus (train 39,299,287 / val 415,832 / test 284,270 tokens).
- **Sources**: Simple English Wikipedia, Tagalog Wikipedia, Tagalog Wikisource, Project Gutenberg public-domain works.
- **Validation**: Tokenizer quality tests (EN/TL/mixed), corpus stats, sample review.
- **Issues found**: None blocking.
- **Final status**: COMPLETE.

## TASK 003.5 — Correctness hardening

- **Objective**: Fix correctness bugs before expensive training; prove the pipeline end-to-end with a preflight run.
- **Major outputs**: Fixed: incorrect special-token IDs; validation sourced from train.bin; AMP clipping scaled gradients; mathematically inconsistent RoPE layout; CUDA RNG state resume issue; checkpoint best/latest semantics; optimizer parameter grouping/dedup; deterministic validation. 50/50 tests. 50-step preflight CUDA run; resume verification (step 50→60).
- **Validation**: pytest 50 passed; smoke test; preflight metrics; resume continuity verified.
- **Issues found**: Real RNG-device bug on resume (ByteTensor RNG states moved to GPU by torch.load) — fixed in restore_rng_state.
- **Final status**: COMPLETE, delivered (architect_task004_preflight_review.zip, TASK_003_5_REPORT.txt).
- **Artifacts**: checkpoints/preflight/ (superseded, do not reuse).

## TASK 004 — FIRST REAL FROM-SCRATCH PRETRAINING (ACCEPTED)

- **Objective**: Production pretraining: 38,379 steps, 8 passes, from scratch, with hardened checkpointing, interruption handling, metrics.
- **Major outputs**: checkpoints/pretrain_v1/{latest.pt, best.pt, metrics.jsonl, run_config.json, generation_samples.txt}; src/train.py hardening (eval-order fix via run_eval_and_save, KeyboardInterrupt save+resume, write_metrics, write_run_config, assert_scratch_dir_empty, compute_grad_norm, skip_nonfinite_grad_step); 59/59 tests; architect_task005_review.zip; TASK_004_REPORT.txt.
- **Results**: 38,379 steps; 314,400,768 tokens; 8.000 passes; best val loss 3.073326 at step 36,500 (~21.61 ppl); final val 3.076171; final train 2.480394; peak VRAM 1.38 GB; median ~17,139 tok/s; English + Filipino continuations readable.
- **Issues found**: 2 hard stops on non-finite gradients (steps 6144, 8022) before the fix; 12 AMP inf-gradient events total, all self-healed after the skip+halve fix; brownout interruption at ~step 26,800 (clean resume from 26,500).
- **Final status**: COMPLETE and ACCEPTED by lead architect.
- **Artifacts**: checkpoints/pretrain_v1/, docs/ai/reports/TASK_004_REPORT.txt.

## TASK 004.5 — SELF-MAINTAINING PROJECT MEMORY, GIT VERSION CONTROL, ZERO-MEMORY AI LEAD HANDOFF (THIS TASK)

- **Objective**: Build permanent operational infrastructure: docs/ai memory system, AGENTS.md, opencode.json, 00_START_HERE.md, .gitignore, git snapshot + milestone tag + push, handoff builder, task finalizer, AI ops tests, AI_LEAD_HANDOFF.zip.
- **Major outputs**: 00_START_HERE.md, AGENTS.md, opencode.json, .gitignore (merged); docs/ai/ (OPERATING_PROTOCOL, CURRENT_STATE, CURRENT_TASK, NEXT_ACTION, PROJECT_STATE.json, CONTINUITY, DECISIONS, TASK_HISTORY, RECOVERY_PROTOCOL, MODEL_STATUS, reports/TASK_004_REPORT.txt, reports/TASK_004_5_REPORT.txt); tools/ai_ops/build_handoff.py, tools/ai_ops/finalize_task.py; tests/test_ai_ops.py; git commit(s) on main; milestone tag task-004-pretraining-complete; AI_LEAD_HANDOFF.zip.
- **Validation**: AI ops tooling tests + full existing suite; ZIP open/hash verification; secret/large-file safety scan.
- **Issues found**: gh CLI not installed (remote push still verified working via origin); unrelated nested repo chatgpt-agent-gateway/ excluded from tracking.
- **Final status**: COMPLETE.
- **Artifacts**: AI_LEAD_HANDOFF.zip, docs/ai/reports/TASK_004_5_REPORT.txt.

## TASK 004.6 — AI OPS TRUST / HANDOFF INTEGRITY HARDENING (THIS TASK)

- **Objective**: Make the AI ops tooling trustworthy for a browser-only AI lead: failed tests must abort finalization; handoff must be a snapshot of exact committed git HEAD bytes (never dirty worktree); accurate working-tree reporting; manifest commit == finalizer commit invariant; no-change finalization still builds the handoff; continuity corrections; .gitignore in handoff; dirty-state investigation.
- **Major outputs**: build_handoff.py now sources every file from `git ls-tree -r` + `git show HEAD:<path>` (exact blob bytes; hashes from those bytes; checkpoint SHA-256 labeled local_untracked_artifact; GIT_STATE.json with head_sha/branch/tracked_worktree_modified_count/staged_change_count/untracked_count/working_tree_clean; working_tree_diff_not_included warning; .gitignore in EXTRA_ROOT_FILES). finalize_task.py now: TESTS_FAILED abort (no stage/commit/push/handoff, non-zero exit), NO_CHANGES path (no empty commit, handoff from current HEAD), HANDOFF_COMMIT_MISMATCH invariant check, fail-closed run_test_suite when venv python missing. .gitignore += data/processed/corpus_text.txt, data/processed/dataset_meta.json (regenerable pipeline artifacts; root cause of the reported dirty state). PROJECT_STATE.json venv typo `.\\venv\\...` → `.\\\\.venv\\Scripts\\python.exe` fixed; updated_at now genuine runtime UTC. tests/test_ai_ops.py: +13 hardening tests.
- **Validation**: full suite 83/83 passed (70 prior + 13 new); finalizer executed on its own reworked code (tests → commit → push → handoff); handoff manifest commit SHA verified == finalizer commit SHA; ZIP opens (testzip clean); checkpoints omitted but SHA-256 recorded (best.pt ba40ad8c…, latest.pt 8c00ff4d… — unchanged, verified).
- **Issues found**: CRLF (core.autocrlf) made disk-byte vs blob-byte hashes differ — resolved by hashing the exact bytes inserted into the ZIP (blob bytes); temp-repo tests pin core.autocrlf=false for determinism. `.git` substring deny blocked `.gitignore` — fixed with `.git/` + explicit allowlist.
- **Final status**: COMPLETE.
- **Artifacts**: AI_LEAD_HANDOFF.zip, docs/ai/reports/TASK_004_6_REPORT.txt.