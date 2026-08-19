# DECISIONS — Architectural & Project Decisions

Chronological record of important decisions and WHY. When a decision is
superseded, mark it as superseded WITH rationale. Never erase history.

## D-001: New project separate from the old MiniGPT demo
The old MiniGPT demo project is a separate repository/history. This project
("chatgpt-like") starts fresh with its own repo, from-scratch philosophy, and
corpus pipeline. WHY: a clean slate for a serious from-scratch foundation-model
proof of concept with strict provenance constraints.

## D-002: User is relay/operator only
The human does not maintain files, run git manually, or remember recovery
prompts. WHY: continuity must survive without human memory; the AI pair
(lead architect + implementation engineer) must be self-sufficient.

## D-003: Strict from-scratch model
No pretrained LLM weights, tokenizer, or embeddings; no distillation; no
external LLM API as the model brain. WHY: the project goal is to demonstrate
building an LLM from first principles on consumer hardware; this also keeps
licensing/provenance clean.

## D-004: Python 3.11 .venv
All work uses `.\.venv\Scripts\python.exe` (Python 3.11). WHY: torch 2.13+cu126
wheel availability and reproducibility on this machine; global Python is
uncontrolled.

## D-005: ~29M parameter first proof-of-concept
GPT-style decoder-only, 29,270,528 params, vocab 8000, d_model 512, 8 layers,
8 heads, ctx 256. WHY: trainable end-to-end on an RTX 3050 4GB within hours,
while exercising the full stack (RoPE, RMSNorm, AMP, grad accumulation,
checkpointing, resume). Scaling comes later in the roadmap.

## D-006: Own 8k SentencePiece tokenizer
SentencePiece BPE, vocab 8000, trained from scratch on our own cleaned corpus.
WHY: vocab 8000 is a good balance for a mixed EN/TL corpus at 29M scale;
SentencePiece handles the multilingual surface without subword bleeding
between scripts.

## D-007: English/Tagalog corpus (~60/40)
Sources: Simple English Wikipedia, Tagalog Wikipedia, Tagalog Wikisource,
Project Gutenberg (public domain). WHY: bilingual EN/TL chat product target;
all sources are human-written and legally clean.

## D-008: Provenance/legal data policy
Every source record is kept in data/manifests (provenance manifest), raw and
cleaned corpora are never committed to git. WHY: reproducibility and legal
safety; only manifests and processed token counts are tracked.

## D-009: Sealed test set
test.bin (284,270 tokens) is SEALED — evaluation only when authorized. WHY:
prevents accidental test-set contamination during development iterations.

## D-010: Code review before expensive training
Each training-adjacent task ends with a review package (zip + report) for the
lead architect before the next expensive stage. WHY: catch bugs before burning
GPU-hours; the browser AI is the reviewer.

## D-011: Corrected RoPE convention
RoPE uses the mathematically consistent cat-based layout (verified: old
repeat_interleave layout deviated ~0.95 max norm; new deviates ~4.8e-7).
WHY: the old layout was mathematically inconsistent (TASK 003.5).

## D-012: ~8-pass TASK 004 run
38,379 steps = 8.000 corpus passes over train.bin with lr 6e-4→6e-5 cosine,
warmup 500, batch 8 × grad-accum 4, ctx 256. WHY: first full pretraining;
8 passes is a reasonable budget for a 29M model on this hardware.

## D-013: Base pretraining before chat tuning
TASK 004 produces a pure base model; chat/instruction tuning (TASK 005) comes
after review. WHY: separate concerns; the base model is the foundation and its
quality must be verified first.

## D-014: AMP non-finite gradient self-healing (supersedes hard stop for grads)
When the GRADIENT norm is non-finite: skip the optimizer step, halve the AMP
scale, drop grads, log warning + metrics record, continue. Non-finite LOSS
remains a hard stop. WHY: GradScaler scale grows unbounded (×2 every 2000 clean
steps), rare fp16 overflow events are non-deterministic and unreproducible, and
hard-stopping + resuming would recur ever more often over a 38K-step run
(observed at steps 6144 and 8022). SUPERSEDES: the TASK 003.5 hard-stop-on-inf-
gradient guard (documented in TASK 004 report; accepted by the lead architect).
No hyperparameters or architecture changed.

## D-015: Future tool/agent integration
Roadmap includes tool-use training, memory, and agent/tool integration
(files/shell/calculator/search). WHY: the product goal is a ChatGPT-LIKE
experience, which is agentic, not just text completion.

## D-016: Git repository as canonical project memory
Git + docs/ai are the authoritative memory; browser chat conversations are NOT.
WHY: continuity must survive chat loss, model switches, and new AI leads.
Established by TASK 004.5.

## D-017: test.bin stays sealed; checkpoints/corpus never committed
*.pt, *.bin, raw/cleaned corpus, and credentials are git-ignored and
safety-scanned by the finalizer. WHY: repo size, secret safety, and artifact
protection (see RECOVERY_PROTOCOL MODE 3 limitation).

## D-018: Handoff ZIP = exact committed git HEAD bytes (TASK 004.6)
AI_LEAD_HANDOFF.zip file members are read from `git show HEAD:<path>`, never
from arbitrary working-tree bytes; manifest SHA-256s are computed from the
exact bytes inserted into the ZIP; checkpoint hashes are labeled
local_untracked_artifact; GIT_STATE.json reports accurate
tracked_worktree_modified_count / staged_change_count / untracked_count /
working_tree_clean and flags working_tree_diff_not_included when dirty. WHY:
a browser-only AI lead must be able to trust the ZIP as a faithful snapshot
of a specific commit, even if the working tree was dirty when it was built.
Test-repo consequence: core.autocrlf=false in temp repos so blob bytes are
deterministic (handoff uses raw blob bytes regardless).

## D-019: Failed tests ABORT finalization (TASK 004.6)
If the required test suite fails, the finalizer stages/commits/pushes nothing,
builds no handoff, reports status TESTS_FAILED, and exits non-zero. WHY:
never claim completion or snapshot a broken state; fail closed. run_test_suite
also fails closed when the venv python is missing.

## D-020: No-change finalization still builds the handoff (TASK 004.6)
When there is nothing safe to commit, the finalizer creates NO empty commit,
reports NO_CHANGES with the current HEAD SHA, skips pushing, and still builds
and verifies AI_LEAD_HANDOFF.zip from that HEAD. WHY: every meaningful task
must end with a valid, verifiable handoff regardless of whether new commits
were needed.

## D-021: Handoff manifest commit SHA == finalizer commit SHA invariant (TASK 004.6)
After building the ZIP, the finalizer re-opens HANDOFF_MANIFEST.json and
fails with HANDOFF_COMMIT_MISMATCH if its git head_sha differs from the
completion commit SHA. WHY: the ZIP must never silently describe a different
state than the commit the finalizer just produced.