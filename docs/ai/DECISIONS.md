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

## D-022: TASK 005 corpus floor — 1M supervised tokens, no fabrication (Part E)
SFT training may only start when >= 1,000,000 usable human-only supervised
target tokens exist. The measured corpus (624,057; Filipino 8.2%) is BELOW
the floor → training STOPPED BEFORE FULL TRAINING and reported to the lead
architect. No synthetic data, no auto-translation, no engineer-authored
Filipino, and no oversampling beyond 2x may be used to pad the corpus; the
natural English-dominant ratio is accepted and reported. WHY: the task
mandates human-only, provenance-clean data and quality over ratio.

## D-023: Assistant-only label masking must be byte-aligned with ids (TASK 005)
The SFT labels array must have exactly the same length as the ids array
(-100 prefix, supervised span over the final assistant target + EOS). The
initial build produced labels SHORTER than ids, shifting supervision onto
USER tokens — caught by the mandated test suite (test_user_tokens_masked),
fixed, and the dataset rebuilt. WHY: a shifted mask silently trains the model
to predict user text; the mandated Part W tests exist precisely to catch this
class of error.

## D-024: TASK 005.1 corpus expansion — Dolly 15K + Taskmaster-1 only (Part A/B)
The expanded corpus adds Databricks Dolly 15K (cc-by-sa-3.0, revision
bdd27f4d…) and Google's Taskmaster-1 (CC BY 4.0, official repo rev
d92cb6af…), both human-generated. No Alpaca/ShareGPT/Vicuna-style synthetic
or LLM-generated data is ever allowed. WHY: strict from-scratch + provenance
cleanliness; 4 human-only sources provide the volume needed to pass the 1M
token gate without fabrication.

## D-025: Taskmaster assistant turns become targets, capped at 4 per conversation
Every accepted assistant turn is a candidate SFT target; up to 4 targets per
conversation are chosen deterministically across the conversation (even
coverage start/middle/end). Earlier assistant turns are context-masked (only
the final assistant response of each example is supervised). Splits are by
CONVERSATION ID (official dialog-ID CSVs for self-dialogs; deterministic
95/5 bucket for woz) — a conversation never crosses train/val. WHY:
task-oriented dialogue is mostly filler confirmation turns; capping keeps
each conversation's signal dense and prevents one domain from dominating.

## D-026: Filipino sampling capped at 4x; effective share lands at 10.5% (Part G)
Filipino examples get a sampling multiplier computed exactly to reach the
15% effective-token target, clamped to [1, 4.0]. With only 51,126 unique fil
tokens the cap is reached and the effective share is 10.5% — the maximum
achievable without fabrication/translation (which remain forbidden). WHY:
the 15–25% band is a target, not a gate; repetition beyond 4x would distort
the epoch distribution.

## D-027: English source balance via up-weighting the other English sources (Part H)
If any English source exceeds 50% of effective English target tokens, the
OTHER English sources are deterministically up-weighted (dominant/others)
iteratively until no source exceeds 50%. WHY: integer "copies" cannot
down-weight a source below 1 copy; up-weighting others is mathematically
equivalent to down-weighting the dominant source and satisfies the task's
"or equivalently" clause. (In the final corpus no source exceeded 50% —
largest share dolly 39.5% — so weights stayed 1.0.)

## D-028: Aya "eng" mislabeled rows rejected deterministically (Part I quality fix)
Some Aya `language_code == "eng"` rows contain Somali/Indonesian/Basque/
German/Turkish text. A deterministic English-check heuristic rejects rows
with non-Latin script, or a >=4-word prompt AND target both containing <2
English function words, or a >=6-word target with a <2-word-hit prompt
(catches short mislabeled prompts). 104 rows rejected; verified 0 non-Latin
and 0 Somali survivors in the final corpus. WHY: mislabeled non-English
content would pollute the English chat signal; the check is deterministic,
tested, and never rewrites data.