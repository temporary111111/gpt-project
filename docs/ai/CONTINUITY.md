# CONTINUITY — Living Project Document

This is the permanent living memory of the project. Keep it structured and
up-to-date. It is NOT a chat log: store decisions, verified results,
constraints, important failures, fixes, current state, and the next action.

## Project goal

Build a ChatGPT-LIKE language model from scratch — a local chat/assistant
experience powered by our own model, tokenizer, and corpus. No pretrained
weights, no pretrained tokenizer, no distillation, no external LLM API as the
model brain.

IMPORTANT distinction:
- ChatGPT-LIKE PRODUCT EXPERIENCE (chat UI, roles, streaming UX) is different
  from ChatGPT-LEVEL FOUNDATION MODEL INTELLIGENCE (frontier-scale pretraining).
- The current 29.27M-parameter model is a PROOF-OF-CONCEPT foundation model.
  It demonstrates the full from-scratch stack; it is not frontier intelligence.

## Strict from-scratch philosophy

- no pretrained LLM weights (no Ollama/Qwen/Llama/DeepSeek/GPT weights as model intelligence)
- no pretrained tokenizer
- no pretrained embeddings
- no LLM distillation
- no external LLM API as final model brain
- DeepSeek/OpenCode is a coding engineer only, never the model brain

## Role separation

| Role | Who | Responsibilities |
|---|---|---|
| USER | human operator | relay/operator only; does not maintain files, run git manually, or remember recovery prompts |
| BROWSER AI | ChatGPT etc. | lead architect / reviewer; no shell access; sends copy-paste-ready tasks |
| DEEPSEEK / OPENCODE | implementation engineer | editing, testing, training, debugging, continuity updates, git, handoffs |

Git repository + docs/ai = AUTHORITATIVE project memory. Chat history is not.

## Hardware & environment

- Windows 11, AMD Ryzen 5 5600H, 8 GB RAM
- RTX 3050 Laptop GPU, 4 GB VRAM (CUDA available)
- Python 3.11, venv: `.\.venv\Scripts\python.exe` (NEVER global Python)
- Project root: C:\Users\dev\Desktop\chatgpt-like
- torch 2.13.0+cu126, numpy 2.4.6, sentencepiece 0.2.2, pytest 9.1.1

## Model architecture

GPT-style decoder-only Transformer:
- 29,270,528 params (all trainable), vocab 8000, d_model 512
- 8 layers, 8 heads, FFN 2048, context 256
- RMSNorm, RoPE (cat-based layout), tied input/output embeddings, dropout 0
- Pre-norm blocks, GPT-2 style residual scaling init
- Files: src/model.py (GPTModel, ModelConfig), src/attention.py (causal MHA + RoPE)

## Tokenizer

- SentencePiece BPE, vocab 8000, trained from scratch on our own corpus
- data/tokenizer/tokenizer_v1.model (+ .vocab + _meta.json)
- Special tokens: <unk>=0 <pad>=1 <bos>=2 <eos>=3 <system>=4 <user>=5 <assistant>=6

## Corpus

- Human-written English (~60%) + Filipino (~40%), legally sourced:
  Simple English Wikipedia, Tagalog Wikipedia, Tagalog Wikisource, Project Gutenberg (public domain)
- train = 39,299,287 tokens; validation = 415,832; test = 284,270 (SEALED)
- Pipeline: scripts/acquire_corpus.py → clean_corpus.py → build_corpus.py → train_tokenizer.py
- data/processed/*.bin are uint16 memmaps (never loaded fully into RAM)

## Completed tasks

- TASK 001: machine/environment audit
- TASK 002: GPT architecture, tokenizer, training pipeline, generation, checkpointing, CUDA, tests
- TASK 003: legal/provenance-aware corpus pipeline + tokenizer_v1 + ~40M-token corpus
- TASK 003.5: correctness hardening (special IDs, val split, AMP clip, RoPE, RNG resume, checkpoint semantics, param groups, deterministic val)
- TASK 004: FIRST real from-scratch pretraining (ACCEPTED) — see below
- TASK 004.5: self-maintaining memory, git, handoff infrastructure
- TASK 004.6: AI Ops trust / handoff integrity hardening (committed-state handoff, test-failure abort, SHA invariant, no-change finalization, continuity corrections)
- TASK 005: Chat/instruction tuning V1 — PIPELINE COMPLETE, TRAINING STOPPED AT PART E GATE (624,057 supervised tokens < 1M floor); see "TASK 005 (SFT) state" below

## Important bugs/fixes (verified)

- Special-token IDs were wrong → fixed (pad=1 bos=2 eos=3 unk=0)
- Validation was sourced from train.bin → fixed (separate validation.bin)
- AMP gradient clipping scaled the gradients → fixed (clip unscaled)
- RoPE layout mathematically inconsistent → fixed (cat convention, verified vs old)
- CUDA RNG resume: torch.load(map_location=cuda) moves ByteTensors; fixed by pinning CPU RNG and passing CPU tensors to torch.cuda.set_rng_state_all
- Checkpoint best/latest save order: latest.pt was saved with the OLD best_val_loss → fixed (update best → save best → save latest, all with updated value)
- Optimizer param grouping/dedup (weight decay on matrices only)
- Deterministic validation
- AMP non-finite GRADIENT norm (12 events in TASK 004): standard self-heal — skip optimizer step, halve GradScaler scale, drop grads, log warning + metrics, continue. Root cause: scale doubles every 2000 clean steps → unbounded growth → rare fp16 overflow. Non-finite LOSS remains a hard stop. This supersedes the TASK 003.5 hard-stop-on-inf-grad behavior (architect-approved).
- SFT assistant-label shift (TASK 005): labels array was SHORTER than ids, shifting supervision onto USER tokens → caught by mandated test test_user_tokens_masked, fixed (full-length -100 prefix + target span), dataset rebuilt (supervised tokens re-measured: 624,057, up from the buggy 449,439).
- OASST validation→train split leak (TASK 005): source split "validation" compared to "val" leaked 217 validation examples into TRAIN → fixed with norm_split(), rebuilt, no-leakage test added.

## Production pretraining result (TASK 004)

- 38,379 optimizer steps; 314,400,768 tokens; 8.000 corpus passes
- best val loss 3.073326 at step 36,500 (perplexity ~21.61)
- final val loss 3.076171; final train loss 2.480394
- peak VRAM 1.38 GB; median ~17,139 tok/s
- base model produces grammatical English + fluent Filipino continuations (generation_samples.txt)

## Current checkpoint locations

- BEST: checkpoints/pretrain_v1/best.pt (step 36,500, val 3.073326)
- LATEST: checkpoints/pretrain_v1/latest.pt (step 38,379, tokens 314,400,768)
- Do NOT modify/delete/move them. Never commit *.pt.

## TASK 005 (SFT) state

- Pipeline (tested, 127/127): scripts/acquire_sft_data.py → build_sft_dataset.py → sft_stats.py; src/sft_dataset.py (SFTDataset, copies/effective sampling), src/sft_train.py (pilot gate, retention guards, resume, atomic checkpoints, dynamic eval interval); configs/sft_chat_v1.json + sft_train_v1.json; scripts/evaluate_chat.py (baseline/compare modes; 6 new held-out multi-turn probes); src/chat.py upgraded (real multi-turn terminal chat, EOS + role-marker stop, clear/exit/system); scripts/audit_sft_samples.py; tools/ai_ops/build_handoff.py Parts J/K (SFT manifests/stats + chat_v1 small files included; chat_v1 best/latest hashed-excluded; data/sft/raw + processed denied).
- Corpus (TASK 005.1): 4 human-only sources — Aya (rev f9ea0458…, 2,568 ex / 180,896 tok), OASST1 (rev fdf72ae0…, 4,266 / 439,354), Dolly (rev bdd27f4d…, cc-by-sa-3.0, 10,926 / 715,354), Taskmaster-1 (rev d92cb6af…, CC BY 4.0, 33,016 / 528,249) → 50,776 examples; UNIQUE_SUPERVISED_TARGET_TOKENS = 1,863,853 (train 1,697,262 / val 166,591) — GATE PASSED (>= 1M). fil unique 51,126 (2.7%); effective fil share 10.5% at 4x cap (D-026); English weights 1.0 (no source >50%, D-027); aya-eng mislabel filter rejected 104 rows (D-028); cross-source exact dups removed 1,160; target unk 0.0; leakage all 0 (incl. TM conversation isolation).
- Baselines (best.pt, eval-only, unchanged): BASELINE_PRETRAIN_VAL_LOSS = 3.093633 (eligible ≤ 3.557678; hard stop > 3.712360); chat baseline greedy EOS 0.0 / repetition 0.68 / fil-in-fil 0.4 / eng-in-eng 0.5.
- Base checkpoint re-verified 2026-08-20: best.pt SHA-256 ba40ad8c…, latest.pt 8c00ff4d… — unchanged. checkpoints/chat_v1/ still has NO .pt (training not started).
- NEXT: SFT pilot (~200 steps, run 1 --max-epochs 1) → retention eligibility → full SFT (run 2 --init-from resume --max-epochs 3, early stop 4 evals) → post-SFT eval (evaluate_chat.py --mode compare; 20 probes + Bruno + 6 new multi-turn) → interactive chat → report + finalize + tag task-005-chat-v1-complete on success.

## Current limitations

- base model NOT chat/instruction tuned (SFT training not yet run; no chat_v1 weights exist)
- repetition, especially greedy decoding
- fluent factual hallucination
- small-model reasoning limitations
- rare replacement-character output for OOV symbols (tokenizer coverage)
- Filipino only 10.5% of EFFECTIVE supervised tokens (4x cap reached; natural ratio, per D-022/D-026)

## Next planned stage

TASK 005.1 training phase (in progress): SFT pilot (~200 optimizer steps,
retention eligibility ≤ 3.557678 / hard stop > 3.712360), then full SFT
(`--init-from resume --max-epochs 3`, early stop after 4 consecutive val
evals without improvement), then post-SFT evaluation
(`scripts/evaluate_chat.py --mode compare` — 20 fixed probes greedy+sampled,
Bruno + 6 new held-out multi-turn probes, EOS/repetition/role-leakage,
fil-in-fil & eng-in-eng), then interactive chat verification, report,
finalize, and milestone tag `task-005-chat-v1-complete` on success.

## Long-term roadmap

base pretraining (DONE) → chat/instruction tuning (TASK 005) → better
evaluation → reasoning curriculum → tool-use training → memory → agent/tool
integration → files/shell/calculator/search → possible larger model scaling

## AI Ops trust rules (TASK 004.6)

- AI_LEAD_HANDOFF.zip is built from EXACT committed git HEAD bytes (`git show HEAD:<path>`), never dirty working-tree bytes; manifest hashes are of the exact bytes in the ZIP; checkpoint hashes are labeled local_untracked_artifact.
- GIT_STATE.json inside the ZIP carries head_sha, branch, tracked_worktree_modified_count, staged_change_count, untracked_count, working_tree_clean, and a working_tree_diff_not_included flag when dirty.
- The finalizer ABORTS with TESTS_FAILED (no stage/commit/push/handoff, non-zero exit) when required tests fail.
- With nothing safe to commit it reports NO_CHANGES, creates no empty commit, and still builds the handoff from current HEAD.
- The finalizer verifies the handoff manifest commit SHA == its own completion commit SHA (HANDOFF_COMMIT_MISMATCH otherwise).
- data/processed/corpus_text.txt + dataset_meta.json are regenerable pipeline artifacts and are git-ignored (keeps the tree clean).

## Git state

- Repository: yes (origin https://github.com/temporary111111/gpt-project.git, branch main)
- Milestone tags: task-004-pretraining-complete
- Checkpoints/corpus binaries are NOT in git (see RECOVERY_PROTOCOL MODE 3 limitation)