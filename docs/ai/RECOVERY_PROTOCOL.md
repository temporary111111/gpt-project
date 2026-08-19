# RECOVERY PROTOCOL — Three Recovery Modes

The project must be recoverable WITHOUT chat history. These are the three
standard recovery paths.

## MODE 1 — Browser AI handoff (normal case)

1. User uploads `AI_LEAD_HANDOFF.zip` to the browser AI (ChatGPT etc.).
2. New AI reads `00_START_HERE.md` first — it explains the purpose and the
   reading order (AGENTS.md → docs/ai/CURRENT_STATE.md → PROJECT_STATE.json →
   CONTINUITY.md → DECISIONS.md → TASK_HISTORY.md → NEXT_ACTION.md → code).
3. The new AI reconstructs state from the package, verifies against code,
   and issues the next implementation task. The user should only need to say
   "Continue." if the UI demands text.

## MODE 2 — Agentic/codebase AI (OpenCode, Cursor, etc.)

1. New AI receives repository access (clone or same working tree).
2. Instruction should effectively be:

       Read AGENTS.md and reconstruct the project from repository state.

3. AGENTS.md + docs/ai/* are auto-loaded (opencode.json instructions) and
   contain everything needed. No chat history required.

## MODE 3 — Full disaster recovery

1. Clone the Git repository (it contains code, configs, tests, docs/ai,
   tokenizer v1 files, and small run metadata).
2. Restore NON-Git large artifacts separately if available (see limitation).
3. Verify environment: `.\.venv\Scripts\python.exe -m pytest tests -v`.

### IMPORTANT LIMITATION (documented honestly)

Normal Git DOES NOT back up:
- large model checkpoints (best.pt / latest.pt — git-ignored by design)
- training binaries (train.bin / validation.bin / test.bin)
- raw/cleaned corpus and downloaded sources

The checkpoints and corpus are excluded from git for size and safety reasons
(see DECISIONS D-017). If the working machine loses these files, they cannot
be restored from git. Keep external backups of checkpoints and data
separately. A later architecture task may implement large-artifact/checkpoint
backup tooling.

### Recovery checklist

- [ ] Git clone/working tree present, branch main, commit matches handoff manifest
- [ ] docs/ai present and consistent (CURRENT_STATE, PROJECT_STATE.json, NEXT_ACTION)
- [ ] .venv recreated: `& 'C:\Users\dev\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv` then `pip install -r requirements.txt`
- [ ] checkpoints/pretrain_v1/best.pt + latest.pt restored from external backup
- [ ] data/processed/*.bin restored from external backup
- [ ] data/tokenizer/tokenizer_v1.model present (tracked in git)
- [ ] tests pass, best.pt loads with 29,270,528 params
- [ ] NEXT_ACTION.md reviewed; do NOT restart completed tasks