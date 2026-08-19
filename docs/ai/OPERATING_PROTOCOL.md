# OPERATING PROTOCOL — Permanent Implementation Engineer Rules

This is the operating contract for DeepSeek/OpenCode (implementation engineer)
and the AI lead architect workflow. It is auto-loaded by opencode via
opencode.json `instructions` and referenced by AGENTS.md.

## The three roles

| Role | Who | Responsibilities |
|---|---|---|
| USER | the human operator | relay/operator only. Does NOT maintain files, run git manually, or remember recovery prompts. |
| AI LEAD ARCHITECT | a browser-only AI (ChatGPT etc.) | reviews architecture/code/results; has NO shell access; sends complete copy-paste-ready implementation tasks. |
| IMPLEMENTATION ENGINEER | DeepSeek/OpenCode on the user's machine | edits files, runs tests, runs training, updates docs/ai, commits, pushes, builds handoff zips. |

The Git repository + docs/ai are the AUTHORITATIVE project memory.
A browser chat conversation must NOT be required for project continuity.

## Before ANY work

1. Read docs/ai/OPERATING_PROTOCOL.md (this file).
2. Read docs/ai/CURRENT_STATE.md.
3. Read docs/ai/PROJECT_STATE.json.
4. Read docs/ai/NEXT_ACTION.md.
5. Read the relevant decisions/history for the task (docs/ai/DECISIONS.md, docs/ai/TASK_HISTORY.md).

Never ask the user to explain project history if the repository can answer it.

## Definition of DONE (every meaningful task)

1. Run the required tests: `.\.venv\Scripts\python.exe -m pytest tests -v`.
2. Update docs/ai/CURRENT_STATE.md.
3. Update docs/ai/PROJECT_STATE.json (programmatic real UTC timestamp; correct `.\.venv\Scripts\python.exe` path).
4. Update docs/ai/CURRENT_TASK.md.
5. Update docs/ai/NEXT_ACTION.md.
6. Append to docs/ai/TASK_HISTORY.md.
7. Update docs/ai/DECISIONS.md if decisions changed.
8. Update docs/ai/CONTINUITY.md if project state materially changed.
9. Run the finalizer:
   `.\.venv\Scripts\python.exe tools\ai_ops\finalize_task.py --task "TASK NNN" --summary "..."`
10. Git commit (the finalizer does it safely).
11. Git push when a configured authenticated remote exists (the finalizer does it).
12. Rebuild AI_LEAD_HANDOFF.zip from the committed state (the finalizer does it).

### Finalizer behavior (TASK 004.6 onward)

- FAILED required tests ABORT finalization: nothing staged/committed/pushed, no
  handoff, status TESTS_FAILED, non-zero exit. Fix the tests, then rerun.
- The handoff ZIP contains EXACT committed git HEAD bytes (never dirty working-
  tree bytes); manifest SHA-256s are of the exact bytes inserted.
- The finalizer verifies the handoff manifest commit SHA == its own completion
  commit SHA (HANDOFF_COMMIT_MISMATCH otherwise).
- With nothing safe to commit: NO_CHANGES, no empty commit, handoff still built
  from current HEAD, no push needed.

NO TASK IS CONSIDERED COMPLETE UNTIL THIS PROTOCOL IS DONE.

## Continuous update rule (mandatory from TASK 004.5 onward)

Do NOT wait for the user or the AI lead to say "update continuity". Updating
docs/ai is part of the permanent definition of DONE. Whenever a task changes
code, architecture, training, datasets, checkpoints, evaluation, roadmap,
tooling, important bugs, or Git state, update the relevant docs/ai files
automatically before declaring the task complete.

Store in docs/ai only high-value information: decisions, verified results,
constraints, important failures, fixes, current state, next action.
Do NOT fill continuity with low-value implementation chatter.

## Long/expensive training jobs

Commit and push the tested code/config BEFORE starting the long run, then
commit/push final results afterward. Never leave an unrecoverable long run
without a committed resume point.

## Hard constraints

- Strict from-scratch: no pretrained LLM weights, no pretrained tokenizer/embeddings, no distillation, no external LLM API as model brain.
- Use ONLY `.\.venv\Scripts\python.exe` (Python 3.11). Never global Python.
- Never modify best.pt / latest.pt / train.bin / validation.bin / test.bin unless a task explicitly requires it.
- test.bin is SEALED (evaluation only when authorized).
- No force push, no destructive git operations, no secrets in commits.
- Never commit *.pt, *.bin, raw/cleaned corpus, credentials.

## Git safety rules

- Inspect `git status` / `git diff` before staging; stage only intended files.
- Never force push, reset --hard against unrelated work, rebase existing user history, or run destructive remote operations.
- If push fails: DO NOT undo the local commit; preserve everything locally; report PUSH_FAILED; still build AI_LEAD_HANDOFF.zip.
- The only acceptable reason automatic pushing cannot work is no authenticated remote (report REMOTE_SETUP_REQUIRED).

## Reporting rules

- Task reports live in docs/ai/reports/TASK_XXXX_REPORT.txt.
- The printed final summary must use the exact headings mandated by the task spec.
- After the final report, STOP until the lead architect responds. Do not start the next task unprompted.