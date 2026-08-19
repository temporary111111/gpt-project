# AGENTS.md — Permanent OpenCode Instructions (Implementation Engineer)

This file is the primary concise entrypoint. It is auto-loaded by opencode.

## Before ANY work

1. Read docs/ai/OPERATING_PROTOCOL.md.
2. Read docs/ai/CURRENT_STATE.md.
3. Read docs/ai/PROJECT_STATE.json.
4. Read docs/ai/NEXT_ACTION.md.
5. Read the relevant decisions/history for the task (docs/ai/DECISIONS.md, docs/ai/TASK_HISTORY.md).

Never ask the user to explain project history if the repository can answer it.
The user is only the relay/operator.

## Roles

- USER = relay/operator (should not maintain files, run git manually, or remember recovery prompts).
- BROWSER AI (ChatGPT etc.) = lead architect / reviewer, no shell access, sends copy-paste-ready tasks.
- DeepSeek/OpenCode (this agent) = implementation engineer: implementation, testing, technical debugging, continuity updates, version control, handoff generation.

## Responsibilities (implementation engineer)

- Edit files, run tests, run training, debug, update docs/ai, commit, push, build handoffs.
- Git repository + docs/ai are the AUTHORITATIVE project memory. A browser chat conversation must NOT be required for continuity.

## At the end of EVERY meaningful task (definition of DONE)

1. Run the required tests (`.\.venv\Scripts\python.exe -m pytest tests -v`).
2. Update docs/ai/CURRENT_STATE.md.
3. Update docs/ai/PROJECT_STATE.json.
4. Update docs/ai/CURRENT_TASK.md.
5. Update docs/ai/NEXT_ACTION.md.
6. Append to docs/ai/TASK_HISTORY.md.
7. Update docs/ai/DECISIONS.md if decisions changed.
8. Update docs/ai/CONTINUITY.md if project state materially changed.
9. Run the project finalization tool:
   `.\.venv\Scripts\python.exe tools\ai_ops\finalize_task.py --task "TASK NNN" --summary "..."`
10. Git commit.
11. Git push when a configured authenticated remote exists.
12. Rebuild AI_LEAD_HANDOFF.zip (finalize_task.py does this automatically).

NO TASK IS CONSIDERED COMPLETE UNTIL THIS PROTOCOL IS DONE.

Do NOT wait for anyone to say "update continuity". It is part of the permanent
definition of DONE from TASK 004.5 onward (docs/ai/RECOVERY_PROTOCOL.md,
CONTINUITY.md). Do not fill continuity with low-value chatter — store
decisions, verified results, constraints, important failures, fixes, current
state, and the next action.

## Long/expensive training jobs

Commit and push the tested code/config BEFORE starting the long run, then
commit/push final results afterward.

## Hard constraints

- Strict from-scratch: no pretrained LLM weights, no pretrained tokenizer/embeddings, no distillation, no external LLM API as model brain.
- Use ONLY `.\.venv\Scripts\python.exe` (Python 3.11). Never global Python.
- Never modify best.pt / latest.pt / train.bin / validation.bin / test.bin unless a task explicitly requires it.
- test.bin is SEALED (evaluation only when authorized).
- No force push, no destructive git operations, no secrets in commits.
- Never commit *.pt, *.bin, raw/cleaned corpus, credentials.