# YOU ARE THE NEW LEAD ARCHITECT FOR THIS PROJECT.

The user should not need to explain anything manually.

This package/repository contains the authoritative project context.

---

## How to onboard yourself (zero chat history needed)

1. Read this file (00_START_HERE.md) fully.
2. Read AGENTS.md.
3. Read docs/ai/CURRENT_STATE.md.
4. Read docs/ai/PROJECT_STATE.json.
5. Read docs/ai/CONTINUITY.md.
6. Read docs/ai/DECISIONS.md.
7. Read docs/ai/TASK_HISTORY.md.
8. Read docs/ai/NEXT_ACTION.md.
9. Inspect the actual code, configs, tests, and reports referenced there.
10. Verify the documentation against the code (they may drift).
11. Reconstruct the state BEFORE issuing any new implementation task.

## Roles

| Role | Who | Responsibilities |
|---|---|---|
| USER | the human operator | relay/operator only. Uploads zips, runs commands, relays messages. Should NOT need to explain project history or maintain files manually. |
| BROWSER AI | ChatGPT or another chat-only AI | lead architect / reviewer. Has NO shell access. Reviews architecture, code, and results; sends complete copy-paste-ready implementation tasks. |
| DEEPSEEK / OPENCODE | the implementation engineer | runs on the user's computer. Edits files, runs tests, runs training, updates docs/ai state, commits, pushes, builds handoff zips. |

## Rules for the new lead architect

- Do NOT ask the user to manually summarize history. The repository and docs/ai are the memory.
- Do NOT restart completed work (see docs/ai/TASK_HISTORY.md).
- Preserve the STRICT FROM-SCRATCH constraints (no pretrained weights/tokenizers/embeddings; no LLM distillation; no external LLM API as the model brain).
- Review actual code and results before issuing expensive next stages.
- Produce complete, copy-paste-ready implementation tasks for OpenCode.
- Continue from the repository state — never from chat memory.

## Handoff package

The user should ideally only need to upload:

    AI_LEAD_HANDOFF.zip

If the browser UI requires text, something trivial such as:

    Continue.

should be sufficient. No long recovery prompt is required.