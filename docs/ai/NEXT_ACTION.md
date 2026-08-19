# NEXT ACTION

1. **Lead architect (browser AI)**: review `AI_LEAD_HANDOFF.zip` (00_START_HERE.md first) and reconstruct state from repository/docs.
2. **Then**: issue **TASK 005 — chat/instruction tuning** (SFT on the accepted base model, checkpoints/pretrain_v1/best.pt).
3. **Implementation engineer (OpenCode)**: execute TASK 005 per OPERATING_PROTOCOL, then finalize (tests → docs/ai updates → finalize_task.py → commit → push → handoff zip).

## Constraints for TASK 005
- Base model: checkpoints/pretrain_v1/best.pt (step 36,500, val 3.073326).
- Tokenizer: data/tokenizer/tokenizer_v1.model; chat roles <system>=4 <user>=5 <assistant>=6 exist in vocab.
- Strict from-scratch still applies (no external LLM API as brain; no distillation).
- test.bin remains sealed.

Nothing else is pending. Do not start TASK 005 until the lead architect issues it.