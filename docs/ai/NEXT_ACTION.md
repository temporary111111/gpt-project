# NEXT ACTION

1. **Lead architect (browser AI)**: review `AI_LEAD_HANDOFF.zip` (00_START_HERE.md first), `docs/ai/reports/TASK_005_REPORT.txt` (STOP-gate report), `data/sft/stats/SFT_DATA_REPORT.md`, and `data/sft/manifests/SOURCES.md`; reconstruct state from the repository/docs.
2. **Decide the TASK 005 corpus path** (Part E gate: 624,057 < 1M supervised tokens):
   - (a) Expand the human-only SFT corpus (e.g., additional legal human-only
     EN/TL instruction/chat datasets) and retarget;
   - (b) Explicitly authorize reduced-corpus SFT (624K tokens, Filipino 8.2%);
   - (c) Other direction.
3. **Then** issue the follow-up order; the implementation engineer will execute it (training runs, if authorized, will use the delivered pipeline as-is: `scripts/build_sft_dataset.py` → `src/sft_train.py`).

## Constraints still in force
- Base model: checkpoints/pretrain_v1/best.pt (step 36,500, val 3.073326; SHA-256 ba40ad8c…, verified unchanged).
- Retention baseline recorded: BASELINE_PRETRAIN_VAL_LOSS = 3.093633 (validation.bin, 50 iters, fp16).
- Tokenizer: data/tokenizer/tokenizer_v1.model; chat roles <system>=4 <user>=5 <assistant>=6.
- Strict from-scratch still applies (no external LLM API as brain; no distillation; no synthetic data).
- test.bin remains sealed.
- 108/108 tests green; finalizer will gate any future completion on them.

Do not start training or TASK 006 until the lead architect orders it.
