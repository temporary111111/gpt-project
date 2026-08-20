# NEXT ACTION

1. **Commit + push the tested pipeline phase (Part Y)** — code, tests, corpus
   stats/manifests, docs (this commit). Base checkpoint SHA re-verified
   (ba40ad8c…); full suite 127/127 green.
2. **Part Q — SFT PILOT (~200 optimizer steps)**, run 1:
   `.\.venv\Scripts\python.exe src\sft_train.py --init-from pretrain --max-epochs 1
   --out-dir checkpoints/chat_v1` (plus base-checkpoint/retention args; see
   src/sft_train.py usage). Gate: base-LM validation loss ≤ 3.557678
   (3.093633 × 1.15) eligible; > 3.712360 (× 1.20) HARD STOP. Resume-proof:
   save latest.pt, relaunch from it, verify continuity.
3. **Part R — FULL SFT**: run 2 `--init-from resume --max-epochs 3` (up to 3
   epochs total; early stop after 4 consecutive val evals without improvement;
   atomic best.pt/latest.pt saves; chat_v1/run_config.json + metrics.jsonl).
4. **Parts T/U/V — POST-SFT EVAL**: `scripts/evaluate_chat.py --mode compare`
   (pretrain_v1/best.pt vs chat_v1/best.pt; 20 fixed probes greedy+sampled;
   Bruno + 6 NEW held-out multi-turn probes; EOS / repetition / role leakage;
   fil-in-fil & eng-in-eng) → `chat_comparison.txt` + `post_sft_eval_metrics.json`.
5. **Part W — INTERACTIVE CHAT**: `src/chat.py` with chat_v1/best.pt
   (multi-turn EN + fil; record transcript).
6. **Continuity + report + finalize**: update docs/ai; write
   `docs/ai/reports/TASK_005_1_REPORT.txt` (exact mandated headings); run
   `tools/ai_ops/finalize_task.py`; on ALL Part Z success criteria create +
   push annotated tag `task-005-chat-v1-complete`; print report; STOP for
   architect review.

## Constraints still in force
- Base model: checkpoints/pretrain_v1/best.pt (SHA-256 ba40ad8c…, verified
  unchanged); latest.pt SHA 8c00ff4d… unchanged.
- Retention baseline: BASELINE_PRETRAIN_VAL_LOSS = 3.093633 (validation.bin,
  50 iters, fp16); eligibility ≤ 3.557678; hard stop > 3.712360.
- Tokenizer: data/tokenizer/tokenizer_v1.model; roles <system>=4 <user>=5
  <assistant>=6.
- SFT corpus: 50,776 examples / UNIQUE 1,863,853 supervised tokens (gate
  passed); effective train 45,724 examples / 1,841,814 tokens (fil 10.5%);
  fil_weight 4.0 cap; English source weights all 1.0.
- Strict from-scratch applies (no external LLM as brain; no synthetic data).
- test.bin remains sealed; raw/processed SFT data never committed.