# SFT DATA REPORT — TASK 005 (chat/instruction tuning V1)

Report generated: 2026-08-20 (UTC). Machine-readable stats:
`data/sft/stats/sft_stats.json`. Provenance: `data/sft/manifests/sources.jsonl`.

## 1. Dataset build pipeline

`scripts/acquire_sft_data.py` (download/filter/raw dump) →
`scripts/build_sft_dataset.py` (normalize / dedup / split / chat-format /
assistant-only labels) → `scripts/sft_stats.py` (aggregate stats).

Build command used:

```
.\.venv\Scripts\python.exe scripts\build_sft_dataset.py --max-target-unk-rate 0.0
```

`--max-target-unk-rate 0.0` rejects EVERY example whose assistant target
contains an unknown token (the tokenizer is expected to cover the corpus well).

## 2. Filtering counters (final build)

| Counter | Value |
|---|---|
| aya_raw (accepted at acquisition) | 2,934 |
| oasst_raw (accepted at acquisition) | 39,751 |
| aya_examples (after build) | 2,934 |
| oasst_examples (after message-tree assembly) | 9,409 |
| oasst_skipped_root_not_prompter | 13 |
| after_quality (empty/corrupted-unicode/pathological rejects) | 12,288 |
| rejected_pathological_repeats | 55 |
| rejected_duplicates (exact (prompt,target) across sources) | 43 |
| after_dedup | 12,245 |
| rejected_eval_probes (prompts reserved for evaluation) | 3 |
| after_probe_exclusion | 12,242 |
| rejected_target_too_long (cannot fit final user turn + target in 256 ctx) | 4,078 |
| rejected_target_unk (target contains <unk>, rate floor 0.0) | 1,253 |
| dropped_oldest_turns (older turns evicted to keep final turn) | 1,748 |
| **final_examples** | **6,911** |

## 3. Final corpus statistics

| Metric | Train | Validation | Total |
|---|---|---|---|
| Examples | 6,573 | 338 | 6,911 |
| Supervised target tokens | 594,179 | 29,878 | **624,057** |

### Language mix by SUPERVISED TARGET TOKENS

| Language | Train | Validation | Total | Share |
|---|---|---|---|---|
| en (OASST1) | 417,121 | 22,492 | 439,613 | 70.4% |
| eng (Aya) | 128,874 | 4,444 | 133,318 | 21.4% |
| fil (Aya) | 48,184 | 2,942 | 51,126 | 8.2% |

The English-dominant ratio is the natural ratio of the human datasets.
Per the task rules, no fabrication/oversampling beyond 2x was used to force a
50/50 split; quality and provenance were prioritized over ratio.

### Tokenizer quality (tokenizer_v1, vocab 8000)

| Metric | Train | Validation |
|---|---|---|
| prompt unk rate | 0.000922 | 0.000922 |
| target unk rate | 0.0 | 0.0 |

After the 0.0 unk floor, no target contains <unk>. Remaining prompt <unk>s are
rare OOV symbols in user prompts; the assistant targets are fully covered.

## 4. Format, masking, truncation

- Chat format per target: `<bos><user>U<assistant>A<eos>` (single or multi-turn).
- Multi-turn: earlier complete turns are context; ONLY the final assistant
  target + EOS receive supervised labels.
- labels = -100 for BOS, user tokens, context turns, role markers, padding.
- Context length 256; the final user turn + final assistant target are always
  kept; older turns are dropped from the oldest first when the window fills.
- Deterministic split: Aya split by stable sha256 bucket (95/5); OASST uses the
  source-provided train/validation splits with `validation` normalized to
  `val` (no message_tree_id crosses). Zero train/val leakage verified by test.

## 5. STOP gate (TASK 005 Part E)

**624,057 usable supervised target tokens < 1,000,000 floor.**

Per Part E: "If less than 1M usable supervised target tokens exist after
filtering: STOP BEFORE FULL TRAINING and report to the lead architect."

=> FULL SFT TRAINING WAS NOT STARTED. This report accompanies the
TASK 005 STOP-gate report to the lead architect.