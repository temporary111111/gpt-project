# SFT DATA REPORT — TASK 005.1 (expanded chat/instruction tuning V1 corpus)

Report generated: 2026-08-20 (UTC). Machine-readable stats:
`data/sft/stats/sft_stats.json`. Provenance: `data/sft/manifests/sources.jsonl`.
Quality audit samples: `data/sft/stats/quality_audit_samples.txt`.

## 1. Dataset build pipeline

`scripts/acquire_sft_data.py` (download/filter/raw dump) →
`scripts/build_sft_dataset.py` (normalize / English-check / dedup / split /
chat-format / assistant-only labels / sampling weights) →
`scripts/sft_stats.py` (aggregate stats).

Build command used:

```
.\.venv\Scripts\python.exe scripts\build_sft_dataset.py
```

`--max-target-unk-rate` default 0.0: EVERY example whose assistant target
contains an unknown token is rejected (target unk rate in the final corpus is
0.0).

## 2. Sources (all human-only; see manifests/SOURCES.md for provenance)

| Source | License | Acquired | Final examples | Unique supervised target tokens |
|---|---|---|---|---|
| Aya (eng+fil originals) | Apache-2.0 | 2,934 | 2,568 | 180,896 |
| OASST1 (en human) | Apache-2.0 | 39,751 msgs | 4,266 | 439,354 |
| Dolly 15K | cc-by-sa-3.0 | 15,010 | 10,926 | 715,354 |
| Taskmaster-1 | CC BY 4.0 | 13,170 convs | 33,016 | 528,249 |
| **Total** | | | **50,776** | **1,863,853** |

## 3. Filtering counters (final build)

| Counter | Value |
|---|---|
| aya_raw / oasst_raw / dolly_raw / taskmaster_raw | 2,934 / 39,751 / 15,010 / 13,170 |
| aya_examples / oasst_examples / dolly_examples / taskmaster_examples | 2,934 / 9,409 / 15,010 / 34,318 |
| oasst_skipped_root_not_prompter | 13 |
| tm_skipped_capped_candidate_turns / tm_skipped_root_not_user | 104,036 / 18,292 |
| rejected_aya_eng_mislabel (mislabeled non-English Aya eng rows) | 104 |
| rejected_pathological_repeats / rejected_corrupted_unicode | 217 / 3 |
| after_quality | 61,347 |
| rejected_duplicates (exact normalized (prompt,target), by source pair) | 1,160 |
| dup_pair detail | aya 43, dolly 15, taskmaster1 1,083, aya\|dolly 2, dolly\|oasst1 1, oasst1\|taskmaster1 16 |
| prompt_only_duplicates (reported, NOT merged) | 6,686 |
| rejected_eval_probes (any user turn matches a held-out eval prompt) | 24 |
| rejected_target_too_long (final user turn + target cannot fit 256 ctx) | 7,954 |
| rejected_target_unk (target contains <unk>) | 1,433 |
| dropped_oldest_turns (older complete turns evicted to keep final turn) | 10,756 |
| **final_examples** | **50,776** |

## 4. Final corpus statistics (UNIQUE, before sampling)

| Metric | Train | Validation | Total |
|---|---|---|---|
| Examples | 43,660 | 7,116 | 50,776 |
| Supervised target tokens | 1,697,262 | 166,591 | **1,863,853** |
| Filipino target tokens | 48,184 | 2,942 | 51,126 |
| English target tokens | 1,649,078 | 163,649 | 1,812,727 |

### Unique language mix by SUPERVISED TARGET TOKENS

| Language | Tokens | Share |
|---|---|---|
| en (OASST1 + Dolly + Taskmaster) | 1,682,957 | 90.3% |
| eng (Aya) | 129,770 | 7.0% |
| fil (Aya) | 51,126 | 2.7% |

### Unique source mix by SUPERVISED TARGET TOKENS (of English 1,812,727)

| Source | Tokens | Share of English |
|---|---|---|
| dolly | 715,354 | 39.5% |
| taskmaster1 | 528,249 | 29.1% |
| oasst1 | 439,354 | 24.2% |
| aya (eng) | 129,770 | 7.2% |

No English source exceeds 50% of effective English tokens, so no source
down-weighting was required (`english_source_weights` all 1.0).

## 5. Sampling weights (EFFECTIVE training mix; SAMPLING ONLY)

| Metric | Value |
|---|---|
| fil_weight | 4.0 (cap reached; no fabrication/translation) |
| effective_train_examples | 45,724 |
| unique_train_examples | 43,660 |
| effective_train_supervised_tokens | 1,841,814 |
| effective_fil_supervised_tokens | 192,736 |
| **effective_fil_share** | **0.1046 (10.5%; max achievable at the 4x cap)** |

Weights are stored per example as `weight` (float) + `copies` (integer) in
`data/sft/processed/sft_train.jsonl`; the training epoch order repeats each
example `copies` times, then shuffles deterministically (seed 1337). The
effective numbers NEVER count toward the token gate.

## 6. Tokenizer quality (tokenizer_v1, vocab 8000) — FINAL corpus

| Metric | Train | Validation |
|---|---|---|
| prompt unk rate | 0.000230 | 0.000113 |
| target unk rate | 0.0 | 0.0 |

All targets containing <unk> were rejected (1,433); remaining prompt <unk>s
are rare OOV symbols in user prompts.

## 7. Format, masking, truncation, splits

- Chat format per target: `<bos><user>U<assistant>A<eos>` (single or multi-turn).
- Multi-turn: earlier complete turns are context; ONLY the final assistant
  target + EOS receive supervised labels; labels = -100 elsewhere.
- Context length 256; final user turn + target always kept; older turns dropped
  oldest-first.
- Splits: Aya + Dolly by stable sha256 bucket 95/5; OASST source-provided
  (no message_tree_id crosses — verified 0 overlap); Taskmaster by official
  conversation-ID CSVs / deterministic woz bucket (verified 0 conversation
  crosses). Zero train/val leakage: `leakage_check` all 0.

## 8. TOKEN GATE (TASK 005.1 Part D/M)

**UNIQUE_SUPERVISED_TARGET_TOKENS = 1,863,853 >= 1,000,000 floor => GATE
PASSED.** (Preferred band 1.2M–3M; hard ceiling 8M — satisfied.) Training
proceeds automatically per Part M.

## 9. Quality audit (TASK 005.1 Part I)

150 deterministic samples (30 each: Aya fil, Aya eng, OASST, Dolly,
Taskmaster) were reviewed manually. All passed the deterministic quality
rules; no example was rewritten. Noted human-data quirks (kept as-is per Part
I): a few OASST playful-dialect answers, a few Dolly factual oddities
(e.g. "Bubal hartebeest is domesticated"). Full dump:
`data/sft/stats/quality_audit_samples.txt`.