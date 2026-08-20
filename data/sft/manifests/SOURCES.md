# SFT Data Sources (TASK 005 + TASK 005.1)

Four human-only datasets were acquired for chat/instruction tuning V1.
NO synthetic data, NO machine-translated data, NO external-LLM-generated
training answers were used anywhere in the pipeline (strict from-scratch rule).
Full machine-readable provenance: `data/sft/manifests/sources.jsonl`.

## 1. CohereLabs/aya_dataset (Aya)

| Field | Value |
|---|---|
| Repository | https://huggingface.co/datasets/CohereLabs/aya_dataset |
| Dataset revision | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` |
| License | Apache-2.0 |
| Split used | `train` ONLY (test split NOT used) |
| Language filter | `language_code` in {eng, fil} |
| Annotation filter | `annotation_type == "original-annotations"` ONLY (human originals) |
| Accepted records | 2,934 (eng 2,164 / fil 770) |
| Rejected | re-annotations 2,251; other languages 197,177 |
| Raw file | `data/sft/raw/aya_eng_fil_original.jsonl` (git-ignored) |
| Raw file SHA-256 | `295405cb3d586fc2c73e80712894eb1d66868a3758c2ce8133c7055d37b8a4d6` |
| Retrieved (UTC) | 2026-08-20T02:21:03.439864+00:00 |

NOTE: this is `aya_dataset` (the human-only annotation dataset), NOT the
machine-generated "Aya Collection". Re-annotations are excluded because they
are not the original human-written annotation of each example.

TASK 005.1 quality fix: some `language_code == "eng"` Aya rows are mislabeled
(Somali/Indonesian/Basque/German/Turkish content). A deterministic
English-check heuristic (non-Latin script, or almost no English function
words in a long prompt AND target) rejected 104 such rows at build time
(`rejected_aya_eng_mislabel`); the final corpus contains 2,568 Aya examples
(fil 729 + eng 1,839).

## 2. OpenAssistant/oasst1

| Field | Value |
|---|---|
| Repository | https://huggingface.co/datasets/OpenAssistant/oasst1 |
| Dataset revision | `fdf72ae0827c1cda404aff25b6603abec9e3399b` |
| License | Apache-2.0 |
| Split used | source `train` + `validation` (kept separate, no tree crosses) |
| Language filter | `lang == "en"` |
| Human-only filters | `synthetic == False`; `model_name` empty; `deleted == False`; `review_result == True`; `tree_state == "ready_for_export"` |
| Label reject list | hate_speech, language_mismatch, not_appropriate, pii, sexual_content, spam, toxicity, violence |
| Accepted records | 39,751 (en human-only messages) |
| Rejected | lang_not_en 47,533; deleted 869; review_failed 685 |
| Raw file | `data/sft/raw/oasst1_en_human_messages.jsonl` (git-ignored) |
| Raw file SHA-256 | `be15d49ece17d8a39b477e5f7b78a8ee9a68fde1adb4c5deb0543d14392a4374` |
| Retrieved (UTC) | 2026-08-20T02:21:03.439864+00:00 |

## 3. Databricks/databricks-dolly-15k (Dolly)

| Field | Value |
|---|---|
| Repository | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| Dataset revision | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |
| License | cc-by-sa-3.0 |
| Split used | `train` (single split) |
| Content | human-generated instruction/response pairs (8 categories) |
| Accepted records | 15,010 (15,011 raw; 1 corrupt-Unicode row rejected) |
| Raw file | `data/sft/raw/dolly_15k.jsonl` (git-ignored) |
| Raw file SHA-256 | `a93626750fc52ceb825556571ec738b8766f3e3474043af9f7086484b6172199` |
| Retrieved (UTC) | 2026-08-20T03:19:03.456942+00:00 |

Conversion (TASK 005.1 Part A): single-turn; `USER = instruction (+
deterministic "\n\n" + context separator when non-empty)`; `ASSISTANT =
response`; no linguistic rewriting; split by stable sha256 bucket 95/5.

## 4. Taskmaster-1 (TM-1-2019)

| Field | Value |
|---|---|
| Repository | https://github.com/google-research-datasets/Taskmaster |
| Source revision | `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` (official repo) |
| License | CC BY 4.0 (TM-1-2019/README.md) |
| Files | `TM-1-2019/self-dialogs.json` + `woz-dialogs.json`; official `train-dev-test/{train,dev,test}.csv` dialog-ID splits (train 6,167 / dev 769 / test 769; cover self-dialogs) |
| Domains | 6 (ordering pizza, restaurant reservation, auto repair, movie tickets, ride hailing, coffee ordering) |
| Accepted conversations | 13,170 (13,215 raw; 4 no_utterances, 36 no_user_or_assistant, ~5 duplicate ids) |
| Raw file | `data/sft/raw/taskmaster1_dialogs.jsonl` (git-ignored) |
| Raw file SHA-256 | `2ef0b08d921882649ae2879aa166a340d2ac991871fcda1a07fbf04c74833ba1` |
| Retrieved (UTC) | 2026-08-20T03:19:03.456942+00:00 |

Conversion (TASK 005.1 Part B): every accepted assistant turn is a candidate
SFT target; up to 4 targets per conversation chosen deterministically ACROSS
the conversation (start/middle/end coverage); earlier assistant turns are
context-masked (only the final assistant response of each example is
supervised); split by CONVERSATION ID (official CSVs; woz fallback
deterministic bucket 95/5) — a conversation never crosses train/val
(104,036 candidate turns were capped away; 18,292 root-not-user rejects).

## Constraints honored

- No ShareGPT / Alpaca / Vicuna / Bactrian / translated / synthetic datasets.
- Raw downloaded files are NOT committed to Git (git-ignored).
- Test split of Aya was never loaded for building the SFT corpus.
- No auto-translation and no engineer-authored Filipino was added; the Filipino
  corpus is exactly the Filipino examples that exist in the human datasets.
- Cross-source exact-duplicate (prompt,target) pairs are removed at build time
  and reported by source pair (`rejected_duplicates`: 1,160).