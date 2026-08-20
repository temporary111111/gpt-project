# SFT Data Sources (TASK 005)

Two human-only datasets were acquired for chat/instruction tuning V1.
NO synthetic data, NO machine-translated data, NO external-LLM-generated
training answers were used anywhere in the pipeline (strict from-scratch rule).

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

Full machine-readable provenance: `data/sft/manifests/sources.jsonl`.

## Constraints honored

- No ShareGPT / Alpaca / Vicuna / Bactrian / translated / synthetic datasets.
- Raw downloaded files are NOT committed to Git (git-ignored).
- Test split of Aya was never loaded for building the SFT corpus.
- No auto-translation and no engineer-authored Filipino was added; the Filipino
  corpus is exactly the Filipino examples that exist in the human datasets.
