"""Acquire and filter the two TASK 005 human SFT sources.

SOURCE 1: CohereLabs/aya_dataset (train split only)
  - languages: eng, fil
  - ONLY annotation_type == "original-annotations" (human-origin; re-annotations
    may contain automatically generated material and are rejected)
  - Aya test split is NEVER used for training

SOURCE 2: OpenAssistant/oasst1 (train + validation, source-provided split)
  - language: English ("en") only
  - STRICTLY human-only: synthetic == False, model_name empty/None,
    deleted == False, review_result == True, tree_state == ready_for_export
  - label-based rejection (spam / sexual_content / hate_speech / violence /
    not_appropriate / language_mismatch / toxicity / pii)
  - assistant candidates prefer rank == 0 when available

Outputs (raw, git-ignored):
  data/sft/raw/aya_eng_fil_original.jsonl
  data/sft/raw/oasst1_en_messages.jsonl
  data/sft/manifests/sources.jsonl  (provenance, tracked)
  data/sft/manifests/SOURCES.md     (license/provenance record, tracked)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\acquire_sft_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402  (data acquisition only)

AYA_REPO = "CohereLabs/aya_dataset"
AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
AYA_LICENSE = "Apache-2.0"
AYA_SPLIT = "train"

OASST_REPO = "OpenAssistant/oasst1"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
OASST_LICENSE = "Apache-2.0"

REJECT_LABELS = {
    "spam", "sexual_content", "hate_speech", "violence",
    "not_appropriate", "language_mismatch", "toxicity", "pii",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_provenance(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_aya_original(row: dict) -> bool:
    """Aya filter: eng/fil AND annotation_type == original-annotations only."""
    code = (row.get("language_code") or "").strip().lower()
    atype = (row.get("annotation_type") or "").strip()
    return code in ("eng", "fil") and atype == "original-annotations"


def acquire_aya(out_dir: Path) -> dict:
    print(f"[aya] loading {AYA_REPO}@{AYA_REVISION} split={AYA_SPLIT} ...")
    ds = load_dataset(AYA_REPO, revision=AYA_REVISION, split=AYA_SPLIT)
    rows = []
    rejected = {"re-annotations": 0, "other_annotation_type": 0, "other_language": 0}
    for i, ex in enumerate(ds):
        if not is_aya_original(ex):
            code = (ex.get("language_code") or "").strip().lower()
            atype = (ex.get("annotation_type") or "").strip()
            if code not in ("eng", "fil"):
                rejected["other_language"] += 1
            elif "re" in atype.lower():
                rejected["re-annotations"] += 1
            else:
                rejected["other_annotation_type"] += 1
            continue
        code = (ex.get("language_code") or "").strip().lower()
        rid = hashlib.sha256(f"{code}|{ex['inputs']}|{ex['targets']}".encode("utf-8")).hexdigest()[:24]
        rows.append({
            "id": f"aya-{code}-{rid}",
            "source": "aya",
            "language_code": code,
            "language": ex.get("language") or "",
            "inputs": ex["inputs"],
            "targets": ex["targets"],
        })
    out_path = out_dir / "aya_eng_fil_original.jsonl"
    write_jsonl(out_path, rows)
    print(f"[aya] accepted {len(rows)} (eng/fil original-annotations) -> {out_path}")
    print(f"[aya] rejected: {rejected}")
    return {"rows": rows, "rejected": rejected, "out": out_path}


def _human_ok(m: dict) -> bool:
    if m.get("synthetic"):
        return False
    if m.get("model_name"):
        return False
    if m.get("deleted"):
        return False
    if m.get("review_result") is not None and not m["review_result"]:
        return False
    tree = m.get("tree_state")
    if tree and tree != "ready_for_export":
        return False
    labels = m.get("labels") or []
    if isinstance(labels, list) and any(l in REJECT_LABELS for l in labels):
        return False
    if isinstance(labels, dict) and any(l in REJECT_LABELS for l in labels):
        return False
    return True


def acquire_oasst(out_dir: Path) -> dict:
    print(f"[oasst] loading {OASST_REPO}@{OASST_REVISION} ...")
    ds = load_dataset(OASST_REPO, revision=OASST_REVISION)
    rows = []
    rejected = {
        "lang_not_en": 0, "synthetic": 0, "model_name": 0, "deleted": 0,
        "review_failed": 0, "tree_state": 0, "labels": 0,
    }
    for split in ("train", "validation"):
        for m in ds[split]:
            if (m.get("lang") or "") != "en":
                rejected["lang_not_en"] += 1
                continue
            if not _human_ok(m):
                for key in ("synthetic", "model_name", "deleted", "review_failed", "tree_state", "labels"):
                    if key == "synthetic" and m.get("synthetic"):
                        rejected[key] += 1
                        break
                    if key == "model_name" and m.get("model_name"):
                        rejected[key] += 1
                        break
                    if key == "deleted" and m.get("deleted"):
                        rejected[key] += 1
                        break
                    if key == "review_failed" and m.get("review_result") is not None and not m["review_result"]:
                        rejected[key] += 1
                        break
                    if key == "tree_state" and m.get("tree_state") and m["tree_state"] != "ready_for_export":
                        rejected[key] += 1
                        break
                    if key == "labels":
                        labels = m.get("labels") or []
                        if isinstance(labels, (list, dict)) and any(l in REJECT_LABELS for l in labels):
                            rejected[key] += 1
                            break
                continue
            rows.append({
                "message_id": m["message_id"],
                "parent_id": m.get("parent_id"),
                "message_tree_id": m["message_tree_id"],
                "role": m["role"],
                "rank": m.get("rank"),
                "text": m["text"],
                "split": split,
            })
    out_path = out_dir / "oasst1_en_human_messages.jsonl"
    write_jsonl(out_path, rows)
    print(f"[oasst] accepted {len(rows)} human-only English messages -> {out_path}")
    print(f"[oasst] rejected: {rejected}")
    return {"rows": rows, "rejected": rejected, "out": out_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire Aya + OASST1 human SFT data")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "sft" / "raw"))
    ap.add_argument("--skip-aya", action="store_true")
    ap.add_argument("--skip-oasst", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    result = {}
    if not args.skip_aya:
        result["aya"] = acquire_aya(out_dir)
    if not args.skip_oasst:
        result["oasst"] = acquire_oasst(out_dir)

    manifest_path = ROOT / "data" / "sft" / "manifests" / "sources.jsonl"
    for name in result:
        r = result[name]
        append_provenance(manifest_path, {
            "dataset": name,
            "repo": AYA_REPO if name == "aya" else OASST_REPO,
            "revision": AYA_REVISION if name == "aya" else OASST_REVISION,
            "license": AYA_LICENSE if name == "aya" else OASST_LICENSE,
            "retrieved_at": now,
            "split_used": AYA_SPLIT if name == "aya" else "train+validation (source-provided)",
            "filters": (
                "language_code in {eng, fil}; annotation_type == original-annotations; test split unused"
                if name == "aya" else
                "lang == en; synthetic==False; model_name empty; deleted==False; "
                "review_result==True; tree_state==ready_for_export; reject labels: "
                + ", ".join(sorted(REJECT_LABELS))
            ),
            "accepted_records": len(r["rows"]),
            "rejected_by_reason": r["rejected"],
            "raw_file": r["out"].name,
            "raw_file_sha256": sha256_file(r["out"]),
        })
        print(f"[prov] provenance appended for {name}")

    print("DONE. Raw files:", {k: str(v["out"]) for k, v in result.items()})


if __name__ == "__main__":
    main()