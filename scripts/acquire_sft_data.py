"""Acquire and filter the TASK 005.1 human SFT sources.

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

SOURCE 3: databricks/databricks-dolly-15k (TASK 005.1)
  - official human-generated instruction/response data (CC BY-SA 3.0)
  - fields: instruction, context, response, category
  - heavy filtering (unk, context-length, dedup) happens in build_sft_dataset.py

SOURCE 4: Taskmaster-1 / TM-1-2019 (TASK 005.1)
  - OFFICIAL Google Research Taskmaster repository, pinned commit
    (CC BY 4.0); TM-1-2019: 13,215 task-based dialogs, 6 domains
  - self-dialogs.json + woz-dialogs.json downloaded raw
  - official train/dev/test dialog-ID CSVs downloaded for conversation-level
    splits (woz IDs not in the CSVs get a deterministic conversation-level bucket)

Outputs (raw, git-ignored):
  data/sft/raw/aya_eng_fil_original.jsonl
  data/sft/raw/oasst1_en_human_messages.jsonl
  data/sft/raw/dolly_15k.jsonl
  data/sft/raw/taskmaster1_dialogs.jsonl
  data/sft/manifests/sources.jsonl  (provenance, tracked)
  data/sft/manifests/SOURCES.md     (license/provenance record, tracked)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\acquire_sft_data.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
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

DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_LICENSE = "cc-by-sa-3.0"

TM_REPO = "google-research-datasets/Taskmaster"
TM_REPO_URL = "https://github.com/google-research-datasets/Taskmaster"
TM_REVISION = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
TM_LICENSE = "cc-by-4.0"
TM_BASE = f"https://raw.githubusercontent.com/google-research-datasets/Taskmaster/{TM_REVISION}/TM-1-2019"
TM_FILES = ["self-dialogs.json", "woz-dialogs.json"]
TM_SPLIT_CSVS = ["train-dev-test/train.csv", "train-dev-test/dev.csv", "train-dev-test/test.csv"]

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


def acquire_dolly(out_dir: Path) -> dict:
    """Raw dump of databricks/databricks-dolly-15k with light validity checks."""
    print(f"[dolly] loading {DOLLY_REPO}@{DOLLY_REVISION} ...")
    ds = load_dataset(DOLLY_REPO, revision=DOLLY_REVISION, split="train")
    rows = []
    rejected = {"empty_instruction": 0, "empty_response": 0, "empty_both": 0,
                "corrupt_unicode": 0, "non_english_expected_mismatch": 0}
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        resp = (ex.get("response") or "").strip()
        if not instr and not resp:
            rejected["empty_both"] += 1
            continue
        if not instr:
            rejected["empty_instruction"] += 1
            continue
        if not resp:
            rejected["empty_response"] += 1
            continue
        if "\ufffd" in instr or "\ufffd" in resp:
            rejected["corrupt_unicode"] += 1
            continue
        rid = hashlib.sha256(f"{instr}|{resp}".encode("utf-8")).hexdigest()[:24]
        rows.append({
            "id": f"dolly-{rid}",
            "source": "dolly",
            "language": "English",
            "lang": "en",
            "category": ex.get("category") or "",
            "instruction": instr,
            "context": (ex.get("context") or "").strip(),
            "response": resp,
        })
    out_path = out_dir / "dolly_15k.jsonl"
    write_jsonl(out_path, rows)
    print(f"[dolly] accepted {len(rows)} -> {out_path}")
    print(f"[dolly] rejected: {rejected}")
    return {"rows": rows, "rejected": rejected, "out": out_path}


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "opencode"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def acquire_taskmaster(out_dir: Path) -> dict:
    """Download official TM-1-2019 JSON dialogs + dialog-ID split CSVs (pinned commit)."""
    print(f"[taskmaster] downloading {TM_REPO_URL} @ {TM_REVISION}")
    downloaded = {}
    for name in TM_FILES + TM_SPLIT_CSVS:
        url = f"{TM_BASE}/{name}"
        data = _http_get(url)
        downloaded[name] = data
        print(f"[taskmaster] downloaded {name} ({len(data):,} bytes)")

    dialogs = []
    for name in TM_FILES:
        dialogs.extend(json.loads(downloaded[name].decode("utf-8")))

    # official dialog-ID split lists (train/dev/test)
    split_lists = {}
    for csv_name in TM_SPLIT_CSVS:
        text = downloaded[csv_name].decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        ids = set()
        for row in reader:
            if row:
                ids.add(row[0].strip())
        split_lists[csv_name.replace(".csv", "").split("/")[-1]] = ids
        print(f"[taskmaster] {csv_name}: {len(ids)} dialog ids")

    # conversation-level split: official CSV when the id is listed, else
    # deterministic conversation-level bucket (95/5). A conversation NEVER
    # appears in both train and validation.
    def conv_split(cid: str) -> str:
        if cid in split_lists["dev"] or cid in split_lists["test"]:
            return "val"
        if cid in split_lists["train"]:
            return "train"
        return "train" if stable_bucket(cid, 100) < 95 else "val"

    rejected = {"no_utterances": 0, "no_user_or_assistant": 0, "bad_structure": 0}
    rows = []
    seen = set()
    for d in dialogs:
        try:
            cid = d["conversation_id"]
            utts = d.get("utterances") or []
        except (KeyError, TypeError):
            rejected["bad_structure"] += 1
            continue
        if cid in seen:
            continue
        seen.add(cid)
        if not utts:
            rejected["no_utterances"] += 1
            continue
        mapped = []
        for u in utts:
            speaker = (u.get("speaker") or "").strip().lower()
            text = (u.get("text") or "").strip()
            if speaker in ("user", "assistant"):
                mapped.append({"index": u.get("index"), "speaker": speaker, "text": text})
            # unknown speakers are skipped (no role fabrication)
        if not any(u["speaker"] == "user" for u in mapped) or \
           not any(u["speaker"] == "assistant" for u in mapped):
            rejected["no_user_or_assistant"] += 1
            continue
        rows.append({
            "id": cid,
            "source": "taskmaster1",
            "language": "English",
            "lang": "en",
            "domain": d.get("domain") or "",
            "file": "self" if cid.startswith("self") else "woz",
            "utterances": mapped,
            "split": conv_split(cid),
        })
    out_path = out_dir / "taskmaster1_dialogs.jsonl"
    write_jsonl(out_path, rows)
    print(f"[taskmaster] accepted {len(rows)} conversations ({len(dialogs)} raw) -> {out_path}")
    print(f"[taskmaster] rejected: {rejected}")
    return {"rows": rows, "rejected": rejected, "out": out_path,
            "raw_checksums": {n: hashlib.sha256(b).hexdigest() for n, b in downloaded.items()},
            "split_list_sizes": {k: len(v) for k, v in split_lists.items()}}


def stable_bucket(key: str, divisor: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % divisor


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire Aya + OASST1 + Dolly + Taskmaster-1 human SFT data")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "sft" / "raw"))
    ap.add_argument("--skip-aya", action="store_true")
    ap.add_argument("--skip-oasst", action="store_true")
    ap.add_argument("--skip-dolly", action="store_true")
    ap.add_argument("--skip-taskmaster", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    result = {}
    if not args.skip_aya:
        result["aya"] = acquire_aya(out_dir)
    if not args.skip_oasst:
        result["oasst"] = acquire_oasst(out_dir)
    if not args.skip_dolly:
        result["dolly"] = acquire_dolly(out_dir)
    if not args.skip_taskmaster:
        result["taskmaster"] = acquire_taskmaster(out_dir)

    manifest_path = ROOT / "data" / "sft" / "manifests" / "sources.jsonl"
    for name in result:
        r = result[name]
        if name == "aya":
            repo, rev, lic, split_used = AYA_REPO, AYA_REVISION, AYA_LICENSE, AYA_SPLIT
            filters = ("language_code in {eng, fil}; annotation_type == original-annotations; test split unused")
        elif name == "oasst":
            repo, rev, lic, split_used = OASST_REPO, OASST_REVISION, OASST_LICENSE, "train+validation (source-provided)"
            filters = ("lang == en; synthetic==False; model_name empty; deleted==False; "
                       "review_result==True; tree_state==ready_for_export; reject labels: "
                       + ", ".join(sorted(REJECT_LABELS)))
        elif name == "dolly":
            repo, rev, lic, split_used = DOLLY_REPO, DOLLY_REVISION, DOLLY_LICENSE, "train"
            filters = ("human-generated instruction/response; light validity only at acquisition "
                       "(empty/corrupt); heavy filtering in build_sft_dataset.py")
        else:  # taskmaster
            repo, rev, lic, split_used = TM_REPO_URL, TM_REVISION, TM_LICENSE, "self-dialogs + woz-dialogs (official train/dev/test dialog-ID CSVs; woz fallback deterministic conv bucket)"
            filters = ("official TM-1-2019; user/assistant speakers only; unknown speakers skipped; "
                       "conversation-level split isolation")
        record = {
            "dataset": name,
            "repo": repo,
            "revision": rev,
            "license": lic,
            "retrieved_at": now,
            "split_used": split_used,
            "filters": filters,
            "accepted_records": len(r["rows"]),
            "rejected_by_reason": r["rejected"],
            "raw_file": r["out"].name,
            "raw_file_sha256": sha256_file(r["out"]),
        }
        if name == "taskmaster":
            record["raw_checksums"] = r["raw_checksums"]
            record["split_list_sizes"] = r["split_list_sizes"]
            record["source_files"] = TM_FILES
        append_provenance(manifest_path, record)
        print(f"[prov] provenance appended for {name}")

    print("DONE. Raw files:", {k: str(v["out"]) for k, v in result.items()})


if __name__ == "__main__":
    main()