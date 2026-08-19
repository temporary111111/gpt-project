"""Manifest and provenance tracking for all acquired sources."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import date
from typing import List, Optional


@dataclass
class SourceRecord:
    source_name: str
    language: str
    source_type: str
    source_url: str
    license: str
    retrieval_date: str
    local_path: str
    raw_size_bytes: int
    status: str
    notes: str = ""
    title: str = ""


def _date_str() -> str:
    return date.today().isoformat()


def append_source_record(path: str, record: SourceRecord) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_source_records(path: str) -> List[SourceRecord]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(SourceRecord(**json.loads(line)))
    return records


def write_sources_md(records: List[SourceRecord], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "# Corpus Sources and Licensing",
        "",
        "Every source used to build the training corpus is recorded here and in",
        "`sources.jsonl`. Text is used only from public-domain or permissively",
        "licensed human-written sources.",
        "",
        "| Source | Language | Type | License | Retrieved | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        title = r.title or r.source_name
        lines.append(
            f"| {title} | {r.language} | {r.source_type} | {r.license} | "
            f"{r.retrieval_date} | {r.status} |"
        )
    lines.append("")
    lines.append("## URLs")
    lines.append("")
    for r in records:
        lines.append(f"- {r.title or r.source_name}: {r.source_url}")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))