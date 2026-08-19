"""Build the self-contained AI_LEAD_HANDOFF.zip for a browser-only AI lead.

The ZIP contains the authoritative project memory (00_START_HERE.md, AGENTS.md,
opencode.json, docs/ai/**), all safe source/config/test files, the latest task
report, small run metadata, README/requirements, and generated
HANDOFF_MANIFEST.json + GIT_STATE.json.

Never includes: .git, .venv, *.pt, *.bin, raw/cleaned corpus, caches,
credentials/secrets, previous handoff zips, or files over the size cap.

Usage:
    .\\.venv\\Scripts\\python.exe tools\\ai_ops\\build_handoff.py
    .\\.venv\\Scripts\\python.exe tools\\ai_ops\\build_handoff.py --out AI_LEAD_HANDOFF.zip --max-file-bytes 5242880
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INCLUDE_ROOTS = [
    "00_START_HERE.md",
    "AGENTS.md",
    "opencode.json",
    "README.md",
    "requirements.txt",
    "docs/ai",
    "configs",
    "src",
    "scripts",
    "tests",
    "tools/ai_ops",
    "checkpoints/pretrain_v1/metrics.jsonl",
    "checkpoints/pretrain_v1/run_config.json",
    "checkpoints/pretrain_v1/generation_samples.txt",
    "data/tokenizer/tokenizer_v1.model",
    "data/tokenizer/tokenizer_v1.vocab",
    "data/tokenizer/tokenizer_v1_meta.json",
]

ARTIFACT_OMIT_LIST = [
    "checkpoints/pretrain_v1/best.pt",
    "checkpoints/pretrain_v1/latest.pt",
]

DENY_SUBSTRINGS = [
    ".git", ".venv", "__pycache__", ".pytest_cache",
    "data/sources", "data/raw", "data/cleaned", "data/processed",
]

DENY_SUFFIXES = (".pyc", ".pyo", ".pt", ".bin", ".tmp", ".zip", ".key", ".pem", ".env")

DENY_NAMES = ("credentials", "secrets", ".env", ".env.", "id_rsa", "id_ed25519")


def _safe(path: str) -> bool:
    """Return True if a relative posix path is safe to include."""
    p = path.replace("\\", "/").lower()
    for sub in DENY_SUBSTRINGS:
        if sub in p:
            return False
    if p.endswith(DENY_SUFFIXES):
        return False
    name = os.path.basename(p)
    if name.startswith(".env") or name in DENY_NAMES or any(name.startswith(n) for n in DENY_NAMES):
        return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(project_root: Path) -> Dict:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True, text=True, timeout=60,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    branch = run("branch", "--show-current")
    sha = run("rev-parse", "HEAD")
    subject = run("log", "-1", "--format=%s")
    remotes_raw = run("remote", "-v")
    remotes: Dict[str, List[str]] = {}
    for line in remotes_raw.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            remotes.setdefault(parts[0], []).append(parts[1])
    dirty = run("status", "--porcelain")
    return {
        "project_root": str(project_root),
        "branch": branch or "unknown",
        "head_sha": sha or "none",
        "last_commit_subject": subject,
        "remotes": remotes,
        "working_tree_dirty": bool(dirty),
        "dirty_file_count": len([l for l in dirty.splitlines() if l.strip()]) if dirty else 0,
    }


def collect_files(project_root: Path, max_file_bytes: int) -> Tuple[List[Path], List[Dict]]:
    included: List[Path] = []
    oversized: List[Dict] = []
    seen = set()
    for rel in INCLUDE_ROOTS:
        p = project_root / rel
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                r = f.relative_to(project_root).as_posix()
                if r in seen or not _safe(r):
                    continue
                seen.add(r)
                if f.stat().st_size > max_file_bytes:
                    oversized.append({"path": r, "size": f.stat().st_size})
                    continue
                included.append(f)
        else:
            r = p.relative_to(project_root).as_posix()
            if r in seen or not _safe(r):
                continue
            seen.add(r)
            if p.stat().st_size > max_file_bytes:
                oversized.append({"path": r, "size": p.stat().st_size})
                continue
            included.append(p)
    return included, oversized


def build_handoff(project_root: Path = PROJECT_ROOT, out_zip: Optional[Path] = None,
                  max_file_bytes: int = 5 * 1024 * 1024,
                  include_pt_hashes: bool = True) -> Dict:
    project_root = Path(project_root)
    if out_zip is None:
        out_zip = project_root / "AI_LEAD_HANDOFF.zip"
    out_zip = Path(out_zip)

    state = git_state(project_root)

    included, oversized = collect_files(project_root, max_file_bytes)

    manifest_files = []
    for f in sorted(included):
        rel = f.relative_to(project_root).as_posix()
        manifest_files.append({
            "path": rel,
            "size": f.stat().st_size,
            "sha256": sha256_file(f),
        })

    omitted_artifacts = []
    if include_pt_hashes:
        for rel in ARTIFACT_OMIT_LIST:
            p = project_root / rel
            if p.exists():
                try:
                    omitted_artifacts.append({
                        "path": rel,
                        "size": p.stat().st_size,
                        "sha256": sha256_file(p),
                        "note": "large artifact NOT included in zip (compute sha256 only)",
                    })
                except OSError as e:
                    omitted_artifacts.append({"path": rel, "error": str(e)})

    proj_state: Dict = {}
    ps_path = project_root / "docs" / "ai" / "PROJECT_STATE.json"
    if ps_path.exists():
        try:
            proj_state = json.loads(ps_path.read_text(encoding="utf-8"))
        except Exception:
            proj_state = {}

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/ai_ops/build_handoff.py",
        "git": state,
        "current_task": proj_state.get("current_task", "unknown"),
        "last_completed_task": proj_state.get("last_completed_task", "unknown"),
        "next_planned_task": proj_state.get("next_planned_task", "unknown"),
        "max_file_bytes": max_file_bytes,
        "files": manifest_files,
        "file_count": len(manifest_files),
        "total_included_bytes": sum(m["size"] for m in manifest_files),
        "omitted_oversized": oversized,
        "omitted_important_artifacts": omitted_artifacts,
    }

    git_state_json = {
        "generated_at": manifest["generated_at"],
        "git": state,
    }

    tmp_zip = out_zip.with_suffix(".zip.tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        z.writestr("GIT_STATE.json", json.dumps(git_state_json, indent=2) + "\n")
        for f in included:
            rel = f.relative_to(project_root).as_posix()
            z.write(f, rel)

    if out_zip.exists():
        out_zip.unlink()
    tmp_zip.replace(out_zip)

    with zipfile.ZipFile(out_zip) as z:
        bad = z.testzip()
    if bad is not None:
        raise RuntimeError(f"handoff zip corrupt member: {bad}")

    return {
        "zip_path": str(out_zip),
        "zip_size": out_zip.stat().st_size,
        "branch": state["branch"],
        "commit_sha": state["head_sha"],
        "file_count": len(manifest_files),
        "omitted_oversized": oversized,
        "omitted_artifacts": omitted_artifacts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AI_LEAD_HANDOFF.zip")
    ap.add_argument("--out", default=None, help="output zip path (default: project root AI_LEAD_HANDOFF.zip)")
    ap.add_argument("--max-file-bytes", type=int, default=5 * 1024 * 1024, help="max included file size (default 5242880)")
    ap.add_argument("--no-pt-hashes", action="store_true", help="skip computing sha256 of omitted .pt checkpoints")
    args = ap.parse_args()

    result = build_handoff(
        PROJECT_ROOT,
        out_zip=Path(args.out) if args.out else None,
        max_file_bytes=args.max_file_bytes,
        include_pt_hashes=not args.no_pt_hashes,
    )
    print(json.dumps(result, indent=2))
    print("AI_LEAD_HANDOFF READY:", result["zip_path"], f"({result['zip_size']} bytes)")


if __name__ == "__main__":
    sys.exit(main())