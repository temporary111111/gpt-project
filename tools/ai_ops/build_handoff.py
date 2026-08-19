"""Build the self-contained AI_LEAD_HANDOFF.zip for a browser-only AI lead.

The ZIP is a snapshot of a SPECIFIC COMMITTED state (Git HEAD). Tracked project
contents are read from Git HEAD itself (git show HEAD:<path>), never from
arbitrary working-tree bytes. HANDOFF_MANIFEST.json + GIT_STATE.json are
generated at build time; they are the only working-tree-produced members.

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

# Small root-level operational files that should be included when tracked.
EXTRA_ROOT_FILES = [".gitignore"]

ARTIFACT_OMIT_LIST = [
    "checkpoints/pretrain_v1/best.pt",
    "checkpoints/pretrain_v1/latest.pt",
]

DENY_SUBSTRINGS = [
    ".git/", ".venv", "__pycache__", ".pytest_cache",
    "data/sources", "data/raw", "data/cleaned", "data/processed",
]

DENY_SUFFIXES = (".pyc", ".pyo", ".pt", ".bin", ".tmp", ".zip", ".key", ".pem", ".env")

DENY_NAMES = ("credentials", "secrets", ".env", ".env.", "id_rsa", "id_ed25519")


def _safe(path: str) -> bool:
    """Return True if a relative posix path is safe to include."""
    p = path.replace("\\", "/").lower()
    name = os.path.basename(p)
    if name in (".gitignore", ".gitattributes"):
        return True
    for sub in DENY_SUBSTRINGS:
        if sub in p:
            return False
    if p.endswith(DENY_SUFFIXES):
        return False
    if name.startswith(".env") or name in DENY_NAMES or any(name.startswith(n) for n in DENY_NAMES):
        return False
    return True


def _is_included(rel: str) -> bool:
    """True if a tracked path falls under INCLUDE_ROOTS or EXTRA_ROOT_FILES."""
    r = rel.replace("\\", "/")
    for root in INCLUDE_ROOTS:
        if r == root or r.startswith(root + "/"):
            return True
    return r in EXTRA_ROOT_FILES


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(project_root: Path, *args: str, binary: bool = False):
    out = subprocess.run(
        ["git", "-C", str(project_root), *args],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.decode(errors='replace')[:300]}")
    return out.stdout if binary else out.stdout.decode("utf-8", errors="replace")


def git_state(project_root: Path) -> Dict:
    """Return rich git state including accurate working-tree counts."""
    try:
        branch = _git(project_root, "branch", "--show-current").strip()
        sha = _git(project_root, "rev-parse", "HEAD").strip()
        subject = _git(project_root, "log", "-1", "--format=%s").strip()
        remotes_raw = _git(project_root, "remote", "-v")
    except RuntimeError:
        branch, sha, subject, remotes_raw = "unknown", "none", "", ""
    remotes: Dict[str, List[str]] = {}
    for line in remotes_raw.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            remotes.setdefault(parts[0], []).append(parts[1])

    status = _git(project_root, "status", "--porcelain", "--untracked-files=all")
    lines = [l for l in status.splitlines() if l.strip()]
    untracked = [l for l in lines if l.startswith("??")]
    staged = [l for l in lines if not l.startswith("??") and l[0] != " "]
    worktree_mod = [l for l in lines if not l.startswith("??") and len(l) >= 2 and l[1] != " "]
    clean = len(lines) == 0
    return {
        "project_root": str(project_root),
        "branch": branch or "unknown",
        "head_sha": sha or "none",
        "last_commit_subject": subject,
        "remotes": remotes,
        "tracked_worktree_modified_count": len(worktree_mod),
        "staged_change_count": len(staged),
        "untracked_count": len(untracked),
        "working_tree_clean": clean,
        "working_tree_dirty": not clean,
        "dirty_file_count": len(lines),
    }


def tracked_files_at(project_root: Path, rev: str = "HEAD") -> List[str]:
    raw = _git(project_root, "ls-tree", "-r", "--name-only", "-z", rev, binary=True)
    return [p for p in raw.decode("utf-8", errors="replace").split("\0") if p]


def blob_at(project_root: Path, path: str, rev: str = "HEAD") -> bytes:
    out = subprocess.run(
        ["git", "-C", str(project_root), "show", f"{rev}:{path}"],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0:
        raise FileNotFoundError(path)
    return out.stdout


def build_handoff(project_root: Path = PROJECT_ROOT, out_zip: Optional[Path] = None,
                  max_file_bytes: int = 5 * 1024 * 1024,
                  include_pt_hashes: bool = True,
                  rev: str = "HEAD") -> Dict:
    project_root = Path(project_root)
    if out_zip is None:
        out_zip = project_root / "AI_LEAD_HANDOFF.zip"
    out_zip = Path(out_zip)

    state = git_state(project_root)
    tracked = tracked_files_at(project_root, rev)

    included: List[Tuple[str, bytes]] = []
    oversized: List[Dict] = []
    excluded: List[Dict] = []
    seen = set()
    for rel in tracked:
        if not _is_included(rel) or not _safe(rel):
            if _is_included(rel) and not _safe(rel):
                excluded.append({"path": rel, "reason": "unsafe pattern"})
            continue
        if rel in seen:
            continue
        seen.add(rel)
        try:
            blob = blob_at(project_root, rel, rev)
        except FileNotFoundError:
            continue
        if len(blob) > max_file_bytes:
            oversized.append({"path": rel, "size": len(blob)})
            continue
        included.append((rel, blob))

    not_tracked_at_head = []
    wanted_files = [r for r in INCLUDE_ROOTS if not (project_root / r).is_dir()] + EXTRA_ROOT_FILES
    tracked_set = set(tracked)
    for rel in wanted_files:
        if rel not in tracked_set and (project_root / rel).exists():
            not_tracked_at_head.append(rel)

    manifest_files = []
    for rel, blob in sorted(included):
        manifest_files.append({"path": rel, "size": len(blob), "sha256": sha256_bytes(blob)})

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
                        "source": "local_untracked_artifact",
                        "note": "large artifact NOT included in zip (sha256 computed from local file only)",
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
        "source_state": "git HEAD committed bytes",
        "git": state,
        "working_tree_diff_not_included": not state["working_tree_clean"],
        "not_tracked_at_head": not_tracked_at_head,
        "current_task": proj_state.get("current_task", "unknown"),
        "last_completed_task": proj_state.get("last_completed_task", "unknown"),
        "next_planned_task": proj_state.get("next_planned_task", "unknown"),
        "max_file_bytes": max_file_bytes,
        "files": manifest_files,
        "file_count": len(manifest_files),
        "total_included_bytes": sum(m["size"] for m in manifest_files),
        "omitted_oversized": oversized,
        "excluded_unsafe": excluded,
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
        for rel, blob in sorted(included):
            z.writestr(rel, blob)

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
        "manifest_commit_sha": state["head_sha"],
        "working_tree_clean": state["working_tree_clean"],
        "file_count": len(manifest_files),
        "omitted_oversized": oversized,
        "omitted_artifacts": omitted_artifacts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AI_LEAD_HANDOFF.zip from committed state")
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