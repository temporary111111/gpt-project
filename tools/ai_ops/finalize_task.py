"""Standard task finalizer for the chatgpt-like project.

Safe deterministic sequence for EVERY task completion:
1. verify project root
2. verify .venv / Python
3. verify required AI continuity files exist
4. check docs/ai state mentions the task (warning only)
5. run the test suite (unless --skip-tests)
6. inspect git status
7. safety-scan candidate files (secrets by filename/pattern, oversized, *.pt, *.bin)
8. refuse anything unsafe
9. stage ONLY safe files explicitly (never `git add -A`)
10. commit with a descriptive task message (no empty commits)
11. push current branch when a configured authenticated remote exists (never force)
12. build AI_LEAD_HANDOFF.zip from the committed state
13. verify the ZIP opens
14. print commit SHA, push status, handoff path and size

If push fails: keep the local commit, report PUSH_FAILED, still build the zip.

Usage:
    .\\.venv\\Scripts\\python.exe tools\\ai_ops\\finalize_task.py --task "TASK 005" --summary "Chat tuning done"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from build_handoff import PROJECT_ROOT, build_handoff, git_state

DENY_SUBSTRINGS = [
    ".venv", "__pycache__", ".pytest_cache", ".git/",
    "data/processed", "data/raw", "data/cleaned", "data/sources",
    "chatgpt-agent-gateway",
]
DENY_SUFFIXES = (".pyc", ".pyo", ".pt", ".bin", ".tmp", ".zip", ".key", ".pem")
DENY_NAMES = ("credentials", "secrets", ".env", "id_rsa", "id_ed25519")

REQUIRED_CONTINUITY_FILES = [
    "AGENTS.md", "00_START_HERE.md", "opencode.json",
    "docs/ai/CURRENT_STATE.md", "docs/ai/PROJECT_STATE.json",
    "docs/ai/CURRENT_TASK.md", "docs/ai/NEXT_ACTION.md",
    "docs/ai/TASK_HISTORY.md", "docs/ai/DECISIONS.md",
    "docs/ai/CONTINUITY.md", "docs/ai/RECOVERY_PROTOCOL.md",
    "docs/ai/OPERATING_PROTOCOL.md",
]


def is_safe_path(rel: str) -> Tuple[bool, str]:
    """Return (safe, reason). Deny secrets/artifacts/oversized by path pattern."""
    p = rel.replace("\\", "/").lower()
    for sub in DENY_SUBSTRINGS:
        if sub in p:
            return False, f"denied path segment: {sub}"
    if p.endswith(DENY_SUFFIXES):
        return False, f"denied extension: {Path(p).suffix}"
    name = os.path.basename(p)
    if name.startswith(".env") or name in DENY_NAMES or any(name.startswith(n) for n in DENY_NAMES):
        return False, f"denied secret-like filename: {name}"
    return True, "ok"


def _expand_entries(project_root: Path, rel: str) -> List[str]:
    """Expand an untracked directory entry into its safe file paths."""
    if rel.endswith("/"):
        rel = rel.rstrip("/")
    p = project_root / rel
    if not p.exists():
        return []
    if p.is_dir():
        out = []
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            parts = f.relative_to(project_root).parts
            if any(part in (".git", "__pycache__", ".pytest_cache", ".venv") for part in parts):
                continue
            out.append(f.relative_to(project_root).as_posix())
        return out
    return [rel]


def scan_candidates(project_root: Path, max_bytes: int = 50 * 1024 * 1024) -> Tuple[List[str], List[Dict]]:
    """Scan `git status --porcelain` output; return (safe_rels, rejected)."""
    out = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=60,
    )
    safe, rejected = [], []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        rel = line[3:].strip()
        rel = rel.split(" -> ")[-1].strip('"')
        if rel.startswith('"'):
            rel = rel[1:-1]
        if not rel:
            continue
        for candidate in _expand_entries(project_root, rel):
            ok, reason = is_safe_path(candidate)
            p = project_root / candidate
            if ok and p.exists() and p.is_file() and p.stat().st_size > max_bytes:
                ok, reason = False, f"oversized ({p.stat().st_size} bytes > {max_bytes})"
            if ok:
                safe.append(candidate)
            else:
                rejected.append({"path": candidate, "reason": reason})
    return safe, rejected


def run_test_suite(project_root: Path, venv_python: Path) -> bool:
    cmd = [str(venv_python), "-m", "pytest", "tests", "-q"]
    out = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=1800)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr[-4000:] if out.stderr else "")
    return out.returncode == 0


def finalize_task(project_root: Path = PROJECT_ROOT, task: str = "",
                  summary: str = "", run_tests: bool = True,
                  max_staged_bytes: int = 50 * 1024 * 1024) -> Dict:
    project_root = Path(project_root)
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    result: Dict = {
        "task": task, "summary": summary,
        "project_root": str(project_root),
        "tests": "skipped", "commit_sha": None, "commit_message": None,
        "push_status": "no_remote_or_no_changes",
        "staged": [], "rejected": [],
        "handoff": None, "errors": [], "warnings": [],
    }

    if not (project_root / ".git").exists():
        result["errors"].append("not a git repository")
        return result
    if not venv_python.exists():
        result["warnings"].append(f"venv python missing: {venv_python}")

    missing = [r for r in REQUIRED_CONTINUITY_FILES if not (project_root / r).exists()]
    if missing:
        result["errors"].append(f"missing continuity files: {missing}")
        return result

    ct = project_root / "docs" / "ai" / "CURRENT_TASK.md"
    th = project_root / "docs" / "ai" / "TASK_HISTORY.md"
    for doc, label in ((ct, "CURRENT_TASK.md"), (th, "TASK_HISTORY.md")):
        if task and doc.exists() and task.lower() not in doc.read_text(encoding="utf-8").lower():
            result["warnings"].append(f"{label} does not mention {task}")

    if run_tests:
        result["tests"] = "passed" if run_test_suite(project_root, venv_python) else "FAILED"

    safe, rejected = scan_candidates(project_root, max_staged_bytes)
    result["staged"] = safe
    result["rejected"] = rejected
    if rejected:
        result["warnings"].append(f"{len(rejected)} unsafe files refused (never staged): "
                                  + ", ".join(r["path"] for r in rejected[:10]))

    if not safe:
        result["warnings"].append("no safe changes to commit (no empty commit created)")
        return result

    add_cmd = ["git", "-C", str(project_root), "add", "--"] + safe
    subprocess.run(add_cmd, check=True, timeout=120)

    msg = f"{task}: {summary}" if task else summary
    commit = subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", msg],
        capture_output=True, text=True, timeout=120,
    )
    if commit.returncode != 0:
        result["errors"].append(f"commit failed: {commit.stderr.strip()}")
        return result
    result["commit_message"] = msg

    state = git_state(project_root)
    result["commit_sha"] = state["head_sha"]

    branch = state["branch"]
    remotes = {k: v[0] for k, v in state["remotes"].items()}
    push_cmd = ["git", "-C", str(project_root), "push", "origin", branch]
    if "origin" in remotes and "--force" not in push_cmd and "-f" not in push_cmd:
        push = subprocess.run(push_cmd, capture_output=True, text=True, timeout=300)
        if push.returncode == 0:
            result["push_status"] = f"pushed to origin/{branch}"
        else:
            result["push_status"] = "PUSH_FAILED (local commit preserved)"
            result["errors"].append(f"push failed: {push.stderr.strip()[:500]}")
    else:
        result["push_status"] = "no_remote (REMOTE_SETUP_REQUIRED)" if "origin" not in remotes else "skipped"

    try:
        handoff = build_handoff(project_root, include_pt_hashes=True)
        result["handoff"] = handoff
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"handoff build failed: {e}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Standard task finalizer")
    ap.add_argument("--task", required=True, help="task id, e.g. TASK 005")
    ap.add_argument("--summary", required=True, help="short completion summary")
    ap.add_argument("--skip-tests", action="store_true", help="do not run pytest")
    ap.add_argument("--max-staged-bytes", type=int, default=50 * 1024 * 1024)
    args = ap.parse_args()

    result = finalize_task(
        PROJECT_ROOT, task=args.task, summary=args.summary,
        run_tests=not args.skip_tests, max_staged_bytes=args.max_staged_bytes,
    )
    print(json.dumps(result, indent=2))
    if result["errors"]:
        print("ERRORS:", "; ".join(result["errors"]))
    sys.exit(1 if result["errors"] else 0)


if __name__ == "__main__":
    sys.exit(main())