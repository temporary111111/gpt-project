"""Tests for the AI ops tooling (TASK 004.5).

All tests use temporary directories/repositories. The real remote is never
touched (push tests use a local bare repository).
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ai_ops"))

from build_handoff import build_handoff, sha256_file  # noqa: E402
from finalize_task import finalize_task, is_safe_path, scan_candidates  # noqa: E402

CONTINUITY_FILES = [
    "docs/ai/CURRENT_STATE.md",
    "docs/ai/PROJECT_STATE.json",
    "docs/ai/CURRENT_TASK.md",
    "docs/ai/NEXT_ACTION.md",
    "docs/ai/TASK_HISTORY.md",
    "docs/ai/DECISIONS.md",
    "docs/ai/CONTINUITY.md",
    "docs/ai/RECOVERY_PROTOCOL.md",
    "docs/ai/OPERATING_PROTOCOL.md",
]

PROJECT_STATE = {
    "current_task": "TASK 004.5",
    "last_completed_task": "TASK 004",
    "next_planned_task": "TASK 005",
}


def git(*args, cwd):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=120)


def make_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "00_START_HERE.md").write_text("# start here\n", encoding="utf-8")
    (proj / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (proj / "opencode.json").write_text('{"$schema": "https://opencode.ai/config.json"}\n', encoding="utf-8")
    (proj / "README.md").write_text("# readme\n", encoding="utf-8")
    (proj / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (proj / "src").mkdir()
    (proj / "src" / "model.py").write_text("class GPTModel: pass\n", encoding="utf-8")
    (proj / "src" / "train.py").write_text("def train(): pass\n", encoding="utf-8")
    (proj / "tests").mkdir()
    (proj / "tests" / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (proj / "docs" / "ai").mkdir(parents=True)
    (proj / "docs" / "ai" / "PROJECT_STATE.json").write_text(json.dumps(PROJECT_STATE), encoding="utf-8")
    for rel in CONTINUITY_FILES:
        if rel != "docs/ai/PROJECT_STATE.json":
            p = proj / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# {rel}\n", encoding="utf-8")
    (proj / "checkpoints" / "pretrain_v1").mkdir(parents=True)
    (proj / "checkpoints" / "pretrain_v1" / "best.pt").write_bytes(b"\x00" * 100)
    (proj / "checkpoints" / "pretrain_v1" / "latest.pt").write_bytes(b"\x01" * 100)
    return proj


def init_git(proj: Path, remote: Path | None = None):
    git("init", "-b", "main", cwd=proj)
    git("config", "user.name", "tester", cwd=proj)
    git("config", "user.email", "tester@example.com", cwd=proj)
    git("add", "-A", cwd=proj)
    git("commit", "-m", "initial", cwd=proj)
    if remote is not None:
        git("init", "--bare", str(remote), cwd=proj)
        git("remote", "add", "origin", str(remote), cwd=proj)
        git("push", "-u", "origin", "main", cwd=proj)


@pytest.fixture()
def proj(tmp_path):
    p = make_project(tmp_path)
    init_git(p)
    return p


def test_handoff_excludes_pt_bin_and_secrets(proj, tmp_path):
    (proj / "docs" / "ai" / "CURRENT_STATE.md").write_text("updated\n", encoding="utf-8")
    (proj / "secret.env").write_text("TOKEN=abc\n", encoding="utf-8")
    (proj / "credentials.txt").write_text("user:pass\n", encoding="utf-8")
    (proj / "data" / "processed").mkdir(parents=True)
    (proj / "data" / "processed" / "train.bin").write_bytes(b"\x00" * 64)
    out = tmp_path / "out.zip"
    build_handoff(proj, out_zip=out, max_file_bytes=10 * 1024 * 1024)
    names = zipfile.ZipFile(out).namelist()
    assert not any(n.endswith(".pt") for n in names)
    assert not any(n.endswith(".bin") for n in names)
    assert not any(n in ("secret.env", "credentials.txt") for n in names)
    assert not any("checkpoints" in n for n in names)


def test_handoff_includes_start_here_and_docs(proj, tmp_path):
    out = tmp_path / "out.zip"
    build_handoff(proj, out_zip=out, max_file_bytes=10 * 1024 * 1024)
    names = set(zipfile.ZipFile(out).namelist())
    assert "00_START_HERE.md" in names
    assert "AGENTS.md" in names
    assert "docs/ai/CURRENT_STATE.md" in names
    assert "docs/ai/PROJECT_STATE.json" in names
    assert "src/model.py" in names
    assert "tests/test_x.py" in names


def test_manifest_hashes_are_correct(proj, tmp_path):
    out = tmp_path / "out.zip"
    build_handoff(proj, out_zip=out, max_file_bytes=10 * 1024 * 1024)
    with zipfile.ZipFile(out) as z:
        manifest = json.loads(z.read("HANDOFF_MANIFEST.json"))
        git_state = json.loads(z.read("GIT_STATE.json"))
    assert manifest["git"]["head_sha"] == git_state["git"]["head_sha"]
    assert manifest["current_task"] == "TASK 004.5"
    assert manifest["next_planned_task"] == "TASK 005"
    by_path = {m["path"]: m for m in manifest["files"]}
    assert sha256_file(proj / "00_START_HERE.md") == by_path["00_START_HERE.md"]["sha256"]
    assert sha256_file(proj / "src" / "model.py") == by_path["src/model.py"]["sha256"]


def test_oversized_files_are_omitted(proj, tmp_path):
    (proj / "src" / "big.py").write_bytes(b"x" * 500)
    out = tmp_path / "out.zip"
    result = build_handoff(proj, out_zip=out, max_file_bytes=100)
    names = zipfile.ZipFile(out).namelist()
    assert "src/big.py" not in names
    assert any(o["path"] == "src/big.py" for o in result["omitted_oversized"])
    assert "src/big.py" in result["omitted_oversized"][0]["path"]


def test_zip_opens_and_is_valid(proj, tmp_path):
    out = tmp_path / "out.zip"
    result = build_handoff(proj, out_zip=out, max_file_bytes=10 * 1024 * 1024)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
    assert out.exists()
    assert result["zip_size"] == out.stat().st_size
    assert result["commit_sha"]


def test_local_git_finalization_works(proj, tmp_path):
    (proj / "NEW_FILE.md").write_text("new\n", encoding="utf-8")
    result = finalize_task(proj, task="TASK 005", summary="test finalize", run_tests=False)
    assert not result["errors"], result["errors"]
    assert result["commit_sha"]
    log = git("log", "-1", "--format=%s", cwd=proj).stdout.strip()
    assert log == "TASK 005: test finalize"
    assert "NEW_FILE.md" in result["staged"]
    assert result["handoff"] is not None
    assert result["handoff"]["commit_sha"] == result["commit_sha"]


def test_finalizer_refuses_secret_files(proj, tmp_path):
    (proj / ".env").write_text("KEY=secret\n", encoding="utf-8")
    (proj / "credentials.txt").write_text("u:p\n", encoding="utf-8")
    (proj / "SAFE.md").write_text("safe\n", encoding="utf-8")
    result = finalize_task(proj, task="TASK 005", summary="safety", run_tests=False)
    rejected = {r["path"] for r in result["rejected"]}
    assert ".env" in rejected
    assert "credentials.txt" in rejected
    assert "SAFE.md" not in rejected
    assert "SAFE.md" in result["staged"]
    tracked = git("ls-files", cwd=proj).stdout
    assert ".env" not in tracked
    assert "credentials.txt" not in tracked


def test_is_safe_path_rejects_secrets_and_artifacts():
    assert is_safe_path("src/model.py") == (True, "ok")
    for bad in ["checkpoints/x/best.pt", "data/processed/train.bin", ".env",
                "env/.env.local", "credentials.json", "secrets.txt", "x.key",
                "a.pem", "data/raw/dump.jsonl", "chatgpt-agent-gateway/a.py",
                "x.pyc", "tmp.tmp"]:
        ok, _ = is_safe_path(bad)
        assert not ok, bad


def test_finalizer_does_not_force_push(proj, tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    init_git(proj, remote=remote)
    (proj / "SAFE2.md").write_text("x\n", encoding="utf-8")
    calls = []

    real_run = subprocess.run

    def capture(*args, **kwargs):
        calls.append(args[0])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture)
    result = finalize_task(proj, task="TASK 005", summary="push test", run_tests=False)
    assert result["push_status"] == "pushed to origin/main"
    assert not result["errors"], result["errors"]
    for call in calls:
        if isinstance(call, list) and call and call[0] == "git":
            assert "--force" not in call
            assert "-f" not in call


def test_finalizer_no_changes_no_empty_commit(proj, tmp_path):
    before = git("rev-parse", "HEAD", cwd=proj).stdout.strip()
    result = finalize_task(proj, task="TASK 005", summary="nothing", run_tests=False)
    assert result["commit_sha"] is None
    assert git("rev-parse", "HEAD", cwd=proj).stdout.strip() == before
    assert any("no empty commit" in w for w in result["warnings"])


def test_commit_sha_recorded_in_handoff(proj, tmp_path):
    (proj / "SAFE3.md").write_text("y\n", encoding="utf-8")
    result = finalize_task(proj, task="TASK 005", summary="sha check", run_tests=False)
    head = git("rev-parse", "HEAD", cwd=proj).stdout.strip()
    assert result["commit_sha"] == head
    with zipfile.ZipFile(result["handoff"]["zip_path"]) as z:
        manifest = json.loads(z.read("HANDOFF_MANIFEST.json"))
    assert manifest["git"]["head_sha"] == head


def test_handoff_omits_pt_hashes_only_when_requested(proj, tmp_path):
    out = tmp_path / "out.zip"
    build_handoff(proj, out_zip=out, max_file_bytes=10 * 1024 * 1024, include_pt_hashes=True)
    with zipfile.ZipFile(out) as z:
        manifest = json.loads(z.read("HANDOFF_MANIFEST.json"))
    pts = [a for a in manifest["omitted_important_artifacts"] if a["path"].endswith(".pt")]
    assert len(pts) == 2
    assert all("sha256" in a for a in pts)