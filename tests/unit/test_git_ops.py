# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.git_ops on a temporary repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.git_ops import (
    CommitIdentity,
    GitError,
    GitSafetyError,
    commit_all,
    create_branch,
    is_git_repo,
    make_run_branch_name,
    recent_log,
    refuse_force,
    refuse_history_rewrite,
    refuse_push,
    reset_to,
    slugify,
    status,
    verify_git_identity,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def test_slugify_basic() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("") == "run"
    assert slugify("a" * 100).startswith("a" * 40)
    assert len(slugify("a" * 100)) == 40


def test_make_run_branch_name_format() -> None:
    name = make_run_branch_name()
    assert name.startswith("agent6/")


def test_is_git_repo_false_for_tmp(tmp_path: Path) -> None:
    assert is_git_repo(tmp_path) is False


def test_status_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    st = status(tmp_path)
    assert st.branch == "main"
    assert st.is_clean is True
    assert st.modified_count == 0


def test_status_dirty_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    st = status(tmp_path)
    assert st.is_clean is False
    assert st.untracked_count == 1


def test_commit_all_returns_sha_and_log(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = commit_all(tmp_path, "add f", trailers={"agent6-step": "x"})
    assert len(sha) == 40
    log = recent_log(tmp_path, n=5)
    assert "add f" in log


def test_create_branch(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    create_branch(tmp_path, "agent6/test")
    assert status(tmp_path).branch == "agent6/test"


def test_refuse_helpers() -> None:
    with pytest.raises(GitSafetyError):
        refuse_push()
    with pytest.raises(GitSafetyError):
        refuse_force()
    with pytest.raises(GitSafetyError):
        refuse_history_rewrite()


def test_status_on_non_repo(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        status(tmp_path)


def test_verify_git_identity_uses_repo_config(tmp_path: Path) -> None:
    _init_repo(tmp_path)  # configures user.name=t, user.email=t@t
    name, email = verify_git_identity(tmp_path, CommitIdentity())
    assert name == "t"
    assert email == "t@t"


def test_verify_git_identity_override_wins(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    name, email = verify_git_identity(tmp_path, CommitIdentity(name="bot", email="bot@example.com"))
    assert name == "bot"
    assert email == "bot@example.com"


def test_verify_git_identity_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate from any global/system git identity by pointing the global
    # config at an empty file.
    empty_cfg = tmp_path / "empty.gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_cfg))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    with pytest.raises(GitError, match="Git identity not configured"):
        verify_git_identity(repo, CommitIdentity())


def test_commit_all_with_identity_overrides_author(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = commit_all(
        tmp_path,
        "add f",
        identity=CommitIdentity(
            name="agent6",
            email="agent6@example.com",
            coauthor="Alice <alice@example.com>",
        ),
    )
    show = subprocess.run(
        ["git", "-C", str(tmp_path), "show", "--no-patch", "--format=%an|%ae|%B", sha],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "agent6|agent6@example.com|" in show
    assert "Co-authored-by: Alice <alice@example.com>" in show


def test_reset_to_soft_keeps_index_and_worktree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    start = status(tmp_path).head_sha
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    sha1 = commit_all(tmp_path, "add a")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    sha2 = commit_all(tmp_path, "add b")
    assert sha2 != start
    reset_to(tmp_path, start, mode="soft")
    assert status(tmp_path).head_sha == start
    # Worktree files survive.
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b\n"
    # Soft reset leaves changes STAGED.
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert set(staged) == {"a.txt", "b.txt"}
    # Orphaned commit object is still alive (reflog keeps it from gc).
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "cat-file", "-t", sha1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "commit"
    )


def test_reset_to_mixed_unstages(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    start = status(tmp_path).head_sha
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    commit_all(tmp_path, "add a")
    reset_to(tmp_path, start, mode="mixed")
    assert status(tmp_path).head_sha == start
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    # Mixed reset leaves changes UNSTAGED (file shows as untracked).
    assert status(tmp_path).untracked_count == 1


def test_reset_to_rejects_hard(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    with pytest.raises(GitError, match="mode must be 'soft' or 'mixed'"):
        reset_to(tmp_path, status(tmp_path).head_sha, mode="hard")


def test_commit_error_surfaces_stdout_when_stderr_empty(tmp_path: Path) -> None:
    """`git commit` writes "nothing to commit, working tree
    clean" to STDOUT, not stderr. `_run` only captured
    stderr, producing error strings like "git commit -m X failed: "
    with no useful detail. The new behaviour must include stdout when
    stderr is empty so the operator gets actionable signal."""
    _init_repo(tmp_path)
    # `commit_all` will stage a no-op and call `git commit`, which exits
    # 1 with "nothing to commit, working tree clean" on STDOUT.
    with pytest.raises(GitError) as excinfo:
        commit_all(tmp_path, "no-op commit on clean repo")
    msg = str(excinfo.value)
    # The detail (from stdout) must be present so callers can pattern-match.
    assert "nothing to commit" in msg.lower()
    # And the prefix must still identify which git invocation failed.
    assert "git commit" in msg
