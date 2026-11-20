# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A nested repository's own git config runs no program under agent6's own git.

A jailed command can make a repository inside the workspace at every isolation level, and
can write the `.git` of a submodule: the nested config, attributes and hooks are workspace
content. git records such a directory as a gitlink and, on some commands, runs git inside
it under that config. Each test arms a nested repository with a filter and a textconv
program and runs the agent6 operations of one route.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import path_safety
from agent6.vcs import review_submodules
from agent6.vcs.git import command as git_command
from agent6.vcs.git import history as git_history
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import policy as git_policy
from agent6.vcs.git import read as git_read


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def _program(tmp_path: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Return a program that records its run and passes its input on, with its record."""
    marker = tmp_path / "ran"
    program = tmp_path / "program"
    program.write_text(f'#!/bin/sh\necho ran >> {marker}\ncat "$@"\n', encoding="utf-8")
    program.chmod(0o755)
    return str(program), marker


def _record_nested(root: pathlib.Path) -> pathlib.Path:
    """Make a repository in `vendor/` and commit it to the project as a submodule.

    Its `.gitmodules` entry asks for a full comparison (`ignore = none`), which outranks
    `diff.ignoreSubmodules`.
    """
    nested = _repo(root / "vendor")
    (nested / ".gitattributes").write_text("* filter=x diff=x\n", encoding="utf-8")
    _git(nested, "add", "-A")
    _git(nested, "commit", "-qm", "attributes")
    (root / ".gitmodules").write_text(
        '[submodule "vendor"]\n\tpath = vendor\n\turl = ./vendor\n\tignore = none\n',
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "vendor")
    return nested


def _commit_nested(root: pathlib.Path, text: str) -> str:
    """Commit a change in the nested repository and record it in the project."""
    (root / "vendor" / "f.txt").write_text(text, encoding="utf-8")
    _git(root / "vendor", "commit", "-qam", text.strip())
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", f"vendor {text.strip()}")
    return _git(root, "rev-parse", "HEAD")


def _arm(nested: pathlib.Path, program: str) -> None:
    """Name the program as the nested repository's filter and textconv."""
    for key in ("filter.x.clean", "filter.x.smudge", "diff.x.textconv"):
        _git(nested, "config", key, program)


def test_a_status_or_a_diff_runs_no_git_inside_a_nested_repository(
    tmp_path: pathlib.Path,
) -> None:
    """A worktree comparison ran `git status` in the nested repository, and its clean filter."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    assert git_read.status(root).is_clean
    assert git_read.modified_paths(root) == []
    assert not git_read.paths_dirty(root, ("vendor",))
    assert git_read.worktree_matches(root, "HEAD", ["vendor"])
    assert git_history.worktree_files(root, "HEAD") == ()
    assert git_history.worktree_patch(root, "HEAD", "vendor", max_bytes=None) == ("", False)
    assert git_command.nul_fields(root, "diff-files", "--name-only", "-z") == [""]
    assert not marker.exists(), "the nested repository's clean filter ran on the host"

    # The commit the nested repository is at still counts: git reads it without running there.
    _git(nested, "-c", "filter.x.clean=", "commit", "-qam", "edited")
    assert git_read.modified_paths(root) == ["vendor"]
    assert not git_read.worktree_matches(root, "HEAD", ["vendor"])
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_a_branch_switch_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """`checkout` and `switch` listed local changes, asking the nested repository for its own."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    git_ops.create_branch(root, "topic")
    git_ops.create_branch(root, "main")  # exists: a plain checkout
    git_ops.switch_branch(root, "topic")
    git_ops.checkout_detached(root, "main")
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_restoring_a_stash_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """`stash apply` ended with a `git status`, which asked the nested repository."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_ops.stash_tracked_changes(root, "kept")
    entry = git_read.find_stash(root, "kept")
    assert entry is not None
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    assert git_ops.restore_stash(root, entry)
    assert (root / "f.txt").read_text(encoding="utf-8") == "v2\n"
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_moving_a_checkout_never_recurses_into_a_submodule(tmp_path: pathlib.Path) -> None:
    """Under `submodule.recurse`, a tree move checked the submodule out with its own filters."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    older = _git(root, "rev-parse", "HEAD")
    newer = _commit_nested(root, "v2\n")
    _git(root, "config", "submodule.recurse", "true")
    _git(root, "config", "submodule.vendor.url", "./vendor")
    _arm(nested, program)

    git_ops.sync_worktree(root, newer, older)
    git_ops.checkout_detached(root, older)
    git_ops.create_branch(root, "topic", start_point=newer)
    git_ops.switch_branch(root, "main")
    assert (nested / "f.txt").read_text(encoding="utf-8") == "v2\n"  # git left it alone
    assert not marker.exists(), "the submodule's own filter ran on the host"


def test_a_diff_never_renders_inside_a_submodule(tmp_path: pathlib.Path) -> None:
    """Under `diff.submodule = diff`, a diff, show or log ran `git diff` inside the submodule."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    older = _git(root, "rev-parse", "HEAD")
    newer = _commit_nested(root, "v2\n")
    _git(root, "config", "diff.submodule", "diff")
    _arm(nested, program)

    assert "Subproject commit" in git_read.commit_diff(root, newer)
    assert "Subproject commit" in git_read.diff_range(root, older, newer)
    assert (
        "Subproject commit"
        in git_history.file_patch(root, older, newer, "vendor", max_bytes=None)[0]
    )
    assert not marker.exists(), "the submodule's own textconv ran on the host"


def test_a_status_summarises_no_submodule(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `status.submoduleSummary`, a status ran `git log` inside each moved submodule."""
    root = _repo(tmp_path / "r")
    nested = _record_nested(root)
    (nested / "f.txt").write_text("v2\n", encoding="utf-8")
    _git(nested, "commit", "-qam", "v2")
    _git(root, "config", "status.submoduleSummary", "true")
    trace = tmp_path / "trace"
    monkeypatch.setenv("GIT_TRACE", str(trace))

    assert "modified:   vendor" in git_command.run(root, "status").stdout
    assert "cd vendor" not in trace.read_text(encoding="utf-8"), "git ran inside the submodule"


def test_a_directory_laid_out_as_a_git_dir_is_no_repository(tmp_path: pathlib.Path) -> None:
    """Below an embedded bare repository, git took that directory's config as the project's."""
    root = _repo(tmp_path / "r")
    embedded = root / "pkg"
    subprocess.run(["git", "init", "-q", "--bare", str(embedded)], check=True)
    (embedded / "src").mkdir()
    _git(embedded, "config", "core.bare", "false")
    _git(embedded, "config", "core.worktree", str(tmp_path))

    assert git_read.toplevel(embedded / "src") is None
    assert not git_read.is_git_repo(embedded / "src")
    assert git_read.toplevel(root) == root


def test_a_review_honors_no_driver_of_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """Under `git.run_repo_drivers`, review's status of a submodule ran that submodule's filter."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    with git_policy.use(git_policy.GitPolicy(honor_repo_drivers=True)):
        dirty = review_submodules.dirty_paths(
            root, ("vendor",), path_safety.host_workspace(root, ())
        )
    assert dirty == ("vendor",)
    assert not marker.exists(), "the nested repository's clean filter ran on the host"
