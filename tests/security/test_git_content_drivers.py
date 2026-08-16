# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A repo-defined git content driver never runs a host command on agent6's ops.

`filter.<n>.clean/smudge/process` and `merge.<n>.driver` defined in a repo's
own `.git/config` are host commands. The clean filter fires on the per-step
auto-commit's `git add`; the merge driver fires on the chain merge's
`merge-tree`. A cloned repo brings them pre-poisoned, and under hardened a
jailed command can write `.git/config` mid-run -- either way, no model action
is needed. agent6 neutralizes each by name unless `git.run_repo_filters` opts
in (the Git-LFS setting, since LFS uses exactly these drivers).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent6 import git_ops


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    yield
    git_ops.set_repo_filter_policy(False)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _git(tmp_path, "config", "user.email", "x@y.z")
    _git(tmp_path, "config", "user.name", "x")
    (tmp_path / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _poison_clean_filter(root: Path, marker: Path) -> None:
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text() + f'\n[filter "pwn"]\n\tclean = touch {marker}\n', encoding="utf-8"
    )
    (root / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")  # a change to stage


def test_the_auto_commit_does_not_run_a_repo_clean_filter(tmp_path: Path) -> None:
    """chain_commit is the live per-step commit; its temp-index `git add` runs
    the clean filter. Off by default: the payload must not fire, and the commit
    still records the (raw) content."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _poison_clean_filter(root, marker)
    git_ops.set_repo_filter_policy(False)
    sha = git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert sha is not None, "the commit did not happen"
    assert not marker.exists(), "the repo's clean filter ran a host command"


def test_run_repo_filters_true_honors_the_driver(tmp_path: Path) -> None:
    """The Git-LFS opt-in: with the knob on, the repo's driver runs (which is
    what LFS needs -- its clean filter turns a big file into a pointer)."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _poison_clean_filter(root, marker)
    git_ops.set_repo_filter_policy(True)
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=None)
    assert marker.exists(), "the driver was neutralized even though the operator opted in"


def test_the_chain_merge_does_not_run_a_repo_merge_driver(tmp_path: Path) -> None:
    """chain_merge merges with `merge-tree --write-tree`, which runs a custom
    merge driver. Off by default the payload must not fire; a neutralized
    driver makes the merge report a conflict, so chain_merge returns None
    (chain + worktree untouched) rather than half-merging."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _git(root, "checkout", "-qb", "other")
    (root / "f.txt").write_text("other\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "o")
    other = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "checkout", "-q", "-")
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text() + f'\n[merge "pwn"]\n\tdriver = touch {marker} && true\n', encoding="utf-8"
    )
    (root / ".gitattributes").write_text("f.txt merge=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("mine\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "m")
    mine = _git(root, "rev-parse", "HEAD").stdout.strip()
    git_ops.set_repo_filter_policy(False)
    result = git_ops.chain_merge(root, other, "merge", ref="refs/agent6/t", fallback_parent=mine)
    assert not marker.exists(), "the repo's merge driver ran a host command"
    assert result is None, "a conflicting neutralized merge should not produce a merge commit"


def test_a_clean_repo_commits_normally_with_filters_off(tmp_path: Path) -> None:
    """The overrides are added per name from the repo's own config, so a repo
    that defines no drivers gets none and commits exactly as before."""
    root = _repo(tmp_path / "r")
    (root / "new.txt").write_text("hello\n", encoding="utf-8")
    git_ops.set_repo_filter_policy(False)
    sha = git_ops.chain_commit(root, "c", ref="refs/agent6/t", fallback_parent=None)
    assert sha is not None


def test_driver_names_enumerate_and_dedup(tmp_path: Path) -> None:
    """A filter with both clean and smudge is one driver, one set of overrides;
    dotted subsection names survive the split."""
    root = _repo(tmp_path / "r")
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text()
        + '\n[filter "lfs"]\n\tclean = git-lfs clean\n\tsmudge = git-lfs smudge\n'
        + '\n[merge "a.b"]\n\tdriver = custom %O %A %B\n',
        encoding="utf-8",
    )
    git_ops.set_repo_filter_policy(False)
    overrides = git_ops._repo_driver_overrides(root)  # pyright: ignore[reportPrivateUsage]
    assert overrides.count("filter.lfs.clean=") == 1, overrides
    assert "filter.lfs.smudge=" in overrides and "filter.lfs.process=" in overrides
    assert "merge.a.b.driver=" in overrides, "a dotted merge-driver name was mis-split"
    git_ops.set_repo_filter_policy(True)
    assert git_ops._repo_driver_overrides(root) == ()  # pyright: ignore[reportPrivateUsage]


def test_a_driver_hidden_behind_an_include_is_still_neutralized(tmp_path: Path) -> None:
    """`git config --local` alone stops at `.git/config`, but a git op follows
    an `[include]` there to a repo-controlled file -- so a filter hidden behind
    one would run while a naive enumeration missed it. The enumeration uses
    `--includes` to match what the op sees (reproduced: without it, the
    include-hidden clean filter fired on the auto-commit)."""
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    marker = tmp_path / "pwned"
    (root / ".git" / "evil.cfg").write_text(
        f'[filter "pwn"]\n\tclean = touch {marker}\n', encoding="utf-8"
    )
    cfg = root / ".git" / "config"
    cfg.write_text(cfg.read_text() + "\n[include]\n\tpath = evil.cfg\n", encoding="utf-8")
    (root / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_ops.set_repo_filter_policy(False)
    assert "filter.pwn.clean=" in git_ops._repo_driver_overrides(root)  # pyright: ignore[reportPrivateUsage]
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "an include-hidden clean filter ran a host command"


def test_the_review_diff_does_not_run_a_repo_clean_filter(tmp_path: Path) -> None:
    """`agent6 review`'s working-tree diff shells out to git directly, and its
    hardening carried the fixed `-c` set without the per-name driver overrides,
    so `git diff HEAD` ran the repo's clean filter on the host."""
    from agent6.ui.cli.review_cmds import (
        _collect_review_diff,  # pyright: ignore[reportPrivateUsage]
    )

    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _poison_clean_filter(root, marker)
    git_ops.set_repo_filter_policy(False)

    proc = _collect_review_diff("git", root, base="", head="HEAD", paths=())
    assert "f.txt" in proc.stdout, proc.stdout
    assert not marker.exists(), "the repo's clean filter ran a host command"
