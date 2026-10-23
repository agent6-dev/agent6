# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A repo-defined git content driver never runs a host command on agent6's ops.

`filter.<n>.clean/smudge/process` and `merge.<n>.driver` in a repo's own `.git/config`
are host commands: the clean filter fires on the per-step auto-commit's `git add`, the
merge driver on the chain merge's `merge-tree`. A cloned repo brings them pre-poisoned,
and under `partial` a jailed command can write `.git/config` mid-run; either way no model
action is needed. agent6 neutralizes each by name unless `git.run_repo_drivers` opts in
(the Git-LFS setting, since LFS uses exactly these drivers).
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Iterator

import pytest

from agent6 import errors
from agent6.vcs import git_ops, git_policy


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    yield
    git_policy.set_repo_driver_policy(False)


def _git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _git(tmp_path, "config", "user.email", "x@y.z")
    _git(tmp_path, "config", "user.name", "x")
    (tmp_path / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _poison_clean_filter(root: pathlib.Path, marker: pathlib.Path) -> None:
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text() + f'\n[filter "pwn"]\n\tclean = touch {marker}\n', encoding="utf-8"
    )
    (root / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")  # a change to stage


def test_the_auto_commit_does_not_run_a_repo_clean_filter(tmp_path: pathlib.Path) -> None:
    """The per-step commit's temp-index `git add` does not run a clean filter by default.

    The payload must not fire, and the commit still records the raw content.
    """
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _poison_clean_filter(root, marker)
    git_policy.set_repo_driver_policy(False)
    sha = git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert sha is not None, "the commit did not happen"
    assert not marker.exists(), "the repo's clean filter ran a host command"


def test_run_repo_drivers_true_honors_the_driver(tmp_path: pathlib.Path) -> None:
    """The Git-LFS opt-in: with the knob on, the repo's driver runs.

    LFS needs it; its clean filter turns a big file into a pointer.
    """
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _poison_clean_filter(root, marker)
    git_policy.set_repo_driver_policy(True)
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=None)
    assert marker.exists(), "the driver was neutralized even though the operator opted in"


def test_the_chain_merge_does_not_run_a_repo_merge_driver(tmp_path: pathlib.Path) -> None:
    """`merge-tree --write-tree` runs no custom merge driver by default.

    A neutralized driver makes the merge report a conflict, so chain_merge returns None
    (chain and worktree untouched) rather than half-merging.
    """
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
    git_policy.set_repo_driver_policy(False)
    result = git_ops.chain_merge(root, other, "merge", ref="refs/agent6/t", fallback_parent=mine)
    assert not marker.exists(), "the repo's merge driver ran a host command"
    assert result is None, "a conflicting neutralized merge should not produce a merge commit"


def test_a_clean_repo_commits_normally_with_drivers_off(tmp_path: pathlib.Path) -> None:
    """A repo that defines no drivers gets no overrides and commits exactly as before.

    The overrides are added per name from the repo's own config.
    """
    root = _repo(tmp_path / "r")
    (root / "new.txt").write_text("hello\n", encoding="utf-8")
    git_policy.set_repo_driver_policy(False)
    sha = git_ops.chain_commit(root, "c", ref="refs/agent6/t", fallback_parent=None)
    assert sha is not None


def test_driver_names_enumerate_and_dedup(tmp_path: pathlib.Path) -> None:
    """A filter with both clean and smudge is one driver, one set of overrides.

    Dotted subsection names survive the split.
    """
    root = _repo(tmp_path / "r")
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text()
        + '\n[filter "lfs"]\n\tclean = git-lfs clean\n\tsmudge = git-lfs smudge\n'
        + '\n[merge "a.b"]\n\tdriver = custom %O %A %B\n',
        encoding="utf-8",
    )
    git_policy.set_repo_driver_policy(False)
    overrides = git_policy.repo_driver_overrides(root)
    assert overrides.count("filter.lfs.clean=") == 1, overrides
    assert "filter.lfs.smudge=" in overrides and "filter.lfs.process=" in overrides
    assert "merge.a.b.driver=" in overrides, "a dotted merge-driver name was mis-split"
    git_policy.set_repo_driver_policy(True)
    assert git_policy.repo_driver_overrides(root) == ()


def test_a_driver_hidden_behind_an_include_is_still_neutralized(tmp_path: pathlib.Path) -> None:
    """The driver enumeration follows `[include]`, as a git op does.

    `git config --local` alone stops at `.git/config`, so a filter hidden behind an include
    to a repo-controlled file would run on the auto-commit while the enumeration missed it.
    """
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
    git_policy.set_repo_driver_policy(False)
    assert "filter.pwn.clean=" in git_policy.repo_driver_overrides(root)
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "an include-hidden clean filter ran a host command"


def test_blame_and_the_other_diff_family_verbs_do_not_run_a_repo_textconv(
    tmp_path: pathlib.Path,
) -> None:
    """The web Git tab's blame does not run a repo-configured `diff.<d>.textconv`.

    The runner forced the builtin renderer for `diff` and `show` only; `blame`, `diff-tree`
    and `log` take the same flags.
    """
    from agent6.vcs import git_files

    root = _repo(tmp_path / "r")
    marker = tmp_path / "pwned"
    _git(root, "config", "diff.pwn.textconv", f"touch {marker}; cat")
    (root / ".gitattributes").write_text("*.txt diff=pwn\n", encoding="utf-8")
    git_policy.set_repo_driver_policy(False)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    rows = git_files.blame_lines(root, head, "f.txt", b"v1\n", lines=1)
    assert rows and rows[0].sha == head
    assert not marker.exists(), "a repo-configured textconv ran as a host command"


def test_the_review_diff_does_not_run_a_repo_clean_filter(tmp_path: pathlib.Path) -> None:
    """`agent6 review`'s working-tree diff carries the per-name driver overrides.

    It shells out to git directly; the fixed `-c` set alone would let `git diff HEAD` run the
    repo's clean filter on the host.
    """
    from agent6.ui.cli import review_cmds

    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _poison_clean_filter(root, marker)
    git_policy.set_repo_driver_policy(False)

    proc = review_cmds._collect_review_diff("git", root, base="", head="HEAD", paths=())
    assert "f.txt" in proc.stdout, proc.stdout
    assert not marker.exists(), "the repo's clean filter ran a host command"


def test_blame_refuses_an_ignore_revs_file_outside_the_repository(tmp_path: pathlib.Path) -> None:
    """Host-side blame does not open a `blame.ignoreRevsFile` outside the repository.

    Git quotes the file's first unparsable line in its error, so a poisoned `.git/config`
    read any host file a line at a time through `/api/git/blame`.
    """
    from agent6.vcs import git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", str(secret))
    git_policy.set_repo_driver_policy(False)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(errors.GitError) as caught:
        git_files.blame_lines(root, head, "f.txt", b"v1\n", lines=1)
    assert "TOKEN-4711" not in str(caught.value)
    assert "blame.ignoreRevsFile" in str(caught.value)


def test_blame_honours_an_ignore_revs_file_inside_the_repository(tmp_path: pathlib.Path) -> None:
    """A repository-relative `blame.ignoreRevsFile` naming a commit still attributes lines."""
    from agent6.vcs import git_files

    root = _repo(tmp_path / "r")
    first = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "reformat")
    second = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / ".git-blame-ignore-revs").write_text(second + "\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", ".git-blame-ignore-revs")
    git_policy.set_repo_driver_policy(False)
    rows = git_files.blame_lines(root, second, "f.txt", b"v2\n", lines=1)
    assert rows[0].sha == first


def test_blame_refuses_an_ignore_revs_file_whose_name_holds_a_newline(
    tmp_path: pathlib.Path,
) -> None:
    """A `blame.ignoreRevsFile` holding a newline was checked as two paths.

    Git reads the value whole, so a file named `a<newline>b` that links outside the repository
    passed a check that split the value into `a` and `b`, two regular files inside it.
    """
    from agent6.vcs import git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    (root / "a").write_text("x\n", encoding="utf-8")
    (root / "b").write_text("y\n", encoding="utf-8")
    (root / "a\nb").symlink_to(secret)
    _git(root, "config", "blame.ignoreRevsFile", "a\nb")
    git_policy.set_repo_driver_policy(False)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(errors.GitError) as caught:
        git_files.blame_lines(root, head, "f.txt", b"v1\n", lines=1)
    assert "TOKEN-4711" not in str(caught.value)
    assert "blame.ignoreRevsFile" in str(caught.value)


def test_blame_reads_a_private_copy_of_the_ignore_revs_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git opened the configured ignore-revs path itself, after the check of the real file.

    A jailed command that swapped the file for a link between the check and git's open had
    git read a host file; blame now runs in a private work tree holding a copy.
    """
    from agent6.vcs import _git_command, git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    first = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "ignore.txt").write_text(first + "\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", "ignore.txt")
    git_policy.set_repo_driver_policy(False)
    real_run = _git_command.run

    def swapping(cwd: pathlib.Path, *args: str, **kwargs: object) -> object:
        if args and args[0] == "blame":
            (root / "ignore.txt").unlink()
            (root / "ignore.txt").symlink_to(secret)
        return real_run(cwd, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(_git_command, "run", swapping)
    rows = git_files.blame_lines(root, first, "f.txt", b"v1\n", lines=1)
    assert rows and "TOKEN" not in repr(rows)
