# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A repo-defined git content driver never runs a host command on agent6's ops.

`filter.<n>.clean/smudge/process` and `merge.<n>.driver` in a repo's own `.git/config`
are host commands: the clean filter fires on the per-turn auto-commit's `git add`, the
merge driver on the chain merge's `merge-tree`. A cloned repo brings them pre-poisoned,
and under `partial` a jailed command can write `.git/config` mid-run; either way no model
action is needed. agent6 neutralizes each by name unless `git.run_repo_drivers` opts in
(the Git-LFS setting, since LFS uses exactly these drivers).
"""

from __future__ import annotations

import errno
import os
import pathlib
import re
import subprocess
from collections.abc import Iterator
from typing import Any

import pytest

from agent6 import errors
from agent6.vcs.git import command as git_command
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import policy as git_policy


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    yield
    git_policy.set_policy(git_policy.GitPolicy())


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


def _blank(key: str) -> str:
    """Return the override that blanks one config key."""
    return f"--config-env={key}={git_policy.BLANK_ENV}"


def _poison_clean_filter(root: pathlib.Path, marker: pathlib.Path) -> None:
    cfg = root / ".git" / "config"
    cfg.write_text(
        cfg.read_text() + f'\n[filter "pwn"]\n\tclean = touch {marker}\n', encoding="utf-8"
    )
    (root / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")  # a change to stage


def test_the_auto_commit_does_not_run_a_repo_clean_filter(tmp_path: pathlib.Path) -> None:
    """The per-turn commit's temp-index `git add` does not run a clean filter by default.

    The payload must not fire, and the commit still records the raw content.
    """
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _poison_clean_filter(root, marker)
    git_policy.set_policy(git_policy.GitPolicy())
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
    git_policy.set_policy(git_policy.GitPolicy(honor_repo_drivers=True))
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
    git_policy.set_policy(git_policy.GitPolicy())
    result = git_ops.chain_merge(root, other, "merge", ref="refs/agent6/t", fallback_parent=mine)
    assert not marker.exists(), "the repo's merge driver ran a host command"
    assert result is None, "a conflicting neutralized merge should not produce a merge commit"


def test_a_clean_repo_commits_normally_with_drivers_off(tmp_path: pathlib.Path) -> None:
    """A repo that defines no drivers gets no overrides and commits exactly as before.

    The overrides are added per name from the repo's own config.
    """
    root = _repo(tmp_path / "r")
    (root / "new.txt").write_text("hello\n", encoding="utf-8")
    git_policy.set_policy(git_policy.GitPolicy())
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
    git_policy.set_policy(git_policy.GitPolicy())
    overrides = git_policy.git_hardening_flags(root)
    assert overrides.count(_blank("filter.lfs.clean")) == 1, overrides
    assert _blank("filter.lfs.smudge") in overrides and _blank("filter.lfs.process") in overrides
    assert _blank("merge.a.b.driver") in overrides, "a dotted merge-driver name was mis-split"
    git_policy.set_policy(git_policy.GitPolicy(honor_repo_drivers=True))
    honored = git_policy.git_hardening_flags(root)
    assert _blank("filter.lfs.clean") not in honored
    assert _blank("merge.a.b.driver") not in honored


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
    git_policy.set_policy(git_policy.GitPolicy())
    assert _blank("filter.pwn.clean") in git_policy.git_hardening_flags(root)
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "an include-hidden clean filter ran a host command"


def test_a_driver_in_a_worktrees_own_config_is_neutralized(tmp_path: pathlib.Path) -> None:
    """The driver enumeration covers `config.worktree`, which `git config --local` skips.

    A filter defined there under `extensions.worktreeConfig` ran on the auto-commit.
    """
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    marker = tmp_path / "pwned"
    _git(root, "config", "extensions.worktreeConfig", "true")
    _git(root, "config", "--worktree", "filter.pwn.clean", f"touch {marker}")
    (root / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_policy.set_policy(git_policy.GitPolicy())
    assert _blank("filter.pwn.clean") in git_policy.git_hardening_flags(root)
    git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "a clean filter in config.worktree ran a host command"


def test_a_driver_in_the_operators_global_config_is_left_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the repository's scopes are blanked; a global filter (Git LFS) is the operator's."""
    root = _repo(tmp_path / "r")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text('[filter "mine"]\n\tclean = cat\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    _git(root, "config", "filter.theirs.clean", "cat")
    git_policy.set_policy(git_policy.GitPolicy())
    assert git_command.run(root, "config", "filter.mine.clean").stdout == "cat\n"
    overrides = git_policy.git_hardening_flags(root)
    assert _blank("filter.theirs.clean") in overrides
    assert _blank("filter.mine.clean") not in overrides


@pytest.mark.parametrize("failure", ["timed out", "did not start", "failed"])
def test_no_driver_runs_where_git_cannot_list_the_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A listing that failed left every driver on, and the per-turn commit ran the clean filter."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _poison_clean_filter(root, marker)
    git_policy.set_policy(git_policy.GitPolicy())
    run = subprocess.run

    def no_listing(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--show-scope" not in argv:
            return run(argv, **kwargs)
        if failure == "timed out":
            raise subprocess.TimeoutExpired(argv, 10)
        if failure == "did not start":
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE))
        return subprocess.CompletedProcess(argv, 128, "", "fatal: its own words\n")

    monkeypatch.setattr(git_policy.subprocess, "run", no_listing)
    with pytest.raises(
        errors.GitError, match=re.escape(f"agent6 runs no git in {root}: ")
    ) as caught:
        git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "the repo's clean filter ran a host command"
    said = {"timed out": "no answer within 10s", "failed": "fatal: its own words"}.get(
        failure, os.strerror(errno.EMFILE)
    )
    assert said in str(caught.value) and said not in caught.value.safe_message


def _define_drivers(root: pathlib.Path, name: str, command: str) -> None:
    """Write a filter and a merge driver into `.git/config`, as a copied repository brings them."""
    quoted = name.replace("\\", "\\\\").replace('"', '\\"')
    with (root / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(f'[filter "{quoted}"]\n\tclean = {command}\n\tsmudge = {command}\n')
        config.write(f'[merge "{quoted}"]\n\tdriver = {command}\n')


def _command_scope(root: pathlib.Path) -> dict[str, str]:
    """Return each key agent6's own overrides set in the repository, with its value."""
    fields = git_command.nul_fields(root, "config", "--show-scope", "-z", "--list")
    pairs = zip(fields[::2], fields[1::2], strict=False)
    return dict(entry.split("\n", 1) for scope, entry in pairs if scope == "command")


@pytest.mark.parametrize(
    "names",
    [
        ("a=b",),
        ("ends=",),
        ("two words",),
        ('quo"te',),
        ("it's",),
        ("tail #",),
        ("x.clean=sh -c 'cat' #",),
        ("x", "x.clean"),
    ],
)
def test_the_overrides_blank_each_driver_key_and_set_no_other(
    tmp_path: pathlib.Path, names: tuple[str, ...]
) -> None:
    """A `-c` override for a driver named `a=b` set `filter.a`: git cut it at its first `=`.

    The driver kept its command, and a name shaped like `x.clean=<command> #` defined one.
    """
    root = _repo(tmp_path / "r")
    git_policy.set_policy(git_policy.GitPolicy())
    fixed = _command_scope(root)
    for name in names:
        _define_drivers(root, name, "false")
    keys = {f"merge.{name}.driver" for name in names} | {
        f"filter.{name}.{command}" for name in names for command in ("clean", "smudge", "process")
    }
    added = {key: value for key, value in _command_scope(root).items() if key not in fixed}
    assert added == dict.fromkeys(keys, "")
    for key in keys:
        assert git_command.run(root, "config", "--get", key).stdout == "\n", key


def test_the_overrides_run_on_a_callers_own_argv_under_git_environment(
    tmp_path: pathlib.Path,
) -> None:
    """`sessions diff` and `ask` put the overrides on an argv of their own.

    git refuses a driver's override where `git_environment` did not set its variable.
    """
    root = _repo(tmp_path / "r")
    _define_drivers(root, "a=b", "false")
    git_policy.set_policy(git_policy.GitPolicy())
    argv = ["git", *git_policy.git_hardening_flags(root), "config", "--get", "filter.a=b.clean"]
    env = git_policy.git_environment()
    ran = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False, env=env)
    assert (ran.returncode, ran.stdout) == (0, "\n")
    del env[git_policy.BLANK_ENV]
    refused = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False, env=env)
    assert refused.returncode == 128 and git_policy.BLANK_ENV in refused.stderr


def _arm_equals_filter(root: pathlib.Path, marker: pathlib.Path) -> None:
    """Give every file a clean filter named `a=b`, and `f.txt` a change to stage."""
    _define_drivers(root, "a=b", f"touch {marker}")
    (root / ".gitattributes").write_text("* filter=a=b\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")


def test_the_per_turn_tree_runs_no_filter_whose_name_holds_an_equals_sign(
    tmp_path: pathlib.Path,
) -> None:
    """The per-turn commit's `git add` ran a clean filter named `a=b`."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _arm_equals_filter(root, marker)
    git_policy.set_policy(git_policy.GitPolicy())
    sha = git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "the repo's clean filter ran a host command"
    assert _git(root, "show", f"{sha}:f.txt").stdout == "v2\n"


def test_the_per_turn_tree_runs_no_filter_whose_name_is_not_utf8(tmp_path: pathlib.Path) -> None:
    """A driver name of bytes that are no UTF-8 made the listing, and every git call, raise.

    The name goes from git's listing back onto git's argv byte for byte.
    """
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    with (root / ".git/config").open("ab") as config:
        config.write(b'[filter "n\xff"]\n\tclean = touch ' + os.fsencode(marker) + b"\n")
    (root / ".gitattributes").write_bytes(b"* filter=n\xff\n")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_policy.set_policy(git_policy.GitPolicy())

    sha = git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)

    assert not marker.exists(), "the repo's clean filter ran a host command"
    assert _git(root, "show", f"{sha}:f.txt").stdout == "v2\n"


def test_a_driver_name_supplies_no_command_to_its_own_override(tmp_path: pathlib.Path) -> None:
    """The override for a filter named `x.clean=<command> #` set `filter.x.clean` to the command."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD").stdout.strip()
    _define_drivers(root, f"x.clean=touch {marker} #", "true")
    (root / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_policy.set_policy(git_policy.GitPolicy())
    sha = git_ops.chain_commit(root, "step", ref="refs/agent6/t", fallback_parent=base)
    assert not marker.exists(), "agent6's own override ran a host command"
    assert _git(root, "show", f"{sha}:f.txt").stdout == "v2\n"


def test_staging_runs_no_filter_whose_name_holds_an_equals_sign(tmp_path: pathlib.Path) -> None:
    """The Git page's stage ran a clean filter named `a=b`."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    _arm_equals_filter(root, marker)
    git_policy.set_policy(git_policy.GitPolicy())
    git_ops.stage_paths(root, ["f.txt"])
    assert not marker.exists(), "the repo's clean filter ran a host command"
    assert _git(root, "show", ":f.txt").stdout == "v2\n"


def test_a_landing_runs_no_merge_driver_whose_name_holds_an_equals_sign(
    tmp_path: pathlib.Path,
) -> None:
    """A landing's `merge-tree` ran a merge driver named `a=b`; blanked, the merge conflicts."""
    marker = tmp_path / "pwned"
    root = _repo(tmp_path / "r")
    target = _git(root, "branch", "--show-current").stdout.strip()
    _define_drivers(root, "a=b", f"touch {marker}")
    (root / ".gitattributes").write_text("f.txt merge=a=b\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "attributes")
    _git(root, "checkout", "-qb", "run")
    (root / "f.txt").write_text("run\n", encoding="utf-8")
    _git(root, "commit", "-qam", "run")
    _git(root, "checkout", "-q", target)
    (root / "f.txt").write_text("target\n", encoding="utf-8")
    _git(root, "commit", "-qam", "target")
    _git(root, "checkout", "-q", "--detach")
    git_policy.set_policy(git_policy.GitPolicy())
    landed = git_ops.plumb_merge(root, target, "run", method="merge", message="land")
    assert not marker.exists(), "the repo's merge driver ran a host command"
    assert landed.conflicted and landed.conflicts == ("f.txt",)


def test_blame_and_the_other_diff_family_verbs_do_not_run_a_repo_textconv(
    tmp_path: pathlib.Path,
) -> None:
    """The web Git tab's blame does not run a repo-configured `diff.<d>.textconv`.

    The runner forced the builtin renderer for `diff` and `show` only; `blame`, `diff-tree`
    and `log` take the same flags.
    """
    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    marker = tmp_path / "pwned"
    _git(root, "config", "diff.pwn.textconv", f"touch {marker}; cat")
    (root / ".gitattributes").write_text("*.txt diff=pwn\n", encoding="utf-8")
    git_policy.set_policy(git_policy.GitPolicy())
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    rows = git_files.blame_lines(root, head, "f.txt", b"v1\n", lines=1)
    assert rows and rows[0].sha == head
    assert not marker.exists(), "a repo-configured textconv ran as a host command"


def test_blame_keeps_an_author_time_python_cannot_date(tmp_path: pathlib.Path) -> None:
    """A commit dated past year 9999 raised `ValueError` out of blame; the row keeps the time."""
    import os

    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_DATE": "@100000000000000 +0000"}
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qam", "future"],
        cwd=root,
        env=env,
        check=True,
    )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    rows = git_files.blame_lines(root, head, "f.txt", b"v2\n", lines=1)
    assert rows[0].sha == head and rows[0].date == "@100000000000000"


def test_blame_refuses_an_ignore_revs_file_outside_the_repository(tmp_path: pathlib.Path) -> None:
    """Host-side blame does not open a `blame.ignoreRevsFile` outside the repository.

    Git quotes the file's first unparsable line in its error, so a poisoned `.git/config`
    read any host file a line at a time through `/api/git/blame`.
    """
    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", str(secret))
    git_policy.set_policy(git_policy.GitPolicy())
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(errors.GitError) as caught:
        git_files.blame_lines(root, head, "f.txt", b"v1\n", lines=1)
    assert "TOKEN-4711" not in str(caught.value)
    assert "blame.ignoreRevsFile" in str(caught.value)


def test_blame_honours_an_ignore_revs_file_inside_the_repository(tmp_path: pathlib.Path) -> None:
    """A repository-relative `blame.ignoreRevsFile` naming a commit still attributes lines."""
    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    first = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "reformat")
    second = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / ".git-blame-ignore-revs").write_text(second + "\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", ".git-blame-ignore-revs")
    git_policy.set_policy(git_policy.GitPolicy())
    rows = git_files.blame_lines(root, second, "f.txt", b"v2\n", lines=1)
    assert rows[0].sha == first


def test_blame_refuses_an_ignore_revs_file_whose_name_holds_a_newline(
    tmp_path: pathlib.Path,
) -> None:
    """A `blame.ignoreRevsFile` holding a newline was checked as two paths.

    Git reads the value whole, so a file named `a<newline>b` that links outside the repository
    passed a check that split the value into `a` and `b`, two regular files inside it.
    """
    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    (root / "a").write_text("x\n", encoding="utf-8")
    (root / "b").write_text("y\n", encoding="utf-8")
    (root / "a\nb").symlink_to(secret)
    _git(root, "config", "blame.ignoreRevsFile", "a\nb")
    git_policy.set_policy(git_policy.GitPolicy())
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
    git read a host file; blame runs in a private work tree holding a copy.
    """
    from agent6.vcs.git import command as git_command
    from agent6.vcs.git import files as git_files

    root = _repo(tmp_path / "r")
    secret = tmp_path / "token.txt"
    secret.write_text("TOKEN-4711\n", encoding="utf-8")
    first = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "ignore.txt").write_text(first + "\n", encoding="utf-8")
    _git(root, "config", "blame.ignoreRevsFile", "ignore.txt")
    git_policy.set_policy(git_policy.GitPolicy())
    real_run = git_command.run

    def swapping(cwd: pathlib.Path, *args: str, **kwargs: object) -> object:
        if args and args[0] == "blame":
            (root / "ignore.txt").unlink()
            (root / "ignore.txt").symlink_to(secret)
        return real_run(cwd, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(git_command, "run", swapping)
    rows = git_files.blame_lines(root, first, "f.txt", b"v1\n", lines=1)
    assert rows and "TOKEN" not in repr(rows)
