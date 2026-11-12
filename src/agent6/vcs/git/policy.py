# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Share git hardening flags and the operator's hook and driver policy."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Generator

from agent6 import child_env, errors


def git() -> str:
    """Return the git executable's path.

    Returns:
        The path of the first `git` on PATH.

    Raises:
        GitError: git is not on PATH.
    """
    executable = shutil.which("git")
    if executable is None:
        raise errors.GitError("git executable not found on PATH")
    return executable


# The oldest git agent6 supports: Debian 13's, the platform floor docs/installation.md names.
GIT_FLOOR: tuple[int, int] = (2, 47)


@functools.cache
def installed_version() -> tuple[int, ...] | None:
    """Return the version of the git on PATH, read once per process.

    Returns:
        The version's numbers, or None when `git --version` cannot run or names none.
    """
    try:
        proc = subprocess.run(
            [git(), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=git_environment(),
        )
    except (errors.GitError, OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", proc.stdout)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def unsupported_note() -> str:
    """Return the line naming an installed git older than `GIT_FLOOR`.

    Returns:
        The line, or "" when git is new enough or its version cannot be read.
    """
    version = installed_version()
    if version is None or version[:2] >= GIT_FLOOR:
        return ""
    found = ".".join(map(str, version))
    floor = ".".join(map(str, GIT_FLOOR))
    return f"unsupported git version {found}: agent6 needs git {floor} or newer"


# Repo-config keys that run a repo-controlled command on the host, blanked with `-c` (highest
# precedence): fsmonitor on every index refresh, diff.external on every diff, gpg.program on
# every commit. None costs correctness: agent6's per-turn commits are unsigned by design.
# Content drivers (filter.*, merge.*.driver) have no blanket switch; `repo_driver_overrides`
# blanks each by name. The prefix keys pin the `a/` and `b/` headers every cited path reads.
_GIT_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.srcPrefix=a/",
    "-c",
    "diff.dstPrefix=b/",
)


@dataclasses.dataclass(frozen=True, slots=True)
class GitPolicy:
    """Which repo-controlled host code agent6's own git operations may run; neither by default.

    Attributes:
        honor_repo_hooks: `git.run_repo_hooks`: fire the repository's `.git/hooks/*`. A hook is
            repo-controlled host code, so honoring one on agent6's own commit is an RCE vector.
        honor_repo_drivers: `git.run_repo_drivers`: honor the content drivers in the
            repository's `.git/config`, the Git-LFS opt-in and an RCE vector for a poisoned clone.
    """

    honor_repo_hooks: bool = False
    honor_repo_drivers: bool = False


# The process's policy, set once at startup from `[git]`. Mutated, never rebound.
_process: dict[str, GitPolicy] = {"policy": GitPolicy()}
# One request of a long-lived process (the web server) under the config that request loaded.
_request: contextvars.ContextVar[GitPolicy | None] = contextvars.ContextVar(
    "agent6_git_policy", default=None
)


def set_policy(policy: GitPolicy) -> None:
    """Set the policy every git operation of this process runs under."""
    _process["policy"] = policy


def current_policy() -> GitPolicy:
    """Return the policy in force: the request's inside `use`, else the process's."""
    return _request.get() or _process["policy"]


@contextlib.contextmanager
def use(policy: GitPolicy) -> Generator[None]:
    """Run the calling context's git operations under a policy of its own.

    A thread started inside it does not inherit it and runs under the process's policy.

    Args:
        policy: The policy of the config this request loaded.

    Yields:
        Nothing; the override ends with the block.
    """
    token = _request.set(policy)
    try:
        yield
    finally:
        _request.reset(token)


# The config keys that name a driver command.
_DRIVER_KEY_RE = r"^(filter\..*\.(clean|smudge|process)|merge\..*\.driver)$"


def repo_driver_overrides(cwd: pathlib.Path) -> tuple[str, ...]:
    """Return the `-c` flags that blank every content driver the repo's own config defines.

    An empty `filter.<n>.clean` passes through and an empty `merge.<n>.driver` reports
    a conflict instead of running. Only the repo's `.git/config` and its includes are
    read: a filter in the operator's `~/.gitconfig` is trusted. Re-read per call, so a
    driver a jailed command writes mid-run is caught too.

    Args:
        cwd: The repository.

    Returns:
        The flags; empty when the policy honors drivers or the repo defines none.
    """
    if current_policy().honor_repo_drivers:
        return ()
    try:
        proc = subprocess.run(
            [
                git(),
                "-C",
                str(cwd),
                "config",
                "--local",
                # A git op follows `[include]` to a repo-controlled file; this enumeration must too.
                "--includes",
                "--name-only",
                "--get-regexp",
                _DRIVER_KEY_RE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    # A filter with both clean and smudge yields two keys for one driver.
    filters: set[str] = set()
    merges: set[str] = set()
    for key in proc.stdout.split():
        if key.startswith("filter."):
            filters.add(key[len("filter.") : key.rindex(".")])
        elif key.startswith("merge."):
            merges.add(key[len("merge.") : key.rindex(".")])
    overrides: list[str] = []
    for name in sorted(filters):
        overrides += [
            "-c",
            f"filter.{name}.clean=",
            "-c",
            f"filter.{name}.smudge=",
            "-c",
            f"filter.{name}.process=",
        ]
    for name in sorted(merges):
        overrides += ["-c", f"merge.{name}.driver="]
    return tuple(overrides)


# Force git's builtin renderer on every diff-family verb (diff, diff-tree, log, show, blame):
# `diff.external` and per-file `diff.<d>.textconv` run host commands the `-c` overrides do not
# cover. Placed after the subcommand.
DIFF_SAFETY_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")


def git_hardening_flags(cwd: pathlib.Path) -> tuple[str, ...]:
    """Return the `-c` overrides every agent6 git invocation carries, placed before the subcommand.

    Public so the callers that shell out to git outside this module carry the same
    hardening; diff and show callers also add `DIFF_SAFETY_FLAGS` after the subcommand.

    Args:
        cwd: The repository, for its content drivers.

    Returns:
        The fixed set, the hooks path unless the policy honors repo hooks, and a blank
        override per content driver the repo defines.
    """
    # /dev/null is not a directory, so git finds no hooks there.
    hooks = () if current_policy().honor_repo_hooks else ("-c", "core.hooksPath=/dev/null")
    return (*_GIT_HARDENING, *hooks, *repo_driver_overrides(cwd))


def git_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build Git's environment without shell overrides of its repository or command policy.

    Args:
        extra: Explicit agent6 settings, including temporary index paths.

    Returns:
        The inherited environment with Git overrides and provider keys removed, and the
        operator's pager set.
    """
    # Inherit only GIT_{ASKPASS,AUTHOR_*,COMMITTER_*,EDITOR,SSH,SSH_COMMAND,SSH_VARIANT,TRACE*}.
    allowed = {"GIT_ASKPASS", "GIT_EDITOR", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_SSH_VARIANT"}
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
        or name in allowed
        or name.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_", "GIT_TRACE"))
    }
    # The operator's pager, never the repository's: GIT_PAGER outranks `core.pager` and
    # `pager.<cmd>`, which a jailed command can write into .git/config under `partial`.
    pager = os.environ.get("GIT_PAGER") or os.environ.get("PAGER") or "less"
    # Disable terminal credential prompts; stable Git prose keeps the stash rescue parseable.
    return child_env.without_provider_keys(
        {
            **inherited,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": pager,
            "LC_ALL": "C",
            **(extra or {}),
        }
    )
