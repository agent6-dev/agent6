# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Share git hardening flags and the operator's hook and driver policy."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

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


# Repo-config keys that run a repo-controlled command on the host, blanked with `-c` (highest
# precedence): fsmonitor on every index refresh, diff.external on every diff, gpg.program on
# every commit. None costs correctness: agent6's per-step commits are unsigned by design.
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

# A repo hook is repo-controlled host code, so honoring one on agent6's own commit is an RCE
# vector; set once from `git.run_repo_hooks` at startup. Mutated, never rebound.
_hook_policy: dict[str, bool] = {"honor_repo_hooks": False}


def set_repo_hook_policy(honor: bool) -> None:
    """Configure whether agent6's own git ops fire the repo's `.git/hooks/*`."""
    _hook_policy["honor_repo_hooks"] = honor


# A driver in `.git/config` is a host command, an RCE vector for a cloned poisoned repo;
# honoring them is the Git-LFS opt-in. Set once from `git.run_repo_drivers` at startup.
_driver_policy: dict[str, bool] = {"honor_repo_drivers": False}


def set_repo_driver_policy(honor: bool) -> None:
    """Configure whether agent6's git ops honor the repo's own content drivers."""
    _driver_policy["honor_repo_drivers"] = honor


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
    if _driver_policy["honor_repo_drivers"]:
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
    hooks = () if _hook_policy["honor_repo_hooks"] else ("-c", "core.hooksPath=/dev/null")
    return (*_GIT_HARDENING, *hooks, *repo_driver_overrides(cwd))


def git_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build Git's environment without shell overrides of its repository or command policy.

    Args:
        extra: Explicit agent6 settings, including temporary index paths.

    Returns:
        The inherited environment with Git overrides and provider keys removed.
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
    # Disable terminal credential prompts; stable Git prose keeps the stash rescue parseable.
    return child_env.without_provider_keys(
        {**inherited, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(extra or {})}
    )
