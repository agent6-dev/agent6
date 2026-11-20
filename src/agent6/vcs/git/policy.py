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


# A nested repository (a submodule, a repository a command made in the workspace) has a config,
# attributes and hooks of its own, all workspace content, and git runs its commands inside one
# under them. These keep git out: no diff renders inside one, no status summarises one, no
# checkout recurses, and a directory laid out as a git dir is a repository only as a `.git`.
_NESTED_REPOSITORY_HARDENING: tuple[str, ...] = (
    "-c",
    "diff.submodule=short",
    "-c",
    "safe.bareRepository=explicit",
    "-c",
    "status.submoduleSummary=false",
    "-c",
    "submodule.recurse=false",
)

# Repo-config keys that run a repo-controlled command on the host, overridden with `-c` (highest
# precedence): fsmonitor on every index refresh, diff.external on every diff, a gpg program on
# a commit (commit.gpgsign) and on a log, show or stash list (log.showSignature),
# gc.recentObjectsHook after a commit or a fetch (maintenance.auto), and a credential helper
# on a fetch's bundle download (fetch.bundleURI). None costs correctness: agent6 signs and
# verifies nothing, and the operator's own git still runs maintenance.
# Content drivers (filter.*, merge.*.driver) have no blanket switch; `hardening` blanks each
# by name. The prefix keys pin the `a/` and `b/` headers every cited path reads.
_GIT_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "log.showSignature=false",
    "-c",
    "maintenance.auto=false",
    "-c",
    "fetch.bundleURI=",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.srcPrefix=a/",
    "-c",
    "diff.dstPrefix=b/",
    *_NESTED_REPOSITORY_HARDENING,
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


# The keys agent6's git answers by name in each repository: a driver command, and the work tree.
_REPO_KEY_RE = r"^(filter\..*\.(clean|smudge|process)|merge\..*\.driver|core\.worktree)$"

# The variable a driver's override reads its empty value from; `git_environment` sets it.
BLANK_ENV = "AGENT6_GIT_BLANK"


@dataclasses.dataclass(frozen=True, slots=True)
class Hardening:
    """What one listing of a repository's own config gives agent6's git there.

    Attributes:
        flags: The overrides every invocation carries, placed before the subcommand:
            the fixed set, the hooks path unless the policy honors repo hooks, and a blank
            override per content driver the repo defines unless the policy honors drivers.
        sets_work_tree: Whether the repo's own config sets `core.worktree`.
    """

    flags: tuple[str, ...]
    sets_work_tree: bool


def _refusal(cwd: pathlib.Path, cause: str, said: str = "") -> errors.GitError:
    """Return the error for a directory agent6 runs no git in.

    On a git older than `GIT_FLOOR`, it opens with the unsupported-version line.

    Args:
        cwd: The directory.
        cause: What agent6 could not learn there.
        said: git's or the OS's own words, kept out of the safe reason.

    Returns:
        The error.
    """
    note = unsupported_note()
    lead = f"{note}\n" if note else ""
    reason = f"{lead}agent6 runs no git in {cwd}: {cause}"
    return errors.GitError(f"{reason} ({said})" if said else reason, safe_message=reason)


def _query(cwd: pathlib.Path, cause: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Ask git one fixed question about a directory, before any other git runs there.

    Args:
        cwd: The directory.
        cause: What a refusal says git could not do.
        *args: git's arguments.

    Returns:
        The finished process.

    Raises:
        GitError: git did not start, gave no answer or was killed.
    """
    try:
        proc = subprocess.run(
            [git(), "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=git_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        error = _refusal(cwd, cause, f"no answer within {exc.timeout:g}s")
        raise error from exc
    except OSError as exc:
        error = _refusal(cwd, cause, str(exc))
        raise error from exc
    if proc.returncode < 0:
        error = _refusal(cwd, cause, f"killed by signal {-proc.returncode}")
        raise error
    return proc


def hardening(cwd: pathlib.Path) -> Hardening:
    """List the repository's own content drivers and work-tree key, in one git call.

    An empty `filter.<n>.clean` passes through and an empty `merge.<n>.driver` reports
    a conflict instead of running. git names the scope of each key: one in the operator's
    global or system config is trusted, and every other scope is the repo's (`.git/config`,
    a worktree's `config.worktree`, their includes). Re-read per call, so a key a jailed
    command writes mid-run is caught too.

    Args:
        cwd: The repository.

    Returns:
        The overrides, and whether the repository names its work tree.

    Raises:
        GitError: git could not list the config.
    """
    # /dev/null is not a directory, so git finds no hooks there.
    hooks = () if current_policy().honor_repo_hooks else ("-c", "core.hooksPath=/dev/null")
    fixed = (*_GIT_HARDENING, *hooks)
    unlisted = "git could not list the repository's config"
    proc = _query(
        cwd,
        unlisted,
        "config",
        "--show-scope",
        # A git op follows `[include]` to a repo-controlled file; this enumeration must too.
        "--includes",
        "-z",
        "--name-only",
        "--get-regexp",
        _REPO_KEY_RE,
    )
    # git exits 1 when no key matches.
    if proc.returncode not in (0, 1):
        error = _refusal(cwd, unlisted, proc.stderr.strip() or f"exit {proc.returncode}")
        raise error
    sets_work_tree = False
    # A filter with both clean and smudge yields two keys for one driver.
    filters: set[str] = set()
    merges: set[str] = set()
    fields = proc.stdout.split("\0")
    for scope, key in zip(fields[::2], fields[1::2], strict=False):
        if scope in ("global", "system"):
            continue
        if key.startswith("filter."):
            filters.add(key[len("filter.") : key.rindex(".")])
        elif key.startswith("merge."):
            merges.add(key[len("merge.") : key.rindex(".")])
        else:
            sets_work_tree = True
    keys: list[str] = []
    if not current_policy().honor_repo_drivers:
        keys = [
            f"filter.{name}.{command}"
            for name in sorted(filters)
            for command in ("clean", "smudge", "process")
        ]
        keys += [f"merge.{name}.driver" for name in sorted(merges)]
    # A driver's name can hold `=`: git cuts `-c` at the first one, `--config-env` at the last.
    overrides = (f"--config-env={key}={BLANK_ENV}" for key in keys)
    return Hardening((*fixed, *overrides), sets_work_tree)


# Force git's builtin renderer on every diff-family verb (diff, diff-tree, log, show, blame):
# `diff.external` and per-file `diff.<d>.textconv` run host commands the `-c` overrides do not
# cover. Placed after the subcommand.
DIFF_SAFETY_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")

# A worktree comparison asks each nested repository whether its files changed by running
# `git status` inside it, under its own config. This compares the commit it is at and no more;
# `diff.ignoreSubmodules` says the same and loses to a `.gitmodules` entry.
_NO_NESTED_STATUS = "--ignore-submodules=dirty"

# The flags each verb carries, placed after the subcommand. `checkout` and `switch` make the
# worktree comparison for the list of local changes that `--quiet` leaves out.
VERB_FLAGS: dict[str, tuple[str, ...]] = {
    "blame": DIFF_SAFETY_FLAGS,
    "checkout": ("--quiet",),
    "diff": (*DIFF_SAFETY_FLAGS, _NO_NESTED_STATUS),
    "diff-files": (_NO_NESTED_STATUS,),
    "diff-tree": DIFF_SAFETY_FLAGS,
    "log": DIFF_SAFETY_FLAGS,
    "show": DIFF_SAFETY_FLAGS,
    "status": (_NO_NESTED_STATUS,),
    "switch": ("--quiet",),
}


def git_hardening_flags(cwd: pathlib.Path) -> tuple[str, ...]:
    """Return the overrides every agent6 git invocation carries, placed before the subcommand.

    Public so the callers that shell out to git outside this module carry the same
    hardening; diff and show callers also add `DIFF_SAFETY_FLAGS` after the subcommand.
    git refuses a driver's override outside `git_environment`, which sets `BLANK_ENV`.

    Args:
        cwd: The repository, for its content drivers.

    Returns:
        The fixed set, the hooks path unless the policy honors repo hooks, and a blank
        override per content driver the repo defines.

    Raises:
        GitError: git could not list the repository's config.
    """
    return hardening(cwd).flags


def work_tree(cwd: pathlib.Path) -> str | None:
    """Return the checkout git works on from a directory, for `GIT_WORK_TREE`.

    Unnamed, git takes the work tree from the repository's `core.worktree`, a key a jailed
    command can write where `.git` is writable. The checkout is the directory holding `cwd`
    whose `.git` is the git dir git found: the one git names, or the parent of a `.git`
    directory when the key names another.

    Args:
        cwd: The directory git runs in.

    Returns:
        The checkout, or None where git finds no work tree: outside a repository, inside a
        git dir, in a bare one.

    Raises:
        GitError: git gave no answer, a path it names holds a line break, or the
            repository is reached through a `.git` file and its `core.worktree` names a
            directory that is not its checkout.
    """
    proc = _query(
        cwd,
        "git could not name the checkout",
        *_GIT_HARDENING,
        "rev-parse",
        "--absolute-git-dir",
        "--show-toplevel",
    )
    if proc.returncode != 0:
        return None
    found = proc.stdout.splitlines()
    if len(found) != 2:
        error = _refusal(cwd, "the path of its git dir or of its checkout holds a line break")
        raise error
    git_dir, named = (pathlib.Path(line).resolve() for line in found)
    here = cwd.resolve()
    for checkout in (named, git_dir.parent):
        if here.is_relative_to(checkout) and _git_dir_of(checkout) == git_dir:
            return str(checkout)
    raise errors.GitError(
        f"core.worktree names {named}, which is not the checkout of {git_dir}; agent6 runs git"
        " on the checkout that holds .git (remove the key: git config --unset core.worktree,"
        " with --worktree in a linked worktree)"
    )


def _git_dir_of(checkout: pathlib.Path) -> pathlib.Path | None:
    """Return the git dir a checkout's `.git` names, or None when it names none."""
    pointer = checkout / ".git"
    if pointer.is_dir():
        return pointer.resolve()
    try:
        # A `.git` file, a linked worktree's or a submodule's: git reads it.
        proc = subprocess.run(
            [git(), "rev-parse", "--resolve-git-dir", str(pointer)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd="/",
            env=git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return pathlib.Path(proc.stdout.strip()).resolve() if proc.returncode == 0 else None


def git_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build Git's environment without shell overrides of its repository or command policy.

    Args:
        extra: Explicit agent6 settings, including temporary index paths.

    Returns:
        The inherited environment with Git overrides and provider keys removed, the
        operator's pager set and `BLANK_ENV` empty.
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
    # No lazy fetch: a promisor remote in .git/config would fetch a missing object through
    # the repository's own transport keys (`core.sshCommand`, a credential helper).
    env = child_env.without_provider_keys(
        {
            **inherited,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_PAGER": pager,
            "LC_ALL": "C",
            **(extra or {}),
        }
    )
    # Set last: no inherited variable and no dropped name changes what a driver's override reads.
    env[BLANK_ENV] = ""
    return env
