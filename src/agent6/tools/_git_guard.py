# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""git-argv guard for run_command: refuse a mutating git subcommand and
reject config-injection (-c alias.*/core.hooksPath ...) that would make git
execute code. Lifted verbatim from dispatch.py to isolate the security screen."""

from __future__ import annotations

from pathlib import Path

from agent6.tools.errors import ToolError

_GIT_MUTATING_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "merge",
        "mv",
        "pull",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "switch",
    }
)
_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset(
    {
        "-C",
        "-c",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
)
_GIT_GLOBAL_OPTIONS_WITH_VALUE_PREFIXES = tuple(
    f"{opt}=" for opt in _GIT_GLOBAL_OPTIONS_WITH_VALUE if opt.startswith("--")
)


def _strip_env_wrapper(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Peel a leading ``env [-i] [-u NAME] [NAME=VALUE...]`` wrapper.

    Best-effort: ``env git clean -fdx`` is the common way to slip a mutating git
    command past the argv[0]=="git" check; strip the wrapper so the refusal
    still applies. Does not (and cannot) catch every wrapper (sh -c, sudo, ...);
    the real protection is the jail RO-binding .git -- this just closes the
    obvious hole.
    """
    if not argv or Path(argv[0]).name != "env":
        return argv
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg in ("-i", "--ignore-environment"):
            idx += 1
        elif arg in ("-u", "--unset"):
            idx += 2  # takes a NAME argument
        elif "=" in arg and not arg.startswith("-"):
            idx += 1  # NAME=VALUE assignment
        else:
            break
    return argv[idx:]


def _git_subcommand(argv: tuple[str, ...]) -> str | None:
    argv = _strip_env_wrapper(argv)
    if not argv or Path(argv[0]).name != "git":
        return None
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--":
            return None
        if arg in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if arg.startswith(_GIT_GLOBAL_OPTIONS_WITH_VALUE_PREFIXES):
            idx += 1
            continue
        if arg.startswith("-"):
            idx += 1
            continue
        return arg
    return None


_GIT_CONFIG_INJECTION_MSG = (
    "run_command refuses `git` with injected config (`-c`, `--config-env`, or a "
    "`GIT_CONFIG_*` env var): an inline `alias.<name>` or `core.hooksPath` can "
    "make git run a forbidden subcommand (push, reset --hard, clean, rebase, ...) "
    "under a benign name, slipping past the mutating-git refusal. Run read-only "
    "git (status, diff, show, log) WITHOUT injected config; change files with "
    "apply_patch / apply_edit."
)


def _refuse_git_config_injection(argv: tuple[str, ...]) -> None:
    """Refuse a git invocation that injects inline config. ``git -c name=value``
    (and ``--config-env``, and ``GIT_CONFIG_*`` set by a leading ``env`` wrapper)
    can define an ``alias.<x>`` or ``core.hooksPath`` that makes git execute a
    FORBIDDEN subcommand under a benign alias name -- e.g.
    ``git -c alias.r='reset --hard' r`` parses as subcommand ``r`` and would
    otherwise slip past :func:`refuse_mutating_git_command`. The read-only git
    the model is allowed never needs injected config, so refuse it outright."""
    git_argv = _strip_env_wrapper(argv)
    if not git_argv or Path(git_argv[0]).name != "git":
        return
    # GIT_CONFIG_* assignments in the leading `env` wrapper: _strip_env_wrapper
    # drops them for subcommand detection, but they are still passed to git.
    wrapper = argv[: len(argv) - len(git_argv)]
    for arg in wrapper:
        if "=" in arg and arg.split("=", 1)[0].startswith("GIT_CONFIG"):
            raise ToolError(_GIT_CONFIG_INJECTION_MSG)
    # `-c` / `--config-env` only inject config when they appear as a GLOBAL
    # option (BEFORE the subcommand): `git -c k=v <sub>`. AFTER the subcommand,
    # `-c` is an ordinary read-only option (combined-diff for `git log/show/diff
    # -c`), so we must stop at the subcommand and not block those. Walk the
    # leading global options exactly as _git_subcommand does.
    idx = 1
    while idx < len(git_argv):
        arg = git_argv[idx]
        if arg == "--":
            return
        if arg in {"-c", "--config-env"} or arg.startswith("--config-env="):
            raise ToolError(_GIT_CONFIG_INJECTION_MSG)
        if arg in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if arg.startswith(_GIT_GLOBAL_OPTIONS_WITH_VALUE_PREFIXES) or arg.startswith("-"):
            idx += 1
            continue
        return  # reached the subcommand; a later `-c` is a read-only option


def refuse_mutating_git_command(argv: tuple[str, ...]) -> None:
    _refuse_git_config_injection(argv)
    subcommand = _git_subcommand(argv)
    if subcommand not in _GIT_MUTATING_SUBCOMMANDS:
        return
    raise ToolError(
        f"run_command refuses mutating git subcommand `git {subcommand}` because "
        ".git/ is protected inside the jail. For revert/recovery, inspect prior "
        "content with `git show HEAD:path/to/file`, then restore it with "
        "apply_patch or apply_edit. Read-only git commands such as status, diff, "
        "show, and log are still allowed."
    )
