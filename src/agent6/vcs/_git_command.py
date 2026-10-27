# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run hardened git commands with shared policy, timeouts and output decoding."""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess

from agent6 import child_env, errors, kinds
from agent6.vcs import git_policy

# Local ops finish in well under a second; this fires only on a stuck filesystem or a held lock.
GIT_TIMEOUT_S = 120.0
# Environment switches that change how git reads every pathspec.
_PATHSPEC_ENV = frozenset(
    {"GIT_GLOB_PATHSPECS", "GIT_ICASE_PATHSPECS", "GIT_LITERAL_PATHSPECS", "GIT_NOGLOB_PATHSPECS"}
)

# How long a timed-out git gets to unlink its lockfiles on SIGTERM before SIGKILL.
GIT_TERM_GRACE_S = 5.0


def spawn(
    cwd: pathlib.Path,
    args: tuple[str, ...],
    env_extra: dict[str, str] | None,
    stdin_text: str | None,
    repo: pathlib.Path | None = None,
) -> tuple[tuple[str, ...], int, bytes, bytes]:
    """Run one hardened git command.

    Args:
        cwd: The directory git runs in: the repository, unless `repo` names it.
        args: The git subcommand and its arguments.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.
        repo: The repository whose drivers are blanked and whose index lock is awaited,
            when git runs elsewhere; None means `cwd`.

    Returns:
        The full argv, the exit code, stdout and stderr as bytes.

    Raises:
        GitError: The command timed out.
    """
    # GIT_TERMINAL_PROMPT=0 fails a credential prompt fast; LC_ALL=C keeps the stash
    # rescue's parse of git's prose valid. The operator's pathspec switches would turn the
    # review's `:(exclude)` magic into literal names, so only a caller's `env_extra` sets them.
    inherited = {k: v for k, v in os.environ.items() if k not in _PATHSPEC_ENV}
    env = child_env.without_provider_keys(
        {**inherited, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env_extra or {})}
    )
    hardening = git_policy.git_hardening_flags(repo or cwd)
    # git 2.53 dies rc=128 on the empty `diff.external` override, so every verb that renders
    # a diff forces the builtin renderer by flag; `blame` runs textconv too.
    argv = list(args)
    if argv and argv[0] in ("blame", "diff", "diff-tree", "log", "show"):
        argv[1:1] = git_policy.DIFF_SAFETY_FLAGS
    # Drivers are blanked on every op: a list of driver-running verbs would enumerate badness.
    full_argv = (git_policy.git(), *hardening, *argv)
    index_lock = (repo or cwd) / ".git" / "index.lock"
    lock_preexisted = index_lock.exists()
    proc = subprocess.Popen(
        full_argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        # Bytes: diff and show emit raw file bytes, and a latin-1 file is not binary to git.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # fsencode: a name decoded with surrogateescape goes back to git byte for byte.
        stdin_bytes = os.fsencode(stdin_text) if stdin_text is not None else None
        out, err = proc.communicate(input=stdin_bytes, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        # git's TERM handler unlinks its own lockfiles, so a lock left afterwards is another git's.
        proc.terminate()
        try:
            proc.communicate(timeout=GIT_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            # SIGKILL skips git's cleanup: clear the lock only when it appeared under this child.
            proc.kill()
            proc.communicate()
            if "GIT_INDEX_FILE" not in env and not lock_preexisted:
                with contextlib.suppress(OSError):
                    index_lock.unlink(missing_ok=True)
        raise errors.GitError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_S:.0f}s"
            " (a stuck filesystem or a held .git/index.lock?)"
        ) from exc
    return full_argv, proc.returncode, out, err


def failure(args: tuple[str, ...], returncode: int, stdout: str, stderr: str) -> errors.GitError:
    """Return a Git failure with its diagnostic and an output-free review reason.

    Args:
        args: The git subcommand and its arguments.
        returncode: The exit code.
        stdout: The command's stdout.
        stderr: The command's stderr.

    Returns:
        The full diagnostic and a safe reason naming only the command and exit status.
    """
    # `git commit` explains most failures on stdout, not stderr.
    stderr_msg = stderr.strip()
    stdout_msg = stdout.strip()
    if stderr_msg and stdout_msg:
        detail = f"{stderr_msg} | stdout: {stdout_msg}"
    else:
        detail = stderr_msg or stdout_msg or f"exit {returncode}"
    return errors.GitError(
        f"git {' '.join(args)} failed: {detail}",
        safe_message=(
            f"git {args[0]} failed (exit {returncode}); its diagnostic is withheld"
            " because it may quote hidden file content."
        ),
    )


def run(
    cwd: pathlib.Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
    stdin_text: str | None = None,
    repo: pathlib.Path | None = None,
) -> kinds.CommandResult:
    """Run one hardened git command in the repository.

    Args:
        cwd: The directory git runs in: the repository, unless `repo` names it.
        *args: The git subcommand and its arguments.
        check: Raise on a non-zero exit.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.
        repo: The repository, when git runs elsewhere; None means `cwd`.

    Returns:
        The command's result, its output decoded lossily.

    Raises:
        GitError: The command failed with check set, or timed out.
    """
    full_argv, returncode, out, err = spawn(cwd, args, env_extra, stdin_text, repo=repo)
    result = kinds.CommandResult(
        argv=full_argv,
        returncode=returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        duration_s=0.0,
    )
    if check and not result.ok:
        error = failure(args, returncode, result.stdout, result.stderr)
        raise error
    return result


def nul_fields(
    cwd: pathlib.Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> list[str]:
    """Run one `-z` listing and return its NUL-separated fields.

    A name that is not valid UTF-8 decodes with surrogateescape, so it reaches a later
    argv or stdin byte for byte; `terminal.printable_name` escapes it for display.

    Args:
        cwd: The repository.
        *args: The git subcommand and its arguments, `-z` among them.
        check: Raise on a non-zero exit.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.

    Returns:
        The fields, the empty one after a trailing NUL included.

    Raises:
        GitError: The command failed with check set, or timed out.
    """
    _, returncode, out, err = spawn(cwd, args, env_extra, stdin_text)
    if check and returncode != 0:
        error = failure(
            args, returncode, out.decode(errors="replace"), err.decode(errors="replace")
        )
        raise error
    return os.fsdecode(out).split("\0")
