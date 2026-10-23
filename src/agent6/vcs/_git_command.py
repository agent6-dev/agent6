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

# How long a timed-out git gets to unlink its lockfiles on SIGTERM before SIGKILL.
GIT_TERM_GRACE_S = 5.0


def spawn(
    cwd: pathlib.Path,
    args: tuple[str, ...],
    env_extra: dict[str, str] | None,
    stdin_text: str | None,
) -> tuple[tuple[str, ...], int, bytes, bytes]:
    """Run one hardened git command.

    Args:
        cwd: The repository.
        args: The git subcommand and its arguments.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.

    Returns:
        The full argv, the exit code, stdout and stderr as bytes.

    Raises:
        GitError: The command timed out.
    """
    # GIT_TERMINAL_PROMPT=0 fails a credential prompt fast; LC_ALL=C keeps the stash
    # rescue's parse of git's prose valid.
    env = child_env.without_provider_keys(
        {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env_extra or {})}
    )
    hardening = git_policy.git_hardening_flags(cwd)
    # git 2.53 dies rc=128 on the empty `diff.external` override, so every verb that renders
    # a diff forces the builtin renderer by flag; `blame` runs textconv too.
    argv = list(args)
    if argv and argv[0] in ("blame", "diff", "diff-tree", "log", "show"):
        argv[1:1] = git_policy.DIFF_SAFETY_FLAGS
    # Drivers are blanked on every op: a list of driver-running verbs would enumerate badness.
    full_argv = (git_policy.git(), *hardening, *argv)
    index_lock = cwd / ".git" / "index.lock"
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
            if not lock_preexisted:
                with contextlib.suppress(OSError):
                    index_lock.unlink(missing_ok=True)
        raise errors.GitError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_S:.0f}s"
            " (a stuck filesystem or a held .git/index.lock?)"
        ) from exc
    return full_argv, proc.returncode, out, err


def failure(args: tuple[str, ...], returncode: int, stdout: str, stderr: str) -> str:
    """Return the message for a failed git command, with its own explanation.

    Args:
        args: The git subcommand and its arguments.
        returncode: The exit code.
        stdout: The command's stdout.
        stderr: The command's stderr.

    Returns:
        `git <args> failed:` and git's explanation, stderr before stdout; the exit code
        when both are empty.
    """
    # `git commit` explains most failures on stdout, not stderr.
    stderr_msg = stderr.strip()
    stdout_msg = stdout.strip()
    if stderr_msg and stdout_msg:
        detail = f"{stderr_msg} | stdout: {stdout_msg}"
    else:
        detail = stderr_msg or stdout_msg or f"exit {returncode}"
    return f"git {' '.join(args)} failed: {detail}"


def run(
    cwd: pathlib.Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> kinds.CommandResult:
    """Run one hardened git command in the repository.

    Args:
        cwd: The repository.
        *args: The git subcommand and its arguments.
        check: Raise on a non-zero exit.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.

    Returns:
        The command's result, its output decoded lossily.

    Raises:
        GitError: The command failed with check set, or timed out.
    """
    full_argv, returncode, out, err = spawn(cwd, args, env_extra, stdin_text)
    result = kinds.CommandResult(
        argv=full_argv,
        returncode=returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        duration_s=0.0,
    )
    if check and not result.ok:
        raise errors.GitError(failure(args, returncode, result.stdout, result.stderr))
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
        raise errors.GitError(
            failure(args, returncode, out.decode(errors="replace"), err.decode(errors="replace"))
        )
    return os.fsdecode(out).split("\0")
