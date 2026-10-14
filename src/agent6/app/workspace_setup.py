# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the environment of an isolated workspace before its first turn.

A `run --worktree` worktree, a fork's worktree, a lane's clone and a machine state's clone
hold tracked files only. `[parallel].setup_command` runs once in each, through the session's
own dispatcher, so it gets the jail every command of that run gets.
"""

from __future__ import annotations

import pathlib
import shlex

from agent6.app import reporter as app_reporter
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import dispatch, results
from agent6.tools import errors as tools_errors

# The lines of the command's output a failure's summary carries.
_TAIL_LINES = 20


def _output_tail(res: results.ExecResult) -> list[str]:
    """Return the last lines of the command's output, stdout before stderr."""
    lines = [line for part in (res.stdout, res.stderr) for line in part.rstrip().splitlines()]
    return lines[-_TAIL_LINES:]


def session_is_isolated(session_dir: pathlib.Path) -> bool:
    """Return whether the session works in a worktree or a lane's clone of its own."""
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return False
    return manifest.worktree is not None or manifest.parallel is not None


def run(
    argv: tuple[str, ...], *, dispatcher: dispatch.ToolDispatcher, reporter: app_reporter.Reporter
) -> str | None:
    """Run the setup command in the session's jail and say what failed.

    Args:
        argv: The setup command.
        dispatcher: The session's dispatcher, whose jail the command runs in.
        reporter: Receives the command line and its output.

    Returns:
        None when the command exited 0 or commands are withheld, else the failure, the
        output tail under it.
    """
    line = shlex.join(argv)
    if dispatcher.command_policy() == "no":
        reporter.note(f"commands are withheld, and the setup command is one ({line}): not run")
        return None
    reporter.note(f"setup command: {line}")
    try:
        res = dispatcher.run_setup(argv)
    except tools_errors.ToolError as exc:
        return f"the setup command did not run ({line}): {exc}"
    tail = _output_tail(res)
    if res.returncode != 0:
        how = "could not start" if res.exec_failed else f"failed (exit {res.returncode})"
        return "\n".join([f"the setup command {how}: {line}", *(f"    {t}" for t in tail)])
    for shown in tail:
        reporter.err(f"         {shown}")
    reporter.note(f"setup command finished in {res.duration_s:.1f}s")
    return None
