# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-session prompt: a CLI session asks for the next input before ending."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from agent6.app._setup import BudgetOverrides, SandboxOverrides
from agent6.directive import steer_problem
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.cli.resume import _cmd_resume
from agent6.viewmodel.listing import scan_session_log

# Free text is the next execution's operator instruction, as `--steer` carries it.
_NEXT_PROMPT = "next (/exit to finish): "
EXIT_COMMAND = "/exit"


def prompting_is_possible() -> bool:
    """Return whether an attended terminal has this process in the foreground.

    A tty is not enough: `agent6 run ... &` keeps one on stdin, and a read from a
    background process group raises SIGTTIN, suspending the job instead of finishing
    it. The foreground check also covers a tty with nobody at it (`docker run -t`).
    """
    if not sys.stdin.isatty():
        return False
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except OSError:
        return False


def follow_up_on_offer(session_dir: Path) -> bool:
    """Return whether the run can take a follow-up execution from here.

    Not after a detach (the run goes on in the background), an `/undo` (the fork it
    named is the continuation) or an `/exit` (asking again would reopen what the
    operator closed); each printed its own line.
    """
    scan = scan_session_log(session_dir / LOGS_NAME)
    return scan.finished and scan.end_reason not in ("undone", "steer_exit")


def end_of_session_prompt(
    *,
    rc: int,
    session_id: str,
    session_dir: Path | None = None,
    ask: Callable[[str], str],
    config_path: Path | None = None,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    model: str = "",
) -> int:
    """Keep the session going from the terminal until `/exit`.

    Each answer runs one resume execution carrying that text as the operator's
    instruction, under the invocation's own overrides. `/exit` or EOF prints the line
    that picks the session back up; nothing is sealed.

    Args:
        rc: The exit code of the execution that just ended.
        session_id: The session.
        session_dir: The session dir; when given, `follow_up_on_offer` decides whether to ask.
        ask: Reads one answer for a prompt.
        config_path: The invocation's `--config`.
        budget_overrides: The invocation's budget flags.
        sandbox_overrides: The invocation's sandbox flags.
        model: The invocation's `--model`.

    Returns:
        The last execution's exit code; an execution that refuses returns its own.
    """
    while True:
        try:
            answer = ask(_NEXT_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            answer = EXIT_COMMAND
        if answer == EXIT_COMMAND:
            print(f"\nresume with:  agent6 resume {session_id}")
            return rc
        if not answer:
            continue
        if (problem := steer_problem(answer)) is not None:
            print(f"[agent6] {problem}", file=sys.stderr)
            continue
        if answer.startswith("/") and len(answer.split()) == 1 and answer != "/undo":
            # A lone slash word is a composer command or a typo; sent, it would cost a model call.
            print(
                f"[agent6] {answer!r} is not sent as a task: this prompt takes a"
                " follow-up instruction (a new execution), /undo, or /exit; slash"
                " commands work in the pause menu and the TUI/web composers.",
                file=sys.stderr,
            )
            continue
        rc = _cmd_resume(
            config_path,
            session_id,
            force=False,
            steer=answer,
            budget_overrides=budget_overrides,
            sandbox_overrides=sandbox_overrides,
            model=model,
        )
        if rc != 0:
            return rc
        if session_dir is not None and not follow_up_on_offer(session_dir):
            return rc
