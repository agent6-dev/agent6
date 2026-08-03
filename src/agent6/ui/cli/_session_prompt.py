# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-session prompt: a CLI session does not end, it asks."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from agent6.ui.cli.resume import _cmd_resume

# Free text is the next leg's operator instruction (what ``--steer`` carries);
# `/exit` finishes. No other verbs until a second one earns its place.
_NEXT_PROMPT = "next (/exit to finish): "
EXIT_COMMAND = "/exit"


def prompting_is_possible() -> bool:
    """Whether the operator is here to answer: an attended terminal this
    process is in the FOREGROUND of.

    The same question every model prompt asks, answered in one place. Without a
    terminal there is nobody to type, so the session prints the resume line
    instead.

    A tty is not enough. `agent6 run ... &` keeps one on stdin, and reading it
    from a background process group raises SIGTTIN, which stops the job: the
    run suspended at the end instead of finishing. The foreground check also
    covers a tty allocated with nobody at it (`docker run -t`, some CI
    runners), where the read would block forever.
    """
    if not sys.stdin.isatty():
        return False
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except OSError:
        return False


def end_of_session_prompt(
    *,
    rc: int,
    session_id: str,
    ask: Callable[[str], str],
    config_path: Path | None = None,
) -> int:
    """Keep the session going from the terminal until ``/exit``.

    Each answer runs one resume leg carrying that text as the operator's
    instruction, so continuing needs no ``agent6 resume <id>`` retyping.
    ``/exit`` (or EOF) stops asking and prints the line that picks the session
    back up: nothing is sealed, and a finished session stays resumable like any
    other. A leg that refuses returns its own code rather than re-prompting
    over the failure.
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
        rc = _cmd_resume(config_path, session_id, force=False, steer=answer)
        if rc != 0:
            return rc
