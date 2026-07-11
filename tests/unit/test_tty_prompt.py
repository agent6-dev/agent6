# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""tty_prompt talks to the controlling terminal (the getpass-style open).

A pty.fork child proves the /dev/tty path end to end: the prompt text lands on
the terminal and the typed reply comes back. This was broken since birth
(``open("/dev/tty", "r+")`` needs a seekable stream), so every prompt silently
used the stdin fallback, and ask_user -- which must never consume piped stdin --
always returned empty answers, even in a foreground interactive run.
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent6.tools.schema import UserQuestion
from agent6.ui.cli.run import _build_questioner  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)


def _drive_pty(child: Any, expect: bytes, reply: bytes) -> int:
    """Fork *child* under a fresh pty, wait for *expect* on the terminal, type
    *reply*, and return the child's exit code."""
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os._exit(child())
    buf = b""
    deadline = time.monotonic() + 15
    try:
        while expect not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if not ready:
                continue
            try:
                buf += os.read(master, 4096)
            except OSError:
                break
        assert expect in buf, f"prompt never appeared on the pty: {buf[-500:]!r}"
        os.write(master, reply)
        _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status)
    finally:
        os.close(master)


def test_tty_prompt_round_trips_on_the_controlling_terminal() -> None:
    def child() -> int:
        from agent6.ui.cli._steer import tty_prompt

        ans = tty_prompt("PICK> ", fall_back_to_stdin=False)
        return 0 if ans == "two" else 13

    assert _drive_pty(child, b"PICK>", b"two\n") == 0


def test_ask_one_stdin_prompts_and_maps_a_digit_to_its_option() -> None:
    def child() -> int:
        from agent6.ui.cli.run import _ask_one_stdin  # pyright: ignore[reportPrivateUsage]

        ans = _ask_one_stdin(UserQuestion(question="Which theme?", options=("alpha", "beta")))
        return 0 if ans == "beta" else 13

    assert _drive_pty(child, b"2) beta", b"2\n") == 0


def test_stdin_questioner_returns_none_without_a_terminal() -> None:
    # A new session has no controlling terminal, the true headless case; run it
    # in a subprocess so an interactively-run pytest (which HAS a /dev/tty)
    # cannot block on a real prompt.
    code = (
        "from agent6.ui.cli.run import _default_stdin_questioner\n"
        "from agent6.tools.schema import UserQuestion\n"
        "q = (UserQuestion(question='anyone there?'),)\n"
        "raise SystemExit(0 if _default_stdin_questioner(q) is None else 13)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        start_new_session=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()[-500:]


def test_questioner_marks_headless_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no front-end and no terminal, ask_user answers empty but says so:
    the question.answer event carries source=headless-default."""
    from agent6.ui.cli import run as run_mod

    def _no_tty(_q: tuple[UserQuestion, ...]) -> tuple[str, ...] | None:
        return None

    monkeypatch.setattr(run_mod, "_default_stdin_questioner", _no_tty)
    emitted: list[tuple[str, dict[str, Any]]] = []

    class _Events:
        def emit(self, event_type: str, **fields: Any) -> None:
            emitted.append((event_type, fields))

    ask = _build_questioner(tmp_path, _Events())  # type: ignore[arg-type]
    answers = ask((UserQuestion(question="pick?", options=("a", "b")),))
    assert answers == ("",)
    answer_events = [f for t, f in emitted if t == "question.answer"]
    assert answer_events and answer_events[0]["source"] == "headless-default"
