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
from agent6.ui.cli._interact import build_questioner

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
        from agent6.ui.cli._interact import ask_one_stdin

        ans = ask_one_stdin(UserQuestion(question="Which theme?", options=("alpha", "beta")))
        return 0 if ans == "beta" else 13

    assert _drive_pty(child, b"2) beta", b"2\n") == 0


def test_ask_one_stdin_cancel_on_tty_returns_none_not_a_reprompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On a tty, cancelling the radio (esc / Ctrl-C) returns None so the caller treats
    # it as unanswered -- it must NOT fall through to a numbered /dev/tty re-prompt
    # for the same question.
    from agent6.ui.cli import _interact as interact_mod
    from agent6.ui.cli._ptk_reader import RadioNav

    fell_back = {"hit": False}

    def _cancelled(*_a: Any, **_k: Any) -> object:  # radio ran, user cancelled
        return RadioNav.CANCEL

    def _boom(*_a: Any, **_k: Any) -> str:
        fell_back["hit"] = True
        return "SHOULD-NOT-BE-USED"

    monkeypatch.setattr(interact_mod, "on_tty", lambda: True)
    monkeypatch.setattr(interact_mod, "radio_select", _cancelled)
    monkeypatch.setattr(interact_mod, "_tty_prompt", _boom)
    got = interact_mod.ask_one_stdin(UserQuestion(question="pick", options=("a", "b")))
    assert got is None
    assert not fell_back["hit"]  # the numbered fallback was NOT invoked


def _series(
    monkeypatch: pytest.MonkeyPatch, radio_script: list[Any], prompt_script: list[str | None]
) -> Any:
    """Patch the widgets with scripted results and return the _interact module, so
    the one-question-at-a-time series logic is tested without a terminal."""
    from agent6.ui.cli import _interact as interact_mod

    radios, prompts = list(radio_script), list(prompt_script)
    monkeypatch.setattr(interact_mod, "on_tty", lambda: True)
    monkeypatch.setattr(interact_mod, "radio_select", lambda *_a, **_k: radios.pop(0))
    monkeypatch.setattr(interact_mod, "ptk_prompt", lambda *_a, **_k: prompts.pop(0))
    monkeypatch.setattr(interact_mod, "_tty_message", lambda *_a, **_k: None)
    return interact_mod


def test_series_answers_in_order_then_submits(monkeypatch: pytest.MonkeyPatch) -> None:
    # q0 is a radio, q1 free text; the review screen submits.
    mod = _series(monkeypatch, ["a", "Submit"], ["hello"])
    qs = (UserQuestion(question="q0", options=("a", "b")), UserQuestion(question="q1"))
    assert mod.default_stdin_questioner(qs) == ("a", "hello")


def test_series_back_revises_the_previous_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # answer q0, go BACK from q1, re-answer q0, answer q1, submit.
    from agent6.ui.cli._ptk_reader import RadioNav

    mod = _series(monkeypatch, ["a", RadioNav.BACK, "b", "c", "Submit"], [])
    qs = (
        UserQuestion(question="q0", options=("a", "b")),
        UserQuestion(question="q1", options=("c", "d")),
    )
    assert mod.default_stdin_questioner(qs) == ("b", "c")


def test_series_skip_leaves_an_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._ptk_reader import RadioNav

    mod = _series(monkeypatch, [RadioNav.SKIP, "x", "Submit"], [])
    qs = (
        UserQuestion(question="q0", options=("a", "b")),
        UserQuestion(question="q1", options=("x", "y")),
    )
    assert mod.default_stdin_questioner(qs) == ("", "x")


def test_series_review_can_go_back_and_revise(monkeypatch: pytest.MonkeyPatch) -> None:
    # "Go back and revise" at the review returns to the last question.
    mod = _series(monkeypatch, ["a", "c", "Go back and revise", "d", "Submit"], [])
    qs = (
        UserQuestion(question="q0", options=("a", "b")),
        UserQuestion(question="q1", options=("c", "d")),
    )
    assert mod.default_stdin_questioner(qs) == ("a", "d")


def test_series_ctrl_c_abandons_unanswered(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._ptk_reader import RadioNav

    mod = _series(monkeypatch, [RadioNav.CANCEL], [])
    qs = (
        UserQuestion(question="q0", options=("a", "b")),
        UserQuestion(question="q1", options=("c", "d")),
    )
    assert mod.default_stdin_questioner(qs) is None


def test_stdin_questioner_returns_none_without_a_terminal() -> None:
    # A new session has no controlling terminal, the true headless case; run it
    # in a subprocess so an interactively-run pytest (which HAS a /dev/tty)
    # cannot block on a real prompt.
    code = (
        "from agent6.ui.cli._interact import default_stdin_questioner\n"
        "from agent6.tools.schema import UserQuestion\n"
        "q = (UserQuestion(question='anyone there?'),)\n"
        "raise SystemExit(0 if default_stdin_questioner(q) is None else 13)\n"
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
    from agent6.ui.cli import _interact as interact_mod

    def _no_tty(_q: tuple[UserQuestion, ...]) -> tuple[str, ...] | None:
        return None

    monkeypatch.setattr(interact_mod, "default_stdin_questioner", _no_tty)
    emitted: list[tuple[str, dict[str, Any]]] = []

    class _Events:
        def emit(self, event_type: str, **fields: Any) -> None:
            emitted.append((event_type, fields))

    ask = build_questioner(tmp_path, _Events())  # type: ignore[arg-type]
    answers = ask((UserQuestion(question="pick?", options=("a", "b")),))
    assert answers == ("",)
    answer_events = [f for t, f in emitted if t == "question.answer"]
    assert answer_events and answer_events[0]["source"] == "headless-default"
