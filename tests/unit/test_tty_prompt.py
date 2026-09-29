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
import pathlib
import pty
import select
import subprocess
import sys
import time
from typing import Any

import pytest

from agent6.tools import operator_prompts, schema
from agent6.ui.cli import _interact

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)


def _drive_pty(child: Any, expect: bytes, reply: bytes) -> int:
    """Fork the child under a fresh pty, wait for the text, type the reply, return its exit code."""
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
        from agent6.ui.cli import _steer

        ans = _steer.tty_prompt("PICK> ", fall_back_to_stdin=False)
        return 0 if ans == "two" else 13

    assert _drive_pty(child, b"PICK>", b"two\n") == 0


def test_tty_prompt_discards_type_ahead() -> None:
    # Text typed before the prompt existed must not be consumed as its answer:
    # a "/detach" typed during the "pausing after this step" window once rode
    # into the next run_command [y/N/a] approval and silently denied it (and a
    # buffered "y" would have silently approved).
    def child() -> int:
        import time as _t

        from agent6.ui.cli import _steer

        _t.sleep(0.5)  # let the parent stuff type-ahead into the pty first
        ans = _steer.tty_prompt("APPROVE> ", fall_back_to_stdin=False)
        return 0 if ans == "y" else 13

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os._exit(child())
    buf = b""
    deadline = time.monotonic() + 15
    try:
        os.write(master, b"/detach\n")  # type-ahead, before any prompt exists
        while b"APPROVE>" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if not ready:
                continue
            try:
                buf += os.read(master, 4096)
            except OSError:
                break
        assert b"APPROVE>" in buf, f"prompt never appeared: {buf[-500:]!r}"
        os.write(master, b"y\n")
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, "type-ahead was consumed as the answer"
    finally:
        os.close(master)


def test_ask_one_stdin_prompts_and_maps_a_digit_to_its_option() -> None:
    def child() -> int:

        ans = _interact.ask_one_stdin(
            schema.UserQuestion(question="Which theme?", options=("alpha", "beta"))
        )
        return 0 if ans == "beta" else 13

    assert _drive_pty(child, b"2) beta", b"2\n") == 0


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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no front-end and no terminal, ask_user answers empty with source=headless-default."""
    from agent6.ui.cli import _interact as interact_mod

    def _no_tty(_q: tuple[schema.UserQuestion, ...], **_kw: object) -> tuple[str, ...] | None:
        return None

    monkeypatch.setattr(interact_mod, "default_stdin_questioner", _no_tty)
    emitted: list[tuple[str, dict[str, Any]]] = []

    class _Events:
        def emit(self, event_type: str, **fields: Any) -> None:
            emitted.append((event_type, fields))

    ask = operator_prompts.OperatorPrompts(
        questioner=_interact.build_questioner(tmp_path),
        journal=_Events().emit,
        session_dir=tmp_path,
    ).ask
    answers = ask((schema.UserQuestion(question="pick?", options=("a", "b")),)).answers
    assert answers == ("",)
    answer_events = [f for t, f in emitted if t == "question.answer"]
    assert answer_events and answer_events[0]["source"] == "headless-default"


def test_a_wait_park_narrates_the_attach_remedy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached-wait session parked on ask_user says where to answer.

    It printed nothing, so a piped `resume` looked hung for 300s.
    """
    from agent6.ui.cli import _interact as interact_mod

    lines: list[str] = []
    monkeypatch.setattr(interact_mod, "tty_message", lines.append)

    def _away(_d: object) -> str:
        return "wait"

    monkeypatch.setattr(interact_mod, "away_mode", _away)

    def _reply(_d: object, _r: object) -> tuple[str, ...]:
        return ("yes",)

    monkeypatch.setattr(interact_mod, "await_frontend_reply", _reply)

    class _Events:
        def emit(self, event_type: str, **fields: Any) -> None:
            pass

    ask = operator_prompts.OperatorPrompts(
        questioner=_interact.build_questioner(tmp_path),
        journal=_Events().emit,
        session_dir=tmp_path,
    ).ask
    ask((schema.UserQuestion(question="pick?", options=("a", "b")),))
    assert any("question awaits a front-end" in ln and "agent6 attach" in ln for ln in lines)

    lines.clear()
    approve = operator_prompts.OperatorPrompts(
        approver=_interact.build_approver(tmp_path),
        journal=_Events().emit,
        session_dir=tmp_path,
    ).approve
    approve("Allow run_command: ls", scope=None)
    assert any("approval awaits a front-end" in ln and "agent6 attach" in ln for ln in lines)


def test_tty_prompt_ends_once_until_holds(tmp_path: pathlib.Path) -> None:
    """A prompt answered by another route ends with None, and the partial line is discarded."""
    flag = tmp_path / "answered"

    def child() -> int:
        from agent6.ui.cli import _steer

        ans = _steer.tty_prompt("PICK> ", fall_back_to_stdin=False, until=flag.exists)
        if ans is not None:
            return 13
        # The half-typed "tw" must not ride into the next prompt as its answer.
        nxt = _steer.tty_prompt("NEXT> ", fall_back_to_stdin=False)
        return 0 if nxt == "ok" else 14

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os._exit(child())
    buf = b""
    deadline = time.monotonic() + 15
    try:
        while b"PICK>" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                buf += os.read(master, 4096)
        assert b"PICK>" in buf, f"prompt never appeared: {buf[-500:]!r}"
        os.write(master, b"tw")  # a partial line, no newline
        time.sleep(0.3)
        flag.write_text("1", encoding="utf-8")
        while b"NEXT>" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                buf += os.read(master, 4096)
        assert b"NEXT>" in buf, f"second prompt never appeared: {buf[-500:]!r}"
        os.write(master, b"ok\n")
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, buf[-500:]
    finally:
        os.close(master)


def test_a_filed_answer_ends_the_terminal_prompt(tmp_path: pathlib.Path) -> None:
    """A foreground run blocked on its terminal takes an answer written over the file bridge.

    The approval and the question both read it and the journal names the source; every other
    seat's "answered" was a lie while the run waited on the terminal.
    """
    from agent6.sessions import ipc

    session_dir = tmp_path / "run"
    session_dir.mkdir()

    def child() -> int:
        emitted: list[dict[str, Any]] = []

        def _emit(event_type: str, **fields: Any) -> None:
            if event_type.endswith(".answer"):
                emitted.append(fields)

        prompts = operator_prompts.OperatorPrompts(
            approver=_interact.build_approver(session_dir),
            questioner=_interact.build_questioner(session_dir),
            journal=_emit,
            session_dir=session_dir,
        )
        approved = prompts.approve("Allow run_command: ls", scope="command")
        answers = prompts.ask((schema.UserQuestion(question="port?"),)).answers
        if not approved or answers != ("9090",):
            return 13
        return 0 if all(f.get("source") == "frontend" for f in emitted) else 14

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os._exit(child())
    buf = b""
    deadline = time.monotonic() + 15
    try:
        while b"[y/N/a/d]" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                buf += os.read(master, 4096)
        assert b"[y/N/a/d]" in buf, f"approval never prompted: {buf[-500:]!r}"
        ipc.write_answer(session_dir, "approval-1", "yes")
        while b"port?" not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                buf += os.read(master, 4096)
        assert b"port?" in buf, f"question never prompted: {buf[-500:]!r}"
        ipc.write_question_answers(session_dir, "question-1", ["9090"])
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, buf[-800:]
        assert b"answered elsewhere" in buf
    finally:
        os.close(master)


def test_the_pause_menu_takes_a_steer_written_while_it_is_open(tmp_path: pathlib.Path) -> None:
    """The Ctrl-C pause menu polls the steer file and takes a front-end's steer."""
    from agent6.sessions import ipc

    session_dir = tmp_path / "run"
    session_dir.mkdir()

    def child() -> int:
        # pytest's capture replaced sys.stdin/stdout; the pty is on fds 0-2.
        sys.stdin = open(0, closefd=False)  # noqa: SIM115
        sys.stdout = open(1, "w", closefd=False)  # noqa: SIM115
        from agent6.ui.cli import _steer_menu

        return 0 if _steer_menu.pause_menu(session_dir) == "from the web" else 13

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os._exit(child())
    buf = b""
    deadline = time.monotonic() + 15

    def drain_until(marker: bytes) -> None:
        nonlocal buf
        while marker not in buf and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                try:
                    buf += os.read(master, 4096)
                except OSError:
                    return

    try:
        drain_until(b"paused:")
        assert b"paused:" in buf, f"the menu never opened: {buf[-500:]!r}"
        os.write(master, b"half a")  # the operator was mid-word
        time.sleep(0.3)
        ipc.submit_steer(session_dir, "from the web")
        drain_until(b"steer arrived")
        if b"steer arrived" not in buf:
            os.kill(pid, 9)  # the menu never looked at the file: do not hang
        _, status = os.waitpid(pid, 0)
        assert b"steer arrived" in buf, f"the menu kept waiting on the keyboard: {buf[-800:]!r}"
        assert os.waitstatus_to_exitcode(status) == 0, buf[-800:]
    finally:
        os.close(master)


def test_detach_away_mode_asks_on_the_terminal_with_stdin_redirected(
    tmp_path: pathlib.Path,
) -> None:
    """A foreground run with stdin redirected still asks the away-mode question on /dev/tty.

    Gated on `sys.stdin.isatty()`, the detach recorded `wait` in silence.
    """

    def child() -> int:
        from agent6.sessions import ipc

        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
        assert not sys.stdin.isatty()
        _interact.prompt_detach_away_mode(tmp_path, (ipc.COMMAND_SCOPE,))
        if ipc.away_mode(tmp_path) != "" or not ipc.session_allow_set(tmp_path, ipc.COMMAND_SCOPE):
            return 13  # the answer typed at the prompt was not recorded
        return 0

    assert _drive_pty(child, b"While away", b"a\n") == 0
