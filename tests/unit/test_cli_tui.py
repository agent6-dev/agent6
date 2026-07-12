# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""run_command approver bridge + TUI auto-spawn gating.

The textual TUI was fully built (modal, writes `approvals/<id>.answer`) but the
workflow side never read those answers and never auto-spawned the dashboard.
These cover the wiring that fixes that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent6.events import EventSink
from agent6.ui.cli import _interact as interactmod
from agent6.ui.cli import _live as livemod


def _events_of(log: Path, type_: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("type") == type_:
            out.append(obj)
    return out


def _live(_d: object) -> bool:
    return True


def _dead(_d: object) -> bool:
    return False


def _ans_yes(_d: object, _pid: object, **_k: object) -> bool:
    return True


def _ans_none(_d: object, _pid: object, **_k: object) -> bool | None:
    return None


def _stdin_no(_p: object) -> str:
    return "no"


def _stdin_yes(_p: object) -> str:
    return "yes"


def _stdin_forbidden(_p: object) -> str:
    pytest.fail("stdin approver must not be used")


def _stdin_session(_p: object) -> str:
    return "session"


def test_approver_uses_tui_answer_when_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("run `ls`?") is True
    assert _events_of(log, "approval.prompt")
    ans = _events_of(log, "approval.answer")[0]
    assert ans["approved"] is True
    assert ans["source"] == "frontend"


def test_approver_does_not_consume_an_answer_written_before_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A premature /api/run/<id>/approve (ids are predictable counters) pre-writes
    # approvals/approval-1.answer before the run reaches its first approval. The
    # approver must clear that stale slot before emitting the prompt, so it is
    # not silently consumed as an auto-approval. Uses the REAL read_answer (short
    # timeout) so this exercises the actual file-bridge ordering.
    import functools

    from agent6.ui.bridge.approval import read_answer, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(read_answer, timeout_s=0.4, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground stdin path
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    write_answer(tmp_path, "approval-1", approved=True)  # the premature POST
    approve = interactmod.build_approver(tmp_path, events)
    # The premature "yes" is cleared before the prompt; read_answer finds nothing
    # and times out, so it falls back to stdin (which denies) -- NOT auto-approved.
    assert approve("run `curl evil`?") is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_consumes_an_answer_written_after_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The legitimate path: the front-end writes the answer only after it renders
    # the emitted prompt. A writer thread does exactly that; the answer is honored.
    import functools
    import threading
    import time

    from agent6.ui.bridge.approval import read_answer, write_answer

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(read_answer, timeout_s=3.0, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)

    def writer() -> None:
        time.sleep(0.3)  # after the prompt is emitted and the poll starts
        write_answer(tmp_path, "approval-1", approved=True)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("run `ls`?") is True
    t.join(timeout=2)
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def _tty(_: object = None) -> bool:
    return True  # simulate a controlling terminal (foreground stdin path)


def test_approver_falls_back_to_stdin_without_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("x") is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_headless_no_frontend_waits_not_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A web/hub-spawned run (no terminal, no away-mode, no front-end attached
    # right now) WAITS for a front-end to attach rather than denying -- deny
    # discards the run's work. A writer thread attaches + answers after a beat.
    import threading
    import time

    from agent6.ui.bridge.approval import write_answer, write_frontend_pid

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    # Use the REAL frontend_is_live: nothing is attached at approve() time (the
    # writer sleeps first), so the approver reaches the wait path; once the
    # writer registers frontend.pid, the wait picks up its answer.
    monkeypatch.setattr(interactmod, "_has_controlling_tty", lambda: False)  # headless
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin

    def attach_and_answer() -> None:
        time.sleep(0.3)
        write_frontend_pid(tmp_path, os.getpid())
        write_answer(tmp_path, "approval-1", approved=True)

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("rm -rf build") is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_approver_session_allows_every_later_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "allow session" (stdin returns "session") approves this command AND every
    # later one without prompting again -- across the run.
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_session)
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("first?") is True
    # A second prompt must NOT reach the stdin approver -- the session marker auto-passes.
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    assert approve("second?") is True
    assert _events_of(log, "approval.answer")[-1]["source"] == "session"


def test_approver_tui_timeout_falls_back_to_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_none)  # TUI died / timed out
    monkeypatch.setattr(interactmod, "_has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_yes)
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("x") is True
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


class _FakeStdout:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _yes() -> bool:
    return True


def _no() -> bool:
    return False


def test_should_spawn_tui_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    def should(**kw: Any) -> bool:
        return livemod.should_spawn_tui(**kw)

    monkeypatch.setattr(livemod, "_tui_available", _yes)
    monkeypatch.setattr(livemod.sys, "stdout", _FakeStdout(tty=True))
    # Headless by default: no --tui -> never spawn.
    assert should(tui=False, interactive=False, mode="run") is False
    # --tui on a TTY with textual + run mode -> spawn.
    assert should(tui=True, interactive=False, mode="run") is True
    # --tui asked for but can't honour -> warn and stay headless.
    assert should(tui=True, interactive=True, mode="run") is False
    assert should(tui=True, interactive=False, mode="plan") is False
    # textual not installed.
    monkeypatch.setattr(livemod, "_tui_available", _no)
    assert should(tui=True, interactive=False, mode="run") is False
    # non-TTY (benches / CI / pipes).
    monkeypatch.setattr(livemod, "_tui_available", _yes)
    monkeypatch.setattr(livemod.sys, "stdout", _FakeStdout(tty=False))
    assert should(tui=True, interactive=False, mode="run") is False


def test_stream_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    def modes(*, tui_enabled: bool) -> tuple[bool, bool]:
        return livemod.stream_modes(tui_enabled=tui_enabled)

    monkeypatch.delenv("AGENT6_FORCE_STREAM", raising=False)
    monkeypatch.delenv("AGENT6_STREAM_TO_LOG", raising=False)

    # Headless, no env: the audited non-streaming path, no console echo.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=False))
    assert modes(tui_enabled=False) == (False, False)

    # Interactive stderr TTY: stream; echo only when the TUI does NOT own the term.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=True))
    assert modes(tui_enabled=False) == (True, True)  # plain ask/plan
    assert modes(tui_enabled=True) == (True, False)  # the TUI renders the deltas

    # AGENT6_FORCE_STREAM (bench/CI): stream AND echo even when headless.
    monkeypatch.setattr(livemod.sys, "stderr", _FakeStdout(tty=False))
    monkeypatch.setenv("AGENT6_FORCE_STREAM", "1")
    assert modes(tui_enabled=False) == (True, True)
    monkeypatch.delenv("AGENT6_FORCE_STREAM")

    # AGENT6_STREAM_TO_LOG (hub-watched headless run): emit the delta EVENTS only,
    # NO console echo -- the dashboard renders them; the stderr temp is discarded.
    monkeypatch.setenv("AGENT6_STREAM_TO_LOG", "1")
    assert modes(tui_enabled=False) == (True, False)


def test_tui_session_disabled_is_noop(tmp_path: Path) -> None:
    # enabled=False must not spawn anything or touch stdout.
    with livemod.tui_session(tmp_path, enabled=False):
        pass
    assert not (tmp_path / "tui_console.log").exists()


def test_spawned_away_default_sets_wait_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A front-end launcher (web/TUI hub) sets AGENT6_DETACHED_AWAY so a spawned,
    # terminal-less run WAITS for a viewer instead of fabricating empty answers.
    from agent6.ui.bridge.approval import away_mode
    from agent6.ui.cli.run import _apply_spawned_away_default  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    _apply_spawned_away_default(tmp_path)
    assert away_mode(tmp_path) == "wait"


def test_spawned_away_default_is_noop_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pure headless run (no launcher, no env) is untouched, keeping its
    # non-hanging default so CI never blocks on an unanswerable question.
    from agent6.ui.bridge.approval import away_mode
    from agent6.ui.cli.run import _apply_spawned_away_default  # pyright: ignore[reportPrivateUsage]

    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    _apply_spawned_away_default(tmp_path)
    assert away_mode(tmp_path) == ""


def test_approver_away_deny_auto_denies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Detach chose "deny all": every run_command is denied without prompting.
    from agent6.ui.bridge.approval import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # must NOT prompt
    set_away_mode(tmp_path, "deny")
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("rm -rf /") is False
    assert _events_of(log, "approval.answer")[0]["source"] == "away-deny"


def test_approver_live_front_end_wins_over_away_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A live front-end (a re-attached watch/TUI/web) is always asked, in its own
    # UI, regardless of the detach away-mode -- away-mode governs only the window
    # when nothing is attached. Even under away="deny", a live front-end answers.
    from agent6.ui.bridge.approval import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)  # a front-end is attached
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)  # and it approved
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # no stdin fall
    set_away_mode(tmp_path, "deny")  # would deny if the front-end did NOT win
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("ls") is True  # the attached front-end approved despite away=deny
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def test_approver_away_wait_blocks_for_a_front_end_when_none_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # away="wait" with NOTHING attached: block until a front-end attaches and
    # answers. A writer thread attaches (frontend.pid) + answers after a beat.
    import threading
    import time

    from agent6.ui.bridge.approval import set_away_mode, write_answer, write_frontend_pid

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin
    set_away_mode(tmp_path, "wait")

    def attach_and_answer() -> None:
        time.sleep(0.3)
        write_frontend_pid(tmp_path, os.getpid())  # a front-end re-attaches
        write_answer(tmp_path, "approval-1", approved=True)  # and answers

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = interactmod.build_approver(tmp_path, events)
    assert approve("ls") is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"
