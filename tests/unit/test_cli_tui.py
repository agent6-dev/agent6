# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""run_command approver bridge + TUI auto-spawn gating.

The textual TUI was fully built (modal, writes `approvals/<id>.answer`) but the
workflow side never read those answers and never auto-spawned the dashboard.
These cover the wiring that fixes that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent6.cli import run as runmod
from agent6.events import EventSink


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
    monkeypatch.setattr(runmod, "frontend_is_live", _live)
    monkeypatch.setattr(runmod, "read_answer", _ans_yes)
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_forbidden)
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
    assert approve("run `ls`?") is True
    assert _events_of(log, "approval.prompt")
    ans = _events_of(log, "approval.answer")[0]
    assert ans["approved"] is True
    assert ans["source"] == "tui"


def test_approver_falls_back_to_stdin_without_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(runmod, "frontend_is_live", _dead)
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_no)
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
    assert approve("x") is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_session_allows_every_later_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "allow session" (stdin returns "session") approves this command AND every
    # later one without prompting again -- across the run.
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(runmod, "frontend_is_live", _dead)
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_session)
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
    assert approve("first?") is True
    # A second prompt must NOT reach the stdin approver -- the session marker auto-passes.
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_forbidden)
    assert approve("second?") is True
    assert _events_of(log, "approval.answer")[-1]["source"] == "session"


def test_approver_tui_timeout_falls_back_to_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(runmod, "frontend_is_live", _live)
    monkeypatch.setattr(runmod, "read_answer", _ans_none)  # TUI died / timed out
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_yes)
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
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
        return runmod._should_spawn_tui(**kw)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(runmod, "_tui_available", _yes)
    monkeypatch.setattr(runmod.sys, "stdout", _FakeStdout(tty=True))
    # Headless by default: no --tui -> never spawn.
    assert should(tui=False, interactive=False, mode="run") is False
    # --tui on a TTY with textual + run mode -> spawn.
    assert should(tui=True, interactive=False, mode="run") is True
    # --tui asked for but can't honour -> warn and stay headless.
    assert should(tui=True, interactive=True, mode="run") is False
    assert should(tui=True, interactive=False, mode="plan") is False
    # textual not installed.
    monkeypatch.setattr(runmod, "_tui_available", _no)
    assert should(tui=True, interactive=False, mode="run") is False
    # non-TTY (benches / CI / pipes).
    monkeypatch.setattr(runmod, "_tui_available", _yes)
    monkeypatch.setattr(runmod.sys, "stdout", _FakeStdout(tty=False))
    assert should(tui=True, interactive=False, mode="run") is False


def test_stream_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    def modes(*, tui_enabled: bool) -> tuple[bool, bool]:
        return runmod._stream_modes(tui_enabled=tui_enabled)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.delenv("AGENT6_FORCE_STREAM", raising=False)
    monkeypatch.delenv("AGENT6_STREAM_TO_LOG", raising=False)

    # Headless, no env: the audited non-streaming path, no console echo.
    monkeypatch.setattr(runmod.sys, "stderr", _FakeStdout(tty=False))
    assert modes(tui_enabled=False) == (False, False)

    # Interactive stderr TTY: stream; echo only when the TUI does NOT own the term.
    monkeypatch.setattr(runmod.sys, "stderr", _FakeStdout(tty=True))
    assert modes(tui_enabled=False) == (True, True)  # plain ask/plan
    assert modes(tui_enabled=True) == (True, False)  # the TUI renders the deltas

    # AGENT6_FORCE_STREAM (bench/CI): stream AND echo even when headless.
    monkeypatch.setattr(runmod.sys, "stderr", _FakeStdout(tty=False))
    monkeypatch.setenv("AGENT6_FORCE_STREAM", "1")
    assert modes(tui_enabled=False) == (True, True)
    monkeypatch.delenv("AGENT6_FORCE_STREAM")

    # AGENT6_STREAM_TO_LOG (hub-watched headless run): emit the delta EVENTS only,
    # NO console echo -- the dashboard renders them; the stderr temp is discarded.
    monkeypatch.setenv("AGENT6_STREAM_TO_LOG", "1")
    assert modes(tui_enabled=False) == (True, False)


def test_tui_session_disabled_is_noop(tmp_path: Path) -> None:
    # enabled=False must not spawn anything or touch stdout.
    with runmod._tui_session(tmp_path, enabled=False):  # pyright: ignore[reportPrivateUsage]
        pass
    assert not (tmp_path / "tui_console.log").exists()


def test_approver_away_deny_auto_denies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Detach chose "deny all": every run_command is denied without prompting.
    from agent6.frontend.approval import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_forbidden)  # must NOT prompt
    set_away_mode(tmp_path, "deny")
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
    assert approve("rm -rf /") is False
    assert _events_of(log, "approval.answer")[0]["source"] == "away-deny"


def test_approver_away_wait_uses_reattached_front_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Detach chose "wait": approval blocks until a reattached front-end answers.
    from agent6.frontend.approval import set_away_mode

    log = tmp_path / "logs.jsonl"
    events = EventSink(log)
    monkeypatch.setattr(runmod, "frontend_is_live", _live)  # a front-end is attached
    monkeypatch.setattr(runmod, "read_answer", _ans_yes)  # and it approved
    monkeypatch.setattr(runmod, "_default_stdin_approver", _stdin_forbidden)  # never falls to stdin
    set_away_mode(tmp_path, "wait")
    approve = runmod._build_approver(tmp_path, events)  # pyright: ignore[reportPrivateUsage]
    assert approve("ls") is True
    assert _events_of(log, "approval.answer")[0]["source"] == "away-wait"
