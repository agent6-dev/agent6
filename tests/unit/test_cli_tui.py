# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run_command approver bridge and the TUI auto-spawn gating.

The harness reads the `approvals/<id>.answer` files the TUI writes and spawns the dashboard.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any

import pytest

from agent6 import events as agent6_events
from agent6.sessions import ipc
from agent6.tools import operator_prompts, schema
from agent6.ui import steer
from agent6.ui.cli import _interact as interactmod
from agent6.ui.cli import _live as livemod


def _events_of(log: pathlib.Path, type_: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("type") == type_:
            out.append(obj)
    return out


def _prompts(
    session_dir: pathlib.Path,
    events: agent6_events.EventSink,
    steer_cell: list[steer.SteerState | None] | None = None,
) -> operator_prompts.OperatorPrompts:
    """The gate over the CLI's own approver and questioner, the pairing a run wires.

    Journaling into events.
    """
    return operator_prompts.OperatorPrompts(
        approver=interactmod.build_approver(session_dir, None, steer_cell),
        questioner=interactmod.build_questioner(session_dir),
        journal=events.emit,
        session_dir=session_dir,
    )


def _live(_d: object) -> bool:
    return True


def _dead(_d: object) -> bool:
    return False


def _ans_yes(_d: object, _pid: object, **_k: object) -> str:
    return "yes"


def _ans_none(_d: object, _pid: object, **_k: object) -> str | None:
    return None


def _stdin_no(_p: object, **_k: object) -> str:
    return "no"


def _stdin_yes(_p: object, **_k: object) -> str:
    return "yes"


def _stdin_forbidden(_p: object, **_k: object) -> str:
    pytest.fail("stdin approver must not be used")


def _stdin_session(_p: object, **_k: object) -> str:
    return "session"


def test_approver_uses_tui_answer_when_live(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    approve = _prompts(tmp_path, events).approve
    assert approve("run `ls`?", scope=ipc.COMMAND_SCOPE) is True
    assert _events_of(log, "approval.prompt")
    ans = _events_of(log, "approval.answer")[0]
    assert ans["approved"] is True
    assert ans["source"] == "frontend"


def test_approver_does_not_consume_an_answer_written_before_the_prompt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A premature approve pre-writes approval-1.answer; the gate clears the stale slot first.
    import functools

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(ipc.read_answer, timeout_s=0.4, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "has_controlling_tty", _tty)  # foreground stdin path
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    ipc.write_answer(tmp_path, "approval-1", "yes")  # the premature POST
    approve = _prompts(tmp_path, events).approve
    # The premature "yes" is cleared; read_answer times out and falls back to stdin, which denies.
    assert approve("run `curl evil`?", scope=ipc.COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_consumes_an_answer_written_after_the_prompt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The legitimate path: the front-end writes the answer after it renders the prompt.
    import functools
    import threading
    import time

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(
        interactmod, "read_answer", functools.partial(ipc.read_answer, timeout_s=3.0, poll_s=0.05)
    )
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)

    def writer() -> None:
        time.sleep(0.3)  # after the prompt is emitted and the poll starts
        ipc.write_answer(tmp_path, "approval-1", "yes")

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    approve = _prompts(tmp_path, events).approve
    assert approve("run `ls`?", scope=ipc.COMMAND_SCOPE) is True
    t.join(timeout=2)
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def _tty(_: object = None) -> bool:
    return True  # simulate a controlling terminal (foreground stdin path)


def test_approver_falls_back_to_stdin_without_tui(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_no)
    approve = _prompts(tmp_path, events).approve
    assert approve("x", scope=ipc.COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "stdin"


def test_approver_headless_no_frontend_waits_not_denies(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No terminal, no away-mode, nothing attached: the run waits for a front-end, never denies.
    import threading
    import time

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    # The real frontend_is_live: nothing is attached at approve() time, so the wait path runs.
    monkeypatch.setattr(interactmod, "has_controlling_tty", lambda: False)  # headless
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin

    def attach_and_answer() -> None:
        time.sleep(0.3)
        ipc.register_frontend(tmp_path, os.getpid())
        ipc.write_answer(tmp_path, "approval-1", "yes")

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("rm -rf build", scope=ipc.COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_approver_session_allows_every_later_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "allow session" approves this command and every later one across the run.
    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_session)
    approve = _prompts(tmp_path, events).approve
    assert approve("first?", scope=ipc.COMMAND_SCOPE) is True
    # A second prompt never reaches the stdin approver: the session marker auto-passes.
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    assert approve("second?", scope=ipc.COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[-1]["source"] == "session"


def test_approver_tui_timeout_falls_back_to_stdin(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)
    monkeypatch.setattr(interactmod, "read_answer", _ans_none)  # TUI died / timed out
    monkeypatch.setattr(interactmod, "has_controlling_tty", _tty)  # foreground
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_yes)
    approve = _prompts(tmp_path, events).approve
    assert approve("x", scope=ipc.COMMAND_SCOPE) is True
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
    # A planning run opens the same view; an ask stays text, its answer being the deliverable.
    assert should(tui=True, interactive=False, mode="plan") is True
    assert should(tui=True, interactive=False, mode="ask") is False
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

    # AGENT6_STREAM_TO_LOG emits the delta events only, no console echo; the dashboard renders them.
    monkeypatch.setenv("AGENT6_STREAM_TO_LOG", "1")
    assert modes(tui_enabled=False) == (True, False)


def test_tui_session_disabled_is_noop(tmp_path: pathlib.Path) -> None:
    # enabled=False must not spawn anything or touch stdout.
    with livemod.tui_session(tmp_path, enabled=False):
        pass
    assert not (tmp_path / "tui_console.log").exists()


def test_spawned_away_default_sets_wait_from_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A front-end launcher sets AGENT6_DETACHED_AWAY so a terminal-less run waits for a viewer.
    from agent6.app import frontend

    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    frontend.apply_spawned_away_default(tmp_path, (ipc.COMMAND_SCOPE,))
    assert ipc.away_mode(tmp_path) == "wait"


def test_spawned_away_default_approve_reuses_session_allow(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT6_DETACHED_AWAY=approve maps to the session-allow marker.

    Like the interactive detach prompt.

    away.mode's vocabulary is deny|wait; "approve" written there falls into the wait branch.
    """
    from agent6.app import frontend

    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "approve")
    frontend.apply_spawned_away_default(tmp_path, (ipc.COMMAND_SCOPE,))
    assert ipc.session_allow_set(tmp_path, ipc.COMMAND_SCOPE) is True
    assert ipc.away_mode(tmp_path) == ""  # approve is never stored in away.mode


def test_set_away_mode_rejects_values_outside_its_vocabulary(tmp_path: pathlib.Path) -> None:
    # away.mode's contract is deny|wait; anything else fails at the writer.

    with pytest.raises(ValueError, match="deny"):
        ipc.set_away_mode(tmp_path, "approve")


def test_spawned_away_default_is_noop_without_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pure headless run keeps its non-hanging default, so CI never blocks on a question.
    from agent6.app import frontend

    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    frontend.apply_spawned_away_default(tmp_path, (ipc.COMMAND_SCOPE,))
    assert ipc.away_mode(tmp_path) == ""


def test_approver_away_deny_auto_denies(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Detach chose "deny all": every run_command is denied without prompting.

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # must NOT prompt
    ipc.set_away_mode(tmp_path, "deny")
    approve = _prompts(tmp_path, events).approve
    assert approve("rm -rf /", scope=ipc.COMMAND_SCOPE) is False
    assert _events_of(log, "approval.answer")[0]["source"] == "away-deny"


def test_approver_live_front_end_wins_over_away_mode(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A live front-end is always asked in its own UI; away-mode governs only the unattended window.

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "frontend_is_live", _live)  # a front-end is attached
    monkeypatch.setattr(interactmod, "read_answer", _ans_yes)  # and it approved
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # no stdin fall
    ipc.set_away_mode(tmp_path, "deny")  # would deny if the front-end did NOT win
    approve = _prompts(tmp_path, events).approve
    assert (
        approve("ls", scope=ipc.COMMAND_SCOPE) is True
    )  # the attached front-end approved despite away=deny
    assert _events_of(log, "approval.answer")[0]["source"] == "frontend"


def test_approver_away_wait_blocks_for_a_front_end_when_none_attached(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # away="wait" with nothing attached blocks until a front-end attaches and answers.
    import threading
    import time

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)  # never stdin
    ipc.set_away_mode(tmp_path, "wait")

    def attach_and_answer() -> None:
        time.sleep(0.3)
        ipc.register_frontend(tmp_path, os.getpid())  # a front-end re-attaches
        ipc.write_answer(tmp_path, "approval-1", "yes")  # and answers

    threading.Thread(target=attach_and_answer, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("ls", scope=ipc.COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_a_stop_request_ends_an_away_wait(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop request breaks an away-wait.

    The questioner returns empty answers and the run parks or denies.

    A run blocked before its start has no step for `stop --after-step` to stop after.
    """
    import threading
    import time

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)
    ipc.set_away_mode(tmp_path, "wait")

    def stop_it() -> None:
        time.sleep(0.3)
        ipc.request_stop(tmp_path)

    threading.Thread(target=stop_it, daemon=True).start()
    ask = _prompts(tmp_path, events).ask
    started = time.monotonic()
    answer = ask((schema.UserQuestion(question="stash?", options=("stash", "cancel")),))
    assert answer.answers == ("",) and answer.unseen
    assert time.monotonic() - started < 10


def test_spawned_away_default_does_not_overwrite_the_operators_choice(
    tmp_path: pathlib.Path,
) -> None:
    """The spawned away default never overwrites the operator's explicit detach choice.

    A `deny` chosen on detach stays `deny` under AGENT6_DETACHED_AWAY=wait.
    """
    import os

    from agent6.app import frontend

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    ipc.set_away_mode(session_dir, "deny")  # the operator's detach answer
    old = os.environ.get("AGENT6_DETACHED_AWAY")
    os.environ["AGENT6_DETACHED_AWAY"] = "wait"  # what the spawned resume carries
    try:
        frontend.apply_spawned_away_default(session_dir, (ipc.COMMAND_SCOPE,))
        assert ipc.away_mode(session_dir) == "deny"
        # With nothing chosen, the launcher's default still applies.
        other = tmp_path / "other"
        other.mkdir()
        frontend.apply_spawned_away_default(other, (ipc.COMMAND_SCOPE,))
        assert ipc.away_mode(other) == "wait"
    finally:
        if old is None:
            del os.environ["AGENT6_DETACHED_AWAY"]
        else:
            os.environ["AGENT6_DETACHED_AWAY"] = old


def test_approver_wait_consumes_a_claimless_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approver wait consumes an answer written with no front-end claim.

    The web UI writes answers without ever registering a claim.
    """
    import threading
    import time

    log = tmp_path / "logs.jsonl"
    events = agent6_events.EventSink(log)
    monkeypatch.setattr(interactmod, "has_controlling_tty", lambda: False)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_forbidden)

    def answer_never_claiming() -> None:
        time.sleep(0.3)
        ipc.write_answer(tmp_path, "approval-1", "yes")  # no register_frontend

    def abort_if_wedged() -> None:
        # A wait loop that never reads a claim-less answer fails fast instead of hanging the suite.
        time.sleep(15)
        ipc.write_steer_answer(tmp_path, "abort")

    threading.Thread(target=answer_never_claiming, daemon=True).start()
    threading.Thread(target=abort_if_wedged, daemon=True).start()
    approve = _prompts(tmp_path, events).approve
    assert approve("run_verify_command", scope=ipc.COMMAND_SCOPE) is True
    assert _events_of(log, "approval.answer")[0]["source"] == "await-frontend"


def test_stdin_approver_renders_the_command_on_its_own_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stdin approver renders the command indented on its own lines.

    A blank line precedes the answer line.
    """
    from agent6.ui.cli import _interact as interactmod

    seen: list[str] = []
    plains: list[object] = []

    def _capture(rendered: str, **kw: object) -> str:
        seen.append(rendered)
        plains.append(kw.get("plain"))
        return "y"

    monkeypatch.setattr(interactmod, "tty_prompt", _capture)
    assert interactmod.default_stdin_approver("Allow run_command: git log --stat -5") == "yes"
    rendered = seen[0]
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    assert plain.startswith("? Allow run_command:\n\n    git log --stat -5\n\n  [y/N/a/d]")
    # The console vocabulary: a bold yellow ? marks the question.
    assert "\x1b[1m\x1b[33m?" in rendered
    # The stdin fallback (stdout may be a pipe) gets the same text unstyled.
    assert plains[0] == plain
    # A prompt without the "<head>: <payload>" shape keeps the one-line form.
    seen.clear()
    interactmod.default_stdin_approver("Proceed?", standing=False)
    assert seen[0] == "Proceed? [y/N]: "


def test_approval_with_a_pause_armed_opens_the_menu_after_the_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval prompt with a pause armed says so and opens the pause menu after the answer.

    An operator prompt counts as a Ctrl-C boundary; the menu's action seeds the next boundary's
    steer.
    """
    from agent6.ui.cli import _interact as interactmod

    events = agent6_events.EventSink(tmp_path / "logs.jsonl")
    notices: list[str] = []
    calls: list[str] = []

    def _steer(armed: bool) -> steer.SteerState:
        return steer.SteerState(
            requested=lambda: armed,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
            armed=lambda: armed,
            prompt_now=lambda: calls.append("menu"),
        )

    def _not_live(_d: pathlib.Path) -> bool:
        return False

    def _no_away(_d: pathlib.Path) -> str | None:
        return None

    def _approve_yes(_p: str, **_k: object) -> str:
        return "yes"

    monkeypatch.setattr(interactmod, "frontend_is_live", _not_live)
    monkeypatch.setattr(interactmod, "away_mode", _no_away)
    monkeypatch.setattr(interactmod, "has_controlling_tty", lambda: True)
    monkeypatch.setattr(interactmod, "tty_message", notices.append)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _approve_yes)

    approve = _prompts(tmp_path, events, [_steer(True)]).approve
    assert approve("Allow run_command: ls", scope="command") is True
    assert calls == ["menu"]
    assert any("pause armed" in n for n in notices)

    calls.clear()
    notices.clear()
    approve = _prompts(tmp_path, events, [_steer(False)]).approve
    assert approve("Allow run_command: ls", scope="command") is True
    assert calls == [] and notices == []


def test_the_prompts_pause_a_console_view_attached_after_they_were_built(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approver and questioner read the console view at prompt time, not at build time.

    The lifecycle builds the gate before the execution attaches the view; a view captured at build
    time pauses nothing, and the heartbeat's line-erase wipes the tty prompt.
    """
    from agent6.ui.cli import _console_view, run

    paused: list[_console_view.ConsoleView] = []
    real_pause = _console_view.ConsoleView.pause

    def _pause(self: _console_view.ConsoleView) -> Any:
        paused.append(self)
        return real_pause(self)

    monkeypatch.setattr(_console_view.ConsoleView, "pause", _pause)
    monkeypatch.setattr(interactmod, "frontend_is_live", _dead)
    monkeypatch.setattr(interactmod, "has_controlling_tty", _tty)
    monkeypatch.setattr(interactmod, "default_stdin_approver", _stdin_yes)

    def _first(_q: tuple[schema.UserQuestion, ...], **_k: object) -> tuple[str, ...]:
        return ("a",)

    monkeypatch.setattr(interactmod, "default_stdin_questioner", _first)
    fe = run.session_frontend()
    events = agent6_events.EventSink(tmp_path / "logs.jsonl")
    prompts = operator_prompts.OperatorPrompts(
        approver=fe.build_approver(tmp_path),
        questioner=fe.build_questioner(tmp_path),
        journal=events.emit,
        session_dir=tmp_path,
    )
    fe.attach_console_view(events)  # the execution attaches the view after the gate exists
    try:
        assert prompts.approve("Allow run_command: ls", scope=ipc.COMMAND_SCOPE) is True
        assert prompts.ask(
            (schema.UserQuestion(question="pick?", options=("a", "b")),)
        ).answers == ("a",)
    finally:
        fe.close_console_view()
    assert len(paused) == 2, "both prompts pause the view the execution attached"


def test_a_dashboard_that_dies_before_the_run_ends_is_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dashboard gone before the run ends is reported with where the output went.

    The run's console output is redirected to tui_console.log while the dashboard owns the terminal;
    an interrupt takes both down together and says nothing.
    """
    import subprocess
    import types

    session_dir = tmp_path / "sess"
    session_dir.mkdir()

    class _Dead:
        returncode = 3

        def poll(self) -> int:
            return 3

        def wait(self, timeout: float | None = None) -> int:
            return 3

    def spawn_dead(argv: list[str], **kwargs: Any) -> _Dead:
        return _Dead()

    monkeypatch.setattr(
        livemod,
        "subprocess",
        types.SimpleNamespace(Popen=spawn_dead, TimeoutExpired=subprocess.TimeoutExpired),
    )
    with livemod.tui_session(session_dir, enabled=True):
        print("run chatter")
    err = capsys.readouterr().err
    assert "dashboard exited with code 3 before the run ended" in err
    assert str(session_dir / "tui_console.log") in err
    assert (session_dir / "tui_console.log").read_text(encoding="utf-8") == "run chatter\n"

    class _Left(_Dead):
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def spawn_left(argv: list[str], **kwargs: Any) -> _Left:
        return _Left()

    monkeypatch.setattr(livemod.subprocess, "Popen", spawn_left)
    with livemod.tui_session(session_dir, enabled=True):
        pass
    assert "the dashboard closed before the run ended" in capsys.readouterr().err

    monkeypatch.setattr(livemod.subprocess, "Popen", spawn_dead)
    with pytest.raises(KeyboardInterrupt), livemod.tui_session(session_dir, enabled=True):
        raise KeyboardInterrupt
    assert "dashboard" not in capsys.readouterr().err


def test_tui_session_degrades_when_console_log_cannot_open(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unopenable tui_console.log degrades to a TUI-less run, like a spawn failure."""
    import subprocess
    import types

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    (session_dir / "tui_console.log").mkdir()  # open("w") fails

    spawned: list[list[str]] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> object:
        spawned.append(argv)  # a spawn before the failing open would be orphaned
        return object()

    monkeypatch.setattr(
        livemod,
        "subprocess",
        types.SimpleNamespace(Popen=fake_popen, TimeoutExpired=subprocess.TimeoutExpired),
    )
    ran = False
    with livemod.tui_session(session_dir, enabled=True):
        ran = True
    assert ran, "the run must continue TUI-less, not abort"
    assert spawned == [], "the log opens before the spawn, so there is nothing to orphan"
    assert "could not start TUI" in capsys.readouterr().err


def test_ctrl_c_kills_a_dashboard_that_ignores_graceful_shutdown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C kills a dashboard that ignores SIGINT and SIGTERM before the parent returns."""
    import subprocess

    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    actions: list[str] = []

    class _Wedged:
        returncode = None
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            if self.waits < 4:
                raise subprocess.TimeoutExpired("tui", timeout or 0)
            return 0

        def send_signal(self, sig: int) -> None:
            actions.append(f"signal:{sig}")

        def terminate(self) -> None:
            actions.append("terminate")

        def kill(self) -> None:
            actions.append("kill")

    def spawn_wedged(_argv: list[str], **_kwargs: Any) -> _Wedged:
        return _Wedged()

    monkeypatch.setattr(livemod.subprocess, "Popen", spawn_wedged)

    with livemod.tui_session(session_dir, enabled=True):
        pass

    assert actions == [f"signal:{livemod.signal.SIGINT}", "terminate", "kill"]


def test_tui_session_restores_the_console_when_a_second_ctrl_c_lands(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second Ctrl-C, landing in the wait for the dashboard, still restores the console."""
    import subprocess
    import sys
    import types

    session_dir = tmp_path / "sess"
    session_dir.mkdir()

    class _Stubborn:
        returncode = 0

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt  # the operator's Ctrl-C, and again in the teardown

        def send_signal(self, sig: int) -> None:
            return None

    def spawn(argv: list[str], **kwargs: Any) -> _Stubborn:
        return _Stubborn()

    monkeypatch.setattr(
        livemod,
        "subprocess",
        types.SimpleNamespace(Popen=spawn, TimeoutExpired=subprocess.TimeoutExpired),
    )
    before = (sys.stdout, sys.stderr)
    with pytest.raises(KeyboardInterrupt), livemod.tui_session(session_dir, enabled=True):
        pass

    assert (sys.stdout, sys.stderr) == before
