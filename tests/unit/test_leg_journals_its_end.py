# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A leg that dies before the loop starts journals session.end BEFORE the
tui_session scope closes: that scope's exit is `_live.tui_session`'s
`proc.wait()`, which blocks on a dashboard that leaves only on a session.end.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import agent6.app._leg as leg_mod
from agent6.app._leg import LegInputs, run_leg
from agent6.app.frontend import FrontendCapabilities
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.layout import SessionLayout
from agent6.ui.acp.frontend import acp_frontend
from agent6.ui.steer import SteerState
from agent6.workflows._session_state import SNAPSHOT_VERSION
from agent6.workflows.loop import SessionResult

# The snapshot resume.py's preflight accepts (load_session_snapshot passes) and
# Conversation.from_wire rejects one leg deeper: a tool_result with no tool_use.
TORN = {
    "version": SNAPSHOT_VERSION,
    "system": "s",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_gone", "content": "ok"}],
        },
    ],
    "tool_calls": 3,
    "next_iteration": 4,
    "root_task_id": None,
    "original_task": "x",
    "verify_command": [],
}


def _returning(value: object) -> Callable[..., object]:
    def stub(*_a: object, **_k: object) -> object:
        return value

    return stub


def test_provider_setup_failure_journals_session_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider setup failure happens before the workflow can journal its own end,
    but the run already has a manifest and worker pid for every surface to read."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-SETUP1")
    layout.ensure()
    events = EventSink(layout.logs_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(leg_mod, "build_session_providers", _fail)
    frontend = MagicMock()
    frontend.stream_modes.return_value = (False, False)
    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
    )

    said: list[str] = []
    with pytest.raises(RuntimeError, match="provider setup failed"):
        run_leg(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=Reporter(out=said.append, err=said.append),
            events=events,
            transcript_sink=MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["type"] == "session.end"
    assert ended["reason"] == "crashed"
    assert ended["iterations"] == 0
    assert any("run crashed" in line for line in said)


def test_gate_setup_failure_closes_the_providers_it_already_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate failure happens after provider construction but before the main teardown scope."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-GATE01")
    layout.ensure()
    events = EventSink(layout.logs_path)
    closed: list[str] = []
    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: closed.append("session"),
    )
    reviser = SimpleNamespace(close=lambda: closed.append("reviser"))

    def _fail_gate(_cfg: Config, _budget: object) -> Config:
        raise RuntimeError("gate setup failed")

    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(reviser))
    frontend = MagicMock()
    frontend.stream_modes.return_value = (False, False)
    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=_fail_gate,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="gate setup failed"):
        run_leg(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    assert closed == ["session", "reviser"]


def test_mcp_setup_failure_journals_session_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP startup is part of a live leg even though the workflow does not exist yet."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-MCPSET")
    layout.ensure()
    events = EventSink(layout.logs_path)
    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("MCP startup failed")

    def _steer_state(*_args: object) -> SteerState:
        return SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _fail)
    monkeypatch.setattr(leg_mod, "chown_to_real_user", _returning(None))
    frontend = MagicMock()
    frontend.stream_modes.return_value = (False, False)
    frontend.make_steer_state.side_effect = _steer_state
    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="MCP startup failed"):
        run_leg(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["type"] == "session.end"
    assert ended["reason"] == "crashed"
    assert ended["iterations"] == 0


def test_a_cleanup_failure_does_not_skip_the_rest_of_the_leg_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed provider close must not strand commands, MCP servers, or ownership work."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-CLOSE1")
    layout.ensure()
    events = EventSink(layout.logs_path)
    closed: list[str] = []

    def _session_close() -> None:
        closed.append("session")
        raise RuntimeError("provider close failed")

    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=_session_close,
    )
    reviser = SimpleNamespace(close=lambda: closed.append("reviser"))
    dispatcher = SimpleNamespace(
        settle_background=lambda: None,
        close=lambda: closed.append("dispatcher"),
    )
    tools = SimpleNamespace(
        curator=None,
        dispatcher=dispatcher,
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    mcp = SimpleNamespace(close=lambda: closed.append("mcp") or ())

    class _Workflow:
        iterations_reached = 1

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> SessionResult:
            events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
            return SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=1,
                tool_calls=0,
            )

    def _steer_state(*_args: object) -> SteerState:
        return SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: closed.append("steer"),
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(reviser))
    monkeypatch.setattr(leg_mod, "build_session_tools", _returning(tools))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _returning(mcp))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))
    monkeypatch.setattr(leg_mod, "Workflow", _Workflow)

    def _chown(_path: Path) -> None:
        closed.append("chown")

    monkeypatch.setattr(leg_mod, "chown_to_real_user", _chown)
    frontend = replace(
        acp_frontend(
            ask=lambda _p, _o, _s, _c, _u=None: None,
            capabilities=FrontendCapabilities(can_ask=False),
            agent6_exe=lambda: "agent6",
            spawn_detached_resume=lambda _cwd, _sid, _flags: "",
        ),
        make_steer_state=_steer_state,
    )
    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="provider close failed"):
        run_leg(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    assert closed == ["steer", "session", "reviser", "dispatcher", "mcp", "chown"]


def test_a_resume_error_journals_session_end_before_the_tui_is_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ResumeError left the loop with no session.end, and the outer handlers
    that journal one for an interrupt or a crash sit past the `tui_session`
    scope, whose exit waits on a dashboard that leaves only on an end it can
    see: `resume --tui` on a torn snapshot hung on its own TUI."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-AAAA11")
    layout.session_dir.mkdir(parents=True)
    snap = layout.session_dir / "loop_state.json"
    snap.write_text(json.dumps(TORN), encoding="utf-8")
    events = EventSink(layout.logs_path)

    # What the co-process TUI could see at the moment `_live.tui_session`'s
    # finally calls proc.wait().
    seen_at_exit: list[list[str]] = []

    class _Recorder(contextlib.AbstractContextManager[None]):
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            lines = (
                layout.logs_path.read_text(encoding="utf-8").splitlines()
                if layout.logs_path.exists()
                else []
            )
            seen_at_exit.append([json.loads(x)["type"] for x in lines if x.strip()])
            return False

    def _tui_session(_dir: Path, _enabled: bool) -> _Recorder:
        return _Recorder()

    def _steer_state(*_a: object) -> SteerState:
        return SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    frontend = replace(
        acp_frontend(
            ask=lambda _p, _o, _s, _c, _u=None: None,
            capabilities=FrontendCapabilities(can_ask=False),
            agent6_exe=lambda: "agent6",
            spawn_detached_resume=lambda _cwd, _sid, _flags: "",
        ),
        tui_session=_tui_session,
        make_steer_state=_steer_state,
    )

    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )
    tools = SimpleNamespace(
        curator=None,
        dispatcher=MagicMock(),
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(leg_mod, "build_session_tools", _returning(tools))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))
    monkeypatch.setattr(leg_mod, "chown_to_real_user", _returning(None))

    inputs = LegInputs(
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=True,
        interactive=False,
        task=None,  # a resumed leg: wf.resume()
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=snap,
        undo_forker=lambda: None,
        prompts=MagicMock(),
        ask_transcript_task=None,
        resuming=True,
    )
    said: list[str] = []
    end = run_leg(
        Config(),
        layout,
        inputs,
        frontend=frontend,
        reporter=Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    assert end.rc == 1
    assert seen_at_exit, "the tui_session scope never closed"
    assert "session.end" in seen_at_exit[0], seen_at_exit[0]
    assert any("resume crashed" in line for line in said)


def _wired_frontend(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str],
    *,
    workflow: type,
    cfg: Config,
    tui_session: Callable[[Path, bool], contextlib.AbstractContextManager[None]] | None = None,
) -> Any:
    """A leg whose providers, tools and merge are recorders: `order` names
    each teardown step as it runs."""
    session = SimpleNamespace(
        budget=MagicMock(),
        rm_role=SimpleNamespace(model="m", provider="p"),
        provider=MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: order.append("session"),
    )
    dispatcher = SimpleNamespace(
        settle_background=lambda: None, close=lambda: order.append("dispatcher")
    )
    tools = SimpleNamespace(
        curator=None,
        dispatcher=dispatcher,
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=cfg,
    )
    monkeypatch.setattr(leg_mod, "build_session_providers", _returning(session))
    monkeypatch.setattr(leg_mod, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(leg_mod, "build_session_tools", _returning(tools))
    monkeypatch.setattr(leg_mod, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(leg_mod, "wants_session_network", _returning(False))

    def _chown(_path: Path) -> None:
        order.append("chown")

    def _merge(*_args: object, **_kwargs: object) -> None:
        order.append("auto_merge")

    monkeypatch.setattr(leg_mod, "chown_to_real_user", _chown)
    monkeypatch.setattr(leg_mod, "finalize_auto_merge", _merge)
    monkeypatch.setattr(leg_mod, "Workflow", workflow)

    def _steer_state(*_args: object) -> SteerState:
        return SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: order.append("steer"),
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    frontend = acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=FrontendCapabilities(can_ask=False),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    if tui_session is None:
        return replace(frontend, make_steer_state=_steer_state)
    return replace(frontend, make_steer_state=_steer_state, tui_session=tui_session)


def _finishing_workflow(iterations: int) -> type:
    class _Workflow:
        iterations_reached = iterations

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> SessionResult:
            return SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=iterations,
                tool_calls=0,
                verified="not_applicable",
            )

    return _Workflow


def test_the_chown_runs_after_the_auto_merge_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under sudo the merge writes the manifest and the transcripts as root;
    the chown is the last step of the teardown, after them, whatever raised."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-ORDER1")
    layout.ensure()
    order: list[str] = []
    cfg = Config.model_validate({"git": {"auto_merge": True}})
    frontend = _wired_frontend(monkeypatch, order, workflow=_finishing_workflow(1), cfg=cfg)
    run_leg(
        cfg,
        layout,
        LegInputs(
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda c, _b: c,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=Reporter(out=lambda _s: None, err=lambda _s: None),
        events=EventSink(layout.logs_path),
        transcript_sink=MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    assert order == ["steer", "session", "dispatcher", "auto_merge", "chown"]


def test_a_raising_dashboard_scope_prints_one_crash_line_and_journals_no_second_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard scope raising after a finished run is the leg's failure,
    not the run's: one crash line, and the run's own end stays its last."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-TUIRAI")
    layout.ensure()
    events = EventSink(layout.logs_path)
    order: list[str] = []

    class _Boom(contextlib.AbstractContextManager[None]):
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            raise RuntimeError("dashboard teardown failed")

    class _Workflow:
        iterations_reached = 3

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> SessionResult:
            events.emit("session.end", reason="finish_session", iterations=3, all_passed=True)
            return SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=3,
                tool_calls=0,
                verified="not_applicable",
            )

    frontend = _wired_frontend(
        monkeypatch, order, workflow=_Workflow, cfg=Config(), tui_session=lambda _d, _e: _Boom()
    )
    said: list[str] = []
    with pytest.raises(RuntimeError, match="dashboard teardown failed"):
        run_leg(
            Config(),
            layout,
            LegInputs(
                session_id=layout.session_id,
                mode="run",
                role="worker",
                isolation="hardened",
                tui_enabled=False,
                interactive=False,
                task="do the thing",
                gate=lambda c, _b: c,
                chain_branch=None,
                base_sha="",
                untracked_at_start=frozenset(),
                resume_state_path=layout.session_dir / "loop_state.json",
                undo_forker=lambda: None,
                prompts=MagicMock(),
                ask_transcript_task=None,
            ),
            frontend=frontend,
            reporter=Reporter(out=said.append, err=said.append),
            events=events,
            transcript_sink=MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )
    assert [line for line in said if "crashed" in line] == ["\n[agent6] run crashed"]
    ends = [
        json.loads(line)
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines()
        if '"session.end"' in line
    ]
    assert [e["reason"] for e in ends] == ["finish_session"]


def test_an_interrupt_after_the_runs_end_leaves_its_result_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C during the background settle, one line after the run journaled
    its own end, journaled a second `session.end` (interrupted, all_passed
    False) that every fold took as the run's, and printed a resume hint for a
    run that had finished."""
    state = tmp_path / "state"
    layout = SessionLayout(state_dir=state, session_id="sess-SETTLE")
    layout.ensure()
    events = EventSink(layout.logs_path)
    order: list[str] = []

    class _Workflow:
        iterations_reached = 3

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> SessionResult:
            events.emit("session.end", reason="finish_session", iterations=3, all_passed=True)
            return SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=3,
                tool_calls=0,
                verified="passed",
            )

    frontend = _wired_frontend(monkeypatch, order, workflow=_Workflow, cfg=Config())
    stubbed_build = leg_mod.build_session_tools

    def _boom() -> None:
        raise KeyboardInterrupt

    def _tools(*args: Any, **kwargs: Any) -> Any:
        tools = stubbed_build(*args, **kwargs)
        tools.dispatcher.settle_background = _boom
        return tools

    monkeypatch.setattr(leg_mod, "build_session_tools", _tools)
    said: list[str] = []
    end = run_leg(
        Config(),
        layout,
        LegInputs(
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda c, _b: c,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    ends = [
        json.loads(line)
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines()
        if '"session.end"' in line
    ]
    assert [(e["reason"], e["all_passed"]) for e in ends] == [("finish_session", True)]
    assert end.rc == 0
    assert "\n[agent6] run had ended; its result stands" in said
    assert not [line for line in said if "resume with:" in line]
