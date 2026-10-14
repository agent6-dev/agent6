# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An execution that dies before the loop starts journals its end before the TUI scope closes.

That scope's exit is `proc.wait()` on a dashboard that leaves only on a session.end.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import subprocess
import types
from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

from agent6 import event_log, git_ops, paths
from agent6.app import _execution, _session, _setup, finalize, reporter
from agent6.app import frontend as app_frontend
from agent6.app import providers as app_providers
from agent6.config import Config
from agent6.harness import _snapshot
from agent6.harness import loop as harness_loop
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui import steer
from agent6.ui.acp import frontend as acp_frontend
from agent6.viewmodel import listing

# The preflight accepts this snapshot and Conversation.from_wire rejects it: a result with no call.
TORN = {
    "version": _snapshot.SNAPSHOT_VERSION,
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider setup failure happens before the harness can journal its own end.

    The run already has a manifest and worker pid for every surface to read.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-SETUP1")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(_session, "build_session_providers", _fail)
    frontend = mock.MagicMock()
    frontend.stream_modes.return_value = (False, False)
    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
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
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
    )

    said: list[str] = []
    with pytest.raises(RuntimeError, match="provider setup failed"):
        _execution.run_execution(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=reporter.Reporter(out=said.append, err=said.append),
            events=events,
            transcript_sink=mock.MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["type"] == "session.end"
    assert ended["reason"] == "crashed"
    assert ended["iterations"] == 0
    assert any("run crashed" in line for line in said)


def test_gate_setup_failure_closes_the_providers_it_already_built(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate failure happens after provider construction but before the main teardown scope."""
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-GATE01")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    closed: list[str] = []
    session = types.SimpleNamespace(
        budget=mock.MagicMock(),
        rm_role=types.SimpleNamespace(model="m", provider="p"),
        provider=mock.MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: closed.append("session"),
    )
    reviser = types.SimpleNamespace(close=lambda: closed.append("reviser"))

    def _fail_gate(_cfg: Config, _budget: object) -> Config:
        raise RuntimeError("gate setup failed")

    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(app_providers, "build_prompt_reviser_provider", _returning(reviser))
    frontend = mock.MagicMock()
    frontend.stream_modes.return_value = (False, False)
    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
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
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="gate setup failed"):
        _execution.run_execution(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=mock.MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    assert closed == ["session", "reviser"]


def test_mcp_setup_failure_journals_session_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP startup is part of a live execution even though the harness does not exist yet."""
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-MCPSET")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    session = types.SimpleNamespace(
        budget=mock.MagicMock(),
        rm_role=types.SimpleNamespace(model="m", provider="p"),
        provider=mock.MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("MCP startup failed")

    def _steer_state(*_args: object) -> steer.SteerState:
        return steer.SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(app_providers, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(_setup, "wants_session_network", _returning(False))
    monkeypatch.setattr(_setup, "start_mcp_manager_if_enabled", _fail)
    monkeypatch.setattr(paths, "chown_to_real_user", _returning(None))
    frontend = mock.MagicMock()
    frontend.stream_modes.return_value = (False, False)
    frontend.make_steer_state.side_effect = _steer_state
    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
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
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="MCP startup failed"):
        _execution.run_execution(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=mock.MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["type"] == "session.end"
    assert ended["reason"] == "crashed"
    assert ended["iterations"] == 0


def test_a_cleanup_failure_does_not_skip_the_rest_of_the_execution_teardown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed provider close must not strand commands, MCP servers, or ownership work."""
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-CLOSE1")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    closed: list[str] = []

    def _session_close() -> None:
        closed.append("session")
        raise RuntimeError("provider close failed")

    session = types.SimpleNamespace(
        budget=mock.MagicMock(),
        rm_role=types.SimpleNamespace(model="m", provider="p"),
        provider=mock.MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=_session_close,
    )
    reviser = types.SimpleNamespace(close=lambda: closed.append("reviser"))
    dispatcher = types.SimpleNamespace(
        settle_background=lambda: None,
        close=lambda: closed.append("dispatcher"),
    )
    tools = types.SimpleNamespace(
        curator=None,
        dispatcher=dispatcher,
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    mcp = types.SimpleNamespace(close=lambda: closed.append("mcp") or ())

    class _Workflow:
        iterations_reached = 1

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=1,
                tool_calls=0,
            )

    def _steer_state(*_args: object) -> steer.SteerState:
        return steer.SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: closed.append("steer"),
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(app_providers, "build_prompt_reviser_provider", _returning(reviser))
    monkeypatch.setattr(_session, "build_session_tools", _returning(tools))
    monkeypatch.setattr(_setup, "start_mcp_manager_if_enabled", _returning(mcp))
    monkeypatch.setattr(_setup, "wants_session_network", _returning(False))
    monkeypatch.setattr(harness_loop, "Harness", _Workflow)

    def _chown(_path: pathlib.Path) -> None:
        closed.append("chown")

    monkeypatch.setattr(paths, "chown_to_real_user", _chown)
    frontend = dataclasses.replace(
        acp_frontend.acp_frontend(
            ask=lambda _p, _o, _s, _c, _u=None: None,
            capabilities=app_frontend.FrontendCapabilities(can_ask=False),
            agent6_exe=lambda: "agent6",
            spawn_detached_resume=lambda _cwd, _sid, _flags: "",
        ),
        make_steer_state=_steer_state,
    )
    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
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
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
    )

    with pytest.raises(RuntimeError, match="provider close failed"):
        _execution.run_execution(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=mock.MagicMock(),
            cwd=tmp_path,
            state_dir=state,
        )

    assert closed == ["steer", "session", "reviser", "dispatcher", "mcp", "chown"]


def test_a_resume_error_journals_session_end_before_the_tui_is_waited_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ResumeError journals a session.end before the TUI scope waits on the dashboard.

    `resume --tui` on a torn snapshot otherwise hangs on its own TUI.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-AAAA11")
    layout.session_dir.mkdir(parents=True)
    snap = layout.session_dir / "loop_state.json"
    snap.write_text(json.dumps(TORN), encoding="utf-8")
    events = event_log.EventSink(layout.logs_path)

    # What the co-process TUI could see when `_live.tui_session`'s finally calls proc.wait().
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

    def _tui_session(_dir: pathlib.Path, _enabled: bool) -> _Recorder:
        return _Recorder()

    def _steer_state(*_a: object) -> steer.SteerState:
        return steer.SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: None,
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    frontend = dataclasses.replace(
        acp_frontend.acp_frontend(
            ask=lambda _p, _o, _s, _c, _u=None: None,
            capabilities=app_frontend.FrontendCapabilities(can_ask=False),
            agent6_exe=lambda: "agent6",
            spawn_detached_resume=lambda _cwd, _sid, _flags: "",
        ),
        tui_session=_tui_session,
        make_steer_state=_steer_state,
    )

    session = types.SimpleNamespace(
        budget=mock.MagicMock(),
        rm_role=types.SimpleNamespace(model="m", provider="p"),
        provider=mock.MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )
    tools = types.SimpleNamespace(
        curator=None,
        dispatcher=mock.MagicMock(),
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(app_providers, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(_session, "build_session_tools", _returning(tools))
    monkeypatch.setattr(_setup, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(_setup, "wants_session_network", _returning(False))
    monkeypatch.setattr(paths, "chown_to_real_user", _returning(None))

    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=True,
        interactive=False,
        task=None,  # a resumed execution: wf.resume()
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha="",
        untracked_at_start=frozenset(),
        resume_state_path=snap,
        undo_forker=lambda: None,
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
        resuming=True,
    )
    said: list[str] = []
    end = _execution.run_execution(
        Config(),
        layout,
        inputs,
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
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
    harness: type,
    cfg: Config,
    tui_session: Callable[[pathlib.Path, bool], contextlib.AbstractContextManager[None]]
    | None = None,
) -> Any:
    """An execution whose providers, tools and merge are recorders; `order` names the teardown."""
    session = types.SimpleNamespace(
        budget=_setup.budget_tracker(cfg),
        rm_role=types.SimpleNamespace(model="m", provider="p"),
        provider=mock.MagicMock(),
        summariser_provider=None,
        review_seats=[],
        close=lambda: order.append("session"),
    )
    dispatcher = types.SimpleNamespace(
        settle_background=lambda: None, close=lambda: order.append("dispatcher")
    )
    tools = types.SimpleNamespace(
        curator=None,
        dispatcher=dispatcher,
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=cfg,
    )
    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(app_providers, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(_session, "build_session_tools", _returning(tools))
    monkeypatch.setattr(_setup, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(_setup, "wants_session_network", _returning(False))

    def _chown(_path: pathlib.Path) -> None:
        order.append("chown")

    def _merge(*_args: object, **_kwargs: object) -> None:
        order.append("auto_merge")

    monkeypatch.setattr(paths, "chown_to_real_user", _chown)
    monkeypatch.setattr(finalize, "finalize_auto_merge", _merge)
    monkeypatch.setattr(harness_loop, "Harness", harness)

    def _steer_state(*_args: object) -> steer.SteerState:
        return steer.SteerState(
            requested=lambda: False,
            clear=lambda: None,
            prompt=lambda: None,
            restore=lambda: order.append("steer"),
            abort_pending=lambda: False,
            interrupt=lambda: False,
            reset_stage=lambda: None,
        )

    frontend = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=app_frontend.FrontendCapabilities(can_ask=False),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    if tui_session is None:
        return dataclasses.replace(frontend, make_steer_state=_steer_state)
    return dataclasses.replace(frontend, make_steer_state=_steer_state, tui_session=tui_session)


def _finishing_workflow(iterations: int) -> type:
    class _Workflow:
        iterations_reached = iterations

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=iterations,
                tool_calls=0,
                verified="not_applicable",
            )

    return _Workflow


def test_the_chown_runs_after_the_auto_merge_writes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under sudo the chown is the last teardown step, after the merge's root writes."""
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-ORDER1")
    layout.ensure()
    order: list[str] = []
    cfg = Config.model_validate({"git": {"auto_merge": True}})
    frontend = _wired_frontend(monkeypatch, order, harness=_finishing_workflow(1), cfg=cfg)
    _execution.run_execution(
        cfg,
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
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
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
        events=event_log.EventSink(layout.logs_path),
        transcript_sink=mock.MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    assert order == ["steer", "session", "dispatcher", "auto_merge", "chown"]


@pytest.mark.parametrize(
    ("reason", "verified", "auto_merge", "merged"),
    [
        ("verify_settled", "passed", True, False),
        ("settled", "not_applicable", True, False),
        ("silent_finish", "passed", True, False),
        ("silent_finish", "not_applicable", True, False),
        ("metric_plateau", "passed", True, False),
        ("metric_plateau", "not_applicable", True, False),
        ("finish_session", "passed", True, True),
        ("finish_session", "not_applicable", True, True),
        ("finish_session", "failed", True, False),
        ("finish_session", "unverified", True, False),
        ("settled", "not_applicable", False, False),
        ("finish_session", "passed", False, False),
    ],
)
def test_auto_merge_leaves_harness_ends_on_the_run_branch_and_says_why(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: _snapshot.SessionEndReason,
    verified: _snapshot.Verification,
    auto_merge: bool,
    merged: bool,
) -> None:
    """Harness ends must keep their branch and explain a skipped automatic merge.

    Only finish_session with a green or absent gate lands on the base branch.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-qm", "base")
    base = git("rev-parse", "main")
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-MERGE1")
    layout.ensure()
    branch = f"agent6/{layout.session_id}"
    git("checkout", "-qb", branch)
    (repo / "a.txt").write_text("run work\n", encoding="utf-8")
    git("commit", "-qam", "step")
    tip = git("rev-parse", branch)
    git("update-ref", git_ops.chain_ref_for(layout.session_id), tip)
    git("checkout", "-q", "main")
    (repo / "a.txt").write_text("run work\n", encoding="utf-8")
    layout.manifest_path.write_text(
        json.dumps(
            {
                "session_id": layout.session_id,
                "mode": "run",
                "user_task": "do the thing",
                "base_branch": "main",
                "base_sha": base,
                "run_branch": branch,
            }
        ),
        encoding="utf-8",
    )
    events = event_log.EventSink(layout.logs_path)

    class _Workflow:
        iterations_reached = 1

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason=reason, iterations=1, verified=verified)
            return _snapshot.SessionResult(
                completed=True,
                reason=reason,
                summary="done",
                iterations=1,
                tool_calls=0,
                verified=verified,
            )

    cfg = Config.model_validate({"git": {"auto_merge": auto_merge}})
    real_merge = finalize.finalize_auto_merge
    frontend = _wired_frontend(monkeypatch, [], harness=_Workflow, cfg=cfg)
    monkeypatch.setattr(finalize, "finalize_auto_merge", real_merge)
    said: list[str] = []
    _execution.run_execution(
        cfg,
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda c, _b: c,
            chain_branch=branch,
            base_sha=base,
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
        cwd=repo,
        state_dir=state,
    )

    manifest = sessions_manifest.read_manifest(layout.session_dir)
    assert git("rev-parse", branch) == tip
    skipped = [line for line in said if "auto_merge skipped:" in line]
    if merged:
        assert git("rev-parse", "main") != base
        assert git("show", "main:a.txt") == "run work"
        assert manifest.merged is not None, "\n".join(said)
        assert any("changes merged into main" in line for line in said)
        assert not any("merge with:" in line for line in said)
    else:
        assert git("rev-parse", "main") == base
        assert manifest.merged is None
        assert any(f"changes are on {branch}" in line for line in said)
        assert any("merge with:  agent6 sessions merge sess-MERGE1" in line for line in said)
    if auto_merge and not merged:
        assert len(skipped) == 1
        assert "\n" not in skipped[0]
        if reason != "finish_session":
            assert reason in skipped[0] and "not finish_session" in skipped[0]
        else:
            assert verified in skipped[0]
    else:
        assert not skipped


def test_a_finish_with_no_commit_is_journaled_and_reads_not_committed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose commits all failed ended "passed" and listed as "passed" while it exited 5."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-qm", "base")
    base = git("rev-parse", "main")
    # The run's edit sits in the tree; no commit ever reached its chain.
    (repo / "a.txt").write_text("run work\n", encoding="utf-8")
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-STRAND")
    layout.ensure()
    branch = f"agent6/{layout.session_id}"
    layout.manifest_path.write_text(
        json.dumps(
            {
                "session_id": layout.session_id,
                "mode": "run",
                "user_task": "do the thing",
                "base_branch": "main",
                "base_sha": base,
                "run_branch": branch,
            }
        ),
        encoding="utf-8",
    )
    events = event_log.EventSink(layout.logs_path)

    class _Workflow:
        iterations_reached = 1

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason="finish_session", iterations=1, all_passed=True)
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=1,
                tool_calls=0,
                verified="passed",
            )

    cfg = Config()
    frontend = _wired_frontend(monkeypatch, [], harness=_Workflow, cfg=cfg)
    said: list[str] = []
    end = _execution.run_execution(
        cfg,
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda c, _b: c,
            chain_branch=branch,
            base_sha=base,
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
        cwd=repo,
        state_dir=state,
    )

    assert end.rc == finalize.EXIT_NO_COMMIT_LANDED
    journal = [json.loads(line) for line in layout.logs_path.read_text().splitlines()]
    assert [e["type"] for e in journal][-2:] == ["session.end", "session.stranded"]
    row = listing.summarize_session_dir(layout.session_dir)
    assert (row.status, row.reason) == ("passed", "not committed")
    assert any("no commit on" in line for line in said), "\n".join(said)


def test_a_raising_dashboard_scope_prints_one_crash_line_and_journals_no_second_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard scope raising after a finished run is the execution's failure, not the run's.

    One crash line, and the run's own end stays its last.

    One crash line, and the run's own end stays its last.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-TUIRAI")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
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

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason="finish_session", iterations=3, all_passed=True)
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=3,
                tool_calls=0,
                verified="not_applicable",
            )

    frontend = _wired_frontend(
        monkeypatch, order, harness=_Workflow, cfg=Config(), tui_session=lambda _d, _e: _Boom()
    )
    said: list[str] = []
    with pytest.raises(RuntimeError, match="dashboard teardown failed"):
        _execution.run_execution(
            Config(),
            layout,
            _execution.ExecutionInputs(
                explicit_leaves=frozenset(),
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
                prompts=mock.MagicMock(),
                ask_transcript_task=None,
            ),
            frontend=frontend,
            reporter=reporter.Reporter(out=said.append, err=said.append),
            events=events,
            transcript_sink=mock.MagicMock(),
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C during the background settle journals no second `session.end`.

    A second end (interrupted) would be taken as the run's by every fold, with a resume hint for a
    run that had finished.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-SETTLE")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    order: list[str] = []

    class _Workflow:
        iterations_reached = 3

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason="finish_session", iterations=3, all_passed=True)
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="done",
                iterations=3,
                tool_calls=0,
                verified="passed",
            )

    frontend = _wired_frontend(monkeypatch, order, harness=_Workflow, cfg=Config())
    stubbed_build = _session.build_session_tools

    def _boom() -> None:
        raise KeyboardInterrupt

    def _tools(*args: Any, **kwargs: Any) -> Any:
        tools = stubbed_build(*args, **kwargs)
        tools.dispatcher.settle_background = _boom
        return tools

    monkeypatch.setattr(_session, "build_session_tools", _tools)
    said: list[str] = []
    end = _execution.run_execution(
        Config(),
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
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
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
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


def test_a_closed_output_pipe_ends_the_run_as_a_stop_not_a_crash(tmp_path: pathlib.Path) -> None:
    """`agent6 run ... | head` journals an interrupted end when the reader goes away.

    The write to the closed pipe raised out of the run and was journaled as `crashed`, so the
    session read "failed · crashed" though the operator had closed its output.
    """
    import json

    layout = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="sess-PIPE")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)

    reason = _execution._journal_escape(events, BrokenPipeError(32, "Broken pipe"), iterations=2)

    ended = [json.loads(line) for line in layout.logs_path.read_text().splitlines()][-1]
    assert reason == "interrupted"
    assert ended["type"] == "session.end" and ended["reason"] == "interrupted"


def test_an_interrupted_end_records_the_tree_the_run_left(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C or SIGTERM end carries `tree` as a declared end does.

    Without it the next resume could not tell the run's in-flight files from the operator's
    edits, and took in whatever the checkout held.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-TERM11")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    order: list[str] = []

    class _Workflow:
        iterations_reached = 2
        chain = types.SimpleNamespace(tree_sha=lambda: "tree-at-the-signal")

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            raise KeyboardInterrupt

    frontend = _wired_frontend(monkeypatch, order, harness=_Workflow, cfg=Config())
    said: list[str] = []
    end = _execution.run_execution(
        Config(),
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
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
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    ends = [
        json.loads(line)
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines()
        if '"session.end"' in line
    ]
    assert [(e["reason"], e["iterations"], e.get("tree")) for e in ends] == [
        ("interrupted", 2, "tree-at-the-signal")
    ]
    assert end.rc == finalize.EXIT_OPERATOR_STOP
    assert "\n[agent6] run interrupted" in said


def _interrupted() -> str:
    raise KeyboardInterrupt


def test_a_second_interrupt_during_the_tree_read_still_journals_the_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tree is read with git inside the escape handler; another Ctrl-C can land there.

    Escaping the handler there would skip the end, and the run would read stale.
    """
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-TERM11")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)
    order: list[str] = []

    class _Workflow:
        iterations_reached = 2
        chain = types.SimpleNamespace(tree_sha=_interrupted)

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            raise KeyboardInterrupt

    frontend = _wired_frontend(monkeypatch, order, harness=_Workflow, cfg=Config())
    said: list[str] = []
    end = _execution.run_execution(
        Config(),
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
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
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
        cwd=tmp_path,
        state_dir=state,
    )
    ends = [
        json.loads(line)
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines()
        if '"session.end"' in line
    ]
    assert [(e["reason"], e["iterations"], e.get("tree")) for e in ends] == [
        ("interrupted", 2, None)
    ]
    assert end.rc == finalize.EXIT_OPERATOR_STOP
    assert "\n[agent6] run interrupted" in said


def test_an_escape_before_the_harness_exists_records_the_tree_too(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash in provider setup journals the tree the run leaves, as a later escape does.

    Without it the end blanked the earlier end's tree and the next resume could not tell the
    run's files from the operator's.
    """
    import subprocess

    from agent6 import git_ops

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "a.txt").write_text("edited\n", encoding="utf-8")
    state = tmp_path / "state"
    layout = sessions_layout.SessionLayout(state_dir=state, session_id="sess-SETUP2")
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(_session, "build_session_providers", _fail)
    frontend = mock.MagicMock()
    frontend.stream_modes.return_value = (False, False)
    inputs = _execution.ExecutionInputs(
        explicit_leaves=frozenset(),
        session_id=layout.session_id,
        mode="run",
        role="worker",
        isolation="hardened",
        tui_enabled=False,
        interactive=False,
        task="do the thing",
        gate=lambda c, _b: c,
        chain_branch=None,
        base_sha=head,
        untracked_at_start=frozenset(),
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=mock.MagicMock(),
        ask_transcript_task=None,
    )
    with pytest.raises(RuntimeError, match="provider setup failed"):
        _execution.run_execution(
            Config(),
            layout,
            inputs,
            frontend=frontend,
            reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
            events=events,
            transcript_sink=mock.MagicMock(),
            cwd=repo,
            state_dir=state,
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["reason"] == "crashed"
    assert ended["tree"] == git_ops.worktree_tree(repo, head, frozenset())


def test_a_plan_under_auto_merge_prints_no_skipped_note(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan or an ask never commits, so `auto_merge` has nothing to skip there."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    state = paths.state_dir(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=state, session_id="plan-MERGE1", subdir="plans"
    )
    layout.ensure()
    events = event_log.EventSink(layout.logs_path)

    class _Workflow:
        iterations_reached = 1

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _task: str) -> _snapshot.SessionResult:
            events.emit("session.end", reason="finish_planning", iterations=1)
            return _snapshot.SessionResult(
                completed=True,
                reason="finish_planning",
                summary="planned",
                iterations=1,
                tool_calls=0,
                verified="not_applicable",
            )

    cfg = Config.model_validate({"git": {"auto_merge": True}})
    frontend = _wired_frontend(monkeypatch, [], harness=_Workflow, cfg=cfg)
    said: list[str] = []
    _execution.run_execution(
        cfg,
        layout,
        _execution.ExecutionInputs(
            explicit_leaves=frozenset(),
            session_id=layout.session_id,
            mode="plan",
            role="planner",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="plan the thing",
            gate=lambda c, _b: c,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=mock.MagicMock(),
            ask_transcript_task=None,
        ),
        frontend=frontend,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=events,
        transcript_sink=mock.MagicMock(),
        cwd=repo,
        state_dir=state,
    )
    assert not [line for line in said if "auto_merge skipped:" in line], said
