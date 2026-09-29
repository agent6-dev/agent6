# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`_execution.detach_to_background`, the one hand-off both lifecycles make after a `/detach`."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import types
from collections.abc import Callable, Sequence
from typing import Any

import pytest

import agent6.app._execution as execution_mod
from agent6 import event_log
from agent6.app import _session, _setup, frontend, providers, reporter
from agent6.config import Config
from agent6.harness import _snapshot
from agent6.harness import loop as harness_loop
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import operator_prompts
from agent6.ui.acp import frontend as acp_frontend


def _frontend(calls: list[tuple[str, Any]], *, spawn_err: str = "") -> frontend.SessionFrontend:
    def _spawn(_cwd: pathlib.Path, sid: str, flags: Sequence[str]) -> str:
        calls.append(("spawn", (sid, list(flags))))
        return spawn_err

    def _ask_away(_dir: pathlib.Path, scopes: tuple[str, ...]) -> None:
        calls.append(("ask", scopes))

    front = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=frontend.FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=_spawn,
    )
    # The ACP front-end has no away-mode prompt; record when the lifecycle asks.
    return dataclasses.replace(front, prompt_detach_away_mode=_ask_away)


def test_ask_policy_is_asked_before_the_spawn_and_the_flags_ride_along(
    tmp_path: pathlib.Path,
) -> None:
    calls: list[tuple[str, Any]] = []
    said: list[str] = []
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    execution_mod.detach_to_background(
        frontend=_frontend(calls),
        cfg=Config(),  # run_commands = ask, nothing granted
        layout=layout,
        cwd=tmp_path,
        flags=["--max-usd", "0.25"],
        reporter=reporter.Reporter(out=said.append, err=said.append),
    )
    assert [c[0] for c in calls] == ["ask", "spawn"]
    assert calls[1][1] == ("runny-one-AAAAAA", ["--max-usd", "0.25"])
    assert any("continues in the background" in s for s in said)
    assert any("agent6 attach runny-one-AAAAAA" in s for s in said)


def test_a_failed_spawn_is_reported_and_never_called_a_continuation(tmp_path: pathlib.Path) -> None:
    """The reattach line prints after the spawn, so a failed spawn never claims a background run."""
    calls: list[tuple[str, Any]] = []
    said: list[str] = []
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    execution_mod.detach_to_background(
        frontend=_frontend(calls, spawn_err="agent6 exe not found"),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=reporter.Reporter(out=said.append, err=said.append),
    )
    assert [c[0] for c in calls] == ["spawn"]  # yes-policy: nothing to ask
    assert said == ["[agent6] agent6 exe not found"]


def test_a_recorded_away_mode_is_the_runs_away_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detached run's away mode is read from the run dir, not only the environment.

    A run detached from a terminal carries the operator's choice in `approvals/away.mode`; a later
    resume from cron or a script reads it there, as the approver does.
    """
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    session_dir = tmp_path / "run"
    session_dir.mkdir()

    assert ipc.effective_away(session_dir) == ""

    ipc.set_away_mode(session_dir, "wait")
    assert ipc.effective_away(session_dir) == "wait"

    # A launcher's env still wins: it is this invocation's own intent.
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "deny")
    assert ipc.effective_away(session_dir) == "deny"


def test_an_invalid_detached_away_env_is_not_an_away_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must not tell preflight that an unattended run has a policy."""
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    monkeypatch.setenv("AGENT6_DETACHED_AWAY", "denny")

    assert ipc.effective_away(session_dir) == ""
    ipc.set_away_mode(session_dir, "wait")
    assert ipc.effective_away(session_dir) == "wait"


def test_a_resume_names_what_the_tree_holds_that_no_commit_does(tmp_path: pathlib.Path) -> None:
    """A fresh run asks about the operator's uncommitted changes rather than sweeping them in.

    Swept in, they land in the run's next auto-commit under the agent's identity and read as the
    run's own work; the crashed execution's own uncommitted tail lands there too, so the prompt
    names them.
    """
    import subprocess as sp

    from agent6 import git_ops

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo,
        check=True,
    )
    base = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    ref = git_ops.chain_ref_for("resume-run-A1")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # the run's own execution-1 work
    git_ops.chain_commit(repo, "iter 1", ref=ref, fallback_parent=base)

    assert git_ops.chain_dirty_paths(repo, ref, base, 5) == []

    (repo / "a.py").write_text("x = 2\n# the operator's note\n", encoding="utf-8")

    assert git_ops.chain_dirty_paths(repo, ref, base, 5) == ["a.py"]


def test_the_worker_pid_survives_the_handoff_and_goes_when_it_fails(tmp_path: pathlib.Path) -> None:
    """The worker pid is cleared after the spawn, so a detaching run never reads as stale."""
    calls: list[tuple[str, Any]] = []
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="runny-one-AAAAAA")
    layout.ensure()
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    ipc.write_worker_pid(layout.session_dir, os.getpid())

    execution_mod.detach_to_background(
        frontend=_frontend(calls),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
    )
    assert ipc.read_worker_pid(layout.session_dir) == os.getpid()  # the child overwrites it

    execution_mod.detach_to_background(
        frontend=_frontend(calls, spawn_err="agent6 exe not found"),
        cfg=cfg,
        layout=layout,
        cwd=tmp_path,
        flags=[],
        reporter=reporter.Reporter(out=lambda _s: None, err=lambda _s: None),
    )
    assert ipc.read_worker_pid(layout.session_dir) is None  # nothing took over: really dead


def _returning(value: object) -> Callable[..., object]:
    def stub(*_a: object, **_k: object) -> object:
        return value

    return stub


def _stub_execution_internals(
    monkeypatch: pytest.MonkeyPatch,
    result: _snapshot.SessionResult | Exception,
    built: dict[str, Any] | None = None,
) -> None:
    class _Workflow:
        iterations_reached = 3

        def __init__(self, **kw: Any) -> None:
            if built is not None:
                built.update(kw)
            self._undo_forker: Callable[[], tuple[str, str] | None] | None = kw[
                "bridge"
            ].undo_forker

        def run(self, _task: str) -> _snapshot.SessionResult:
            if isinstance(result, Exception):
                raise result
            if result.reason == "undone" and self._undo_forker is not None:
                self._undo_forker()  # what the loop does before an `undone` end
            return result

    session = types.SimpleNamespace(
        budget=types.SimpleNamespace(
            format_summary=lambda: "[agent6] cost $0.01", estimate_usd=lambda: (0.01, False)
        ),
        rm_role=types.SimpleNamespace(model="fake/model"),
        provider=None,
        summariser_provider=None,
        review_seats=[],
        close=lambda: None,
    )
    tools = types.SimpleNamespace(
        curator=None,
        dispatcher=types.SimpleNamespace(settle_background=lambda: None, close=lambda: None),
        compact_drop_at_chars=1,
        compact_summarise_at_chars=1,
        keep_recent_chars=1,
        cfg=Config(),
    )
    monkeypatch.setattr(_session, "build_session_providers", _returning(session))
    monkeypatch.setattr(providers, "build_prompt_reviser_provider", _returning(None))
    monkeypatch.setattr(_setup, "wants_session_network", _returning(False))
    monkeypatch.setattr(_setup, "start_mcp_manager_if_enabled", _returning(None))
    monkeypatch.setattr(_session, "build_session_tools", _returning(tools))
    monkeypatch.setattr(harness_loop, "Harness", _Workflow)
    monkeypatch.setattr(_session, "session_facts_provider", _returning(lambda: None))


def test_a_loop_crash_prints_the_end_that_it_journals(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loop exception is a crashed session and exit 1, said before the CLI reports the error."""
    _stub_execution_internals(monkeypatch, RuntimeError("provider stream broke"))
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="crash-one-AAAAAA"
    )
    layout.ensure()
    said: list[str] = []
    front = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=frontend.FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()

    with pytest.raises(RuntimeError, match="provider stream broke"):
        execution_mod.run_execution(
            Config(),
            layout,
            execution_mod.ExecutionInputs(
                session_id=layout.session_id,
                mode="run",
                role="worker",
                isolation="hardened",
                tui_enabled=False,
                interactive=False,
                task="do the thing",
                gate=lambda cfg, _b: cfg,
                chain_branch=None,
                base_sha="",
                untracked_at_start=frozenset(),
                resume_state_path=layout.session_dir / "loop_state.json",
                undo_forker=lambda: None,
                prompts=operator_prompts.OperatorPrompts(session_dir=layout.session_dir),
                ask_transcript_task=None,
            ),
            frontend=front,
            reporter=reporter.Reporter(out=said.append, err=said.append),
            events=event_log.EventSink(layout.logs_path),
            transcript_sink=None,  # type: ignore[arg-type]
            cwd=cwd,
            state_dir=tmp_path / "state",
        )

    ended = json.loads(layout.logs_path.read_text(encoding="utf-8").splitlines()[-1])
    assert ended["reason"] == "crashed"
    assert any("run crashed" in line for line in said)


def test_a_detached_ask_execution_hands_the_run_over_instead_of_answering_with_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/detach` at the pause menu is offered in every mode, the ask branch included.

    Otherwise a detached ask prints the loop's bookkeeping line as the model's answer and never asks
    the caller to spawn the continuation.
    """
    _stub_execution_internals(
        monkeypatch,
        _snapshot.SessionResult(
            completed=False,
            reason="detached",
            summary="operator detached at iter 3; resuming in the background",
            iterations=3,
            tool_calls=7,
        ),
    )
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="asky-one-AAAAAA"
    )
    layout.ensure()
    saved: list[tuple[str, str]] = []
    said: list[str] = []

    def _save(_layout: sessions_layout.SessionLayout, question: str, answer: str) -> None:
        saved.append((question, answer))

    front = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=frontend.FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = execution_mod.run_execution(
        Config(),
        layout,
        execution_mod.ExecutionInputs(
            session_id=layout.session_id,
            mode="ask",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="what does this repo do?",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=operator_prompts.OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task="what does this repo do?",
        ),
        frontend=dataclasses.replace(front, save_ask_transcript=_save),
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=event_log.EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert (end.rc, end.detach_requested) == (0, True)
    assert saved == []
    assert not any("operator detached at iter" in s for s in said)


def test_an_undone_ask_execution_names_the_fork_instead_of_answering_with_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/undo` at the pause menu is offered in every mode, the ask branch included.

    Otherwise an undone ask prints the loop's bookkeeping line as the model's answer and never names
    the fork to continue from.
    """
    _stub_execution_internals(
        monkeypatch,
        _snapshot.SessionResult(
            completed=False,
            reason="undone",
            summary="operator undid the last message at iter 3",
            iterations=3,
            tool_calls=7,
        ),
    )
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="asky-two-AAAAAA"
    )
    layout.ensure()
    saved: list[tuple[str, str]] = []
    said: list[str] = []

    def _save(_layout: sessions_layout.SessionLayout, question: str, answer: str) -> None:
        saved.append((question, answer))

    front = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=frontend.FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = execution_mod.run_execution(
        Config(),
        layout,
        execution_mod.ExecutionInputs(
            session_id=layout.session_id,
            mode="ask",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="what does this repo do?",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: ("fork-two-BBBBBB", "what does this repo do?"),
            prompts=operator_prompts.OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task="what does this repo do?",
        ),
        frontend=dataclasses.replace(front, save_ask_transcript=_save),
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=event_log.EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert end.rc == 0
    assert saved == []
    assert any("continue as fork-two-BBBBBB" in s for s in said)
    assert not any("operator undid" in s for s in said)


def test_a_surface_without_the_revise_choice_skips_revision(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A surface with no selector skips the revise_prompt choice; None is not a quit."""
    built: dict[str, Any] = {}
    _stub_execution_internals(
        monkeypatch,
        _snapshot.SessionResult(
            completed=True, reason="finish_session", summary="ok", iterations=1, tool_calls=0
        ),
        built,
    )
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="acp-one-AAAAAA"
    )
    layout.ensure()
    said: list[str] = []
    front = acp_frontend.acp_frontend(
        ask=lambda _p, _o, _s, _c, _u=None: None,
        capabilities=frontend.FrontendCapabilities(),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _sid, _flags: "",
    )
    assert front.select_revised_prompt is None
    cwd = tmp_path / "repo"
    cwd.mkdir()

    end = execution_mod.run_execution(
        Config.model_validate({"prompt": {"revise_prompt": "interactive"}}),
        layout,
        execution_mod.ExecutionInputs(
            session_id=layout.session_id,
            mode="run",
            role="worker",
            isolation="hardened",
            tui_enabled=False,
            interactive=False,
            task="do the thing",
            gate=lambda cfg, _b: cfg,
            chain_branch=None,
            base_sha="",
            untracked_at_start=frozenset(),
            resume_state_path=layout.session_dir / "loop_state.json",
            undo_forker=lambda: None,
            prompts=operator_prompts.OperatorPrompts(session_dir=layout.session_dir),
            ask_transcript_task=None,
        ),
        frontend=front,
        reporter=reporter.Reporter(out=said.append, err=said.append),
        events=event_log.EventSink(layout.logs_path),
        transcript_sink=None,  # type: ignore[arg-type]
        cwd=cwd,
        state_dir=tmp_path / "state",
    )

    assert end.rc == 0
    assert built["revision"].mode == "off"
    assert any("this surface has none" in s for s in said)
