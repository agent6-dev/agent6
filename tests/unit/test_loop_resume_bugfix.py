# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resume pins: the end-of-iteration snapshot, the completion scalars, the final checkpoint.

The snapshot never replays executed tools, the completion scalars survive a resume, and the final
checkpoint commits a dirty worktree on a gated run's success exit.
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.harness._chain import RunChain
from agent6.harness._compactor import Compactor
from agent6.harness._conversation import Conversation
from agent6.harness._metric import MetricGuard
from agent6.harness._metric import MetricSample as _MetricSample
from agent6.harness._provider_call import CallSettings
from agent6.harness._snapshot import (
    SNAPSHOT_VERSION,
    SessionSnapshot,
    load_session_snapshot,
)
from agent6.harness.loop import (
    Harness,
    LoopState,
)
from agent6.tools.results import ExecResult, RawResult

# The `[git]` surface the loop reads, empty as a real Config carries it unset.
_GIT_STUB = SimpleNamespace(
    control="agent6",
    commit_per_step=True,
    commit=SimpleNamespace(
        checkpoint=SimpleNamespace(message="agent6"), name="", email="", trailer=""
    ),
)


def _silent(_: str) -> None:
    return None


def _wf(
    root: Path | None = None,
    *,
    ref: str | None = None,
    fallback_parent: str | None = None,
    **kw: Any,
) -> Harness:
    """Return a loop over the root with a chain wired as run.py wires it; no root, no chain."""
    if root is not None and fallback_parent is None:
        fallback_parent = (
            sp.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    defaults: dict[str, Any] = {
        "chain": RunChain(
            root or Path("/tmp"),
            ref=ref or ("refs/agent6/test" if root is not None else None),
            fallback_parent=fallback_parent,
        ),
        "config": MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            harness=MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_retries=2,
            ),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    return Harness(**defaults)


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


# --- #12: completion-relevant scalars round-trip + restore -----------------


def test_snapshot_persists_completion_scalars(tmp_path: Path) -> None:
    """verify_ever_passed, gateless_ever_edited and the metric summary survive the snapshot."""
    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=2)
    state.verify.ever_passed = True
    state.verify.scoped = True
    state.settled.gateless_ever_edited = True
    state.metric.history.append(_MetricSample(label="x", score=27.0, returncode=0, at_ceiling=True))
    state.system, state.tool_calls, state.root_task_id = "s", 2, None
    wf._save_resume_snapshot(state, [], next_iteration=4)  # pyright: ignore[reportPrivateUsage]
    loaded = load_session_snapshot(snap)
    assert loaded.verify_ever_passed is True
    assert loaded.verify_scoped is True
    assert loaded.gateless_ever_edited is True
    assert loaded.metric_best_score == 27.0
    assert loaded.metric_at_ceiling is True


def test_snapshot_preserves_run_lifetime_memory_finish_state(tmp_path: Path) -> None:
    """A resume keeps a prior red, the memory notices and the once-only finish deferral."""
    from agent6.harness.loop import restore_completion_state

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.ever_failed = True
    state.memory.written = True
    state.memory.flip_nudged = True
    state.memory.finish_nudged = True
    state.system, state.tool_calls, state.root_task_id = "s", 0, None
    wf._save_resume_snapshot(state, [], next_iteration=3)  # pyright: ignore[reportPrivateUsage]

    loaded = load_session_snapshot(snap)
    fresh = LoopState(original_task="t", tool_calls=0)
    restore_completion_state(fresh, loaded)

    assert fresh.verify.ever_failed is True
    assert fresh.memory.written is True
    assert fresh.memory.flip_nudged is True
    assert fresh.memory.finish_nudged is True


def test_completed_prose_turn_is_snapshotted_before_the_boundary(tmp_path: Path) -> None:
    """A prose turn plus its nudge is a completed iteration, so a stop there resumes after it."""
    from agent6.harness._snapshot import SessionResult
    from agent6.providers import ProviderResponse

    repo = tmp_path / "repo"
    _git_repo(repo)
    snap_path = tmp_path / "loop_state.json"
    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="answer in prose",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "answer in prose"}]},
    )
    wf = _wf(
        root=repo,
        provider=provider,
        resume_state_path=snap_path,
        call=CallSettings(retry_count=0, retry_delay_s=0.0),
        max_iterations=5,
    )
    stopped = SessionResult(
        completed=False, reason="interactive_stop", summary="", iterations=1, tool_calls=0
    )

    def stop_at_boundary(*_a: object, **_k: object) -> SessionResult:
        return stopped

    with mock.patch.object(Harness, "_operator_boundary", stop_at_boundary):
        result = wf.run("do the task")
    assert result.reason == "interactive_stop"
    loaded = load_session_snapshot(snap_path)
    assert loaded.next_iteration == 2  # the prose turn is a COMPLETED iteration
    dumped = json.dumps(loaded.messages)
    assert "answer in prose" in dumped  # the model's turn survives the stop
    assert "[harness]" in dumped  # and so does the nudge that answered it


def test_snapshot_persists_and_restores_parallel_group_counter(tmp_path: Path) -> None:
    """The /parallel group counter is run-lifetime state, persisted like the completion scalars.

    Lane ids and the imported branches embed it; a reset rebuilds a prior group's ids.
    """
    from agent6.harness.loop import (
        restore_completion_state,
    )

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=0)
    state.parallel_groups_dispatched = 2
    state.system, state.tool_calls, state.root_task_id = "s", 0, None
    wf._save_resume_snapshot(state, [], next_iteration=4)  # pyright: ignore[reportPrivateUsage]
    loaded = load_session_snapshot(snap)
    assert loaded.parallel_groups_dispatched == 2

    fresh = LoopState(original_task="t", tool_calls=0)
    restore_completion_state(fresh, loaded)
    assert fresh.parallel_groups_dispatched == 2  # the next dispatch is p3, not p1


def test_snapshot_persists_and_restores_pins(tmp_path: Path) -> None:
    """Operator /pin instructions are run-lifetime state; an older snapshot loads with none."""
    from agent6.harness.loop import (
        restore_completion_state,
    )

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = LoopState(original_task="t", tool_calls=0)
    state.pins.extend(["never touch schema files", "goal:\nship X"])
    state.system, state.tool_calls, state.root_task_id = "s", 0, None
    wf._save_resume_snapshot(state, [], next_iteration=4)  # pyright: ignore[reportPrivateUsage]
    loaded = load_session_snapshot(snap)
    assert loaded.pins == ("never touch schema files", "goal:\nship X")

    fresh = LoopState(original_task="t", tool_calls=0)
    restore_completion_state(fresh, loaded)
    assert fresh.pins == ["never touch schema files", "goal:\nship X"]

    # Pre-pins snapshot (no `pins` key) still loads: additive default.
    raw = json.loads(snap.read_text(encoding="utf-8"))
    del raw["pins"]
    snap.write_text(json.dumps(raw), encoding="utf-8")
    assert load_session_snapshot(snap).pins == ()


def test_pre_version_bump_snapshot_refused_loudly(tmp_path: Path) -> None:
    """A snapshot from an older SNAPSHOT_VERSION refuses to resume or fork with a clear reason."""
    import pytest

    snap = tmp_path / "loop_state.json"
    snap.write_text(
        json.dumps(
            {
                "version": 1,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="predates a state-format change"):
        load_session_snapshot(snap)


def test_malformed_snapshot_shapes_fail_loud(tmp_path: Path) -> None:
    """A wrong-shape snapshot (null, list, scalar, missing key) raises a clean ValueError."""
    import pytest

    snap = tmp_path / "loop_state.json"
    for bad in ("null", "[]", "123", '"str"'):
        snap.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_session_snapshot(snap)
    # current version, wrong internals: missing required keys, and a non-list messages
    snap.write_text(json.dumps({"version": SNAPSHOT_VERSION, "system": "s"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed run-state snapshot"):
        load_session_snapshot(snap)
    snap.write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION,
                "system": "s",
                "messages": "oops",
                "tool_calls": 0,
                "next_iteration": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="messages"):
        load_session_snapshot(snap)


def test_resume_seeds_state_from_snapshot_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_drive_loop` restores verify_ever_passed and an at-ceiling metric sample on resume.

    One iteration that finishes at once; the loop saw the restored history, so no early-finish
    rejection.
    """
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run")

    captured: dict[str, Any] = {}
    orig = MetricGuard.at_ceiling

    def _spy(guard: MetricGuard) -> bool:
        captured["at_ceiling"] = orig(guard)
        return captured["at_ceiling"]

    monkeypatch.setattr(MetricGuard, "at_ceiling", _spy)
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            metric_best_score=27.0,
            metric_at_ceiling=True,
        ),
    )
    assert result.completed is True
    assert result.reason == "finish_session"
    # The early-finish guard consulted the restored at-ceiling history.
    assert captured.get("at_ceiling") is True


class _EventCapture:
    def __init__(self, path: Path = Path("logs.jsonl")) -> None:
        self.events: list[dict[str, Any]] = []
        self.path = path  # EventSink.path: the log file the emits land in

    def emit(self, event_type: str, /, **fields: Any) -> None:
        self.events.append({"type": event_type, **fields})


def test_resume_reannounces_restored_pins_for_the_read_model() -> None:
    """A resumed execution emits loop.pin.restored with the snapshot's pins.

    A fork's fresh log has no pin.added events, so the surfaces would show zero pins.
    """
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            pins=("keep A", "ship X"),
        ),
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["keep A", "ship X"]
    assert restored[0]["count"] == 2


def test_resume_start_carries_the_execution_identity(tmp_path: Path) -> None:
    """loop.resume.start opens a resumed or forked execution's log with session_id and mode."""
    from agent6.harness._snapshot import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()
    (start,) = [e for e in ev.events if e["type"] == "loop.resume.start"]
    assert start["session_id"] == "tidy-otter-AB12CD"
    assert start["mode"] == "run"


def test_resume_with_no_pins_still_corrects_a_stale_pin_added() -> None:
    """A pin lost to a crash is re-added from the fold on resume, even with an empty snapshot.

    The fold replaces on the corrective event; guarding it on a non-empty list let the stale pin
    stand.
    """
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
            pins=(),
        ),
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored, "an empty restore must still be announced"
    assert restored[0]["pins"] == []
    assert restored[0]["count"] == 0


# --- #3: end-of-iteration snapshot (no replay of executed tools) -----------


def test_snapshot_written_after_tool_dispatch_advances_iteration(tmp_path: Path) -> None:
    """After a full iteration the snapshot advances to the next iteration with the executed turn."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    snap = repo / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    # Iter 1: a run_command tool_use (a side effect). Iter 2: finish_session.
    provider.call.side_effect = [
        SimpleNamespace(
            text="",
            tool_uses=({"id": "a1", "name": "run_command", "input": {"command": "echo hi"}},),
            refused={},
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "a1",
                        "name": "run_command",
                        "input": {"command": "echo hi"},
                    }
                ]
            },
        ),
        SimpleNamespace(
            text="",
            tool_uses=({"id": "f1", "name": "finish_session", "input": {"summary": "done"}},),
            refused={},
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "finish_session",
                        "input": {"summary": "x"},
                    }
                ]
            },
        ),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = ExecResult(
        returncode=0, stdout="hi", stderr="", duration_s=0.0, exec_failed=False
    )
    dispatcher.set_run_root_node_id = MagicMock()

    events: list[dict[str, Any]] = []
    wf = _wf(
        root=repo,
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        resume_state_path=snap,
    )
    orig_save = wf._save_resume_snapshot  # pyright: ignore[reportPrivateUsage]
    orig_call = provider.call
    orig_compact = Compactor.compact

    def _spy_save(state: Any, messages: list[dict[str, Any]], **kw: Any) -> None:
        orig_save(state, messages, **kw)
        events.append(
            {
                "kind": "save",
                "next_iteration": kw["next_iteration"],
                "messages": json.loads(json.dumps(messages)),
            }
        )

    def _spy_call(**kw: Any) -> Any:
        events.append({"kind": "provider_call"})
        return orig_call(**kw)

    def _spy_compact(compactor: Compactor, msgs: Any, state: Any, **kw: Any) -> bool:
        events.append({"kind": "compact"})
        return orig_compact(compactor, msgs, state, **kw)

    wf._save_resume_snapshot = _spy_save  # type: ignore[method-assign]
    provider.call = _spy_call
    compact_spy = mock.patch.object(Compactor, "compact", _spy_compact)
    compact_spy.start()
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        ),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="go",
    )
    compact_spy.stop()

    # The snapshot advancing to next_iteration=2 is written at the end of iter 1: no crash window.
    kinds = [ev["kind"] for ev in events]
    first_call = kinds.index("provider_call")
    second_compact = next(i for i, k in enumerate(kinds) if k == "compact" and i > first_call)
    end_of_iter_saves = [
        ev
        for i, ev in enumerate(events)
        if first_call < i < second_compact and ev["kind"] == "save"
    ]
    assert end_of_iter_saves, (
        "expected an end-of-iteration snapshot between the 1st provider call"
        " and iter-2's compaction (the post-tool-dispatch crash window)"
    )
    post = [s for s in end_of_iter_saves if s["next_iteration"] == 2]
    assert post, "end-of-iteration snapshot must advance next_iteration to 2"
    msgs = post[0]["messages"]
    assert any(m.get("role") == "assistant" for m in msgs), "assistant turn must be snapshotted"
    has_tool_result = any(
        m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        for m in msgs
    )
    assert has_tool_result, "executed tool_result must be in the advanced snapshot"


# --- #10: final checkpoint commits a dirty worktree on a gated run ---------


def test_final_checkpoint_commits_dirty_worktree_on_gated_run(tmp_path: Path) -> None:
    """An uncommitted run_command edit on a gated run is captured by the final checkpoint."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("pytest", "-q"),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    emitted: list[tuple[str, dict[str, Any]]] = []

    class _Sink:
        def emit(self, event_type: str, **fields: Any) -> None:
            emitted.append((event_type, fields))

    wf = _wf(root=repo, config=config, mode="run", events=_Sink())
    # Worker edited a file via run_command; never re-verified, never committed.
    (repo / "edit.txt").write_text("a real edit\n")
    head_before = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf.checkpoints.final(iteration=5)

    head_after = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head_after == head_before, "the operator's HEAD never moves"
    chain = sp.run(
        ["git", "rev-parse", "refs/agent6/test"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert chain != head_before, "the chain must capture the edit on exit"
    shown = sp.run(
        ["git", "show", "refs/agent6/test:edit.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert shown == "a real edit\n"
    subject = sp.run(
        ["git", "log", "-1", "--pretty=%s", "refs/agent6/test"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "checkpoint" in subject
    # A diff.updated is emitted too, so the folds count the final commit.
    kinds = [k for k, _ in emitted]
    assert "loop.auto_commit" in kinds and "diff.updated" in kinds


def test_final_checkpoint_noop_when_clean_or_not_run_mode(tmp_path: Path) -> None:
    """No commit when the tree is clean, and never in non-run mode."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("pytest",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    head = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf_clean = _wf(root=repo, config=config, mode="run")
    wf_clean.checkpoints.final(iteration=1)
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )

    # Dirty tree, but plan mode -> still no commit.
    (repo / "edit.txt").write_text("plan-mode edit\n")
    wf_plan = _wf(root=repo, config=config, mode="plan")
    wf_plan.checkpoints.final(iteration=1)
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )


def test_a_forked_execution_reports_the_elisions_its_context_carries() -> None:
    """A fork re-announces its elision markers: it copies the checkpoint but not logs.jsonl."""
    from agent6.harness._compaction import ELISION_GIST_PREFIX, ELISION_PREFIX

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={"content": [{"type": "tool_use", "id": "t1", "name": "finish_session", "input": {}}]},
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture()
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run", events=ev)
    # A restored context carrying two bare elisions and one distilled gist.
    restored = Conversation.from_wire(
        [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": f"{ELISION_PREFIX}: x"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "b", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "b", "content": f"{ELISION_PREFIX}: y"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": f"{ELISION_GIST_PREFIX}: z",
                    }
                ],
            },
        ]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=restored,
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        resume_from=SessionSnapshot(
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=(),
        ),
    )
    restored_ev = [e for e in ev.events if e["type"] == "loop.compact.restored"]
    assert restored_ev, "the restored context's elisions were never announced"
    assert restored_ev[0]["elided"] == 3
    assert restored_ev[0]["gists"] == 1


def test_initial_pins_seed_a_fresh_run_out_of_band() -> None:
    """A `/parallel` lane inherits the coordinator's pins through `--pin`, not a task prefix.

    Seeding emits the same replace-fold event a restore does and keeps the pins in state.
    """
    from agent6.providers import ProviderResponse

    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    config = MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        harness=MagicMock(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=(),
            verify_when="never",
            verify_retries=2,
        ),
    )
    ev = _EventCapture()
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        initial_pins=("never touch schema files",),
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=conversation,
        tool_calls=0,
        start_iteration=1,
        original_task="go",
        root_task_id=None,
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["never touch schema files"]
    # The block the worker sees is the SAME one a restart re-shows.
    first_call = provider.call.call_args_list[0]
    messages = first_call.kwargs.get("messages") or first_call.args[1]
    flat = str(messages)
    assert "PINNED operator instructions (verbatim):" in flat
    assert "never touch schema files" in flat


def test_initial_pins_honor_the_cap_and_skip_empties() -> None:
    """`--pin` seeds pins through the same cap and non-empty check as `/pin`."""
    from agent6.harness._operator import PINS_MAX_CHARS
    from agent6.providers import ProviderResponse

    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "d"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "d"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    config = MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        harness=MagicMock(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=(),
            verify_when="never",
            verify_retries=2,
        ),
    )
    ev = _EventCapture()
    huge = "x" * (PINS_MAX_CHARS + 1)
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        initial_pins=("keep this", "", huge, "   "),  # 1 good, 1 empty, 1 over-cap, 1 blank
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=conversation,
        tool_calls=0,
        start_iteration=1,
        original_task="go",
        root_task_id=None,
    )
    restored = [e for e in ev.events if e["type"] == "loop.pin.restored"]
    assert restored and restored[0]["pins"] == ["keep this"]  # only the fitting one
    refused = [e for e in ev.events if e["type"] == "loop.pin.refused"]
    assert len(refused) == 3  # the empty, the over-cap, and the blank


def test_a_gate_swapped_between_executions_is_announced_to_the_worker(tmp_path: Path) -> None:
    """The system prompt is the run's, frozen at its start.

    Config that gains a verify command between executions swaps what judges the work; the notice
    says so.
    """
    from agent6.harness._snapshot import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=("pytest", "-q"),
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("make", "check"),  # the operator pinned one since
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()

    (swap,) = [e for e in ev.events if e["type"] == "loop.verify_swapped"]
    assert swap["was"] == ["pytest", "-q"] and swap["now"] == ["make", "check"]
    sent = provider.call.call_args.kwargs["messages"]
    told = json.dumps(sent)
    assert "was `pytest -q`" in told and "now `make check`" in told


def test_an_adopted_gate_carries_into_the_next_execution(tmp_path: Path) -> None:
    """A resumed execution starts with the gate the run adopted, so no swap notice fires."""
    from agent6.harness._snapshot import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=("pytest", "-q"),  # adopted in execution one; the config names none
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    dispatcher.adopt_verify_command.return_value = True
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()

    dispatcher.adopt_verify_command.assert_called_once_with(("pytest", "-q"))
    assert [e for e in ev.events if e["type"] == "loop.verify_swapped"] == []
    (carried,) = [e for e in ev.events if e["type"] == "loop.verify_inferred"]
    assert carried["command"] == ["pytest", "-q"] and carried["source"] == "resumed"
    # The execution's own snapshot names the carried gate as the one in force.
    written = json.loads(snap_path.read_text(encoding="utf-8"))
    assert tuple(written["verify_command"]) == ("pytest", "-q")


def test_a_green_verdict_survives_a_resume_after_the_run_committed(tmp_path: Path) -> None:
    """The snapshot's `head_sha` is the run's chain tip; the carry asks for dirt relative to it."""
    import subprocess

    from agent6.git_ops import chain_commit, chain_tip
    from agent6.git_ops import status as git_status
    from agent6.harness._snapshot import SessionSnapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "x.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    chain = "refs/agent6/run1/head"
    wf = _wf(
        root=repo,
        config=Config.model_validate({"harness": {"verify_command": ["true"]}}),
        ref=chain,
        fallback_parent=base,
    )
    # Execution one: the worker edits, the gate goes green, the harness chain-commits.
    (repo / "x.txt").write_text("the run's work\n", encoding="utf-8")
    assert chain_commit(repo, "iter 1", ref=chain, fallback_parent=base) is not None
    snap = SessionSnapshot.model_validate(
        {
            "system": "s",
            "messages": [],
            "tool_calls": 3,
            "next_iteration": 4,
            "root_task_id": None,
            "original_task": "t",
            "verify_command": ("true",),
            "last_verify_ok": True,
            "edited_since_verify": False,
            "head_sha": wf.chain.checkpoint_head_sha(),
        }
    )
    assert snap.head_sha == chain_tip(repo, chain) != git_status(repo).head_sha

    state = LoopState(original_task="t", tool_calls=0)
    wf._carry_verify_verdict(state, snap)  # pyright: ignore[reportPrivateUsage]

    assert state.verify.last_ok is True
    assert state.verify.green_and_untouched is True


def test_a_gate_withheld_between_executions_is_no_swap_for_the_worker(tmp_path: Path) -> None:
    """An execution that cannot run commands drops its gate before the loop sees the config.

    No notice and no swap event: no command can run, that one included.
    """
    from agent6.harness._snapshot import SessionSnapshot as _Snap

    session_dir = tmp_path / "sessions" / "runs" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    snap_path = session_dir / "loop_state.json"
    snap_path.write_text(
        _Snap(
            system="s",
            messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            tool_calls=0,
            next_iteration=3,
            root_task_id=None,
            original_task="go",
            verify_command=("pytest", "-q"),
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),  # dropped at execution start: commands are withheld
            metric=SimpleNamespace(goal="maximize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_session", "input": {"summary": "done"}},),
        refused={},
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_session",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})
    dispatcher.command_policy.return_value = "no"
    ev = _EventCapture(path=session_dir / "logs.jsonl")
    wf = _wf(
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        events=ev,
        resume_state_path=snap_path,
    )
    wf.resume()

    assert not [e for e in ev.events if e["type"] == "loop.verify_swapped"]
    told = json.dumps(provider.call.call_args.kwargs["messages"])
    assert "changed between executions" not in told
