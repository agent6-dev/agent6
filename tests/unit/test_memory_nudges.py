# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-run memory write nudges: the one-shot add_memory advisory at the
first red-to-green verify flip, and the once-deferred finish_run backstop
after such a recovery. Both fire only in run mode with a memory store wired,
and only while nothing has been recorded (bench/longhorizon FINDINGS #2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.tools.results import AddMemoryResult, ExecResult
from agent6.workflows._conversation import AssistantTurn, Notice
from agent6.workflows._nudges import MEMORY_FINISH_NUDGE, MEMORY_FLIP_NUDGE
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
    _TurnState,  # pyright: ignore[reportPrivateUsage]
)


def _wf(**kw: Any) -> Workflow:
    kw.setdefault("state_dir", Path("/tmp/state"))
    return Workflow(
        root=Path("/tmp"),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        **kw,
    )


def _state(**kw: Any) -> _LoopState:
    return _LoopState(original_task="t", tool_calls=0, **kw)


def _turn(iteration: int = 1, **kw: Any) -> _TurnState:
    return _TurnState(iteration=iteration, resp=MagicMock(), assistant=AssistantTurn((), ()), **kw)


def _verify(wf: Workflow, state: _LoopState, turn: _TurnState, rc: int) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "run_verify_command",
        ExecResult(returncode=rc, stdout="", stderr="", duration_s=0.0, exec_failed=False),
    )


def _notice_texts(turn: _TurnState) -> list[str]:
    return [item.text for item in turn.tool_results if isinstance(item, Notice)]


def test_flip_advisory_fires_once_at_first_red_green_flip() -> None:
    wf = _wf()
    state = _state()
    fail = _turn(1)
    _verify(wf, state, fail, rc=1)
    assert state.verify_ever_failed is True
    assert fail.verify_flipped_green is False

    flip = _turn(2)
    _verify(wf, state, flip, rc=0)
    assert flip.verify_flipped_green is True
    wf._turn_notices(state, flip)  # pyright: ignore[reportPrivateUsage]
    assert MEMORY_FLIP_NUDGE in _notice_texts(flip)
    assert state.memory_flip_nudged is True

    # A second recovery does not re-nudge.
    again = _turn(3)
    _verify(wf, state, again, rc=1)
    _verify(wf, state, again, rc=0)
    assert again.verify_flipped_green is True
    wf._turn_notices(state, again)  # pyright: ignore[reportPrivateUsage]
    assert MEMORY_FLIP_NUDGE not in _notice_texts(again)


def test_flip_advisory_needs_a_prior_red_verify() -> None:
    wf = _wf()
    state = _state()
    green = _turn(1)
    _verify(wf, state, green, rc=0)
    assert green.verify_flipped_green is False
    wf._turn_notices(state, green)  # pyright: ignore[reportPrivateUsage]
    assert _notice_texts(green) == []


def test_flip_advisory_suppressed_without_store_write_or_run_mode() -> None:
    for wf, state_kw in (
        (_wf(state_dir=None), {}),
        (_wf(), {"memory_written": True}),
        (_wf(mode="ask"), {}),
    ):
        state = _state(last_verify_ok=False, verify_ever_failed=True, **state_kw)
        flip = _turn(2)
        _verify(wf, state, flip, rc=0)
        wf._turn_notices(state, flip)  # pyright: ignore[reportPrivateUsage]
        assert _notice_texts(flip) == []


def test_add_memory_dispatch_marks_memory_written() -> None:
    wf = _wf()
    state = _state()
    turn = _turn(1)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "add_memory",
        AddMemoryResult(id="x" * 26, scope="facts", created_at="2026"),
    )
    assert state.memory_written is True


def test_finish_gate_defers_once_then_honours() -> None:
    wf = _wf()
    state = _state(verify_ever_failed=True, last_verify_ok=True)

    first = _turn(5, finish_signal="done", finish_payload={"k": "v"})
    wf._gate_memory_finish(state, first)  # pyright: ignore[reportPrivateUsage]
    assert first.finish_signal is None
    assert first.finish_payload is None
    assert MEMORY_FINISH_NUDGE in _notice_texts(first)
    assert state.memory_finish_nudged is True

    second = _turn(6, finish_signal="done")
    wf._gate_memory_finish(state, second)  # pyright: ignore[reportPrivateUsage]
    assert second.finish_signal == "done"
    assert _notice_texts(second) == []


def test_finish_gate_quiet_without_a_recovery_or_after_a_write() -> None:
    wf = _wf()
    cases = [
        # Verify never failed: a smooth run is never interrogated.
        (wf, _state(last_verify_ok=True)),
        # Still red at finish: nothing proven to record.
        (wf, _state(verify_ever_failed=True, last_verify_ok=False)),
        # The worker already recorded something.
        (wf, _state(verify_ever_failed=True, last_verify_ok=True, memory_written=True)),
        # No memory store wired.
        (_wf(state_dir=None), _state(verify_ever_failed=True, last_verify_ok=True)),
        # Not a run-mode workflow.
        (_wf(mode="ask"), _state(verify_ever_failed=True, last_verify_ok=True)),
    ]
    for gated_wf, state in cases:
        turn = _turn(5, finish_signal="done")
        gated_wf._gate_memory_finish(state, turn)  # pyright: ignore[reportPrivateUsage]
        assert turn.finish_signal == "done"
        assert _notice_texts(turn) == []
        assert state.memory_finish_nudged is False
