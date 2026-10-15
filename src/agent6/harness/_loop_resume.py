# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Restore and persist execution snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent6 import git_ops, portable
from agent6.harness import _compaction, _conversation, _loop_state, _metric, _operator, _snapshot
from agent6.prompts import revision as prompts_revision

if TYPE_CHECKING:
    from agent6.harness import loop


def fold_interrupted_turn(
    harness: loop.Harness, snapshot: _snapshot.SessionSnapshot
) -> tuple[_snapshot.SessionSnapshot, tuple[str, ...]]:
    """Advance the snapshot past a turn the last execution died in, and drop the marker.

    The advanced snapshot is on disk before the marker goes, so no later death re-runs
    the turn's calls.

    Args:
        harness: The execution's harness.
        snapshot: The snapshot the execution resumes from.

    Returns:
        The snapshot to resume from, and the names of the calls that never returned.

    Raises:
        OSError: The advanced snapshot could not be written.
    """
    assert harness.resume_state_path is not None
    marker = harness.resume_state_path.parent / _snapshot.TURN_IN_FLIGHT_NAME
    flight = _snapshot.read_turn_marker(marker)
    interrupted: tuple[str, ...] = ()
    if flight is not None and flight.iteration == snapshot.next_iteration:
        try:
            snapshot, interrupted = _snapshot.fold_interrupted_turn(snapshot, flight)
        except ValueError as exc:
            harness.log(
                f"LOOP: dropped an in-flight turn record the snapshot does not lead to: {exc}"
            )
        else:
            portable.atomic_write(harness.resume_state_path, snapshot.model_dump_json())
    _snapshot.clear_turn_marker(marker)
    return snapshot, interrupted


def seed_carryover(
    harness: loop.Harness,
    state: _loop_state.LoopState,
    conversation: _conversation.Conversation,
    resume_from: _snapshot.SessionSnapshot | None,
) -> None:
    """Seed the execution's carried state and announce it for the read model.

    A resumed or forked execution re-announces its restored pins and elision counters,
    even when empty: the fold replaces on these events, so the surfaces never list a pin no
    restart will re-inject. A fresh run seeded with pins takes the same path.

    Args:
        harness: The execution's harness.
        state: The execution's state.
        conversation: The conversation the pins are announced into.
        resume_from: The snapshot a resumed execution restores; None for a fresh run.
    """
    if resume_from is not None:
        _loop_state.restore_completion_state(state, resume_from)
        state.verify.adopted = harness.adopted_on_resume
        carry_verify_verdict(harness, state, resume_from)
        harness.emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
        elided, gists = _compaction.count_elisions(conversation)
        harness.emit("loop.compact.restored", elided=elided, gists=gists)
    elif harness.initial_pins:
        # The pin owner applies the cap and the non-empty check; a refused pin is logged.
        for pin in harness.initial_pins:
            if not _operator.try_pin(state.pins, pin):
                harness.log(
                    f"  --pin refused (empty or over the {_operator.PINS_MAX_CHARS}-char cap)"
                )
                harness.emit("loop.pin.refused", chars=len(pin), limit=_operator.PINS_MAX_CHARS)
        harness.emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
        if state.pins:
            conversation.notice(prompts_revision.pinned_block(state.pins))


def carry_adopted_gate(
    harness: loop.Harness, snapshot: _snapshot.SessionSnapshot
) -> tuple[str, ...]:
    """Carry the gate a gateless run adopted in an earlier execution.

    Args:
        harness: The execution's harness.
        snapshot: The snapshot the execution resumes from.

    Returns:
        The snapshot's command when the config names none and the jail can run it; `()`
        otherwise, and the execution-start notice reads the gate as swapped.
    """
    argv = tuple(snapshot.verify_command)
    if harness.gate.configured or not argv or not harness.dispatcher.adopt_verify_command(argv):
        return ()
    harness.log(f"LOOP: verify gate carried from the last execution: {' '.join(argv)}")
    harness.emit(
        "loop.verify_inferred",
        command=list(argv),
        source="resumed",
        adopted_at=snapshot.next_iteration,
    )
    return argv


def carry_verify_verdict(
    harness: loop.Harness, state: _loop_state.LoopState, snap: _snapshot.SessionSnapshot
) -> None:
    """Carry the prior execution's verify observation when it still describes this tree.

    The chain tip must be the snapshot's and the worktree clean; an operator commit or edit
    in between fails closed, so the execution starts unobserved rather than wrongly green
    or red. `baseline_ok` is about the base commit, so a resume that moved the base (its
    earlier work landed) carries none.

    Args:
        harness: The execution's harness.
        state: The execution's state.
        snap: The snapshot the execution resumes from.
    """
    if snap.base_sha == harness.chain.base_sha:
        state.verify.baseline_ok = snap.baseline_ok
    state.standing.tools_mark = snap.standing_tools_mark
    state.standing.fruitless = snap.standing_fruitless
    state.ok_tool_calls = snap.ok_tool_calls
    if snap.last_verify_ok is None or not snap.head_sha:
        return
    try:
        if snap.head_sha != harness.chain.checkpoint_head_sha():
            return
        dirty = harness.chain.is_dirty()
    except (git_ops.GitError, OSError):
        return
    if not dirty:
        state.verify.last_ok = snap.last_verify_ok
        state.verify.edited_since = snap.edited_since_verify


def save_resume_snapshot(
    harness: loop.Harness,
    state: _loop_state.LoopState,
    messages: list[dict[str, Any]],
    *,
    next_iteration: int,
    write_checkpoint: bool = False,
) -> None:
    """Write the resume snapshot, and the turn's checkpoint when asked.

    Written before each model call and again after the turn's tool results land, so a
    crash after a non-idempotent tool resumes from after it. Every write advances the
    latest pointer; only the pre-call write owns `checkpoints/<next_iteration>.json`, so
    `fork --at-turn N` has one meaning. Atomic, so a crash mid-write keeps the prior
    snapshot. A None `resume_state_path` writes nothing.

    Args:
        harness: The execution's harness.
        state: The execution's state.
        messages: The conversation in wire form.
        next_iteration: The iteration the snapshot resumes at.
        write_checkpoint: Also write the turn's checkpoint file.
    """
    if harness.resume_state_path is None:
        return
    goal = harness.metrics.goal
    best = _metric.best_metric_sample(state.metric.history, goal=goal) if goal is not None else None
    snapshot = _snapshot.SessionSnapshot(
        system=state.system,
        messages=messages,
        tool_calls=state.tool_calls,
        next_iteration=next_iteration,
        root_task_id=state.root_task_id,
        original_task=state.original_task,
        operator_task=state.operator_task,
        base_sha=harness.chain.base_sha,
        verify_command=harness.gate.command(state.verify),
        review_rejections_total=state.gates.review_total,
        verify_ever_passed=state.verify.ever_passed,
        verify_ever_failed=state.verify.ever_failed,
        gateless_ever_edited=state.settled.gateless_ever_edited,
        parallel_groups_dispatched=state.parallel_groups_dispatched,
        pins=tuple(state.pins),
        metric_best_score=best.score if best is not None else None,
        metric_at_ceiling=state.metric.at_ceiling(),
        last_verify_ok=state.verify.last_ok,
        edited_since_verify=state.verify.edited_since,
        baseline_ok=state.verify.baseline_ok,
        verify_scoped=state.verify.scoped,
        verify_withheld=state.verify.withheld,
        memory_written=state.memory.written,
        memory_flip_nudged=state.memory.flip_nudged,
        standing_tools_mark=state.standing.tools_mark,
        standing_fruitless=state.standing.fruitless,
        ok_tool_calls=state.ok_tool_calls,
        head_sha=harness.chain.checkpoint_head_sha(),
        graph_version=harness.checkpoint_graph_version(),
    )
    blob = snapshot.model_dump_json()
    # Recovery state, not run output: an unwritable state dir warns once and never aborts.
    try:
        # The checkpoint first, then the latest pointer, so a fork keeps a durable target.
        if write_checkpoint:
            cp_dir = harness.resume_state_path.parent / "checkpoints"
            portable.atomic_write(cp_dir / f"{next_iteration:04d}.json", blob)
        portable.atomic_write(harness.resume_state_path, blob)
    except OSError as exc:
        if not harness.snapshot_write_failed:
            harness.snapshot_write_failed = True
            harness.log(
                f"LOOP: WARNING could not persist resume snapshot ({exc}); "
                "resume/fork are unavailable for this run, continuing anyway"
            )


def mark_turn(harness: loop.Harness, turn: _loop_state.TurnState) -> None:
    """Write the in-flight marker: the turn's calls and the results that returned so far.

    Args:
        harness: The execution's harness.
        turn: This turn's state.
    """
    if harness.resume_state_path is None:
        return
    _snapshot.write_turn_marker(
        harness.resume_state_path.parent / _snapshot.TURN_IN_FLIGHT_NAME,
        _snapshot.TurnInFlight(
            iteration=turn.iteration,
            assistant=turn.assistant.raw_content,
            results=tuple(
                item.content
                for item in turn.tool_results
                if isinstance(item, _conversation.ToolResultItem)
            ),
        ),
    )
