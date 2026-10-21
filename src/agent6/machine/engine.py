# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine engine: a pure reducer loop driven by journaled facts.

Every impure step is a `World` method; an observed result passes through `reduce`, then is
journaled before the returned blackboard replaces the current one. On restart the recorded
facts replay through the same reducer to rebuild the position (crash recovery); with
`live=False` and no world the replay alone reproduces the recorded path offline.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal

from agent6.machine import _contracts, _execution, schema
from agent6.machine import journal as machine_journal
from agent6.machine import template as machine_template


def _now_iso() -> str:
    """Return the current UTC instant as an ISO-8601 timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")


def _emit_notify(
    state: schema.StateSpec,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    state_name: str,
) -> None:
    """Journal a state's `notify` message on entry and fire the operator hook.

    Presentation only: the caller swallows a render error, so a notify never flips a
    terminal's status.

    Args:
        state: The state being entered; one without a `notify` emits nothing.
        blackboard: The values the message renders against.
        journal: The instance's journal.
        world: The world, whose operator hook fires.
        state_name: The state's name.
    """
    if state.notify is None:
        return
    message = machine_template.render_string(
        machine_template.parse_template(state.notify.message), blackboard, where="notify"
    )
    journal.append(
        machine_journal.MachineNotify(
            ts=_now_iso(), state=state_name, message=message, level=state.notify.level
        )
    )
    world.notify("notify", state_name, message, state.notify.level)


def _terminal_reason(state: schema.TerminalState, blackboard: Mapping[str, object]) -> str:
    """Return a terminal's `reason`, rendered against the blackboard as `notify` is.

    Presentation only: a reference that cannot render keeps the template in the text, with
    the error beside it, and never changes the status.

    Args:
        state: The terminal state.
        blackboard: The values the reason renders against.

    Returns:
        The rendered reason.
    """
    try:
        return machine_template.render_string(
            machine_template.parse_template(state.reason), blackboard, where="reason"
        )
    except _execution.STATE_RUNTIME_ERRORS as exc:
        return f"{state.reason} (reason not rendered: {exc})"


def _emit_end(
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    *,
    status: Literal["ok", "failed"],
    reason: str,
    state: str,
    transitions: int,
    unbooked: machine_journal.AgentFact | None = None,
) -> _contracts.MachineResult:
    """Journal a `machine.end` and fire the operator notify hook for it.

    Args:
        journal: The instance's journal.
        world: The world.
        status: `ok` or `failed`.
        reason: Why the machine ended.
        state: The state it ended in.
        transitions: The transitions taken.
        unbooked: An agent slice whose StepEvent was never written (its capture could not be
            reduced); its spend rides on the end event.

    Returns:
        The result the end records.
    """
    end = machine_journal.MachineEnd(
        ts=_now_iso(),
        status=status,
        reason=reason,
        state=state,
        transitions=transitions,
        usd=unbooked.usd if unbooked is not None else 0.0,
        usd_partial=unbooked.usd_partial if unbooked is not None else False,
        input_tokens=unbooked.input_tokens if unbooked is not None else 0,
        output_tokens=unbooked.output_tokens if unbooked is not None else 0,
    )
    journal.append(end)
    world.notify("end", state, reason, status)
    return _contracts.MachineResult.from_end(end)


def _end_failed(
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    state: str,
    transitions: int,
    exc: Exception,
    unbooked: machine_journal.AgentFact | None = None,
) -> _contracts.MachineResult:
    """Journal a failed end for a runtime state error.

    Args:
        journal: The instance's journal.
        world: The world, whose operator hook fires for the end.
        state: The state that failed.
        transitions: The transitions taken.
        exc: The state error, quoted in the end's reason.
        unbooked: An agent slice whose StepEvent was never written (its capture could not be
            reduced); its spend rides on the end event.

    Returns:
        The failed result.
    """
    return _emit_end(
        journal,
        world,
        status="failed",
        reason=f"state {state!r}: {exc}",
        state=state,
        transitions=transitions,
        unbooked=unbooked,
    )


@dataclasses.dataclass(slots=True)
class _EngineState:
    """The bookkeeping threaded through the replay and the live loop.

    Attributes:
        spec: The machine.
        journal: The instance's journal.
        world: The world; None for a replay.
        exit_on_wait: Park at a wait that is not ready instead of blocking.
        blackboard: The current blackboard, folded by the replay then advanced live.
        state: The current state's name.
        transitions: The transitions taken.
        spent_usd: The agent spend so far, for the max_usd check.
    """

    spec: schema.MachineSpec
    journal: machine_journal.MachineJournal
    world: _contracts.World | None
    exit_on_wait: bool
    blackboard: dict[str, Any]
    state: str
    transitions: int = 0
    spent_usd: float = 0.0


_STATE_FACT_KINDS: dict[type[schema.StateSpec], type[machine_journal.Fact]] = {
    schema.ToolState: machine_journal.ToolFact,
    schema.AgentState: machine_journal.AgentFact,
    schema.WaitState: machine_journal.WaitFact,
    schema.BranchState: machine_journal.BranchFact,
}


def _declared_gotos(state: schema.StateSpec) -> frozenset[str]:
    """Return every destination the state can legally journal."""
    if isinstance(state, schema.BranchState):
        return frozenset(clause.goto for clause in state.when)
    if isinstance(state, schema.ToolState | schema.AgentState | schema.WaitState):
        return frozenset(state.on.values())
    return frozenset()


def _validate_recorded_route(
    state: schema.StateSpec,
    event: machine_journal.StepEvent,
    blackboard: Mapping[str, object],
    remedy: str,
) -> None:
    """Hold a replayed event's route to the outcome its fact determines.

    Args:
        state: The state at the replayed position.
        event: The journaled step.
        blackboard: The replayed blackboard before the step.
        remedy: The fresh-start hint appended to each error.

    Raises:
        EngineError: The recorded clause, label or goto disagrees with the fact.
    """
    if isinstance(state, schema.BranchState) and isinstance(event.fact, machine_journal.BranchFact):
        clause_index, expected_label, expected_goto = _execution.route_branch(state, blackboard)
        if event.fact.clause_index != clause_index:
            raise _contracts.EngineError(
                f"journal branch fact selects clause {event.fact.clause_index} at seq"
                f" {event.seq}, but the replayed blackboard selects clause {clause_index}."
                f"{remedy}"
            )
    elif isinstance(state, schema.ToolState) and isinstance(event.fact, machine_journal.ToolFact):
        expected_label = _execution.tool_outcome(event.fact)
        expected_goto = state.on[expected_label]
    elif isinstance(state, schema.AgentState) and isinstance(event.fact, machine_journal.AgentFact):
        expected_label = event.fact.outcome
        expected_goto = state.on[expected_label]
    elif isinstance(state, schema.WaitState) and isinstance(event.fact, machine_journal.WaitFact):
        expected_label = event.fact.woke_by
        expected_goto = state.on[expected_label]
    else:  # the caller's fact-kind check makes this unreachable
        raise _contracts.EngineError(f"journal fact at seq {event.seq} cannot be routed")
    if event.label != expected_label:
        raise _contracts.EngineError(
            f"journal records label {event.label!r} at seq {event.seq}, but its fact"
            f" implies label {expected_label!r}.{remedy}"
        )
    if event.goto != expected_goto:
        raise _contracts.EngineError(
            f"journal records goto {event.goto!r} at seq {event.seq}, but its fact"
            f" implies goto {expected_goto!r}.{remedy}"
        )


def _rebuild_from_journal(eng: _EngineState, events: list[Any]) -> None:
    """Replay the recorded steps through the reducer, advancing the engine state in place.

    Every recorded field is held to the machine: the step's state matches the replayed
    position, seqs are contiguous, the fact kind fits the state and the goto is a declared
    edge.

    Args:
        eng: The engine state at the initial position.
        events: The journal's events.

    Raises:
        EngineError: The journal does not fit the machine, or a step cannot be reduced.
    """
    spec = eng.spec
    blackboard = eng.blackboard
    state = eng.state
    transitions = eng.transitions
    spent_usd = eng.spent_usd
    remedy = f" {machine_journal.fresh_start_remedy(eng.journal.root)}"
    for event in events:
        if isinstance(event, machine_journal.AttemptSpend):
            spent_usd += event.usd  # a crashed attempt's slice counts; it moves no position
            continue
        if not isinstance(event, machine_journal.StepEvent):
            continue
        state_spec = spec.states.get(state)
        if state_spec is None:
            raise _contracts.EngineError(
                f"journal references state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started)."
                f"{remedy}"
            )
        if event.state != state:
            raise _contracts.EngineError(
                f"journal diverges at seq {event.seq}: it records state"
                f" {event.state!r}, the replayed position is {state!r}.{remedy}"
            )
        if event.seq != transitions:
            raise _contracts.EngineError(
                f"journal is not contiguous: expected seq {transitions}, found"
                f" {event.seq} (state {state!r}).{remedy}"
            )
        expected_kind = _STATE_FACT_KINDS.get(type(state_spec))
        if expected_kind is None or not isinstance(event.fact, expected_kind):
            raise _contracts.EngineError(
                f"journal fact at seq {event.seq} is {type(event.fact).__name__},"
                f" which state {state!r} cannot produce.{remedy}"
            )
        if event.goto not in _declared_gotos(state_spec):
            raise _contracts.EngineError(
                f"journal goto {event.goto!r} at seq {event.seq} is not an edge"
                f" state {state!r} declares.{remedy}"
            )
        try:
            _validate_recorded_route(state_spec, event, blackboard, remedy)
            blackboard = _execution.reduce(spec, state_spec, event.fact, blackboard)
        except _execution.STATE_RUNTIME_ERRORS as exc:
            raise _contracts.EngineError(
                f"cannot replay journaled step at state {state!r}: {exc}"
            ) from exc
        if isinstance(event.fact, machine_journal.AgentFact):
            spent_usd += event.fact.usd
        state = event.goto
        transitions = event.seq + 1
    eng.blackboard = blackboard
    eng.state = state
    eng.transitions = transitions
    eng.spent_usd = spent_usd


def _run_live_loop(eng: _EngineState) -> _contracts.MachineResult:  # noqa: C901, PLR0911, PLR0912, PLR0915  # one branch per state kind
    """Continue live from where the journal ends.

    One state per iteration: execute, reduce, journal the fact, advance; until a terminal, a
    budget cap, a runtime state error, a stop request or an `--exit-on-wait` park.

    Args:
        eng: The engine state after the replay.

    Returns:
        How the drive ended.

    Raises:
        EngineError: No world, a state the loaded machine no longer declares, or a fault a
            state's data cannot explain.
    """
    spec = eng.spec
    journal = eng.journal
    exit_on_wait = eng.exit_on_wait
    world = eng.world
    if world is None:  # pragma: no cover - defensive
        raise _contracts.EngineError("live execution requires a World")
    blackboard = eng.blackboard
    state = eng.state
    transitions = eng.transitions
    spent_usd = eng.spent_usd
    while True:
        # A stop request parks here: the fact in flight is journaled, no end is written.
        if machine_journal.stop_requested(journal.root):
            machine_journal.clear_stop_request(journal.root)
            return _contracts.MachineResult(
                "stopped", "stop requested by the operator", state, transitions
            )
        current = spec.states.get(state)
        if current is None:
            raise _contracts.EngineError(
                f"journal resumes at state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started)."
                f" {machine_journal.fresh_start_remedy(journal.root)}"
            )
        # A notify fires on entry, at least once across a crash; an armed wait is not a fresh
        # entry, else every --exit-on-wait tick would re-page the operator for one park.
        already_parked = False
        if isinstance(current, schema.WaitState):
            pending = journal.read_pending_wait()
            already_parked = (
                pending is not None and pending.state == state and pending.seq == transitions
            )
        if not already_parked:
            try:
                _emit_notify(current, blackboard, journal, world, state)
            except _execution.STATE_RUNTIME_ERRORS as exc:
                # Non-fatal, never silent: the failure is journaled and the hook told.
                fail = f"notify failed: {exc}"
                with contextlib.suppress(machine_journal.JournalError):
                    journal.append(
                        machine_journal.MachineNotify(
                            ts=_now_iso(), state=state, message=fail, level="error"
                        )
                    )
                world.notify("notify", state, fail, "error")
        if isinstance(current, schema.TerminalState):
            result = _emit_end(
                journal,
                world,
                status=current.status,
                reason=_terminal_reason(current, blackboard),
                state=state,
                transitions=transitions,
            )
            journal.write_snapshot(
                machine_journal.Snapshot(seq=transitions, state=state, blackboard=blackboard)
            )
            return result
        if transitions >= spec.budget.max_transitions:
            reason = f"max_transitions ({spec.budget.max_transitions}) reached"
            return _emit_end(
                journal, world, status="failed", reason=reason, state=state, transitions=transitions
            )
        usd_limit = spec.budget.max_usd
        if usd_limit is not None and spent_usd >= usd_limit:
            reason = f"max_usd (${usd_limit}) exceeded (spent ~${spent_usd:.4f})"
            return _emit_end(
                journal, world, status="failed", reason=reason, state=state, transitions=transitions
            )
        remaining_usd = None if usd_limit is None else max(usd_limit - spent_usd, 0.0)

        try:
            if exit_on_wait and isinstance(current, schema.WaitState):
                fired = _execution.fire_persisted_wait(
                    current, blackboard, journal, world, state, transitions
                )
                if fired is None:
                    pending = journal.read_pending_wait()
                    if pending is not None and pending.wake_epoch is not None:
                        detail = f"until {pending.wake_at}"
                    else:
                        detail = "until a signal poke"
                    return _contracts.MachineResult("waiting", detail, state, transitions)
                label, goto, fact = fired
            elif isinstance(current, schema.WaitState):
                blocked = _execution.block_on_wait(
                    current, blackboard, journal, world, state, transitions
                )
                if blocked is None:
                    machine_journal.clear_stop_request(journal.root)
                    return _contracts.MachineResult(
                        "stopped", "stop requested by the operator", state, transitions
                    )
                label, goto, fact = blocked
            else:
                label, goto, fact = _execution.execute(
                    spec,
                    current,
                    blackboard,
                    world,
                    seq=transitions,
                    state_name=state,
                    remaining_usd=remaining_usd,
                )
        except _execution.STATE_RUNTIME_ERRORS as exc:
            return _end_failed(journal, world, state, transitions, exc)
        if isinstance(fact, machine_journal.WaitFact) and fact.woke_by == "signal":
            world.materialize_poke(fact.payload)
        # The capture is reduced before the step is journaled: a fact that cannot be reduced would
        # re-crash every later replay, so the machine ends here and the spend rides on the end.
        try:
            next_blackboard = _execution.reduce(spec, current, fact, blackboard)
        except _execution.STATE_RUNTIME_ERRORS as exc:
            return _end_failed(
                journal,
                world,
                state,
                transitions,
                exc,
                unbooked=fact if isinstance(fact, machine_journal.AgentFact) else None,
            )
        journal.append(
            machine_journal.StepEvent(
                ts=_now_iso(),
                seq=transitions,
                state=state,
                label=label,
                goto=goto,
                fact=fact,
            )
        )
        # The wake record and the poke's claim go only once their transition is durable.
        if isinstance(fact, machine_journal.WaitFact):
            journal.clear_pending_wait()
            if fact.woke_by == "signal":
                journal.ack_signal()
        blackboard = next_blackboard
        if isinstance(fact, machine_journal.AgentFact):
            spent_usd += fact.usd
        transitions += 1
        journal.write_snapshot(
            machine_journal.Snapshot(seq=transitions, state=goto, blackboard=blackboard)
        )
        state = goto


def drive(
    spec: schema.MachineSpec,
    journal: machine_journal.MachineJournal,
    world: _contracts.World | None,
    *,
    live: bool,
    exit_on_wait: bool = False,
    base_commit: str | None = None,
) -> _contracts.MachineResult:
    """Run or replay a machine against its journal.

    Args:
        spec: The machine.
        journal: The instance's journal.
        world: The world; ignored by a replay.
        live: Recover from the journal and continue (`machine run`), or only reconstruct the
            recorded path (`machine replay`).
        exit_on_wait: At the first wait that is not ready, persist its instant and return
            `waiting` for an external scheduler to re-invoke.
        base_commit: The starting commit, recorded only when creating the instance.

    Returns:
        How the drive ended.

    Raises:
        EngineError: The journal was started by another machine, is malformed, or disagrees
            with its own replay.
    """
    events = journal.read()
    if events and not isinstance(events[0], machine_journal.MachineBegin):
        raise _contracts.EngineError(
            "journal must start with a machine.begin event."
            f" {machine_journal.fresh_start_remedy(journal.root)}"
        )
    if any(isinstance(event, machine_journal.MachineBegin) for event in events[1:]):
        raise _contracts.EngineError(
            "journal contains more than one machine.begin event."
            f" {machine_journal.fresh_start_remedy(journal.root)}"
        )
    if any(isinstance(event, machine_journal.MachineEnd) for event in events[:-1]):
        raise _contracts.EngineError(
            "journal contains events after machine.end."
            f" {machine_journal.fresh_start_remedy(journal.root)}"
        )

    # The instance is keyed by the machine id alone, so another file can land on the journal.
    begin = events[0] if events else None
    if isinstance(begin, machine_journal.MachineBegin) and (
        begin.machine != spec.machine or begin.version != spec.version
    ):
        raise _contracts.EngineError(
            f"this journal was started by machine {begin.machine!r} v{begin.version},"
            f" but the file declares {spec.machine!r} v{spec.version}. A different"
            " machine reused the id, or the file changed since this instance began."
            f" {machine_journal.fresh_start_remedy(journal.root)}"
        )

    if not events and live:
        journal.ensure_dirs()
        journal.begin(machine=spec.machine, version=spec.version, base_commit=base_commit)

    eng = _EngineState(
        spec=spec,
        journal=journal,
        world=world,
        exit_on_wait=exit_on_wait,
        blackboard=_execution.initial_blackboard(spec),
        state=spec.initial,
    )
    _rebuild_from_journal(eng, events)

    end = events[-1] if events and isinstance(events[-1], machine_journal.MachineEnd) else None
    if end is not None:
        if end.state != eng.state or end.transitions != eng.transitions:
            raise _contracts.EngineError(
                "journal machine.end disagrees with its replayed position:"
                f" records state {end.state!r} after {end.transitions} transitions,"
                f" replay reaches {eng.state!r} after {eng.transitions}."
                f" {machine_journal.fresh_start_remedy(journal.root)}"
            )
        return _contracts.MachineResult.from_end(end)

    if not live:
        current = spec.states.get(eng.state)
        if isinstance(current, schema.TerminalState):
            return _contracts.MachineResult(
                current.status,
                _terminal_reason(current, eng.blackboard),
                eng.state,
                eng.transitions,
            )
        return _contracts.MachineResult(
            "incomplete", "journal ends before a terminal state", eng.state, eng.transitions
        )

    return _run_live_loop(eng)
