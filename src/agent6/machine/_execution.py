# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Execute machine states and reduce their observed facts."""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from typing import Any, Literal

from agent6.machine import _contracts, _semantics, predicate, schema
from agent6.machine import journal as machine_journal
from agent6.machine import template as machine_template

# A data-driven state failure ends the machine cleanly, never as a poison StepEvent.
STATE_RUNTIME_ERRORS = (
    _contracts.StateRuntimeError,
    predicate.PredicateError,
    machine_template.TemplateError,
)


def initial_blackboard(spec: schema.MachineSpec) -> dict[str, Any]:
    """Return the blackboard of a fresh instance: every variable at its declared value."""
    blackboard: dict[str, Any] = {}
    for name, var in spec.vars.operator.items():
        blackboard[name] = var.value
    for name, var in spec.vars.code.items():
        blackboard[name] = var.default
    for name, var in spec.vars.agent.items():
        blackboard[name] = var.default
    return blackboard


def apply_capture(
    spec: schema.MachineSpec,
    state: schema.ToolState,
    stdout: str,
    blackboard: dict[str, Any],
) -> None:
    """Apply a tool's capture to the blackboard in place.

    Raises:
        StateRuntimeError: The stdout is not JSON, or does not match the output schema; the
            capture gate halts the machine before a poison fact is journaled.
    """
    capture = state.capture
    if capture is None:
        return
    try:
        result_obj: Any = machine_journal.scrub_lone_surrogates(json.loads(stdout))
    except json.JSONDecodeError as exc:
        raise _contracts.StateRuntimeError(
            f"tool stdout is not valid JSON for capture: {exc}"
        ) from exc
    if state.output_schema is not None:
        problems = _semantics.validate_record_payload(
            spec.schemas, state.output_schema, result_obj, where="tool stdout"
        )
        if problems:
            raise _contracts.StateRuntimeError(
                "tool stdout does not match output_schema: " + "; ".join(problems)
            )
    if capture.stdout_json is not None:
        blackboard[capture.stdout_json] = result_obj
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": result_obj}
        for target, template_text in capture.set.items():
            template = machine_template.parse_template(template_text)
            blackboard[target] = machine_template.render_value(
                template, scope, where=f"state capture.set.{target}"
            )


def apply_agent_capture(state: schema.AgentState, payload: Any, blackboard: dict[str, Any]) -> None:
    """Apply an agent's capture of its validated payload to the blackboard in place."""
    capture = state.capture
    if capture.finish_json is not None:
        blackboard[capture.finish_json] = payload
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": payload}
        for target, template_text in capture.set.items():
            template = machine_template.parse_template(template_text)
            blackboard[target] = machine_template.render_value(
                template, scope, where=f"agent capture.set.{target}"
            )


def reduce(
    spec: schema.MachineSpec,
    state: schema.StateSpec,
    fact: machine_journal.Fact,
    blackboard: dict[str, Any],
) -> dict[str, Any]:
    """Apply a journaled fact to the blackboard.

    Args:
        spec: The machine.
        state: The state that produced the fact.
        fact: The fact.
        blackboard: The blackboard before the step.

    Returns:
        A new blackboard.

    Raises:
        StateRuntimeError: The captured value does not match its schema or cannot render.
    """
    updated = dict(blackboard)
    if (
        isinstance(state, schema.ToolState)
        and isinstance(fact, machine_journal.ToolFact)
        and not fact.timed_out
        and fact.exit_code == 0
    ):
        apply_capture(spec, state, fact.stdout, updated)
    elif (
        isinstance(state, schema.AgentState)
        and isinstance(fact, machine_journal.AgentFact)
        and fact.outcome == "ok"
    ):
        problems = _semantics.validate_record_payload(
            spec.schemas, state.output_schema, fact.payload, where="agent payload"
        )
        if problems:
            raise _contracts.StateRuntimeError(
                "agent payload does not match output_schema: " + "; ".join(problems)
            )
        apply_agent_capture(state, fact.payload, updated)
    return updated


def route_branch(
    state: schema.BranchState, blackboard: Mapping[str, object]
) -> tuple[int, str, str]:
    """Return the (clause index, label, goto) of the first clause that fires.

    Raises:
        EngineError: No clause fired, which `validate_semantics` rules out.
    """
    for index, clause in enumerate(state.when):
        if clause.else_ is not None:
            return index, "else", clause.goto
        assert clause.if_ is not None
        if predicate.evaluate(predicate.parse_predicate(clause.if_), blackboard):
            return index, clause.if_, clause.goto
    raise _contracts.EngineError(f"branch fell through with no matching clause: {state.when!r}")


def is_forever(state: schema.WaitState) -> bool:
    """Return whether the wait has no timer and parks until a poke."""
    return state.every_secs is None and state.until is None


def compute_wake(state: schema.WaitState, blackboard: Mapping[str, object], now: float) -> float:
    """Return the wait's absolute wake instant.

    Raises:
        StateRuntimeError: The rendered timing is not a positive integer or an ISO-8601
            instant, or the wait has no timer.
    """
    if state.every_secs is not None:
        rendered = machine_template.render_string(
            machine_template.parse_template(state.every_secs), blackboard, where="every_secs"
        )
        try:
            seconds = int(rendered)
        except ValueError as exc:
            raise _contracts.StateRuntimeError(
                f"`every_secs` did not render to an integer: {rendered!r}"
            ) from exc
        if seconds < 1:
            raise _contracts.StateRuntimeError(f"`every_secs` must be >= 1: {seconds}")
        return now + seconds
    if state.until is not None:
        rendered = machine_template.render_string(
            machine_template.parse_template(state.until), blackboard, where="until"
        )
        try:
            moment = datetime.datetime.fromisoformat(rendered)
        except ValueError as exc:
            raise _contracts.StateRuntimeError(
                f"`until` is not an ISO-8601 instant: {rendered!r}"
            ) from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=datetime.UTC)
        return moment.timestamp()
    raise _contracts.StateRuntimeError("a timerless `wait` has no wake instant to compute")


def arm_pending_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    state_name: str,
    seq: int,
) -> machine_journal.PendingWait:
    """Return the pending wait of this visit, arming a fresh one when none is.

    The absolute instant is persisted before anything waits on it, so a resume compares
    against the same instant; the seq tells this visit from an earlier visit's uncleared
    record.
    """
    pending = journal.read_pending_wait()
    if pending is None or pending.state != state_name or pending.seq != seq:
        wake = None if is_forever(state) else compute_wake(state, blackboard, world.now())
        pending = machine_journal.PendingWait(state=state_name, wake_epoch=wake, seq=seq)
        journal.write_pending_wait(pending)
    return pending


def block_on_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    state_name: str,
    seq: int,
) -> tuple[str, str, machine_journal.Fact] | None:
    """Block on a wait in the foreground.

    The driver clears the pending record once the transition is journaled: a stale one would
    suppress the notify on re-entry and reuse its instant under a later `--exit-on-wait`.

    Returns:
        The (label, goto, fact) of the wake, or None when a stop request interrupted the
        sleep, leaving the wait armed and nothing journaled.
    """
    pending = arm_pending_wait(state, blackboard, journal, world, state_name, seq)
    woke = world.sleep_until(pending.wake_epoch)
    if woke.woke_by == "stop":
        return None
    return (
        woke.woke_by,
        state.on[woke.woke_by],
        machine_journal.WaitFact(
            wake_epoch=pending.wake_epoch, woke_by=woke.woke_by, payload=woke.payload
        ),
    )


def fire_persisted_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: _contracts.World,
    state_name: str,
    seq: int,
) -> tuple[str, str, machine_journal.Fact] | None:
    """Arm or fire a wait without blocking (`--exit-on-wait`).

    Returns:
        The (label, goto, fact) when a poke arrived or the instant has passed, or None when
        the wait is not ready, its record persisted for the caller to yield on.
    """
    pending = arm_pending_wait(state, blackboard, journal, world, state_name, seq)
    signaled, payload = journal.take_signal()
    if signaled:
        return (
            "signal",
            state.on["signal"],
            machine_journal.WaitFact(
                wake_epoch=pending.wake_epoch, woke_by="signal", payload=payload
            ),
        )
    if pending.wake_epoch is not None and world.now() >= pending.wake_epoch:
        return (
            "tick",
            state.on["tick"],
            machine_journal.WaitFact(wake_epoch=pending.wake_epoch, woke_by="tick"),
        )
    return None


def tool_outcome(
    fact: machine_journal.ToolFact | _contracts.ToolExecResult,
) -> Literal["ok", "nonzero", "timeout"]:
    """Return the label a tool result routes on."""
    if fact.timed_out:
        return "timeout"
    if fact.exit_code != 0:
        return "nonzero"
    return "ok"


def agent_outcome(
    spec: schema.MachineSpec, state: schema.AgentState, result: _contracts.AgentExecResult
) -> Literal["ok", "failed", "budget_exhausted", "timeout"]:
    """Return the label an agent result routes on; `ok` needs a payload matching the schema."""
    if result.reason == "budget_exhausted":
        return "budget_exhausted"
    if result.reason == "timeout":
        return "timeout"
    if result.reason == "finish_session" and result.payload is not None:
        problems = _semantics.validate_record_payload(
            spec.schemas, state.output_schema, result.payload, where="finish_session payload"
        )
        if not problems:
            return "ok"
    return "failed"


def agent_usd_cap(state_cap: float | None, remaining: float | None) -> float | None:
    """Return the smaller of the state's own cap and the machine's unspent budget.

    The aggregate gate guards only state starts, so the child needs this cap to stay under
    the machine's max_usd.
    """
    if state_cap is None:
        return remaining
    if remaining is None:
        return state_cap
    return min(state_cap, remaining)


def execute(
    spec: schema.MachineSpec,
    state: schema.StateSpec,
    blackboard: Mapping[str, object],
    world: _contracts.World,
    *,
    seq: int = 0,
    state_name: str = "",
    remaining_usd: float | None = None,
) -> tuple[str, str, machine_journal.Fact]:
    """Execute one tool, branch or agent state against the world.

    Args:
        spec: The machine.
        state: The state to execute.
        blackboard: The current blackboard.
        world: The world.
        seq: The transition, for the agent state's own log.
        state_name: The state's name, likewise.
        remaining_usd: The machine's unspent budget, or None when uncapped.

    Returns:
        The label, the goto and the fact to journal.

    Raises:
        EngineError: The state is a terminal.
    """
    if isinstance(state, schema.ToolState):
        argv = machine_template.render_command(state.command, blackboard, where="command")
        result = world.run_tool(
            tuple(argv),
            float(state.timeout_secs),
            state=state_name,
            pass_env=state.pass_env,
            step_seq=seq,
        )
        label = tool_outcome(result)
        fact: machine_journal.Fact = machine_journal.ToolFact(
            exit_code=result.exit_code,
            stdout=result.stdout,
            timed_out=result.timed_out,
            stderr=result.stderr,
        )
        return label, state.on[label], fact
    if isinstance(state, schema.BranchState):
        index, label, goto = route_branch(state, blackboard)
        return label, goto, machine_journal.BranchFact(clause_index=index)
    if isinstance(state, schema.AgentState):
        prompt = machine_template.render_string(
            machine_template.parse_template(state.prompt), blackboard, where="agent prompt"
        )
        result = world.run_agent(
            _contracts.AgentRequest(
                prompt=prompt,
                timeout_s=float(state.timeout_secs),
                max_usd=agent_usd_cap(state.max_usd, remaining_usd),
                mode=state.mode,
                state_name=state_name,
                step_seq=seq,
                output_schema=state.output_schema,
                schemas=spec.schemas,
            )
        )
        outcome = agent_outcome(spec, state, result)
        payload = result.payload if outcome == "ok" else None
        return (
            outcome,
            state.on[outcome],
            machine_journal.AgentFact(
                outcome=outcome,
                reason=result.reason,
                payload=payload,
                usd=result.usd,
                usd_partial=result.usd_partial,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )
    raise _contracts.EngineError(f"cannot execute terminal state directly: {state!r}")
