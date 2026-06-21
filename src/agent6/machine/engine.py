# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine engine: a pure reducer loop driven by journaled facts (§5.1).

The engine executes one state at a time. The *only* impure step is
:meth:`World.run_tool` / :meth:`World.sleep_until` / :meth:`World.now`; its
result is written to the journal as a fact before the blackboard is reduced.
``reduce`` and ``next_state`` are pure, so:

* **Crash recovery**, on restart, recorded facts are replayed through the
  same pure reducer to rebuild the blackboard and position, then execution
  continues live from the last completed step.
* **Replay**, the identical reconstruction runs with ``live=False`` and no
  ``World`` at all, reproducing the recorded path offline for backtesting.

Phase 2 implements the four deterministic state kinds, ``tool``, ``branch``,
``wait``, ``terminal``. Phase 3 adds the ``agent`` kind, which runs a normal
agent6 loop through an injected :class:`World.run_agent` and captures the
schema-validated ``finish_run`` payload into the blackboard.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from agent6.machine.journal import (
    AgentFact,
    BranchFact,
    Fact,
    MachineEnd,
    MachineJournal,
    PendingWait,
    Snapshot,
    StepEvent,
    ToolFact,
    WaitFact,
)
from agent6.machine.model import (
    AgentState,
    BranchState,
    MachineSpec,
    StateSpec,
    TerminalState,
    ToolState,
    WaitState,
    validate_finish_payload,
)
from agent6.machine.predicate import evaluate, parse_predicate
from agent6.machine.template import parse_template, render_command, render_string, render_value
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.types import JailPolicy, SandboxProfile

__all__ = [
    "AgentExecResult",
    "AgentRequest",
    "EngineError",
    "LiveWorld",
    "MachineResult",
    "ToolExecResult",
    "World",
    "drive",
]


class EngineError(Exception):
    """Raised when a machine cannot be executed (bad data, unsupported kind)."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------
# The world boundary, the only impure surface.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolExecResult:
    """The observable result of running one `tool` command."""

    exit_code: int
    stdout: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """What the engine asks the world to run for one `agent` state."""

    prompt: str
    timeout_s: float
    # Optional per-state overrides mirrored from ``AgentState``. ``None``
    # means "fall back to the effective config" in the world implementation.
    # `model` is optional too: a `machine run` agent state always sets it
    # (AgentState.model is min_length=1), but `machine create`'s authoring
    # agent has no state and must INHERIT the operator's worker model -- an
    # empty-string override there overwrote the worker model with "" and failed
    # min_length validation, breaking the command outright.
    model: str | None = None
    provider: str | None = None
    thinking: str | None = None
    temperature: float | None = None
    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    # Workflow mode for the nested loop: "agent" (default) for a machine
    # `agent` state, a read-only structured-output judge; "run" for an agent
    # state that opted into coding work; "machine" for the `machine create`
    # authoring agent. machine_agent maps anything else to "run".
    mode: str = "agent"


@dataclass(frozen=True, slots=True)
class AgentExecResult:
    """The observable result of one agent loop.

    ``reason`` is the agent loop's stop reason (e.g. ``"finish_run"``,
    ``"budget_exhausted"``, ``"timeout"``, ``"max_iterations"``); ``payload`` is
    the structured object the agent passed to ``finish_run`` (``None`` if it
    never called it or passed no structured result). ``usd`` and the token
    counts report the slice this agent loop spent, summed into machine-level
    spend for ``machine status`` (§6).
    """

    reason: str
    payload: dict[str, Any] | None
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class World(Protocol):
    """Everything the engine is allowed to observe from the outside."""

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, allow_network: bool = False
    ) -> ToolExecResult: ...

    def run_agent(self, request: AgentRequest) -> AgentExecResult: ...

    def now(self) -> float: ...

    def sleep_until(self, wake_epoch: float) -> Literal["tick", "signal"]: ...


_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")


@dataclass(frozen=True, slots=True)
class LiveWorld:
    """Production :class:`World`: tools go through the jail, waits really sleep.

    A ``wait`` blocks in-process until its absolute instant (§4.3) or until an
    operator drops a ``signal`` file in the machine directory, whichever comes
    first. Because the wake instant is journaled, a future persisted-wake
    driver replays the identical file with no format change.

    ``agent`` states are delegated to an injected ``agent_runner`` so the engine
    module need not import the provider / workflow stack; the CLI wires the real
    runner (loading the effective config, building a provider and the loop). When no
    runner is configured, reaching an ``agent`` state fails loudly.
    """

    cwd: Path
    journal: MachineJournal
    agent_runner: Callable[[AgentRequest], AgentExecResult] | None = None
    poll_interval_s: float = 0.5
    profile: SandboxProfile = "strict"
    # Paths made read-only in every tool jail, the running machine's own
    # `.asm.toml` + `scripts/` bundle, so a tool can't rewrite its own machine
    # logic or bundled scripts mid-run (set by the CLI).
    protect_paths: tuple[Path, ...] = ()
    # The machine's persistent, writable scratch dir: granted RW in every tool
    # jail and surfaced to scripts as $AGENT6_MACHINE_DATA_DIR. It lives out of
    # the workspace (under the per-repo state dir) and persists across
    # iterations, so it is where a `tool` keeps DURABLE state (a built venv,
    # caches). cwd is writable too, but it is the repo, not durable machine
    # state. Set by the CLI to <instance>/data.
    data_dir: Path | None = None

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, allow_network: bool = False
    ) -> ToolExecResult:
        # The engine is the host-netns supervisor, so an opt-in tool's jail
        # gets the host network (it inherits the engine's netns); a non-opt-in
        # tool gets a fresh empty netns. Whether opt-in is permitted at all is
        # gated by the CLI at startup (sandbox.tool_network), so by the time we
        # run, `allow_network` is authoritative.
        env_list = [(key, os.environ[key]) for key in _SAFE_ENV_KEYS if key in os.environ]
        # Writable HOME for toolchain caches (go/cargo/pip); the jail's /tmp is
        # writable on both profiles. Mirrors the run_command jail env.
        env_list.append(("HOME", "/tmp/agent6-home"))  # noqa: S108 - resolved inside the jail
        extra_rw: tuple[Path, ...] = ()
        if self.data_dir is not None:
            # Grant RW on the data dir + tell the script where it is. This is the
            # portable way to persist across iterations (hardened tool jails are
            # otherwise read-only); the journal still records every transition.
            #
            # Export the data dir RELATIVE to cwd, not the host abspath: under
            # `strict` the jail pivots cwd to /workspace, so the host abspath
            # doesn't exist inside the jail, but the relative path resolves
            # against the (jail-set) cwd on every profile. Fall back to abspath
            # if it somehow isn't under cwd.
            try:
                data_value = str(self.data_dir.relative_to(self.cwd))
            except ValueError:
                data_value = str(self.data_dir)
            env_list.append(("AGENT6_MACHINE_DATA_DIR", data_value))
            extra_rw = (self.data_dir,)
        policy = JailPolicy(
            cwd=self.cwd,
            argv=argv,
            profile=self.profile,
            env=tuple(env_list),
            allow_network=allow_network,
            extra_protect_paths=self.protect_paths,
            extra_rw_paths=extra_rw,
            timeout_s=float(timeout_s),
        )
        try:
            result = run_in_jail(policy)
        except subprocess.TimeoutExpired:
            return ToolExecResult(exit_code=124, stdout="", timed_out=True)
        except JailUnavailableError as exc:
            raise EngineError(f"jail unavailable: {exc}") from exc
        return ToolExecResult(exit_code=result.returncode, stdout=result.stdout, timed_out=False)

    def run_agent(self, request: AgentRequest) -> AgentExecResult:
        if self.agent_runner is None:
            raise EngineError("machine reached an `agent` state but no agent runner is configured")
        return self.agent_runner(request)

    def now(self) -> float:
        return time.time()

    def sleep_until(self, wake_epoch: float) -> Literal["tick", "signal"]:
        while True:
            if self.journal.take_signal():
                return "signal"
            remaining = wake_epoch - time.time()
            if remaining <= 0:
                return "tick"
            time.sleep(min(remaining, self.poll_interval_s))


# --------------------------------------------------------------------------
# Result.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MachineResult:
    status: Literal["ok", "failed", "incomplete", "waiting"]
    reason: str
    state: str
    transitions: int


# --------------------------------------------------------------------------
# Pure blackboard helpers.
# --------------------------------------------------------------------------


def initial_blackboard(spec: MachineSpec) -> dict[str, Any]:
    blackboard: dict[str, Any] = {}
    for name, var in spec.vars.operator.items():
        blackboard[name] = var.value
    for name, var in spec.vars.code.items():
        blackboard[name] = var.default
    for name, var in spec.vars.agent.items():
        blackboard[name] = var.default
    return blackboard


def _apply_capture(state: ToolState, stdout: str, blackboard: dict[str, Any]) -> None:
    capture = state.capture
    if capture is None:
        return
    try:
        result_obj: Any = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise EngineError(f"tool stdout is not valid JSON for capture: {exc}") from exc
    if capture.stdout_json is not None:
        blackboard[capture.stdout_json] = result_obj
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": result_obj}
        for target, template_text in capture.set.items():
            template = parse_template(template_text)
            blackboard[target] = render_value(template, scope, where=f"state capture.set.{target}")


def _apply_agent_capture(state: AgentState, payload: Any, blackboard: dict[str, Any]) -> None:
    capture = state.capture
    if capture.finish_json is not None:
        blackboard[capture.finish_json] = payload
        return
    if capture.set is not None:
        scope: dict[str, Any] = {**blackboard, "result": payload}
        for target, template_text in capture.set.items():
            template = parse_template(template_text)
            blackboard[target] = render_value(template, scope, where=f"agent capture.set.{target}")


def reduce(state: StateSpec, fact: Fact, blackboard: dict[str, Any]) -> dict[str, Any]:
    """Apply a journaled *fact* to the blackboard, returning a new dict."""
    updated = dict(blackboard)
    if (
        isinstance(state, ToolState)
        and isinstance(fact, ToolFact)
        and not fact.timed_out
        and fact.exit_code == 0
    ):
        _apply_capture(state, fact.stdout, updated)
    elif isinstance(state, AgentState) and isinstance(fact, AgentFact) and fact.outcome == "ok":
        _apply_agent_capture(state, fact.payload, updated)
    return updated


# --------------------------------------------------------------------------
# Pure branch routing.
# --------------------------------------------------------------------------


def _route_branch(state: BranchState, blackboard: Mapping[str, object]) -> tuple[int, str, str]:
    for index, clause in enumerate(state.when):
        if clause.else_ is not None:
            return index, "else", clause.goto
        assert clause.if_ is not None
        if evaluate(parse_predicate(clause.if_), blackboard):
            return index, clause.if_, clause.goto
    # validate_semantics guarantees a final `else`, so this is unreachable.
    raise EngineError(f"branch fell through with no matching clause: {state.when!r}")


# --------------------------------------------------------------------------
# Wait timing.
# --------------------------------------------------------------------------


def _compute_wake(state: WaitState, blackboard: Mapping[str, object], now: float) -> float:
    if state.every_secs is not None:
        rendered = render_string(parse_template(state.every_secs), blackboard, where="every_secs")
        try:
            seconds = int(rendered)
        except ValueError as exc:
            raise EngineError(f"`every_secs` did not render to an integer: {rendered!r}") from exc
        if seconds < 0:
            raise EngineError(f"`every_secs` is negative: {seconds}")
        return now + seconds
    if state.until is not None:
        rendered = render_string(parse_template(state.until), blackboard, where="until")
        try:
            moment = datetime.fromisoformat(rendered)
        except ValueError as exc:
            raise EngineError(f"`until` is not an ISO-8601 instant: {rendered!r}") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.timestamp()
    raise EngineError(
        "`cron` wait timing is not implemented in the v1 runtime (Phase 4 persisted-wake)"
    )


def _fire_persisted_wait(
    state: WaitState,
    blackboard: Mapping[str, object],
    journal: MachineJournal,
    world: World,
    state_name: str,
) -> tuple[str, str, Fact] | None:
    """Arm-or-fire a `wait` without blocking (``--exit-on-wait``, §6).

    On first reaching the wait, the absolute wake instant is computed once and
    persisted so re-invocations compare against the same instant. Returns the
    ``(label, goto, fact)`` triple when the wait fires (a signal arrived or the
    instant has passed), clearing the persisted record; returns ``None`` when
    the wait is not yet ready, leaving the record persisted for the caller to
    yield on.
    """
    pending = journal.read_pending_wait()
    if pending is None or pending.state != state_name:
        wake = _compute_wake(state, blackboard, world.now())
        pending = PendingWait(state=state_name, wake_epoch=wake)
        journal.write_pending_wait(pending)
    if journal.take_signal():
        journal.clear_pending_wait()
        return (
            "signal",
            state.on["signal"],
            WaitFact(wake_epoch=pending.wake_epoch, woke_by="signal"),
        )
    if world.now() >= pending.wake_epoch:
        journal.clear_pending_wait()
        return "tick", state.on["tick"], WaitFact(wake_epoch=pending.wake_epoch, woke_by="tick")
    return None


# --------------------------------------------------------------------------
# One impure step.
# --------------------------------------------------------------------------


def _agent_outcome(
    spec: MachineSpec, state: AgentState, result: AgentExecResult
) -> Literal["ok", "failed", "budget_exhausted", "timeout"]:
    if result.reason == "budget_exhausted":
        return "budget_exhausted"
    if result.reason == "timeout":
        return "timeout"
    if result.reason == "finish_run" and result.payload is not None:
        problems = validate_finish_payload(spec, state.output_schema, result.payload)
        if not problems:
            return "ok"
    return "failed"


def _execute(
    spec: MachineSpec, state: StateSpec, blackboard: Mapping[str, object], world: World
) -> tuple[str, str, Fact]:
    if isinstance(state, ToolState):
        argv = render_command(state.command, blackboard, where="command")
        # Under the explicit-only model a tool reaches the network iff it set
        # allow_network = "allow" (the operator-set ceiling + hardened limits are
        # enforced as machine-run startup refusals). "auto"/"block" → isolated.
        result = world.run_tool(
            tuple(argv),
            float(state.timeout_secs),
            allow_network=state.allow_network == "allow",
        )
        if result.timed_out:
            label = "timeout"
        elif result.exit_code != 0:
            label = "nonzero"
        else:
            label = "ok"
        fact: Fact = ToolFact(
            exit_code=result.exit_code, stdout=result.stdout, timed_out=result.timed_out
        )
        return label, state.on[label], fact
    if isinstance(state, WaitState):
        wake = _compute_wake(state, blackboard, world.now())
        woke_by = world.sleep_until(wake)
        return woke_by, state.on[woke_by], WaitFact(wake_epoch=wake, woke_by=woke_by)
    if isinstance(state, BranchState):
        index, label, goto = _route_branch(state, blackboard)
        return label, goto, BranchFact(clause_index=index)
    if isinstance(state, AgentState):
        prompt = render_string(parse_template(state.prompt), blackboard, where="agent prompt")
        result = world.run_agent(
            AgentRequest(
                # "inherit" -> no override (None), so the world uses the
                # operator's effective worker model.
                model=None if state.model == "inherit" else state.model,
                prompt=prompt,
                timeout_s=float(state.timeout_secs),
                provider=state.provider,
                thinking=state.thinking,
                temperature=state.temperature,
                max_usd=state.usd_limit,
                max_input_tokens=state.max_input_tokens,
                max_output_tokens=state.max_output_tokens,
                # Per-state: "agent" (default) is a read-only structured-output
                # judge; "run" lets the state do real coding work (opt-in).
                mode=state.mode,
            )
        )
        outcome = _agent_outcome(spec, state, result)
        payload = result.payload if outcome == "ok" else None
        return (
            outcome,
            state.on[outcome],
            AgentFact(
                outcome=outcome,
                reason=result.reason,
                payload=payload,
                usd=result.usd,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
        )
    raise EngineError(f"cannot execute terminal state directly: {state!r}")


# --------------------------------------------------------------------------
# The driver.
# --------------------------------------------------------------------------


def drive(  # noqa: PLR0911, PLR0912
    spec: MachineSpec,
    journal: MachineJournal,
    world: World | None,
    *,
    live: bool,
    exit_on_wait: bool = False,
) -> MachineResult:
    """Run or replay *spec* against its *journal*.

    With ``live=True`` (``machine run``) the engine recovers from any existing
    journal, then continues to a terminal state, appending new facts. With
    ``live=False`` (``machine replay``) it only reconstructs the recorded path
    and reports where the journal ends; *world* is ignored.

    With ``exit_on_wait=True`` (``machine run --exit-on-wait``) the engine makes
    all the progress it can, but the first time it reaches a ``wait`` that is
    not yet ready it persists the absolute wake instant and returns a
    ``"waiting"`` result instead of blocking, for an external scheduler
    (systemd timer / cron) to re-invoke and resume (§6).
    """
    events = journal.read()
    if events and isinstance(events[-1], MachineEnd):
        end = events[-1]
        return MachineResult(end.status, end.reason, end.state, end.transitions)

    blackboard = initial_blackboard(spec)
    state = spec.initial
    transitions = 0
    spent_usd = 0.0

    if not events and live:
        journal.ensure_dirs()
        journal.begin(machine=spec.machine, version=spec.version)

    # Rebuild from recorded facts (recovery / replay).
    for event in events:
        if not isinstance(event, StepEvent):
            continue
        blackboard = reduce(spec.states[state], event.fact, blackboard)
        if isinstance(event.fact, AgentFact):
            spent_usd += event.fact.usd
        state = event.goto
        transitions = event.seq + 1

    if not live:
        current = spec.states.get(state)
        if isinstance(current, TerminalState):
            return MachineResult(current.status, current.reason, state, transitions)
        return MachineResult(
            "incomplete", "journal ends before a terminal state", state, transitions
        )

    if world is None:  # pragma: no cover - defensive
        raise EngineError("live execution requires a World")

    while True:
        current = spec.states[state]
        if isinstance(current, TerminalState):
            journal.append(
                MachineEnd(
                    ts=_now_iso(),
                    status=current.status,
                    reason=current.reason,
                    state=state,
                    transitions=transitions,
                )
            )
            journal.write_snapshot(Snapshot(seq=transitions, state=state, blackboard=blackboard))
            return MachineResult(current.status, current.reason, state, transitions)
        if transitions >= spec.budget.max_transitions:
            reason = f"max_transitions ({spec.budget.max_transitions}) exceeded"
            journal.append(
                MachineEnd(
                    ts=_now_iso(),
                    status="failed",
                    reason=reason,
                    state=state,
                    transitions=transitions,
                )
            )
            return MachineResult("failed", reason, state, transitions)
        usd_limit = spec.budget.usd_limit
        if usd_limit is not None and spent_usd >= usd_limit:
            reason = (
                f"{spec.budget.usd_field_name} (${usd_limit}) exceeded (spent ~${spent_usd:.4f})"
            )
            journal.append(
                MachineEnd(
                    ts=_now_iso(),
                    status="failed",
                    reason=reason,
                    state=state,
                    transitions=transitions,
                )
            )
            return MachineResult("failed", reason, state, transitions)

        if exit_on_wait and isinstance(current, WaitState):
            fired = _fire_persisted_wait(current, blackboard, journal, world, state)
            if fired is None:
                pending = journal.read_pending_wait()
                wake = pending.wake_epoch if pending is not None else world.now()
                return MachineResult(
                    "waiting", f"waiting in {state!r} until {wake}", state, transitions
                )
            label, goto, fact = fired
        else:
            label, goto, fact = _execute(spec, current, blackboard, world)
        journal.append(
            StepEvent(
                ts=_now_iso(),
                seq=transitions,
                state=state,
                label=label,
                goto=goto,
                fact=fact,
            )
        )
        blackboard = reduce(current, fact, blackboard)
        if isinstance(fact, AgentFact):
            spent_usd += fact.usd
        transitions += 1
        journal.write_snapshot(Snapshot(seq=transitions, state=goto, blackboard=blackboard))
        state = goto
