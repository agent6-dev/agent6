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
import json
import pathlib
import shutil
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import pydantic

from agent6 import kinds, paths, portable
from agent6.machine import _semantics, predicate, schema
from agent6.machine import journal as machine_journal
from agent6.machine import template as machine_template
from agent6.sandbox import jail
from agent6.sessions import layout

__all__ = [
    "AgentExecResult",
    "AgentRequest",
    "EngineError",
    "LiveWorld",
    "MachineResult",
    "ToolExecResult",
    "WaitWake",
    "World",
    "drive",
]


class EngineError(Exception):
    """A machine cannot be executed: bad data, or a journal that does not fit it."""


class StateRuntimeError(EngineError):
    """A state met invalid data despite the load-time checks."""


# A data-driven state failure ends the machine cleanly, never as a poison StepEvent.
_STATE_RUNTIME_ERRORS = (
    StateRuntimeError,
    predicate.PredicateError,
    machine_template.TemplateError,
)


def _now_iso() -> str:
    """Return the current UTC instant as an ISO-8601 timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")


@dataclasses.dataclass(frozen=True, slots=True)
class ToolExecResult:
    """The observable result of running one `tool` command."""

    exit_code: int
    stdout: str
    timed_out: bool
    stderr: str = ""


class AgentRequest(pydantic.BaseModel):
    """What the engine asks the world to run for one `agent` state.

    The `request` block of `request.json` across the machine-agent subprocess boundary
    (the envelope is `MachineAgentRequest` in `app/machine_agent.py`); bytes pinned by
    `tests/unit/test_machine_agent_ipc.py`.

    Attributes:
        prompt: The rendered prompt.
        timeout_s: The wall-clock cap.
        model: The state's model, or None to inherit the operator's worker model (the
            `machine create` authoring agent has no state).
        provider: The `[providers.*]` entry, or None for the effective config's.
        effort: The reasoning effort override, or None.
        temperature: The sampling override, or None.
        max_usd: The slice's spend cap, or None.
        max_tokens_fallback: The unmetered token bound, or None.
        mode: The nested loop's mode: `agent` (a read-only structured judge) or `run`.
        state_name: The state, for its own watchable logs.jsonl; "" for the authoring agent.
        step_seq: The transition, likewise; 0 for the authoring agent.
        output_schema: The finish contract the execution refuses a non-conforming result by.
        schemas: The spec's schema table verbatim, so nested records resolve execution side.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    prompt: str
    timeout_s: float
    model: str | None = None
    provider: str | None = None
    effort: str | None = None
    temperature: float | None = None
    max_usd: float | None = None
    max_tokens_fallback: int | None = None
    mode: str = "agent"
    state_name: str = ""
    step_seq: int = 0
    output_schema: str | None = None
    schemas: dict[str, dict[str, schema.FieldSpec]] = pydantic.Field(default_factory=dict)


class AgentExecResult(pydantic.BaseModel):
    """The observable result of one agent loop.

    `result.json` across the machine-agent subprocess boundary (written by `run_one`,
    validated back in `app/machine_agent.py`); bytes pinned by
    `tests/unit/test_machine_agent_ipc.py`.

    Attributes:
        reason: The loop's stop reason: `finish_session`, `budget_exhausted`, `timeout`,
            `max_iterations`.
        payload: The structured object passed to `finish_session`, or None.
        usd: The slice's spend, summed into the machine's.
        usd_partial: `usd` is a known under-estimate (an unpriced model).
        input_tokens: The slice's input tokens.
        output_tokens: The slice's output tokens.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    reason: str
    payload: dict[str, Any] | None
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    @pydantic.field_validator("payload")
    @classmethod
    def _scrub_payload(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Replace lone surrogates, which `json.loads` accepts and `model_dump_json` refuses.

        Unscrubbed, the subprocess dies before writing `result.json` and the host routes a
        finished agent to its `on.failed` edge.

        Returns:
            The scrubbed payload, or None.
        """
        return value if value is None else machine_journal.scrub_lone_surrogates(value)


@dataclasses.dataclass(frozen=True, slots=True)
class WaitWake:
    """How a `wait` woke.

    Attributes:
        woke_by: A clock `tick`, an operator `signal` poke, or a `stop` request that
            interrupted the sleep (the wait stays armed, nothing is journaled).
        payload: The JSON a poke carried, journaled in the WaitFact; None for a bare poke
            or a tick.
    """

    woke_by: Literal["tick", "signal", "stop"]
    payload: Any = None


class World(Protocol):
    """Everything the engine may observe from the outside."""

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        network: kinds.NetworkMode = "none",
        pass_env: tuple[str, ...] = (),
    ) -> ToolExecResult:
        """Run one tool command."""
        ...

    def run_agent(self, request: AgentRequest) -> AgentExecResult:
        """Run one agent loop."""
        ...

    def now(self) -> float:
        """Return the clock as an epoch."""
        ...

    def sleep_until(self, wake_epoch: float | None) -> WaitWake:
        """Block until the instant or a poke; None is a timerless wait, until a poke."""
        ...

    def materialize_poke(self, payload: Any) -> None:
        """Write a poke payload where the next tool can read it."""
        ...

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        """Fire the operator notify hook for a `notify` message or the `machine.end`."""
        ...


# (argv, timeout_s, network, pass_env) -> the jail policy the shared builder produced.
ToolPolicyFactory = Callable[
    [tuple[str, ...], float, kinds.NetworkMode, tuple[str, ...]], kinds.JailPolicy
]


def _state_log_seq(p: pathlib.Path) -> int:
    """Return the seq of a `<seq>-<state>` log dir name, so the sort is numeric."""
    prefix = p.name.split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else -1


def _prune_state_logs(root: pathlib.Path, *, keep: int) -> None:
    """Keep the newest per-state log dirs, leaving room for the one about to be written.

    Best effort; the journal keeps the full history regardless.

    Args:
        root: The per-state log root.
        keep: The number to keep, including the next; 0 prunes nothing.
    """
    if keep == 0:
        return
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=_state_log_seq)
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - keep + 1)]:
        shutil.rmtree(stale, ignore_errors=True)


@dataclasses.dataclass(frozen=True, slots=True)
class LiveWorld:
    """The production world: tools go through the jail, waits sleep.

    The agent runner, the tool policy and the jail runner are injected by the CLI, so this
    module imports neither the harness nor config nor git.

    Attributes:
        cwd: The workspace.
        journal: The instance's journal.
        agent_runner: Runs an agent state given its request and its own logs.jsonl path;
            reaching an agent state without one fails loudly.
        poll_interval_s: How often a blocking wait checks for a poke or a stop.
        tool_policy: Builds each tool jail's policy from the shared builder with the machine
            deltas (bundle protect paths, the data dir grant), so a tool is confined like a
            run command.
        state_log_root: Where each agent state writes `<seq>-<state>/logs.jsonl`; None
            disables per-state logs.
        state_log_keep: How many per-state log dirs to keep (`[machine].state_log_keep`);
            0 keeps all.
        notify_hook: The operator's `[machine.notify].on_event` argv, run on the host; None
            for no hook (the front-ends still render the journaled events).
        data_dir: The machine's persistent scratch dir, `$AGENT6_MACHINE_DATA_DIR` to a tool,
            outside the workspace; where a tool keeps durable state.
        jail_runner: Executes one tool policy; the CLI overrides it for `mode = "run"`
            machines so tools run in the machine's own tree. None is the plain jail.
        on_wait: Fired once as a wait starts to block, so the foreground run names where it
            parked; it cannot affect the sleep.
    """

    cwd: pathlib.Path
    journal: machine_journal.MachineJournal
    agent_runner: Callable[[AgentRequest, pathlib.Path | None], AgentExecResult] | None = None
    poll_interval_s: float = 0.5
    tool_policy: ToolPolicyFactory | None = None
    state_log_root: pathlib.Path | None = None
    state_log_keep: int = 50
    notify_hook: Callable[[str, str, str, str], None] | None = None
    data_dir: pathlib.Path | None = None
    jail_runner: Callable[[kinds.JailPolicy], kinds.CommandResult] | None = None
    on_wait: Callable[[], None] | None = None

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        network: kinds.NetworkMode = "none",
        pass_env: tuple[str, ...] = (),
    ) -> ToolExecResult:
        """Run one tool command in its jail.

        The CLI gated `network` and `pass_env` at startup; the policy builder clamps the
        network to what the isolation level provides.

        Args:
            argv: The rendered command.
            timeout_s: The wall-clock cap.
            network: `host` or `none`.
            pass_env: The operator environment variables the command receives.

        Returns:
            The exit code, output and whether the command timed out.

        Raises:
            EngineError: No tool policy is wired, or the jail is unavailable.
        """
        if self.tool_policy is None:
            raise EngineError("LiveWorld has no tool_policy factory wired")
        try:
            policy = self.tool_policy(tuple(argv), float(timeout_s), network, pass_env)
            result = (self.jail_runner or jail.run_in_jail)(policy)
        except jail.JailUnavailableError as exc:
            raise EngineError(f"jail unavailable: {exc}") from exc
        # run_in_jail returns rc 124 on a timeout and never raises TimeoutExpired.
        return ToolExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            timed_out=result.returncode == 124,
            stderr=result.stderr,
        )

    def run_agent(self, request: AgentRequest) -> AgentExecResult:
        """Run one agent loop through the injected runner.

        Returns:
            The loop's result.

        Raises:
            EngineError: No agent runner is configured.
        """
        if self.agent_runner is None:
            raise EngineError("machine reached an `agent` state but no agent runner is configured")
        return self.agent_runner(request, self._state_log(request))

    def _state_log(self, request: AgentRequest) -> pathlib.Path | None:
        """Return the agent state's own logs.jsonl path, pruning old ones first, or None."""
        if self.state_log_root is None or not request.state_name:
            return None
        _prune_state_logs(self.state_log_root, keep=self.state_log_keep)
        return (
            self.state_log_root / f"{request.step_seq:04d}-{request.state_name}" / layout.LOGS_NAME
        )

    def now(self) -> float:
        """Return the wall clock as an epoch."""
        return time.time()

    def sleep_until(self, wake_epoch: float | None) -> WaitWake:
        """Block until the wake instant, a poke or a stop request, whichever comes first.

        Args:
            wake_epoch: The instant; None is a timerless wait, until a poke.

        Returns:
            How the wait woke.
        """
        if self.on_wait is not None:
            with contextlib.suppress(OSError):  # a dead terminal never stops a machine
                self.on_wait()
        while True:
            signaled, payload = self.journal.take_signal()
            if signaled:
                return WaitWake("signal", payload)
            if machine_journal.stop_requested(self.journal.root):
                return WaitWake("stop")  # the pending wait stays armed for the next run
            if wake_epoch is None:
                time.sleep(self.poll_interval_s)
                continue
            remaining = wake_epoch - time.time()
            if remaining <= 0:
                return WaitWake("tick")
            time.sleep(min(remaining, self.poll_interval_s))

    def materialize_poke(self, payload: Any) -> None:
        """Write a poke's payload to `$AGENT6_MACHINE_DATA_DIR/poke.json` for the next tool.

        Atomic, and called before the StepEvent is appended, so a durable step implies a
        durable file. A no-op without a data dir.
        """
        if self.data_dir is None:
            return
        paths.mkdir_for_real_user(self.data_dir)
        portable.atomic_write(self.data_dir / "poke.json", json.dumps(payload, sort_keys=True))

    def notify(self, kind: str, state: str, message: str, level: str) -> None:
        """Fire the operator hook, when one is wired."""
        if self.notify_hook is not None:
            self.notify_hook(kind, state, message, level)


@dataclasses.dataclass(frozen=True, slots=True)
class MachineResult:
    """How a drive ended.

    Attributes:
        status: `ok` or `failed` from a journaled end; `incomplete` for a replay that ends
            before a terminal; `waiting` for an `--exit-on-wait` park; `stopped` for a stop.
        reason: The terminal's reason, the failure, or where the machine parked.
        state: The state the machine is in.
        transitions: The transitions taken.
    """

    status: Literal["ok", "failed", "incomplete", "waiting", "stopped"]
    reason: str
    state: str
    transitions: int

    @classmethod
    def from_end(cls, end: machine_journal.MachineEnd) -> MachineResult:
        """Return the result a journaled end records."""
        return cls(end.status, end.reason, end.state, end.transitions)


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


def _apply_capture(
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
        raise StateRuntimeError(f"tool stdout is not valid JSON for capture: {exc}") from exc
    if state.output_schema is not None:
        problems = _semantics.validate_record_payload(
            spec.schemas, state.output_schema, result_obj, where="tool stdout"
        )
        if problems:
            raise StateRuntimeError(
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


def _apply_agent_capture(
    state: schema.AgentState, payload: Any, blackboard: dict[str, Any]
) -> None:
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
        _apply_capture(spec, state, fact.stdout, updated)
    elif (
        isinstance(state, schema.AgentState)
        and isinstance(fact, machine_journal.AgentFact)
        and fact.outcome == "ok"
    ):
        problems = _semantics.validate_record_payload(
            spec.schemas, state.output_schema, fact.payload, where="agent payload"
        )
        if problems:
            raise StateRuntimeError(
                "agent payload does not match output_schema: " + "; ".join(problems)
            )
        _apply_agent_capture(state, fact.payload, updated)
    return updated


def _route_branch(
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
    raise EngineError(f"branch fell through with no matching clause: {state.when!r}")


def _is_forever(state: schema.WaitState) -> bool:
    """Return whether the wait has no timer and parks until a poke."""
    return state.every_secs is None and state.until is None


def _compute_wake(state: schema.WaitState, blackboard: Mapping[str, object], now: float) -> float:
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
            raise StateRuntimeError(
                f"`every_secs` did not render to an integer: {rendered!r}"
            ) from exc
        if seconds < 1:
            raise StateRuntimeError(f"`every_secs` must be >= 1: {seconds}")
        return now + seconds
    if state.until is not None:
        rendered = machine_template.render_string(
            machine_template.parse_template(state.until), blackboard, where="until"
        )
        try:
            moment = datetime.datetime.fromisoformat(rendered)
        except ValueError as exc:
            raise StateRuntimeError(f"`until` is not an ISO-8601 instant: {rendered!r}") from exc
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=datetime.UTC)
        return moment.timestamp()
    raise StateRuntimeError("a timerless `wait` has no wake instant to compute")


def _arm_pending_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: World,
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
        wake = None if _is_forever(state) else _compute_wake(state, blackboard, world.now())
        pending = machine_journal.PendingWait(state=state_name, wake_epoch=wake, seq=seq)
        journal.write_pending_wait(pending)
    return pending


def _block_on_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: World,
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
    pending = _arm_pending_wait(state, blackboard, journal, world, state_name, seq)
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


def _fire_persisted_wait(
    state: schema.WaitState,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: World,
    state_name: str,
    seq: int,
) -> tuple[str, str, machine_journal.Fact] | None:
    """Arm or fire a wait without blocking (`--exit-on-wait`).

    Returns:
        The (label, goto, fact) when a poke arrived or the instant has passed, or None when
        the wait is not ready, its record persisted for the caller to yield on.
    """
    pending = _arm_pending_wait(state, blackboard, journal, world, state_name, seq)
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


def _tool_outcome(
    fact: machine_journal.ToolFact | ToolExecResult,
) -> Literal["ok", "nonzero", "timeout"]:
    """Return the label a tool result routes on."""
    if fact.timed_out:
        return "timeout"
    if fact.exit_code != 0:
        return "nonzero"
    return "ok"


def _agent_outcome(
    spec: schema.MachineSpec, state: schema.AgentState, result: AgentExecResult
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


def _agent_usd_cap(state_cap: float | None, remaining: float | None) -> float | None:
    """Return the smaller of the state's own cap and the machine's unspent budget.

    The aggregate gate guards only state starts, so the child needs this cap to stay under
    the machine's max_usd.
    """
    if state_cap is None:
        return remaining
    if remaining is None:
        return state_cap
    return min(state_cap, remaining)


def _execute(
    spec: schema.MachineSpec,
    state: schema.StateSpec,
    blackboard: Mapping[str, object],
    world: World,
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
        # A tool reaches the network only when it set `host`; startup refused what the operator
        # did not grant. `auto` is a network of its own: a state's processes die with the state.
        result = world.run_tool(
            tuple(argv),
            float(state.timeout_secs),
            network="host" if state.network == "host" else "none",
            pass_env=state.pass_env,
        )
        label = _tool_outcome(result)
        fact: machine_journal.Fact = machine_journal.ToolFact(
            exit_code=result.exit_code,
            stdout=result.stdout,
            timed_out=result.timed_out,
            stderr=result.stderr,
        )
        return label, state.on[label], fact
    if isinstance(state, schema.BranchState):
        index, label, goto = _route_branch(state, blackboard)
        return label, goto, machine_journal.BranchFact(clause_index=index)
    if isinstance(state, schema.AgentState):
        prompt = machine_template.render_string(
            machine_template.parse_template(state.prompt), blackboard, where="agent prompt"
        )
        result = world.run_agent(
            AgentRequest(
                model=None if state.model == "inherit" else state.model,
                prompt=prompt,
                timeout_s=float(state.timeout_secs),
                provider=state.provider,
                effort=state.effort,
                temperature=state.temperature,
                max_usd=_agent_usd_cap(state.max_usd, remaining_usd),
                max_tokens_fallback=state.max_tokens_fallback,
                mode=state.mode,
                state_name=state_name,
                step_seq=seq,
                output_schema=state.output_schema,
                schemas=spec.schemas,
            )
        )
        outcome = _agent_outcome(spec, state, result)
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
    raise EngineError(f"cannot execute terminal state directly: {state!r}")


def _emit_notify(
    state: schema.StateSpec,
    blackboard: Mapping[str, object],
    journal: machine_journal.MachineJournal,
    world: World,
    state_name: str,
) -> None:
    """Journal a state's `notify` message on entry and fire the operator hook.

    Presentation only: the caller swallows a render error, so a notify never flips a
    terminal's status.
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


def _emit_end(
    journal: machine_journal.MachineJournal,
    world: World,
    *,
    status: Literal["ok", "failed"],
    reason: str,
    state: str,
    transitions: int,
    unbooked: machine_journal.AgentFact | None = None,
) -> MachineResult:
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
    return MachineResult.from_end(end)


def _end_failed(
    journal: machine_journal.MachineJournal,
    world: World,
    state: str,
    transitions: int,
    exc: Exception,
    unbooked: machine_journal.AgentFact | None = None,
) -> MachineResult:
    """Journal a failed end for a runtime state error.

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
    world: World | None
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

    Raises:
        EngineError: The recorded clause, label or goto disagrees with the fact.
    """
    if isinstance(state, schema.BranchState) and isinstance(event.fact, machine_journal.BranchFact):
        clause_index, expected_label, expected_goto = _route_branch(state, blackboard)
        if event.fact.clause_index != clause_index:
            raise EngineError(
                f"journal branch fact selects clause {event.fact.clause_index} at seq"
                f" {event.seq}, but the replayed blackboard selects clause {clause_index}."
                f"{remedy}"
            )
    elif isinstance(state, schema.ToolState) and isinstance(event.fact, machine_journal.ToolFact):
        expected_label = _tool_outcome(event.fact)
        expected_goto = state.on[expected_label]
    elif isinstance(state, schema.AgentState) and isinstance(event.fact, machine_journal.AgentFact):
        expected_label = event.fact.outcome
        expected_goto = state.on[expected_label]
    elif isinstance(state, schema.WaitState) and isinstance(event.fact, machine_journal.WaitFact):
        expected_label = event.fact.woke_by
        expected_goto = state.on[expected_label]
    else:  # the caller's fact-kind check makes this unreachable
        raise EngineError(f"journal fact at seq {event.seq} cannot be routed")
    if event.label != expected_label:
        raise EngineError(
            f"journal records label {event.label!r} at seq {event.seq}, but its fact"
            f" implies label {expected_label!r}.{remedy}"
        )
    if event.goto != expected_goto:
        raise EngineError(
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
    remedy = " Archive the instance directory to start fresh."
    for event in events:
        if isinstance(event, machine_journal.AttemptSpend):
            spent_usd += event.usd  # a crashed attempt's slice counts; it moves no position
            continue
        if not isinstance(event, machine_journal.StepEvent):
            continue
        state_spec = spec.states.get(state)
        if state_spec is None:
            raise EngineError(
                f"journal references state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started);"
                f" archive the instance directory to start fresh: {eng.journal.root}"
            )
        if event.state != state:
            raise EngineError(
                f"journal diverges at seq {event.seq}: it records state"
                f" {event.state!r}, the replayed position is {state!r}.{remedy}"
            )
        if event.seq != transitions:
            raise EngineError(
                f"journal is not contiguous: expected seq {transitions}, found"
                f" {event.seq} (state {state!r}).{remedy}"
            )
        expected_kind = _STATE_FACT_KINDS.get(type(state_spec))
        if expected_kind is None or not isinstance(event.fact, expected_kind):
            raise EngineError(
                f"journal fact at seq {event.seq} is {type(event.fact).__name__},"
                f" which state {state!r} cannot produce.{remedy}"
            )
        if event.goto not in _declared_gotos(state_spec):
            raise EngineError(
                f"journal goto {event.goto!r} at seq {event.seq} is not an edge"
                f" state {state!r} declares.{remedy}"
            )
        try:
            _validate_recorded_route(state_spec, event, blackboard, remedy)
            blackboard = reduce(spec, state_spec, event.fact, blackboard)
        except _STATE_RUNTIME_ERRORS as exc:
            raise EngineError(f"cannot replay journaled step at state {state!r}: {exc}") from exc
        if isinstance(event.fact, machine_journal.AgentFact):
            spent_usd += event.fact.usd
        state = event.goto
        transitions = event.seq + 1
    eng.blackboard = blackboard
    eng.state = state
    eng.transitions = transitions
    eng.spent_usd = spent_usd


def _run_live_loop(eng: _EngineState) -> MachineResult:  # noqa: C901, PLR0911, PLR0912, PLR0915  # one branch per state kind
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
        raise EngineError("live execution requires a World")
    blackboard = eng.blackboard
    state = eng.state
    transitions = eng.transitions
    spent_usd = eng.spent_usd
    while True:
        # A stop request parks here: the fact in flight is journaled, no end is written.
        if machine_journal.stop_requested(journal.root):
            machine_journal.clear_stop_request(journal.root)
            return MachineResult("stopped", "stop requested by the operator", state, transitions)
        current = spec.states.get(state)
        if current is None:
            raise EngineError(
                f"journal resumes at state {state!r}, which the loaded machine no"
                " longer declares (the file was edited since this instance started);"
                f" archive the instance directory to start fresh: {journal.root}"
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
            except _STATE_RUNTIME_ERRORS as exc:
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
                reason=current.reason,
                state=state,
                transitions=transitions,
            )
            journal.write_snapshot(
                machine_journal.Snapshot(seq=transitions, state=state, blackboard=blackboard)
            )
            return result
        if transitions >= spec.budget.max_transitions:
            reason = f"max_transitions ({spec.budget.max_transitions}) exceeded"
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
                fired = _fire_persisted_wait(
                    current, blackboard, journal, world, state, transitions
                )
                if fired is None:
                    pending = journal.read_pending_wait()
                    if pending is not None and pending.wake_epoch is not None:
                        detail = f"until {pending.wake_at}"
                    else:
                        detail = "until a signal poke"
                    return MachineResult(
                        "waiting", f"waiting in {state!r} {detail}", state, transitions
                    )
                label, goto, fact = fired
            elif isinstance(current, schema.WaitState):
                blocked = _block_on_wait(current, blackboard, journal, world, state, transitions)
                if blocked is None:
                    machine_journal.clear_stop_request(journal.root)
                    return MachineResult(
                        "stopped", "stop requested by the operator", state, transitions
                    )
                label, goto, fact = blocked
            else:
                label, goto, fact = _execute(
                    spec,
                    current,
                    blackboard,
                    world,
                    seq=transitions,
                    state_name=state,
                    remaining_usd=remaining_usd,
                )
        except _STATE_RUNTIME_ERRORS as exc:
            return _end_failed(journal, world, state, transitions, exc)
        if isinstance(fact, machine_journal.WaitFact) and fact.woke_by == "signal":
            world.materialize_poke(fact.payload)
        # The capture is reduced before the step is journaled: a fact that cannot be reduced would
        # re-crash every later replay, so the machine ends here and the spend rides on the end.
        try:
            next_blackboard = reduce(spec, current, fact, blackboard)
        except _STATE_RUNTIME_ERRORS as exc:
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
    world: World | None,
    *,
    live: bool,
    exit_on_wait: bool = False,
) -> MachineResult:
    """Run or replay a machine against its journal.

    Args:
        spec: The machine.
        journal: The instance's journal.
        world: The world; ignored by a replay.
        live: Recover from the journal and continue (`machine run`), or only reconstruct the
            recorded path (`machine replay`).
        exit_on_wait: At the first wait that is not ready, persist its instant and return
            `waiting` for an external scheduler to re-invoke.

    Returns:
        How the drive ended.

    Raises:
        EngineError: The journal was started by another machine, is malformed, or disagrees
            with its own replay.
    """
    events = journal.read()
    if events and not isinstance(events[0], machine_journal.MachineBegin):
        raise EngineError(
            "journal must start with a machine.begin event;"
            f" archive the instance directory to start fresh: {journal.root}"
        )
    if any(isinstance(event, machine_journal.MachineBegin) for event in events[1:]):
        raise EngineError(
            "journal contains more than one machine.begin event;"
            f" archive the instance directory to start fresh: {journal.root}"
        )
    if any(isinstance(event, machine_journal.MachineEnd) for event in events[:-1]):
        raise EngineError(
            "journal contains events after machine.end;"
            f" archive the instance directory to start fresh: {journal.root}"
        )

    # The instance is keyed by the machine id alone, so another file can land on the journal.
    begin = events[0] if events else None
    if isinstance(begin, machine_journal.MachineBegin) and (
        begin.machine != spec.machine or begin.version != spec.version
    ):
        raise EngineError(
            f"this journal was started by machine {begin.machine!r} v{begin.version},"
            f" but the file declares {spec.machine!r} v{spec.version}. A different"
            " machine reused the id, or the file changed since this instance began;"
            f" archive the instance directory to start fresh: {journal.root}"
        )

    if not events and live:
        journal.ensure_dirs()
        journal.begin(machine=spec.machine, version=spec.version)

    eng = _EngineState(
        spec=spec,
        journal=journal,
        world=world,
        exit_on_wait=exit_on_wait,
        blackboard=initial_blackboard(spec),
        state=spec.initial,
    )
    _rebuild_from_journal(eng, events)

    end = events[-1] if events and isinstance(events[-1], machine_journal.MachineEnd) else None
    if end is not None:
        if end.state != eng.state or end.transitions != eng.transitions:
            raise EngineError(
                "journal machine.end disagrees with its replayed position:"
                f" records state {end.state!r} after {end.transitions} transitions,"
                f" replay reaches {eng.state!r} after {eng.transitions};"
                f" archive the instance directory to start fresh: {journal.root}"
            )
        return MachineResult.from_end(end)

    if not live:
        current = spec.states.get(eng.state)
        if isinstance(current, schema.TerminalState):
            return MachineResult(current.status, current.reason, eng.state, eng.transitions)
        return MachineResult(
            "incomplete", "journal ends before a terminal state", eng.state, eng.transitions
        )

    return _run_live_loop(eng)
