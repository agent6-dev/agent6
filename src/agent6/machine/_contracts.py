# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine world protocol, execution records and errors."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol

import pydantic

from agent6.machine import journal as machine_journal
from agent6.machine import schema


class EngineError(Exception):
    """A machine cannot be executed: bad data, or a journal that does not fit it."""


class StateRuntimeError(EngineError):
    """A state met invalid data despite the load-time checks."""


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
        effort: The reasoning effort the `machine create` authoring agent runs at, or None; a
            state sets its own in its `config` table.
        max_usd: The slice's spend cap, or None.
        max_plan_points: The slice's plan-points cap, or None.
        mode: The nested loop's mode: `agent` (a read-only structured judge) or `run`.
        state_name: The state, for its own watchable logs.jsonl; "" for the authoring agent.
        step_seq: The transition, likewise; 0 for the authoring agent.
        output_schema: The finish contract the execution refuses a non-conforming result by.
        schemas: The spec's schema table verbatim, so nested records resolve execution side.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    prompt: str
    timeout_s: float
    effort: str | None = None
    max_usd: float | None = None
    max_plan_points: float | None = None
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
            `max_turns`.
        payload: The structured object passed to `finish_session`, or None.
        usd: The slice's spend, summed into the machine's.
        usd_partial: `usd` is a known under-estimate (an unpriced model).
        input_tokens: The slice's input tokens.
        output_tokens: The slice's output tokens.
        plan_consumed: The slice's plan points, summed into `machine create`'s ledger.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    reason: str
    payload: dict[str, Any] | None
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    plan_consumed: float = 0.0

    @pydantic.field_validator("payload")
    @classmethod
    def _scrub_payload(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Replace lone surrogates, which `json.loads` accepts and `model_dump_json` refuses.

        Unscrubbed, the subprocess dies before writing `result.json` and the host routes a
        finished agent to its `on.failed` edge.

        Args:
            value: The `finish_session` payload.

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
        state: str = "",
        pass_env: tuple[str, ...] = (),
        step_seq: int = 0,
    ) -> ToolExecResult:
        """Run one tool command under the named state's sandbox.

        Args:
            argv: The rendered command.
            timeout_s: The wall-clock cap.
            state: The state's name.
            pass_env: The operator environment variables the command receives.
            step_seq: The transition number, for the state's log directory.

        Returns:
            The exit code, output and whether the command timed out.
        """
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
