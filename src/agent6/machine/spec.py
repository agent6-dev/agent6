# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The parse boundary of a `.asm.toml` machine file.

Pydantic (`extra="forbid", frozen=True, strict=True`) catches the shape; `_semantics` enforces
the cross-cutting rules. Every violation is a load-time error aggregated into MachineError,
so `agent6 machine check` prints them all at once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "AgentState",
    "BranchState",
    "Edge",
    "MachineError",
    "MachineSpec",
    "NotifySpec",
    "StateSpec",
    "TerminalState",
    "ToolState",
    "TypeRef",
    "WaitState",
    "edges",
    "parse_type",
    "reachable_states",
    "type_str",
]

# Strict: TOML supplies native scalars, so a quoted number is refused; tuple fields opt out.
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_StrTuple = Annotated[tuple[str, ...], Field(strict=False)]
_NonEmptyStrTuple = Annotated[tuple[str, ...], Field(strict=False, min_length=1)]

IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")

_SCALARS = ("str", "int", "float", "bool")
# `parse_type` resolves these before the declared schemas, so a schema so named is unreachable.
BUILTIN_TYPE_NAMES = frozenset({*_SCALARS, "json"})
RESERVED_NAMES = frozenset({"vars", "operator", "code", "agent", "result"})

AGENT_LABELS = frozenset({"ok", "failed", "budget_exhausted", "timeout"})
TOOL_LABELS = frozenset({"ok", "nonzero", "timeout"})
WAIT_LABELS = frozenset({"tick", "signal"})


class MachineError(Exception):
    """A machine file does not load and validate cleanly.

    Attributes:
        problems: Every diagnostic, in order.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


@dataclass(frozen=True, slots=True)
class ScalarT:
    """A scalar type: one of `str`, `int`, `float`, `bool`."""

    name: str


@dataclass(frozen=True, slots=True)
class ListT:
    """A list of one scalar type."""

    elem: str


@dataclass(frozen=True, slots=True)
class JsonT:
    """Any JSON value."""


@dataclass(frozen=True, slots=True)
class RecordT:
    """A record of a declared schema."""

    name: str


TypeRef = ScalarT | ListT | JsonT | RecordT


class TypeParseError(Exception):
    """A type declaration names no scalar, list, `json` or declared schema."""


def parse_type(text: str, schema_names: frozenset[str]) -> TypeRef:
    """Parse a type declaration.

    Args:
        text: The declaration: a scalar, `json`, `list[<scalar>]` or a schema name.
        schema_names: The declared schemas.

    Returns:
        The type.

    Raises:
        TypeParseError: The text names none of those.
    """
    if text in _SCALARS:
        return ScalarT(text)
    if text == "json":
        return JsonT()
    list_match = _LIST_RE.match(text)
    if list_match:
        elem = list_match.group(1)
        if elem not in _SCALARS:
            raise TypeParseError(
                f"list element type must be a scalar (str/int/float/bool), got {elem!r}"
            )
        return ListT(elem)
    if text in schema_names:
        return RecordT(text)
    raise TypeParseError(f"unknown type {text!r}")


def type_str(t: TypeRef) -> str:
    """Return the type as a problem line names it."""
    if isinstance(t, ScalarT):
        return t.name
    if isinstance(t, ListT):
        return f"list[{t.elem}]"
    if isinstance(t, JsonT):
        return "json"
    return f"record {t.name!r}"


def type_decl(t: TypeRef) -> str:
    """Return the value an author writes in a `type = "..."` declaration."""
    return t.name if isinstance(t, RecordT) else type_str(t)


def _normalize_field(value: Any) -> Any:
    """Return a bare type string as `{type = ...}`; anything else unchanged."""
    if isinstance(value, str):
        return {"type": value}
    return value


class FieldSpec(BaseModel):
    """One schema field: its type, whether it is optional, and an enum for a `str`."""

    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    optional: bool = False
    enum: _StrTuple | None = None


_FieldSpecT = Annotated[FieldSpec, BeforeValidator(_normalize_field)]


def _normalize_notify(value: Any) -> Any:
    """Return a bare message string as `{message = ...}`; anything else unchanged."""
    if isinstance(value, str):
        return {"message": value}
    return value


class NotifySpec(BaseModel):
    """A state's `notify`: a templated message journaled on entry and sent to the hook.

    Presentation only; it adds no edge. Authors write `notify = "msg"` or
    `notify = { message = "msg", level = "warn" }`.
    """

    model_config = _MODEL_CONFIG

    message: str = Field(min_length=1)
    level: Literal["info", "warn", "error"] = "info"


_NotifySpecT = Annotated[NotifySpec, BeforeValidator(_normalize_notify)]


class OperatorVar(BaseModel):
    """A `[vars.operator]` variable: a typed constant."""

    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    value: Any


class MutableVar(BaseModel):
    """A `[vars.code]` or `[vars.agent]` variable: a typed default its owner's states write."""

    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    default: Any


class VarsSection(BaseModel):
    """The three owner tables of the blackboard, one read namespace."""

    model_config = _MODEL_CONFIG

    operator: dict[str, OperatorVar] = Field(default_factory=dict)
    code: dict[str, MutableVar] = Field(default_factory=dict)
    agent: dict[str, MutableVar] = Field(default_factory=dict)


def _finite_usd(v: float) -> float:
    """Return the cap, refusing an infinite one, which passes `gt=0.0` and never binds.

    Raises:
        ValueError: The cap is not finite.
    """
    if not math.isfinite(v):
        raise ValueError("max_usd must be a finite cap")
    return v


_FiniteUsd = Annotated[float, AfterValidator(_finite_usd)]


class BudgetSpec(BaseModel):
    """The machine's spend bounds.

    Attributes:
        max_usd: The cap on cumulative metered spend, or None; an unpriced model is bounded
            per state by `[budget].max_tokens_fallback` instead.
        max_transitions: The cap on state hops; always binds.
    """

    model_config = _MODEL_CONFIG

    max_usd: _FiniteUsd | None = Field(default=None, gt=0.0)
    max_transitions: int = Field(gt=0)


class Capture(BaseModel):
    """How a state writes its result: one whole-value target, or `set` templates per target."""

    model_config = _MODEL_CONFIG

    stdout_json: str | None = None
    finish_json: str | None = None
    set: dict[str, str] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Capture:
        """Return the capture, requiring exactly one of its three modes.

        Raises:
            ValueError: None or several modes are set.
        """
        present = [
            name
            for name, value in (
                ("stdout_json", self.stdout_json),
                ("finish_json", self.finish_json),
                ("set", self.set),
            )
            if value is not None
        ]
        if len(present) != 1:
            raise ValueError(
                "capture must declare exactly one of `stdout_json`, `finish_json`, or `set`"
                f" (found: {present or 'none'})"
            )
        return self


class WhenClause(BaseModel):
    """One branch clause: an `if` predicate or the final `else`, and its `goto`."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    if_: str | None = Field(default=None, alias="if")
    else_: bool | None = Field(default=None, alias="else")
    goto: str = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> WhenClause:
        """Return the clause, requiring exactly one of `if` and `else`, and `else = true`.

        Raises:
            ValueError: The clause declares both, neither, or `else = false`.
        """
        if (self.if_ is None) == (self.else_ is None):
            raise ValueError("a `when` clause must declare exactly one of `if` or `else`")
        if self.else_ is not None and self.else_ is not True:
            raise ValueError("`else` must be `true` when present")
        return self


class AgentState(BaseModel):
    """An `agent` state: one agent6 loop whose `finish_session` payload is captured.

    Attributes:
        kind: The discriminator.
        notify: The message journaled on entry, or None.
        model: The model, or `inherit` for the operator's worker model (a hardcoded model
            passes `machine check` and can die at run time).
        mode: `agent`, a read-only structured judge; or `run`, with the coding tools.
        prompt: The prompt template.
        output_schema: The schema the payload must match.
        capture: How the payload is written to the blackboard.
        timeout_secs: The wall-clock cap.
        on: The destination per outcome label.
        provider: The `[providers.*]` entry by name (never a secret), or None to inherit.
        effort: The reasoning effort override, or None to inherit.
        temperature: The sampling override, or None to inherit.
        max_usd: The slice's spend cap, or None to inherit the config's.
        max_tokens_fallback: The unmetered token bound (-1 unlimited, 0 refuse), or None to
            inherit.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["agent"]
    notify: _NotifySpecT | None = None
    model: str = Field(default="inherit", min_length=1)
    mode: Literal["agent", "run"] = "agent"
    prompt: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    capture: Capture
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    provider: str | None = None
    effort: Literal["off", "low", "medium", "high", "xhigh", "max"] | None = None
    temperature: float | None = None
    max_usd: _FiniteUsd | None = Field(default=None, gt=0.0)
    max_tokens_fallback: int | None = Field(default=None, ge=-1)


class ToolState(BaseModel):
    """A `tool` state: one jailed command whose JSON stdout may be captured.

    Attributes:
        kind: The discriminator.
        notify: The message journaled on entry, or None.
        command: The argv templates.
        output_schema: The schema the stdout must match, or None for no shape.
        capture: How the stdout is written to the blackboard, or None.
        timeout_secs: The wall-clock cap.
        on: The destination per outcome label.
        network: What the jail joins. `auto`: a network of its own where the isolation
            level can give one, else the host's with a warning. `host`: the machine's,
            granted only by the operator's `sandbox.network`, else the run is refused
            naming this state. `none`: a network of its own, or the run is refused. No
            `session`: a state's processes die with the state.
        pass_env: The operator environment variables the command receives, each of which
            `[machine].pass_env` must allow or the run is refused at startup.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["tool"]
    notify: _NotifySpecT | None = None
    command: _NonEmptyStrTuple
    output_schema: str | None = None
    capture: Capture | None = None
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    network: Literal["auto", "host", "none"] = "auto"
    pass_env: _StrTuple = ()

    @field_validator("pass_env")
    @classmethod
    def _env_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        """Return the names, each a valid environment variable name.

        Raises:
            ValueError: A name is not one.
        """
        for name in names:
            if not _ENV_NAME_RE.fullmatch(name):
                raise ValueError(f"pass_env names an invalid environment variable name {name!r}")
        return names


_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _seconds_as_str(value: object) -> object:
    """Return a bare integer `every_secs` as its template string; a float stays refused."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


class WaitState(BaseModel):
    """A `wait` state: parks until an interval, an instant, or a poke."""

    model_config = _MODEL_CONFIG

    kind: Literal["wait"]
    notify: _NotifySpecT | None = None
    every_secs: Annotated[str, BeforeValidator(_seconds_as_str)] | None = None
    until: str | None = None
    on: dict[str, str]


class BranchState(BaseModel):
    """A `branch` state: routes on the first `when` clause that fires; the last is `else`."""

    model_config = _MODEL_CONFIG

    kind: Literal["branch"]
    notify: _NotifySpecT | None = None
    when: Annotated[tuple[WhenClause, ...], Field(strict=False, min_length=1)]


class TerminalState(BaseModel):
    """A `terminal` state: ends the machine with a status and a reason."""

    model_config = _MODEL_CONFIG

    kind: Literal["terminal"]
    notify: _NotifySpecT | None = None
    status: Literal["ok", "failed"]
    reason: str = Field(min_length=1)


StateSpec = Annotated[
    AgentState | ToolState | WaitState | BranchState | TerminalState,
    Field(discriminator="kind"),
]


# The operator-only policy a machine `[config]` overlay may never carry; the loader and
# `config set --machine-file` both refuse off this one set, and each leaf says why.
PROTECTED_OVERLAY_TABLES: tuple[str, ...] = ("providers", "sandbox", "presets", "mcp")
PROTECTED_OVERLAY_LEAVES: dict[str, str] = {
    "machine.notify": "the notify hook runs an operator argv on the host outside the jail",
    "notify.on_complete": "the completion hook runs an operator argv on the host outside the jail",
    "git.run_repo_hooks": (
        "honoring the repo's .git/hooks runs repo-controlled code on the host outside the jail"
    ),
    "git.run_repo_filters": (
        "honoring the repo's content drivers runs repo-controlled code on the host outside the jail"
    ),
    "machine.pass_env": (
        "it decides which of the operator's environment variables reach a tool jail"
    ),
    "prompt.system_prompt_file": (
        "the system prompt is read from the host, outside the jail, and sent to the provider"
    ),
}


def protected_overlay_key_error(key: str) -> str | None:
    """Return why a dotted machine-overlay key is operator-only, or None when it is allowed."""
    for table in PROTECTED_OVERLAY_TABLES:
        if key == table or key.startswith(f"{table}."):
            return (
                f"machine [config] overlays must not set {table}.*:"
                " connections/secrets, sandbox policy, strategy presets, and MCP"
                " servers are operator-only (global/repo config)"
            )
    for dotted, why in PROTECTED_OVERLAY_LEAVES.items():
        if key == dotted or key.startswith(f"{dotted}."):
            return f"machine [config] overlays must not set {dotted}: {why} (operator-only)"
    return None


def protected_overlay_error(config: dict[str, Any]) -> str | None:
    """Return the refusal for the first operator-only key in a machine overlay, or None."""
    for head, value in config.items():
        if problem := protected_overlay_key_error(head):
            return problem
        if isinstance(value, dict):
            for leaf in value:
                if problem := protected_overlay_key_error(f"{head}.{leaf}"):
                    return problem
    return None


class MachineSpec(BaseModel):
    """A parsed `.asm.toml` machine.

    Attributes:
        machine: The machine's id.
        version: The file format version.
        initial: The entry state.
        budget: The spend bounds.
        vars: The blackboard's owner tables.
        schemas: The record schemas.
        states: The named states.
        config: The run's highest-precedence config layer, an ordinary config fragment minus
            the operator-only policy `PROTECTED_OVERLAY_*` refuses.
    """

    model_config = _MODEL_CONFIG

    machine: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: Literal[1]
    initial: str = Field(min_length=1)
    budget: BudgetSpec
    vars: VarsSection = Field(default_factory=VarsSection)
    schemas: dict[str, dict[str, _FieldSpecT]] = Field(default_factory=dict)
    states: dict[str, StateSpec]
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _forbid_protected_overlay_tables(self) -> MachineSpec:
        """Return the machine, refusing an overlay that carries operator-only policy.

        Raises:
            ValueError: The overlay sets a protected table or leaf.
        """
        if problem := protected_overlay_error(self.config):
            raise ValueError(problem)
        return self


@dataclass(frozen=True, slots=True)
class Edge:
    """One labelled transition of the machine graph."""

    src: str
    dst: str
    label: str


def edges(spec: MachineSpec) -> tuple[Edge, ...]:
    """Return every labelled edge of the machine graph."""
    out: list[Edge] = []
    for name, state in spec.states.items():
        if isinstance(state, BranchState):
            for clause in state.when:
                label = clause.if_ if clause.if_ is not None else "else"
                out.append(Edge(src=name, dst=clause.goto, label=label))
        elif isinstance(state, (AgentState, ToolState, WaitState)):
            for label, target in state.on.items():
                out.append(Edge(src=name, dst=target, label=label))
    return tuple(out)


def reachable_states(spec: MachineSpec) -> frozenset[str]:
    """Return the states reachable from `initial` along declared edges."""
    adjacency: dict[str, list[str]] = {name: [] for name in spec.states}
    for edge in edges(spec):
        if edge.dst in adjacency:
            adjacency[edge.src].append(edge.dst)
    seen: set[str] = set()
    if spec.initial in spec.states:
        stack = [spec.initial]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current])
    return frozenset(seen)
