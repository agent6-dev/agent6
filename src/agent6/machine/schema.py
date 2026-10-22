# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The parse boundary of a `.asm.toml` machine file.

Pydantic (`extra="forbid", frozen=True, strict=True`) catches the shape; `_semantics` enforces
the cross-cutting rules. Every violation is a load-time error aggregated into MachineError,
so `agent6 machine check` prints them all at once.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Annotated, Any, Literal

import pydantic

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
# allow_inf_nan=False: an infinite max_usd passes gt=0.0 and never binds.
_MODEL_CONFIG = pydantic.ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_StrTuple = Annotated[tuple[str, ...], pydantic.Field(strict=False)]
_NonEmptyStrTuple = Annotated[tuple[str, ...], pydantic.Field(strict=False, min_length=1)]

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


@dataclasses.dataclass(frozen=True, slots=True)
class ScalarT:
    """A scalar type: one of `str`, `int`, `float`, `bool`."""

    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class ListT:
    """A list of one scalar type."""

    elem: str


@dataclasses.dataclass(frozen=True, slots=True)
class JsonT:
    """Any JSON value."""


@dataclasses.dataclass(frozen=True, slots=True)
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


class FieldSpec(pydantic.BaseModel):
    """One schema field: its type, whether it is optional, and an enum for a `str`."""

    model_config = _MODEL_CONFIG

    type: str = pydantic.Field(min_length=1)
    optional: bool = False
    enum: _StrTuple | None = None


_FieldSpecT = Annotated[FieldSpec, pydantic.BeforeValidator(_normalize_field)]


def _normalize_notify(value: Any) -> Any:
    """Return a bare message string as `{message = ...}`; anything else unchanged."""
    if isinstance(value, str):
        return {"message": value}
    return value


class NotifySpec(pydantic.BaseModel):
    """A state's `notify`: a templated message journaled on entry and sent to the hook.

    Presentation only; it adds no edge. Authors write `notify = "msg"` or
    `notify = { message = "msg", level = "warn" }`.
    """

    model_config = _MODEL_CONFIG

    message: str = pydantic.Field(min_length=1)
    level: Literal["info", "warn", "error"] = "info"


_NotifySpecT = Annotated[NotifySpec, pydantic.BeforeValidator(_normalize_notify)]


class OperatorVar(pydantic.BaseModel):
    """A `[vars.operator]` variable: a typed constant."""

    model_config = _MODEL_CONFIG

    type: str = pydantic.Field(min_length=1)
    value: Any


class MutableVar(pydantic.BaseModel):
    """A `[vars.code]` or `[vars.agent]` variable: a typed default its owner's states write."""

    model_config = _MODEL_CONFIG

    type: str = pydantic.Field(min_length=1)
    default: Any


class VarsSection(pydantic.BaseModel):
    """The three owner tables of the blackboard, one read namespace."""

    model_config = _MODEL_CONFIG

    operator: dict[str, OperatorVar] = pydantic.Field(default_factory=dict)
    code: dict[str, MutableVar] = pydantic.Field(default_factory=dict)
    agent: dict[str, MutableVar] = pydantic.Field(default_factory=dict)


class BudgetSpec(pydantic.BaseModel):
    """The machine's spend bounds.

    Attributes:
        max_usd: The cap on cumulative metered spend, or None; an unpriced model is bounded
            per state by `[budget].max_tokens_fallback` instead.
        max_transitions: The cap on state hops; always binds.
    """

    model_config = _MODEL_CONFIG

    max_usd: float | None = pydantic.Field(default=None, gt=0.0)
    max_transitions: int = pydantic.Field(gt=0)


class Capture(pydantic.BaseModel):
    """How a state writes its result: one whole-value target, or `set` templates per target."""

    model_config = _MODEL_CONFIG

    stdout_json: str | None = None
    finish_json: str | None = None
    set: dict[str, str] | None = None

    @pydantic.model_validator(mode="after")
    def _exactly_one(self) -> Capture:
        """Return the capture, requiring exactly one of its three modes.

        Returns:
            The capture unchanged.

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


class WhenClause(pydantic.BaseModel):
    """One branch clause: an `if` predicate or the final `else`, and its `goto`."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    if_: str | None = pydantic.Field(default=None, alias="if")
    else_: bool | None = pydantic.Field(default=None, alias="else")
    goto: str = pydantic.Field(min_length=1)

    @pydantic.model_validator(mode="after")
    def _exactly_one(self) -> WhenClause:
        """Return the clause, requiring exactly one of `if` and `else`, and `else = true`.

        Returns:
            The clause unchanged.

        Raises:
            ValueError: The clause declares both, neither, or `else = false`.
        """
        if (self.if_ is None) == (self.else_ is None):
            raise ValueError("a `when` clause must declare exactly one of `if` or `else`")
        if self.else_ is not None and self.else_ is not True:
            raise ValueError("`else` must be `true` when present")
        return self


class AgentState(pydantic.BaseModel):
    """An `agent` state: one agent6 loop whose `finish_session` payload is captured.

    Attributes:
        kind: The discriminator.
        notify: The message journaled on entry, or None.
        mode: `agent`, a read-only structured judge; or `run`, with the coding tools.
        prompt: The prompt template.
        output_schema: The schema the payload must match.
        capture: How the payload is written to the blackboard.
        timeout_s: The wall-clock cap.
        on: The destination per outcome label.
        max_usd: The slice's spend cap, or None to inherit the config's; a field because the
            engine clamps it to what the machine budget has left.
        config: The state's own config layer, above the machine's `[config]`: the model
            (`models.worker`) and every other knob of the state.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["agent"]
    notify: _NotifySpecT | None = None
    mode: Literal["agent", "run"] = "agent"
    prompt: str = pydantic.Field(min_length=1)
    output_schema: str = pydantic.Field(min_length=1)
    capture: Capture
    timeout_s: int = pydantic.Field(gt=0)
    on: dict[str, str]
    max_usd: float | None = pydantic.Field(default=None, gt=0.0)
    config: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="after")
    def _one_spelling_for_the_cap(self) -> AgentState:
        """Return the state, refusing a `max_usd` that its `config` table also sets.

        Returns:
            The state unchanged.

        Raises:
            ValueError: The cap is set both ways.
        """
        budget = self.config.get("budget")
        if self.max_usd is not None and isinstance(budget, dict) and "max_usd" in budget:
            raise ValueError("`max_usd` and `config.budget.max_usd` set the same thing; keep one")
        return self


class ToolState(pydantic.BaseModel):
    """A `tool` state: one jailed command whose JSON stdout may be captured.

    Attributes:
        kind: The discriminator.
        notify: The message journaled on entry, or None.
        command: The argv templates.
        output_schema: The schema the stdout must match, or None for no shape.
        capture: How the stdout is written to the blackboard, or None.
        timeout_s: The wall-clock cap.
        on: The destination per outcome label.
        pass_env: The operator environment variables the command receives, each of which
            `[machine].pass_env` must allow or the run is refused at startup.
        config: The state's own config layer, above the machine's `[config]`; a tool state
            reads its `sandbox` table.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["tool"]
    notify: _NotifySpecT | None = None
    command: _NonEmptyStrTuple
    output_schema: str | None = None
    capture: Capture | None = None
    timeout_s: int = pydantic.Field(gt=0)
    on: dict[str, str]
    pass_env: _StrTuple = ()
    config: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("pass_env")
    @classmethod
    def _env_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        """Return the names, each a valid environment variable name.

        Args:
            names: The state's `pass_env`.

        Returns:
            The names unchanged.

        Raises:
            ValueError: A name is not one.
        """
        for name in names:
            if not _ENV_NAME_RE.fullmatch(name):
                raise ValueError(f"pass_env names an invalid environment variable name {name!r}")
        return names


_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _seconds_as_str(value: object) -> object:
    """Return a bare integer `delay_s` as its template string; a float stays refused."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


class WaitState(pydantic.BaseModel):
    """A `wait` state: parks for a one-shot delay, until an instant, or until a poke."""

    model_config = _MODEL_CONFIG

    kind: Literal["wait"]
    notify: _NotifySpecT | None = None
    delay_s: Annotated[str, pydantic.BeforeValidator(_seconds_as_str)] | None = None
    until: str | None = None
    on: dict[str, str]


class BranchState(pydantic.BaseModel):
    """A `branch` state: routes on the first `when` clause that fires; the last is `else`."""

    model_config = _MODEL_CONFIG

    kind: Literal["branch"]
    notify: _NotifySpecT | None = None
    when: Annotated[tuple[WhenClause, ...], pydantic.Field(strict=False, min_length=1)]


class TerminalState(pydantic.BaseModel):
    """A `terminal` state: ends the machine with a status and a reason.

    Attributes:
        kind: The discriminator.
        notify: The message journaled on entry, or None.
        status: `ok` or `failed`.
        reason: Why the machine ended, a template rendered against the blackboard at the end.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["terminal"]
    notify: _NotifySpecT | None = None
    status: Literal["ok", "failed"]
    reason: str = pydantic.Field(min_length=1)


StateSpec = Annotated[
    AgentState | ToolState | WaitState | BranchState | TerminalState,
    pydantic.Field(discriminator="kind"),
]


# The operator-only policy no overlay may carry; the loader and `config set --machine-file`
# both refuse off this one set, and each leaf says why.
PROTECTED_OVERLAY_TABLES: tuple[str, ...] = ("mcp", "presets", "providers")
PROTECTED_OVERLAY_LEAVES: dict[str, str] = {
    "machine.notify": "the notify hook runs an operator argv on the host outside the jail",
    "notify.on_complete": "the completion hook runs an operator argv on the host outside the jail",
    "git.run_repo_hooks": (
        "honoring the repo's .git/hooks runs repo-controlled code on the host outside the jail"
    ),
    "git.run_repo_drivers": (
        "honoring the repo's content drivers runs repo-controlled code on the host outside the jail"
    ),
    "machine.pass_env": (
        "it decides which of the operator's environment variables reach a tool jail"
    ),
    "prompt.system_prompt_file": (
        "the system prompt is read from the host, outside the jail, and sent to the provider"
    ),
    "skills.extra_dirs": (
        "a skill directory is read on the host, outside the jail, and sent to the provider"
    ),
    "workspace.base_dir": (
        "the supervisor creates and removes every state's working tree under it, on the host"
    ),
    "budget.allow_paid_credits": "it lets a plan's paid credits be spent",
}
# The supervisor reads these once for the whole machine, so one state cannot change them.
STATE_OVERLAY_KEYS: tuple[str, ...] = ("git.commit", "machine", "parallel", "workspace")


def protected_overlay_key_error(key: str, state: str = "") -> str | None:
    """Return why a dotted overlay key is operator-only, or None when it is allowed.

    Args:
        key: The dotted key.
        state: The state whose `config` table holds it; "" for the machine's `[config]`.

    Returns:
        The refusal message; None when the key is allowed.
    """
    where = f"[states.{state}.config]" if state else "machine [config] overlays"
    for table in PROTECTED_OVERLAY_TABLES:
        if key == table or key.startswith(f"{table}."):
            return (
                f"{where} must not set {table}.*:"
                " connections/secrets, strategy presets, and MCP servers are"
                " operator-only (global/repo config)"
            )
    for dotted, why in PROTECTED_OVERLAY_LEAVES.items():
        if key == dotted or key.startswith(f"{dotted}."):
            return f"{where} must not set {dotted}: {why} (operator-only)"
    return None


def protected_overlay_error(config: dict[str, Any], *, state: str = "") -> str | None:
    """Return the refusal for the first key an overlay may not carry, or None.

    Args:
        config: The machine's `[config]` table, or one state's `config` table.
        state: The state whose table it is; "" for the machine's.

    Returns:
        The refusal message; None when every key is allowed.
    """
    for head, value in config.items():
        keys = [head, *(f"{head}.{leaf}" for leaf in value)] if isinstance(value, dict) else [head]
        for key in keys:
            if state and key in STATE_OVERLAY_KEYS:
                return (
                    f"[states.{state}.config] must not set {key}.*: it is read once for the"
                    " whole machine; set it in the machine's [config]"
                )
            if problem := protected_overlay_key_error(key, state):
                return problem
    return None


class MachineSpec(pydantic.BaseModel):
    """A parsed `.asm.toml` machine.

    Attributes:
        machine: The machine's id.
        version: The file format version.
        initial: The entry state.
        budget: The spend bounds.
        vars: The blackboard's owner tables.
        schemas: The record schemas.
        states: The named states.
        config: The config layer above the operator's, an ordinary config fragment minus the
            operator-only policy `PROTECTED_OVERLAY_*` refuses; a state's `config` sits above it.
    """

    model_config = _MODEL_CONFIG

    machine: str = pydantic.Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: Literal[1]
    initial: str = pydantic.Field(min_length=1)
    budget: BudgetSpec
    vars: VarsSection = pydantic.Field(default_factory=VarsSection)
    schemas: dict[str, dict[str, _FieldSpecT]] = pydantic.Field(default_factory=dict)
    states: dict[str, StateSpec]
    config: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="after")
    def _forbid_protected_overlay_tables(self) -> MachineSpec:
        """Return the machine, refusing an overlay that carries operator-only policy.

        Returns:
            The machine unchanged.

        Raises:
            ValueError: The overlay sets a protected table or leaf.
        """
        if problem := protected_overlay_error(self.config):
            raise ValueError(problem)
        for name, state in self.states.items():
            if isinstance(state, (AgentState, ToolState)) and (
                problem := protected_overlay_error(state.config, state=name)
            ):
                raise ValueError(problem)
        return self


@dataclasses.dataclass(frozen=True, slots=True)
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
