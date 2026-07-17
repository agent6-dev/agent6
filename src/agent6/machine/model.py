# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse and validate a `.asm.toml` machine file into a `MachineSpec`.

The parse boundary is pydantic v2 (`extra="forbid", frozen=True`),
exactly like `agent6.config`. Structural shape is caught by pydantic;
the cross-cutting rules from the spec (§4.5), global name uniqueness
across owner subtables, the ownership wall, reference/field type-checking,
total branches, reachability, are enforced by :func:`validate_semantics`.

Every violation is a *load-time* error, aggregated into
:class:`MachineError` so `agent6 machine check` can print them all at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

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

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")

_SCALARS = ("str", "int", "float", "bool")
RESERVED_NAMES = frozenset({"vars", "operator", "code", "agent", "result"})

AGENT_LABELS = frozenset({"ok", "failed", "budget_exhausted", "timeout"})
TOOL_LABELS = frozenset({"ok", "nonzero", "timeout"})
WAIT_LABELS = frozenset({"tick", "signal"})


class MachineError(Exception):
    """Raised when a machine file does not load and validate cleanly.

    ``problems`` is the full, ordered list of diagnostics.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


# --------------------------------------------------------------------------
# Type system
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalarT:
    name: str  # one of _SCALARS


@dataclass(frozen=True, slots=True)
class ListT:
    elem: str  # one of _SCALARS


@dataclass(frozen=True, slots=True)
class JsonT:
    pass


@dataclass(frozen=True, slots=True)
class RecordT:
    name: str


TypeRef = ScalarT | ListT | JsonT | RecordT


class TypeParseError(Exception):
    pass


def parse_type(text: str, schema_names: frozenset[str]) -> TypeRef:
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
    if isinstance(t, ScalarT):
        return t.name
    if isinstance(t, ListT):
        return f"list[{t.elem}]"
    if isinstance(t, JsonT):
        return "json"
    return f"record {t.name!r}"


# --------------------------------------------------------------------------
# Pydantic parse models (trust boundary)
# --------------------------------------------------------------------------


def _normalize_field(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": value}
    return value


class FieldSpec(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    optional: bool = False
    enum: tuple[str, ...] | None = None


_FieldSpecT = Annotated[FieldSpec, BeforeValidator(_normalize_field)]


def _normalize_notify(value: Any) -> Any:
    if isinstance(value, str):
        return {"message": value}
    return value


class NotifySpec(BaseModel):
    """A state's optional `notify`: a templated message emitted on entry.

    Presentation only (§4.3): entering the state journals a `machine.notify`
    event and fires the operator notify hook; it adds no edge and no control
    flow. Authors write ``notify = "msg"`` (level defaults to "info") or
    ``notify = { message = "msg", level = "warn" }``.
    """

    model_config = _MODEL_CONFIG

    message: str = Field(min_length=1)
    level: Literal["info", "warn", "error"] = "info"


_NotifySpecT = Annotated[NotifySpec, BeforeValidator(_normalize_notify)]


class OperatorVar(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    value: Any


class MutableVar(BaseModel):
    model_config = _MODEL_CONFIG

    type: str = Field(min_length=1)
    default: Any


class VarsSection(BaseModel):
    model_config = _MODEL_CONFIG

    operator: dict[str, OperatorVar] = Field(default_factory=dict)
    code: dict[str, MutableVar] = Field(default_factory=dict)
    agent: dict[str, MutableVar] = Field(default_factory=dict)


class BudgetSpec(BaseModel):
    """Whole-machine spend bounds. `max_transitions` always binds.

    The USD limit is optional, at most one of the two: `max_usd` is hard
    (`machine run` refuses up front when a covered agent state's model has
    no price data); `best_effort_usd_limit` binds only when spend is
    measurable, for unpriced or local models. Spend is metered as an
    estimate (reported cost, else price times tokens).
    """

    model_config = _MODEL_CONFIG

    max_usd: float | None = Field(default=None, gt=0.0)
    best_effort_usd_limit: float | None = Field(default=None, gt=0.0)
    max_transitions: int = Field(gt=0)

    @model_validator(mode="after")
    def _at_most_one_usd(self) -> BudgetSpec:
        if self.max_usd is not None and self.best_effort_usd_limit is not None:
            raise ValueError(
                "[budget] may set at most one of `max_usd` (hard cap, machine run"
                " refuses unpriced models) and `best_effort_usd_limit` (enforced"
                " when spend is measurable)"
            )
        return self

    @property
    def usd_limit(self) -> float | None:
        return self.max_usd if self.max_usd is not None else self.best_effort_usd_limit

    @property
    def usd_field_name(self) -> str:
        return "max_usd" if self.max_usd is not None else "best_effort_usd_limit"


class Capture(BaseModel):
    model_config = _MODEL_CONFIG

    stdout_json: str | None = None
    finish_json: str | None = None
    set: dict[str, str] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Capture:
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
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    if_: str | None = Field(default=None, alias="if")
    else_: bool | None = Field(default=None, alias="else")
    goto: str = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> WhenClause:
        if (self.if_ is None) == (self.else_ is None):
            raise ValueError("a `when` clause must declare exactly one of `if` or `else`")
        if self.else_ is not None and self.else_ is not True:
            raise ValueError("`else` must be `true` when present")
        return self


class AgentState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["agent"]
    # Optional templated message emitted on entry (§4.3); presentation only.
    notify: _NotifySpecT | None = None
    # "inherit" (the default) uses the operator's effective worker model, so a
    # machine need not hardcode a model the operator may not have configured,
    # the #1 way an LLM-authored machine passed `machine check` but died at run
    # time. Set an explicit provider/model only to pin a specific one.
    model: str = Field(default="inherit", min_length=1)
    # "agent" (default): a read-only structured-output judge, classify/score/
    # decide and return a finish_run result; cannot edit the repo. Set "run" for
    # an agent state that must do real coding work (edit/verify/commit tools).
    mode: Literal["agent", "run"] = "agent"
    prompt: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    capture: Capture
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    # Optional per-state overrides for how this agent loop is driven. When
    # unset each falls back to the effective config (machine ``[config]``
    # overlay < repo < global < defaults). ``provider`` selects which
    # ``[providers.*]`` entry backs the call; ``thinking`` and ``temperature``
    # tune reasoning/sampling; the budget caps bound this single agent slice.
    # Secrets/connection keys are never expressed here, only the provider
    # *name*, which must already exist in the effective config.
    provider: str | None = None
    thinking: Literal["off", "low", "medium", "high"] | None = None
    temperature: float | None = None
    # Same contract as [budget]: `max_usd` is hard (machine run refuses when
    # this state's model is unpriced), `best_effort_usd_limit` binds when
    # spend is measurable. At most one; both unset means no per-state cap.
    max_usd: float | None = Field(default=None, gt=0.0)
    best_effort_usd_limit: float | None = Field(default=None, gt=0.0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _at_most_one_usd(self) -> AgentState:
        if self.max_usd is not None and self.best_effort_usd_limit is not None:
            raise ValueError(
                "an agent state may set at most one of `max_usd` and `best_effort_usd_limit`"
            )
        return self

    @property
    def usd_limit(self) -> float | None:
        return self.max_usd if self.max_usd is not None else self.best_effort_usd_limit


class ToolState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["tool"]
    notify: _NotifySpecT | None = None
    command: tuple[str, ...] = Field(min_length=1)
    output_schema: str | None = None
    capture: Capture | None = None
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    # This tool's network stance for its jailed subprocess:
    #  - ``auto`` (default): no network; isolated (empty netns) where the profile
    #    can (``strict``), tolerant where it can't (``hardened`` shares the host
    #    netns), the deterministic, offline default that runs anywhere.
    #  - ``allow``: wants the host network. Granted only if the operator permits
    #    it via ``sandbox.tool_network`` (``only_explicit_states`` or ``allow``);
    #    under ``block`` the run is refused naming this state. Enforceable because
    #    the machine engine is a host-netns supervisor: the tool's jail can reach
    #    the network while the agent states stay confined to the provider API.
    #  - ``block``: no network, REQUIRED, refuse on ``hardened`` (which can't
    #    guarantee per-tool isolation), unlike ``auto`` which tolerates it.
    # The tool only *declares*; whether ``allow`` is granted is the operator's
    # call (``sandbox.tool_network``, read from global/repo config, never a
    # machine overlay).
    allow_network: Literal["auto", "allow", "block"] = "auto"


class WaitState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["wait"]
    notify: _NotifySpecT | None = None
    every_secs: str | None = None
    until: str | None = None
    cron: str | None = None
    on: dict[str, str]


class BranchState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["branch"]
    notify: _NotifySpecT | None = None
    when: tuple[WhenClause, ...] = Field(min_length=1)


class TerminalState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["terminal"]
    notify: _NotifySpecT | None = None
    status: Literal["ok", "failed"]
    reason: str = Field(min_length=1)


StateSpec = Annotated[
    AgentState | ToolState | WaitState | BranchState | TerminalState,
    Field(discriminator="kind"),
]


class MachineSpec(BaseModel):
    """A validated `.asm.toml` machine definition: budget, typed `schemas`, the
    named `states` graph, and an optional agent6 `[config]` overlay whose
    operator-only security tables (providers/sandbox/profiles) are refused so an
    untrusted machine file cannot weaken the sandbox."""

    model_config = _MODEL_CONFIG

    machine: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: Literal[1]
    initial: str = Field(min_length=1)
    budget: BudgetSpec
    vars: VarsSection = Field(default_factory=VarsSection)
    schemas: dict[str, dict[str, _FieldSpecT]] = Field(default_factory=dict)
    states: dict[str, StateSpec]
    # Machine-level agent6 config overlay. Anything set here layers on top of
    # the effective repo/global/default config for the duration of the
    # machine run (``machine[config]`` is the highest-precedence layer). It is
    # an ordinary agent6 config fragment, most knobs ``agent6 config show``
    # lists are valid, but it MUST NOT carry operator-only security policy
    # (see ``_forbid_protected_overlay_tables``): ``[providers.*]`` (endpoints
    # + api-key env names + secrets), ``[sandbox.*]`` (the jail: network
    # egress incl. allow_urls, run_commands, .git protection), and
    # ``[profiles.*]`` (presets that DEFINE that same sandbox/providers/notify
    # policy) are read only from the global/repo config, never a (possibly
    # untrusted) machine file. Unset keys simply read through to the lower layers.
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _forbid_protected_overlay_tables(self) -> MachineSpec:
        # A machine file may be LLM-drafted (`machine create`), shared, or
        # otherwise untrusted, yet its `[config]` overlay is the highest
        # config layer at run time. So it must not carry operator-only
        # security policy: `[providers.*]` (endpoints + api-key env names),
        # `[sandbox.*]` (the jail itself, network egress incl. allow_urls,
        # run_commands, and .git protection), or `[profiles.*]` (which DEFINE
        # sandbox/providers/notify presets: the profile the operator selects by
        # name in the global/repo config is resolved from every layer including
        # this overlay, so an overlay-defined `[profiles.<selected>]` would splice
        # operator-only policy, even a host `[machine.notify]` argv, straight into
        # the effective config). Those are read only from the global/repo config;
        # an overlay that sets them is rejected at load so a machine can never
        # weaken the sandbox the operator chose.
        for table in ("providers", "sandbox", "profiles"):
            if table in self.config:
                raise ValueError(
                    f"machine `[config]` overlay must not declare `[{table}.*]`:"
                    " connections/secrets, sandbox policy, and profile presets are"
                    " operator decisions set in the global/repo config, never in a"
                    " .asm.toml file"
                )
        # The machine notify hook runs an operator argv on the host OUTSIDE the
        # jail, so it must never come from a (possibly LLM-drafted) machine file.
        machine_overlay = self.config.get("machine")
        if isinstance(machine_overlay, dict) and "notify" in machine_overlay:
            raise ValueError(
                "machine `[config]` overlay must not declare `[machine.notify]`:"
                " the notify hook runs an operator argv outside the jail and is"
                " set only in the global/repo config, never in a .asm.toml file"
            )
        # `git.run_repo_hooks` decides whether a `mode="run"` state's auto-commit
        # honors the repo's `.git/hooks/*`, which is repo-controlled code that runs
        # on the HOST outside the jail (a host-RCE vector). Secure-by-default false;
        # a machine file must not be able to flip it on. Other `[git]` keys (commit
        # identity) are harmless overlay knobs and stay allowed.
        git_overlay = self.config.get("git")
        if isinstance(git_overlay, dict) and "run_repo_hooks" in git_overlay:
            raise ValueError(
                "machine `[config]` overlay must not set `git.run_repo_hooks`:"
                " honoring the repo's .git/hooks runs repo-controlled code on the"
                " host outside the jail; it is an operator decision in the"
                " global/repo config, never in a .asm.toml file"
            )
        return self


# --------------------------------------------------------------------------
# Graph edges
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    label: str


def edges(spec: MachineSpec) -> tuple[Edge, ...]:
    """Every directed, labelled edge in the machine graph."""
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
    """States reachable from ``initial`` following declared edges."""
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


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
