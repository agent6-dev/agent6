# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse and validate a `.asm.toml` machine file into a `MachineSpec`.

The parse boundary is pydantic v2 (`extra="forbid", frozen=True`),
exactly like `agent6.config`. Structural shape is caught by pydantic;
the cross-cutting rules from the spec (§4.5) — global name uniqueness
across owner subtables, the ownership wall, reference/field type-checking,
total branches, reachability — are enforced by :func:`validate_semantics`.

Every violation is a *load-time* error, aggregated into
:class:`MachineError` so `agent6 machine check` can print them all at once.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from agent6.machine.predicate import PredicateError, Reference, parse_predicate
from agent6.machine.template import Interp, Template, TemplateError, parse_template

__all__ = [
    "AgentState",
    "BranchState",
    "Edge",
    "MachineError",
    "MachineSpec",
    "StateSpec",
    "TerminalState",
    "ToolState",
    "WaitState",
    "edges",
    "load_machine",
    "reachable_states",
    "validate_finish_payload",
    "validate_semantics",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")

_SCALARS = ("str", "int", "float", "bool")
_RESERVED_NAMES = frozenset({"vars", "operator", "code", "agent", "result"})

_AGENT_LABELS = frozenset({"ok", "failed", "budget_exhausted", "timeout"})
_TOOL_LABELS = frozenset({"ok", "nonzero", "timeout"})
_WAIT_LABELS = frozenset({"tick", "signal"})


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


class _TypeParseError(Exception):
    pass


def _parse_type(text: str, schema_names: frozenset[str]) -> TypeRef:
    if text in _SCALARS:
        return ScalarT(text)
    if text == "json":
        return JsonT()
    list_match = _LIST_RE.match(text)
    if list_match:
        elem = list_match.group(1)
        if elem not in _SCALARS:
            raise _TypeParseError(
                f"list element type must be a scalar (str/int/float/bool), got {elem!r}"
            )
        return ListT(elem)
    if text in schema_names:
        return RecordT(text)
    raise _TypeParseError(f"unknown type {text!r}")


def _type_str(t: TypeRef) -> str:
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
    model_config = _MODEL_CONFIG

    max_usd: float = Field(gt=0.0)
    max_transitions: int = Field(gt=0)


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
    # "inherit" (the default) uses the operator's effective worker model, so a
    # machine need not hardcode a model the operator may not have configured —
    # the #1 way an LLM-authored machine passed `machine check` but died at run
    # time. Set an explicit provider/model only to pin a specific one.
    model: str = Field(default="inherit", min_length=1)
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
    # Secrets/connection keys are never expressed here — only the provider
    # *name*, which must already exist in the effective config.
    provider: str | None = None
    thinking: Literal["off", "low", "medium", "high"] | None = None
    temperature: float | None = None
    max_usd: float | None = Field(default=None, gt=0.0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)


class ToolState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["tool"]
    command: tuple[str, ...] = Field(min_length=1)
    output_schema: str | None = None
    capture: Capture | None = None
    timeout_secs: int = Field(gt=0)
    on: dict[str, str]
    # This tool's network stance for its jailed subprocess:
    #  - ``auto`` (default): no network; isolated (empty netns) where the profile
    #    can (``strict``), tolerant where it can't (``hardened`` shares the host
    #    netns) — the deterministic, offline default that runs anywhere.
    #  - ``allow``: wants the host network. Granted only if the operator permits
    #    it via ``sandbox.tool_network`` (``only_explicit_states`` or ``allow``);
    #    under ``block`` the run is refused naming this state. Enforceable because
    #    the machine engine is a host-netns supervisor: the tool's jail can reach
    #    the network while the agent states stay confined to the provider API.
    #  - ``block``: no network, REQUIRED — refuse on ``hardened`` (which can't
    #    guarantee per-tool isolation), unlike ``auto`` which tolerates it.
    # The tool only *declares*; whether ``allow`` is granted is the operator's
    # call (``sandbox.tool_network``, read from global/repo config, never a
    # machine overlay).
    allow_network: Literal["auto", "allow", "block"] = "auto"


class WaitState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["wait"]
    every_secs: str | None = None
    until: str | None = None
    cron: str | None = None
    on: dict[str, str]


class BranchState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["branch"]
    when: tuple[WhenClause, ...] = Field(min_length=1)


class TerminalState(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["terminal"]
    status: Literal["ok", "failed"]
    reason: str = Field(min_length=1)


StateSpec = Annotated[
    AgentState | ToolState | WaitState | BranchState | TerminalState,
    Field(discriminator="kind"),
]


class MachineSpec(BaseModel):
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
    # an ordinary agent6 config fragment — most knobs ``agent6 config show``
    # lists are valid — but it MUST NOT carry operator-only security policy
    # (see ``_forbid_protected_overlay_tables``): ``[providers.*]`` (endpoints
    # + api-key env names + secrets) and ``[sandbox.*]`` (the jail: network
    # egress incl. allow_urls, run_commands, .git/.agent6 protection) are read
    # only from the global/repo config, never a (possibly untrusted) machine
    # file. Unset keys simply read through to the lower layers.
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _forbid_protected_overlay_tables(self) -> MachineSpec:
        # A machine file may be LLM-drafted (`machine create`), shared, or
        # otherwise untrusted, yet its `[config]` overlay is the highest
        # config layer at run time. So it must not carry operator-only
        # security policy: `[providers.*]` (endpoints + api-key env names) or
        # `[sandbox.*]` (the jail itself — network egress incl. allow_urls,
        # run_commands, and .git/.agent6 protection). Those are read only from
        # the global/repo config; an overlay that sets them is rejected at load
        # so a machine can never weaken the sandbox the operator chose.
        for table in ("providers", "sandbox"):
            if table in self.config:
                raise ValueError(
                    f"machine `[config]` overlay must not declare `[{table}.*]` —"
                    " connections/secrets and sandbox policy are operator"
                    " decisions set in the global/repo config, never in a"
                    " .asm.toml file"
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


def load_machine(path: Path) -> MachineSpec:
    """Load, parse, and fully validate the `.asm.toml` file at *path*.

    Raises :class:`MachineError` aggregating every diagnostic. Never
    returns a partially-valid machine.
    """
    if not path.is_file():
        raise MachineError([f"machine file not found: {path}"])
    text = path.read_text(encoding="utf-8")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise MachineError([f"not valid TOML ({path}): {exc}"]) from exc
    precheck = _precheck(raw)
    if precheck:
        raise MachineError(precheck)
    try:
        spec = MachineSpec.model_validate(raw)
    except ValidationError as exc:
        raise MachineError(_format_validation_error(exc)) from exc
    problems = validate_semantics(spec)
    if problems:
        raise MachineError(problems)
    return spec


def _precheck(raw: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    vars_section = raw.get("vars")
    if isinstance(vars_section, dict):
        for key in vars_section:
            if key not in ("operator", "code", "agent"):
                problems.append(
                    f"`vars.{key}` has no owner subtable; put it in"
                    " `[vars.operator]`, `[vars.code]`, or `[vars.agent]`"
                )
    return problems


def _format_validation_error(err: ValidationError) -> list[str]:
    problems: list[str] = []
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        problems.append(f"{loc}: {issue['msg']} (type={issue['type']})")
    return problems


# --------------------------------------------------------------------------
# Semantic validation (§4.5)
# --------------------------------------------------------------------------


def validate_semantics(spec: MachineSpec) -> list[str]:
    """Run every cross-cutting rule the pydantic shape cannot express."""
    problems: list[str] = []
    schema_names = frozenset(spec.schemas)

    for sname in spec.schemas:
        if not _IDENT_RE.match(sname):
            problems.append(f"schema name {sname!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")

    schemas, schema_problems = _resolve_schemas(spec, schema_names)
    problems.extend(schema_problems)

    var_types, var_owner, var_problems = _resolve_vars(spec, schema_names, schemas)
    problems.extend(var_problems)

    if spec.initial not in spec.states:
        problems.append(f"initial state {spec.initial!r} is not a declared state")

    for name, state in spec.states.items():
        if not _IDENT_RE.match(name):
            problems.append(f"state name {name!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")
        problems.extend(_validate_state(name, state, var_types, var_owner, schemas, schema_names))

    problems.extend(_validate_graph(spec))
    return problems


def _resolve_schemas(
    spec: MachineSpec, schema_names: frozenset[str]
) -> tuple[dict[str, dict[str, TypeRef]], list[str]]:
    problems: list[str] = []
    resolved: dict[str, dict[str, TypeRef]] = {}
    for sname, fields in spec.schemas.items():
        resolved_fields: dict[str, TypeRef] = {}
        for fname, field in fields.items():
            if not _IDENT_RE.match(fname):
                problems.append(f"schema {sname!r}: field name {fname!r} is not a valid identifier")
            try:
                ftype = _parse_type(field.type, schema_names)
            except _TypeParseError as exc:
                problems.append(f"schema {sname!r}.{fname}: {exc}")
                continue
            if field.enum is not None and ftype != ScalarT("str"):
                problems.append(f"schema {sname!r}.{fname}: `enum` is only valid on `str` fields")
            resolved_fields[fname] = ftype
        resolved[sname] = resolved_fields
    problems.extend(_detect_schema_cycles(resolved))
    return resolved, problems


def _detect_schema_cycles(resolved: dict[str, dict[str, TypeRef]]) -> list[str]:
    problems: list[str] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in done or name not in resolved:
            return
        if name in visiting:
            cycle = " -> ".join((*trail, name))
            problems.append(f"record schema cycle: {cycle}")
            return
        visiting.add(name)
        for ftype in resolved[name].values():
            if isinstance(ftype, RecordT):
                visit(ftype.name, (*trail, name))
        visiting.discard(name)
        done.add(name)

    for name in resolved:
        visit(name, ())
    return problems


def _resolve_vars(
    spec: MachineSpec,
    schema_names: frozenset[str],
    schemas: dict[str, dict[str, TypeRef]],
) -> tuple[dict[str, TypeRef], dict[str, str], list[str]]:
    problems: list[str] = []
    var_types: dict[str, TypeRef] = {}
    var_owner: dict[str, str] = {}

    declared: dict[str, str] = {}
    owners: tuple[tuple[str, dict[str, Any]], ...] = (
        ("operator", dict(spec.vars.operator)),
        ("code", dict(spec.vars.code)),
        ("agent", dict(spec.vars.agent)),
    )
    for owner, table in owners:
        for vname, varspec in table.items():
            if not _IDENT_RE.match(vname):
                problems.append(
                    f"variable name {vname!r} in `[vars.{owner}]` is not a valid identifier"
                    " (^[a-z][a-z0-9_]*$)"
                )
            if vname in _RESERVED_NAMES:
                problems.append(f"variable name {vname!r} is reserved and may not be used")
            if vname in declared:
                problems.append(
                    f"variable {vname!r} declared in both `[vars.{declared[vname]}]` and"
                    f" `[vars.{owner}]`; the three owner subtables share one read namespace"
                )
                continue
            declared[vname] = owner
            var_owner[vname] = owner
            try:
                vtype = _parse_type(varspec.type, schema_names)
            except _TypeParseError as exc:
                problems.append(f"variable {vname!r} in `[vars.{owner}]`: {exc}")
                continue
            var_types[vname] = vtype
            value = varspec.value if owner == "operator" else varspec.default
            problems.extend(_check_value(value, vtype, schemas, f"variable {vname!r}"))
    return var_types, var_owner, problems


def _check_value(
    value: Any, t: TypeRef, schemas: dict[str, dict[str, TypeRef]], label: str
) -> list[str]:
    if isinstance(t, ScalarT):
        return _check_scalar(value, t.name, label)
    if isinstance(t, ListT):
        if not isinstance(value, list):
            return [f"{label}: expected list, got {_py_type(value)}"]
        problems: list[str] = []
        for index, element in enumerate(value):
            problems.extend(_check_scalar(element, t.elem, f"{label}[{index}]"))
        return problems
    if isinstance(t, JsonT):
        return _check_json(value, label)
    # RecordT: a default/value is a placeholder (the example uses `{}` for a
    # required-field record), so we do not require presence — but any field
    # that *is* present must be known and well-typed.
    if not isinstance(value, dict):
        return [f"{label}: expected object for record {t.name!r}, got {_py_type(value)}"]
    problems = []
    fields = schemas.get(t.name, {})
    for key, sub in value.items():
        if not isinstance(key, str) or key not in fields:
            problems.append(f"{label}: unknown field {key!r} for record {t.name!r}")
            continue
        problems.extend(_check_value(sub, fields[key], schemas, f"{label}.{key}"))
    return problems


def _check_scalar(value: Any, name: str, label: str) -> list[str]:
    if name == "bool":
        ok = isinstance(value, bool)
    elif name == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif name == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:  # str
        ok = isinstance(value, str)
    if not ok:
        return [f"{label}: expected {name}, got {_py_type(value)}"]
    return []


def _check_json(value: Any, label: str) -> list[str]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return []
    if isinstance(value, list):
        problems: list[str] = []
        for index, element in enumerate(value):
            problems.extend(_check_json(element, f"{label}[{index}]"))
        return problems
    if isinstance(value, dict):
        problems = []
        for key, sub in value.items():
            if not isinstance(key, str):
                problems.append(f"{label}: json object keys must be strings, got {_py_type(key)}")
            problems.extend(_check_json(sub, f"{label}.{key}"))
        return problems
    return [f"{label}: value is not JSON-serializable ({_py_type(value)})"]


def _py_type(value: Any) -> str:
    return type(value).__name__


# --------------------------------------------------------------------------
# Runtime payload validation (the agent `finish_run` trust boundary)
# --------------------------------------------------------------------------


def validate_finish_payload(spec: MachineSpec, schema_name: str, payload: Any) -> list[str]:
    """Strictly validate an agent `finish_run` *payload* against a record schema.

    Stricter than the load-time placeholder check on variable defaults: every
    non-optional field must be present, `enum` constraints are enforced, and
    nested records recurse. Presumes *spec* already passed `validate_semantics`,
    so its schema graph is well-formed. Returns an empty list when the payload
    conforms, or a list of human-readable problems otherwise.
    """
    schema_names = frozenset(spec.schemas)
    return _check_record_strict(
        payload, schema_name, spec.schemas, schema_names, "finish_run payload"
    )


def _check_record_strict(
    value: Any,
    schema_name: str,
    raw_schemas: dict[str, dict[str, FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    fields = raw_schemas.get(schema_name)
    if fields is None:
        return [f"{label}: unknown schema {schema_name!r}"]
    if not isinstance(value, dict):
        return [f"{label}: expected object for record {schema_name!r}, got {_py_type(value)}"]
    problems: list[str] = []
    for fname, field in fields.items():
        if fname not in value:
            if not field.optional:
                problems.append(f"{label}: missing required field {fname!r}")
            continue
        problems.extend(
            _check_field_value(value[fname], field, raw_schemas, schema_names, f"{label}.{fname}")
        )
    for key in value:
        if key not in fields:
            problems.append(f"{label}: unknown field {key!r} for record {schema_name!r}")
    return problems


def _check_field_value(
    value: Any,
    field: FieldSpec,
    raw_schemas: dict[str, dict[str, FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    try:
        ftype = _parse_type(field.type, schema_names)
    except _TypeParseError as exc:  # pragma: no cover - spec already validated
        return [f"{label}: {exc}"]
    if isinstance(ftype, RecordT):
        return _check_record_strict(value, ftype.name, raw_schemas, schema_names, label)
    problems = _check_value(value, ftype, {}, label)
    if not problems and field.enum is not None and value not in field.enum:
        problems.append(f"{label}: {value!r} is not one of enum {list(field.enum)}")
    return problems


# --------------------------------------------------------------------------
# Reference / template resolution
# --------------------------------------------------------------------------


def _resolve_ref_type(
    ref: Reference,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
) -> tuple[TypeRef | None, str | None]:
    if ref.root == "result":
        if result_type is None:
            return None, f"`result` is not navigable here ({ref.dotted!r})"
        current: TypeRef = result_type
    else:
        looked_up = var_types.get(ref.root)
        if looked_up is None:
            return None, f"unknown variable {ref.root!r}"
        current = looked_up
    for key in ref.path:
        if not isinstance(current, RecordT):
            return None, f"cannot navigate into {_type_str(current)} at {ref.dotted!r}"
        fields = schemas.get(current.name, {})
        if key not in fields:
            return None, f"record {current.name!r} has no field {key!r} (in {ref.dotted!r})"
        current = fields[key]
    return current, None


def _validate_template(
    text: str,
    *,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
    allow_splice: bool,
    where: str,
) -> list[str]:
    try:
        template = parse_template(text)
    except TemplateError as exc:
        return [f"{where}: {exc}"]
    problems: list[str] = []
    for part in template.parts:
        if not isinstance(part, Interp):
            continue
        ref_type, error = _resolve_ref_type(part.ref, var_types, schemas, result_type)
        if error is not None:
            problems.append(f"{where}: {error}")
            continue
        assert ref_type is not None
        problems.extend(
            _check_interp_filter(part, ref_type, template, allow_splice=allow_splice, where=where)
        )
    return problems


def _check_interp_filter(
    part: Interp,
    ref_type: TypeRef,
    template: Template,
    *,
    allow_splice: bool,
    where: str,
) -> list[str]:
    if part.filt == "json":
        return []
    if part.filt == "len":
        if isinstance(ref_type, ScalarT) and ref_type.name != "str":
            return [
                f"{where}: `| len` does not apply to {_type_str(ref_type)} ({part.ref.dotted!r})"
            ]
        return []
    # Bare reference (no filter): must be a scalar, unless it is a lone
    # list reference spliced into argv (§4.4).
    if isinstance(ref_type, ScalarT):
        return []
    if allow_splice and isinstance(ref_type, ListT) and template.is_lone_ref:
        return []
    return [
        f"{where}: bare reference to {_type_str(ref_type)} ({part.ref.dotted!r});"
        " apply `| json` or, for a list in argv, splice it as a standalone element"
    ]


# --------------------------------------------------------------------------
# State validation
# --------------------------------------------------------------------------


def _validate_state(
    name: str,
    state: StateSpec,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    if isinstance(state, AgentState):
        return _validate_agent(name, state, var_types, var_owner, schemas, schema_names)
    if isinstance(state, ToolState):
        return _validate_tool(name, state, var_types, var_owner, schemas, schema_names)
    if isinstance(state, WaitState):
        return _validate_wait(name, state, var_types, schemas)
    if isinstance(state, BranchState):
        return _validate_branch(name, state, var_types, schemas)
    return []  # TerminalState: shape is fully checked by pydantic


def _validate_on(name: str, on: dict[str, str], expected: frozenset[str]) -> list[str]:
    got = frozenset(on)
    problems: list[str] = []
    for missing in sorted(expected - got):
        problems.append(f"state {name!r}: `on` is missing outcome {missing!r}")
    for extra in sorted(got - expected):
        problems.append(
            f"state {name!r}: `on` has unknown outcome {extra!r} (allowed: {sorted(expected)})"
        )
    return problems


def _validate_agent(
    name: str,
    state: AgentState,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    problems = _validate_on(name, state.on, _AGENT_LABELS)
    if state.output_schema not in schema_names:
        problems.append(
            f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
        )
        result_type: TypeRef | None = None
    else:
        result_type = RecordT(state.output_schema)
    problems.extend(
        _validate_template(
            state.prompt,
            var_types=var_types,
            schemas=schemas,
            result_type=None,
            allow_splice=False,
            where=f"state {name!r} prompt",
        )
    )
    if state.capture.stdout_json is not None:
        problems.append(f"state {name!r}: an `agent` capture uses `finish_json`, not `stdout_json`")
    problems.extend(
        _validate_capture(
            name,
            state.capture,
            owner="agent",
            var_types=var_types,
            var_owner=var_owner,
            schemas=schemas,
            result_type=result_type,
            whole_type=result_type,
        )
    )
    return problems


def _validate_tool(
    name: str,
    state: ToolState,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    problems = _validate_on(name, state.on, _TOOL_LABELS)
    result_type: TypeRef | None = None
    if state.output_schema is not None:
        if state.output_schema not in schema_names:
            problems.append(
                f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
            )
        else:
            result_type = RecordT(state.output_schema)
    for index, element in enumerate(state.command):
        problems.extend(
            _validate_template(
                element,
                var_types=var_types,
                schemas=schemas,
                result_type=None,
                allow_splice=True,
                where=f"state {name!r} command[{index}]",
            )
        )
    if state.capture is not None:
        if state.capture.finish_json is not None:
            problems.append(
                f"state {name!r}: a `tool` capture uses `stdout_json`, not `finish_json`"
            )
        if state.capture.stdout_json is not None and state.output_schema is not None:
            problems.append(
                f"state {name!r}: `stdout_json` whole-capture is opaque; drop `output_schema`"
                " or use `set` field-capture"
            )
        problems.extend(
            _validate_capture(
                name,
                state.capture,
                owner="code",
                var_types=var_types,
                var_owner=var_owner,
                schemas=schemas,
                result_type=result_type,
                whole_type=JsonT(),
            )
        )
    return problems


def _validate_capture(
    name: str,
    capture: Capture,
    *,
    owner: str,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
    whole_type: TypeRef | None,
) -> list[str]:
    problems: list[str] = []
    whole_target = capture.stdout_json if owner == "code" else capture.finish_json
    if whole_target is not None:
        problems.extend(_check_capture_target(name, whole_target, owner, var_owner))
        target_type = var_types.get(whole_target)
        if target_type is not None and whole_type is not None and target_type != whole_type:
            problems.append(
                f"state {name!r}: capture target {whole_target!r} has type"
                f" {_type_str(target_type)} but the captured value is {_type_str(whole_type)}"
            )
    if capture.set is not None:
        for target, template in capture.set.items():
            problems.extend(_check_capture_target(name, target, owner, var_owner))
            problems.extend(
                _validate_set_assignment(
                    name,
                    target,
                    template,
                    var_types=var_types,
                    schemas=schemas,
                    result_type=result_type,
                )
            )
    return problems


def _check_capture_target(
    name: str, target: str, owner: str, var_owner: dict[str, str]
) -> list[str]:
    actual = var_owner.get(target)
    if actual is None:
        return [f"state {name!r}: capture target {target!r} is not a declared variable"]
    if actual != owner:
        return [
            f"state {name!r}: a `{owner}` state may only write `[vars.{owner}]` variables,"
            f" but {target!r} is owned by `[vars.{actual}]`"
        ]
    return []


def _validate_set_assignment(
    name: str,
    target: str,
    template: str,
    *,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
) -> list[str]:
    where = f"state {name!r} capture.set.{target}"
    try:
        parsed = parse_template(template)
    except TemplateError as exc:
        return [f"{where}: {exc}"]
    target_type = var_types.get(target)
    # A lone, filter-less interpolation captures the referenced *value* with
    # its native type (the only way a non-string value reaches the
    # blackboard); its type must match the target variable.
    if parsed.is_lone_ref:
        interp = parsed.parts[0]
        assert isinstance(interp, Interp)
        source_type, error = _resolve_ref_type(interp.ref, var_types, schemas, result_type)
        if error is not None:
            return [f"{where}: {error}"]
        if target_type is not None and source_type is not None and source_type != target_type:
            return [
                f"{where}: assigns {_type_str(source_type)} to {target!r} of type"
                f" {_type_str(target_type)}"
            ]
        return []
    # Otherwise the assignment renders to a string, so the target must be str.
    problems = _validate_template(
        template,
        var_types=var_types,
        schemas=schemas,
        result_type=result_type,
        allow_splice=False,
        where=where,
    )
    if target_type is not None and target_type != ScalarT("str"):
        problems.append(
            f"{where}: a rendered template yields a string but {target!r} has type"
            f" {_type_str(target_type)}"
        )
    return problems


def _validate_wait(
    name: str,
    state: WaitState,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    problems = _validate_on(name, state.on, _WAIT_LABELS)
    timings = [
        timing
        for timing, value in (
            ("every_secs", state.every_secs),
            ("until", state.until),
            ("cron", state.cron),
        )
        if value is not None
    ]
    if len(timings) != 1:
        problems.append(
            f"state {name!r}: a `wait` must declare exactly one of `every_secs`, `until`,"
            f" or `cron` (found: {timings or 'none'})"
        )
    if state.cron is not None:
        # `cron` is parsed but the v1 runtime cannot fire it -- `machine run`
        # would raise mid-run. Reject it at load so `machine check`/`test`
        # catch it up front instead of failing only when the wait is reached.
        problems.append(
            f"state {name!r}: `cron` wait timing is not yet implemented"
            " (reserved for a future persisted-wake runtime); use `every_secs` or `until`"
        )
    for timing, value in (
        ("every_secs", state.every_secs),
        ("until", state.until),
        ("cron", state.cron),
    ):
        if value is None:
            continue
        problems.extend(
            _validate_template(
                value,
                var_types=var_types,
                schemas=schemas,
                result_type=None,
                allow_splice=False,
                where=f"state {name!r} {timing}",
            )
        )
    return problems


def _validate_branch(
    name: str,
    state: BranchState,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    problems: list[str] = []
    last_index = len(state.when) - 1
    for index, clause in enumerate(state.when):
        if clause.else_ is not None and index != last_index:
            problems.append(f"state {name!r}: an `else` clause must be the final `when` clause")
        if clause.if_ is not None:
            problems.extend(_validate_predicate(name, clause.if_, var_types, schemas))
    if state.when[last_index].else_ is None:
        problems.append(
            f"state {name!r}: branch is not total (no final `else`);"
            " add `{ else = true, goto = ... }`"
        )
    return problems


def _validate_predicate(
    name: str,
    source: str,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    try:
        predicate = parse_predicate(source)
    except PredicateError as exc:
        return [f"state {name!r}: predicate {source!r}: {exc}"]
    problems: list[str] = []
    for ref in predicate.references:
        _, error = _resolve_ref_type(ref, var_types, schemas, None)
        if error is not None:
            problems.append(f"state {name!r}: predicate {source!r}: {error}")
    return problems


def _validate_graph(spec: MachineSpec) -> list[str]:
    problems: list[str] = []
    for edge in edges(spec):
        if edge.dst not in spec.states:
            problems.append(
                f"state {edge.src!r}: transition target {edge.dst!r} is not a declared state"
            )
    reachable = reachable_states(spec)
    for name in spec.states:
        if name not in reachable:
            problems.append(f"state {name!r} is unreachable from initial state {spec.initial!r}")
    return problems
