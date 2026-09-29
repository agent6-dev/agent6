# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Load and semantically validate a `.asm.toml` machine.

The shapes are pydantic (`spec`); this module adds the cross-cutting load-time rules (name
uniqueness, the ownership wall, reference and field type checks, total branches,
reachability) plus `load_machine` and the runtime `validate_record_payload`. Every violation
is aggregated into one MachineError.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import pathlib
import tomllib
from collections.abc import Mapping
from typing import Any, Literal

import pydantic

from agent6.machine import predicate as machine_predicate
from agent6.machine import spec as machine_spec
from agent6.machine import template as machine_template


def load_machine(path: pathlib.Path) -> machine_spec.MachineSpec:
    """Load, parse and fully validate a `.asm.toml` file.

    Args:
        path: The machine file.

    Returns:
        The validated machine; never a partially valid one.

    Raises:
        MachineError: Every diagnostic, aggregated.
    """
    if not path.is_file():
        raise machine_spec.MachineError([f"machine file not found: {path}"])
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise machine_spec.MachineError([f"not valid TOML ({path}): {exc}"]) from exc
    except UnicodeDecodeError as exc:
        raise machine_spec.MachineError([f"not valid UTF-8 ({path}): {exc}"]) from exc
    except OSError as exc:
        raise machine_spec.MachineError([f"cannot be read ({path}): {exc}"]) from exc
    precheck = _precheck(raw)
    if precheck:
        raise machine_spec.MachineError(precheck)
    try:
        spec = machine_spec.MachineSpec.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise machine_spec.MachineError(_format_validation_error(exc)) from exc
    problems = validate_semantics(spec)
    if problems:
        raise machine_spec.MachineError(problems)
    return spec


def _precheck(raw: dict[str, Any]) -> list[str]:
    """Return the problems pydantic would report less readably: a var outside an owner table."""
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


def _format_validation_error(err: pydantic.ValidationError) -> list[str]:
    """Return one line per pydantic issue, located by its dotted path."""
    problems: list[str] = []
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        problems.append(f"{loc}: {issue['msg']} (type={issue['type']})")
    return problems


@dataclasses.dataclass(frozen=True, slots=True)
class _Env:
    """The resolved declarations the state rules read.

    Attributes:
        var_types: Each variable's type.
        var_owner: Each variable's owner subtable.
        var_values: Each variable's declared value or default.
        schemas: The record schemas, fields resolved to types.
    """

    var_types: dict[str, machine_spec.TypeRef]
    var_owner: dict[str, str]
    var_values: dict[str, Any]
    schemas: dict[str, dict[str, machine_spec.TypeRef]]


def _resolve_env(spec: machine_spec.MachineSpec) -> tuple[_Env, list[str]]:
    """Resolve the schemas and variables.

    Returns:
        The env and the problems found.
    """
    schema_names = frozenset(spec.schemas)
    schemas, problems = _resolve_schemas(spec, schema_names)
    var_types, var_owner, var_values, var_problems = _resolve_vars(spec, schema_names, schemas)
    problems.extend(var_problems)
    return _Env(var_types, var_owner, var_values, schemas), problems


def validate_semantics(spec: machine_spec.MachineSpec) -> list[str]:
    """Run every cross-cutting rule the pydantic shape cannot express.

    Args:
        spec: The parsed machine.

    Returns:
        The problems, in order; empty when the machine is valid.
    """
    problems: list[str] = []

    for sname in spec.schemas:
        if not machine_spec.IDENT_RE.fullmatch(sname):
            problems.append(f"schema name {sname!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")
        elif sname in machine_spec.BUILTIN_TYPE_NAMES:
            # `parse_type` resolves the built-ins first, so such a schema could never be named.
            problems.append(
                f"schema name {sname!r} is a built-in type, so nothing could reference it"
            )

    env, env_problems = _resolve_env(spec)
    problems.extend(env_problems)

    if spec.initial not in spec.states:
        problems.append(f"initial state {spec.initial!r} is not a declared state")

    for name, state in spec.states.items():
        if not machine_spec.IDENT_RE.fullmatch(name):
            problems.append(f"state name {name!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")
        problems.extend(_validate_state(name, state, env))

    problems.extend(_validate_graph(spec))
    return problems


def _resolve_schemas(
    spec: machine_spec.MachineSpec, schema_names: frozenset[str]
) -> tuple[dict[str, dict[str, machine_spec.TypeRef]], list[str]]:
    """Resolve each schema's field types.

    Returns:
        The schemas and the problems found.
    """
    problems: list[str] = []
    resolved: dict[str, dict[str, machine_spec.TypeRef]] = {}
    for sname, fields in spec.schemas.items():
        resolved_fields: dict[str, machine_spec.TypeRef] = {}
        for fname, field in fields.items():
            if not machine_spec.IDENT_RE.fullmatch(fname):
                problems.append(f"schema {sname!r}: field name {fname!r} is not a valid identifier")
            try:
                ftype = machine_spec.parse_type(field.type, schema_names)
            except machine_spec.TypeParseError as exc:
                problems.append(f"schema {sname!r}.{fname}: {exc}")
                continue
            if field.enum is not None:
                if not field.enum:
                    problems.append(
                        f"schema {sname!r}.{fname}: `enum` must contain at least one value"
                    )
                if ftype != machine_spec.ScalarT("str"):
                    problems.append(
                        f"schema {sname!r}.{fname}: `enum` is only valid on `str` fields"
                    )
            resolved_fields[fname] = ftype
        resolved[sname] = resolved_fields
    problems.extend(_detect_schema_cycles(resolved))
    return resolved, problems


def _detect_schema_cycles(resolved: dict[str, dict[str, machine_spec.TypeRef]]) -> list[str]:
    """Return one problem per cycle among record schemas."""
    problems: list[str] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        """Depth-first visit; a name met while visiting closes a cycle."""
        if name in done or name not in resolved:
            return
        if name in visiting:
            cycle = " -> ".join((*trail, name))
            problems.append(f"record schema cycle: {cycle}")
            return
        visiting.add(name)
        for ftype in resolved[name].values():
            if isinstance(ftype, machine_spec.RecordT):
                visit(ftype.name, (*trail, name))
        visiting.discard(name)
        done.add(name)

    for name in resolved:
        visit(name, ())
    return problems


def _resolve_vars(
    spec: machine_spec.MachineSpec,
    schema_names: frozenset[str],
    schemas: dict[str, dict[str, machine_spec.TypeRef]],
) -> tuple[dict[str, machine_spec.TypeRef], dict[str, str], dict[str, Any], list[str]]:
    """Resolve the three owner tables into one namespace.

    Returns:
        The types, the owners, the values and the problems found.
    """
    problems: list[str] = []
    var_types: dict[str, machine_spec.TypeRef] = {}
    var_owner: dict[str, str] = {}
    var_values: dict[str, Any] = {}

    declared: dict[str, str] = {}
    owners: tuple[tuple[str, dict[str, Any]], ...] = (
        ("operator", dict(spec.vars.operator)),
        ("code", dict(spec.vars.code)),
        ("agent", dict(spec.vars.agent)),
    )
    for owner, table in owners:
        for vname, varspec in table.items():
            if not machine_spec.IDENT_RE.fullmatch(vname):
                problems.append(
                    f"variable name {vname!r} in `[vars.{owner}]` is not a valid identifier"
                    " (^[a-z][a-z0-9_]*$)"
                )
            if vname in machine_spec.RESERVED_NAMES:
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
                vtype = machine_spec.parse_type(varspec.type, schema_names)
            except machine_spec.TypeParseError as exc:
                problems.append(f"variable {vname!r} in `[vars.{owner}]`: {exc}")
                continue
            var_types[vname] = vtype
            value = varspec.value if owner == "operator" else varspec.default
            var_values[vname] = value
            problems.extend(
                _check_value(
                    value,
                    vtype,
                    schemas,
                    f"variable {vname!r}",
                    raw_schemas=spec.schemas,
                )
            )
    return var_types, var_owner, var_values, problems


def fixture_problems(spec: machine_spec.MachineSpec, fixture: dict[str, Any]) -> list[str]:
    """Return the problems in a `--blackboard` fixture.

    Every key names a declared variable and every value satisfies its type, the checks the
    declared defaults get.

    Args:
        spec: The machine.
        fixture: The fixture's values by variable.

    Returns:
        The problems; empty when the fixture is valid.
    """
    env, _ = _resolve_env(spec)
    problems: list[str] = []
    for name, value in fixture.items():
        if name not in env.var_types:
            problems.append(f"blackboard fixture: {name!r} is not a declared variable")
            continue
        problems.extend(
            _check_value(
                value,
                env.var_types[name],
                env.schemas,
                f"fixture {name!r}",
                raw_schemas=spec.schemas,
            )
        )
    return problems


def _check_value(
    value: Any,
    t: machine_spec.TypeRef,
    schemas: dict[str, dict[str, machine_spec.TypeRef]],
    label: str,
    *,
    raw_schemas: dict[str, dict[str, machine_spec.FieldSpec]] | None = None,
) -> list[str]:
    """Return the problems of a declared value against its type.

    A record value is a placeholder: presence is not required, but every present field
    must be known and well typed.
    """
    if isinstance(t, machine_spec.ScalarT):
        return _check_scalar(value, t.name, label)
    if isinstance(t, machine_spec.ListT):
        if not isinstance(value, list):
            return [f"{label}: expected list, got {_py_type(value)}"]
        problems: list[str] = []
        for index, element in enumerate(value):
            problems.extend(_check_scalar(element, t.elem, f"{label}[{index}]"))
        return problems
    if isinstance(t, machine_spec.JsonT):
        return _check_json(value, label)
    if not isinstance(value, dict):
        return [f"{label}: expected object for record {t.name!r}, got {_py_type(value)}"]
    problems = []
    fields = schemas.get(t.name, {})
    for key, sub in value.items():
        if not isinstance(key, str) or key not in fields:
            problems.append(f"{label}: unknown field {key!r} for record {t.name!r}")
            continue
        field_problems = _check_value(
            sub, fields[key], schemas, f"{label}.{key}", raw_schemas=raw_schemas
        )
        if not field_problems and raw_schemas is not None:
            field_problems = _enum_problems(sub, raw_schemas[t.name][key], f"{label}.{key}")
        problems.extend(field_problems)
    return problems


def _enum_problems(value: Any, field: machine_spec.FieldSpec, label: str) -> list[str]:
    """Return the `enum` problem of a value, the check a default and a runtime field share."""
    if field.enum is None or value in field.enum:
        return []
    return [f"{label}: {value!r} is not one of enum {list(field.enum)}"]


def _check_scalar(value: Any, name: str, label: str) -> list[str]:
    """Return the problem of a value against a scalar type; a bool is never an int."""
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
    """Return the problems of a value against `json`: serializable, string keys."""
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
    """Return the value's type name, for a problem line."""
    return type(value).__name__


def validate_record_payload(
    schemas: dict[str, dict[str, machine_spec.FieldSpec]],
    schema_name: str,
    payload: Any,
    *,
    where: str,
) -> list[str]:
    """Strictly validate a captured payload against a record schema.

    The one runtime capture gate, for an agent's `finish_session` payload (engine side and
    the execution's in-run check) and a tool's parsed stdout: every required field present,
    enums enforced, nested records recursed. The schemas have passed `validate_semantics`.

    Args:
        schemas: The machine's schema table.
        schema_name: The schema the payload must match.
        payload: The captured value.
        where: Names the capture in each problem.

    Returns:
        The problems; empty when the payload conforms.
    """
    return _check_record_strict(payload, schema_name, schemas, frozenset(schemas), where)


def _check_record_strict(
    value: Any,
    schema_name: str,
    raw_schemas: dict[str, dict[str, machine_spec.FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    """Return the problems of a value against a schema, every required field present."""
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
    field: machine_spec.FieldSpec,
    raw_schemas: dict[str, dict[str, machine_spec.FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    """Return the problems of one field's value, recursing into a record."""
    try:
        ftype = machine_spec.parse_type(field.type, schema_names)
    except machine_spec.TypeParseError as exc:  # pragma: no cover - spec already validated
        return [f"{label}: {exc}"]
    if isinstance(ftype, machine_spec.RecordT):
        return _check_record_strict(value, ftype.name, raw_schemas, schema_names, label)
    return _check_value(value, ftype, {}, label) or _enum_problems(value, field, label)


def _resolve_ref_type(
    ref: machine_predicate.Reference, env: _Env, result_type: machine_spec.TypeRef | None
) -> tuple[machine_spec.TypeRef | None, str | None]:
    """Resolve a reference's type.

    Returns:
        The type and None, or None and the problem.
    """
    if ref.root == "result":
        if result_type is None:
            return None, f"`result` is not navigable here ({ref.dotted!r})"
        current: machine_spec.TypeRef = result_type
    else:
        looked_up = env.var_types.get(ref.root)
        if looked_up is None:
            return None, f"unknown variable {ref.root!r}"
        current = looked_up
    for key in ref.path:
        if not isinstance(current, machine_spec.RecordT):
            return None, f"cannot navigate into {machine_spec.type_str(current)} at {ref.dotted!r}"
        fields = env.schemas.get(current.name, {})
        if key not in fields:
            return None, f"record {current.name!r} has no field {key!r} (in {ref.dotted!r})"
        current = fields[key]
    return current, None


def _validate_template(
    text: str,
    env: _Env,
    *,
    result_type: machine_spec.TypeRef | None,
    allow_splice: bool,
    where: str,
) -> list[str]:
    """Return the problems of a template: every reference resolves and every filter applies."""
    try:
        template = machine_template.parse_template(text)
    except machine_template.TemplateError as exc:
        return [f"{where}: {exc}"]
    problems: list[str] = []
    for part in template.parts:
        if not isinstance(part, machine_template.Interp):
            continue
        ref_type, error = _resolve_ref_type(part.ref, env, result_type)
        if error is not None:
            problems.append(f"{where}: {error}")
            continue
        assert ref_type is not None
        if part.filt == "json":
            continue
        if part.filt == "len":
            if isinstance(ref_type, machine_spec.ScalarT) and ref_type.name != "str":
                problems.append(
                    f"{where}: `| len` does not apply to {machine_spec.type_str(ref_type)} "
                    f"({part.ref.dotted!r})"
                )
            continue
        # A bare reference must be a scalar, unless a lone list reference is spliced into argv.
        if isinstance(ref_type, machine_spec.ScalarT):
            continue
        if allow_splice and isinstance(ref_type, machine_spec.ListT) and template.is_lone_ref:
            continue
        problems.append(
            f"{where}: bare reference to {machine_spec.type_str(ref_type)} ({part.ref.dotted!r});"
            " apply `| json` or, for a list in argv, splice it as a standalone element"
        )
    return problems


def _validate_state(name: str, state: machine_spec.StateSpec, env: _Env) -> list[str]:
    """Return one state's problems; every kind's `notify` template is checked first."""
    problems: list[str] = []
    if state.notify is not None:
        problems.extend(
            _validate_template(
                state.notify.message,
                env,
                result_type=None,
                allow_splice=False,
                where=f"state {name!r} notify",
            )
        )
    if isinstance(state, machine_spec.AgentState):
        problems.extend(_validate_agent(name, state, env))
    elif isinstance(state, machine_spec.ToolState):
        problems.extend(_validate_tool(name, state, env))
    elif isinstance(state, machine_spec.WaitState):
        problems.extend(_validate_wait(name, state, env))
    elif isinstance(state, machine_spec.BranchState):
        problems.extend(_validate_branch(name, state, env))
    return problems  # a terminal's shape is fully checked by pydantic


def _validate_on(name: str, on: dict[str, str], expected: frozenset[str]) -> list[str]:
    """Return the problems of an `on` table against the kind's outcome labels."""
    got = frozenset(on)
    problems: list[str] = []
    for missing in sorted(expected - got):
        problems.append(f"state {name!r}: `on` is missing outcome {missing!r}")
    for extra in sorted(got - expected):
        problems.append(
            f"state {name!r}: `on` has unknown outcome {extra!r} (allowed: {sorted(expected)})"
        )
    return problems


def _validate_agent(name: str, state: machine_spec.AgentState, env: _Env) -> list[str]:
    """Return an agent state's problems."""
    problems = _validate_on(name, state.on, machine_spec.AGENT_LABELS)
    if state.output_schema not in env.schemas:
        problems.append(
            f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
        )
        result_type: machine_spec.TypeRef | None = None
    else:
        result_type = machine_spec.RecordT(state.output_schema)
    problems.extend(
        _validate_template(
            state.prompt,
            env,
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
            env,
            owner="agent",
            result_type=result_type,
            whole_type=result_type,
        )
    )
    return problems


def _validate_tool(name: str, state: machine_spec.ToolState, env: _Env) -> list[str]:
    """Return a tool state's problems."""
    problems = _validate_on(name, state.on, machine_spec.TOOL_LABELS)
    result_type: machine_spec.TypeRef | None = None
    if state.output_schema is not None:
        if state.output_schema not in env.schemas:
            problems.append(
                f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
            )
        else:
            result_type = machine_spec.RecordT(state.output_schema)
    for index, element in enumerate(state.command):
        problems.extend(
            _validate_template(
                element,
                env,
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
                env,
                owner="code",
                result_type=result_type,
                whole_type=machine_spec.JsonT(),
            )
        )
    return problems


def _validate_capture(
    name: str,
    capture: machine_spec.Capture,
    env: _Env,
    *,
    owner: str,
    result_type: machine_spec.TypeRef | None,
    whole_type: machine_spec.TypeRef | None,
) -> list[str]:
    """Return a capture's problems: the ownership wall and the target types."""
    problems: list[str] = []
    whole_target = capture.stdout_json if owner == "code" else capture.finish_json
    if whole_target is not None:
        problems.extend(_check_capture_target(name, whole_target, owner, env.var_owner))
        target_type = env.var_types.get(whole_target)
        if target_type is not None and whole_type is not None and target_type != whole_type:
            problems.append(
                f"state {name!r}: capture target {whole_target!r} has "
                "type"
                f" {machine_spec.type_str(target_type)} but the captured value is "
                f"{machine_spec.type_str(whole_type)};"
                f' declare it as type = "{machine_spec.type_decl(whole_type)}"'
            )
    if capture.set is not None:
        for target, template in capture.set.items():
            problems.extend(_check_capture_target(name, target, owner, env.var_owner))
            problems.extend(
                _validate_set_assignment(name, target, template, env, result_type=result_type)
            )
    return problems


def _check_capture_target(
    name: str, target: str, owner: str, var_owner: dict[str, str]
) -> list[str]:
    """Return the ownership-wall problem of a capture target."""
    actual = var_owner.get(target)
    if actual is None:
        return [
            f"state {name!r}: capture target {target!r} is not a declared variable;"
            f" declare it in [vars.{owner}]"
        ]
    if actual != owner:
        return [
            f"state {name!r}: a `{owner}` state may only write `[vars.{owner}]` variables,"
            f" but {target!r} is owned by `[vars.{actual}]`"
        ]
    return []


def _validate_set_assignment(
    name: str, target: str, template: str, env: _Env, *, result_type: machine_spec.TypeRef | None
) -> list[str]:
    """Return a `capture.set` assignment's problems: a lone reference keeps its type, else str."""
    where = f"state {name!r} capture.set.{target}"
    try:
        parsed = machine_template.parse_template(template)
    except machine_template.TemplateError as exc:
        return [f"{where}: {exc}"]
    target_type = env.var_types.get(target)
    if parsed.is_lone_ref:
        interp = parsed.parts[0]
        assert isinstance(interp, machine_template.Interp)
        source_type, error = _resolve_ref_type(interp.ref, env, result_type)
        if error is not None:
            return [f"{where}: {error}"]
        if target_type is not None and source_type is not None and source_type != target_type:
            return [
                f"{where}: assigns {machine_spec.type_str(source_type)} to {target!r} of type"
                f" {machine_spec.type_str(target_type)};"
                f' declare it as type = "{machine_spec.type_decl(source_type)}"'
            ]
        return []
    problems = _validate_template(
        template, env, result_type=result_type, allow_splice=False, where=where
    )
    if target_type is not None and target_type != machine_spec.ScalarT("str"):
        problems.append(
            f"{where}: a rendered template yields a string but {target!r} has type"
            f' {machine_spec.type_str(target_type)}; declare it as type = "str"'
            " (only a lone {{ var }} keeps a value's type)"
        )
    return problems


def _validate_wait(name: str, state: machine_spec.WaitState, env: _Env) -> list[str]:
    """Return a wait state's problems; a timerless wait declares only `signal`."""
    timings = [
        timing
        for timing, value in (
            ("every_secs", state.every_secs),
            ("until", state.until),
        )
        if value is not None
    ]
    if len(timings) > 1:
        problems = [
            f"state {name!r}: a `wait` may declare at most one of `every_secs` or"
            f" `until` (found: {timings})"
        ]
    elif not timings:
        problems = _validate_on(name, state.on, machine_spec.WAIT_LABELS - frozenset({"tick"}))
    else:
        problems = _validate_on(name, state.on, machine_spec.WAIT_LABELS)
    for timing, value in (
        ("every_secs", state.every_secs),
        ("until", state.until),
    ):
        if value is None:
            continue
        problems.extend(
            _validate_template(
                value,
                env,
                result_type=None,
                allow_splice=False,
                where=f"state {name!r} {timing}",
            )
        )
    if state.every_secs is not None:
        problems.extend(_timing_problems(name, "every_secs", state.every_secs, env, kind="int"))
    if state.until is not None:
        problems.extend(_timing_problems(name, "until", state.until, env, kind="iso"))
    return problems


def _static_ref_value(ref: machine_predicate.Reference, var_values: dict[str, Any]) -> Any:
    """Return the declared value a reference resolves to, or None."""
    current: Any = var_values.get(ref.root)
    for key in ref.path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _timing_literal_problems(
    name: str, key: str, literal: str, kind: Literal["int", "iso"]
) -> list[str]:
    """Return the problems of a constant timing: a positive integer, or an ISO-8601 instant."""
    if kind == "int":
        try:
            seconds = int(literal)
        except ValueError:
            return [
                f"state {name!r}: `{key}` must be an integer literal or an int variable"
                f" reference; got {literal!r}"
            ]
        if seconds < 1:
            return [
                f"state {name!r}: `{key}` must be >= 1; got {seconds}"
                " (a zero or negative interval would busy-loop)"
            ]
        return []
    try:
        datetime.datetime.fromisoformat(literal)
    except ValueError:
        return [
            f"state {name!r}: `{key}` must be an ISO-8601 instant or a str variable"
            f" reference; got {literal!r}"
        ]
    return []


def _timing_problems(
    name: str, key: str, text: str, env: _Env, *, kind: Literal["int", "iso"]
) -> list[str]:
    """Return the value problems of a wait timing template.

    A literal is range or format checked; a lone reference is type checked, and an
    operator-owned constant is checked statically; a composite is the engine's to check.

    Args:
        name: The state's name.
        key: `every_secs` or `until`.
        text: The timing template.
        env: The resolved declarations.
        kind: `int` for `every_secs`, `iso` for `until`.

    Returns:
        The problems.
    """
    problems: list[str] = []
    try:
        template = machine_template.parse_template(text)
    except machine_template.TemplateError:
        return problems  # `_validate_template` reported it
    if not any(isinstance(part, machine_template.Interp) for part in template.parts):
        literal = "".join(part for part in template.parts if isinstance(part, str))
        return _timing_literal_problems(name, key, literal, kind)
    if not template.is_lone_ref:
        return problems  # composite: the engine validates the rendered value
    interp = template.parts[0]
    assert isinstance(interp, machine_template.Interp)
    ref = interp.ref
    ref_type, error = _resolve_ref_type(ref, env, None)
    if error is None:
        assert ref_type is not None
        expected = machine_spec.ScalarT("int") if kind == "int" else machine_spec.ScalarT("str")
        if ref_type != expected:
            noun = "an int" if kind == "int" else "a str"
            problems.append(
                f"state {name!r}: `{key}` must reference {noun} variable, not "
                f"{machine_spec.type_str(ref_type)}"
            )
        elif env.var_owner.get(ref.root) == "operator":
            value = _static_ref_value(ref, env.var_values)
            if kind == "int":
                if isinstance(value, int) and not isinstance(value, bool) and value < 1:
                    problems.append(
                        f"state {name!r}: `{key}` must be >= 1; {ref.dotted!r} is {value}"
                        " (a zero or negative interval would busy-loop)"
                    )
            elif isinstance(value, str):  # kind == "iso"
                try:
                    datetime.datetime.fromisoformat(value)
                except ValueError:
                    problems.append(
                        f"state {name!r}: `{key}` must be an ISO-8601 instant;"
                        f" {ref.dotted!r} is {value!r}"
                    )
    return problems


def _validate_branch(name: str, state: machine_spec.BranchState, env: _Env) -> list[str]:
    """Return a branch state's problems: each predicate, and a final `else`."""
    problems: list[str] = []
    last_index = len(state.when) - 1
    for index, clause in enumerate(state.when):
        if clause.else_ is not None and index != last_index:
            problems.append(f"state {name!r}: an `else` clause must be the final `when` clause")
        if clause.if_ is not None:
            problems.extend(_validate_predicate(name, clause.if_, env))
    if state.when[last_index].else_ is None:
        problems.append(
            f"state {name!r}: branch is not total (no final `else`);"
            " add `{ else = true, goto = ... }`"
        )
    return problems


def _validate_predicate(name: str, source: str, env: _Env) -> list[str]:
    """Return a predicate's problems: its grammar, its references and its `len()` arguments."""
    try:
        predicate = machine_predicate.parse_predicate(source)
    except machine_predicate.PredicateError as exc:
        return [f"state {name!r}: predicate {source!r}: {exc}"]
    problems: list[str] = []
    for ref in predicate.references:
        _, error = _resolve_ref_type(ref, env, None)
        if error is not None:
            # An author writes TOML's `true`, which the Python parser reads as an undeclared name.
            if not ref.path and ref.root in {"true", "false", "null", "none"}:
                error += (
                    " (predicates use Python literals True/False/None, not TOML"
                    " true/false/null; for a bool var write the bare name, e.g. `flag`)"
                )
            problems.append(f"state {name!r}: predicate {source!r}: {error}")
    problems.extend(_predicate_len_problems(name, source, predicate.tree.body, env))
    return problems


def _predicate_len_problems(name: str, source: str, body: ast.expr, env: _Env) -> list[str]:
    """Return the `len()` calls whose argument has no length, as the `| len` filter check does."""
    problems: list[str] = []
    for node in ast.walk(body):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
        ):
            continue
        arg = node.args[0]
        ref = (
            machine_predicate.as_reference(arg)
            if isinstance(arg, (ast.Name, ast.Attribute))
            else None
        )
        if ref is not None:
            ftype, error = _resolve_ref_type(ref, env, None)
            if error is None and isinstance(ftype, machine_spec.ScalarT) and ftype.name != "str":
                problems.append(
                    f"state {name!r}: predicate {source!r}: `len()` does not apply to"
                    f" {machine_spec.type_str(ftype)} ({ref.dotted!r})"
                )
        elif isinstance(arg, ast.Constant) and not isinstance(arg.value, str):
            problems.append(
                f"state {name!r}: predicate {source!r}: `len()` does not apply to literal"
                f" {arg.value!r}"
            )
    return problems


def _validate_graph(spec: machine_spec.MachineSpec) -> list[str]:
    """Return the graph's problems: undeclared targets and unreachable states."""
    problems: list[str] = []
    for edge in machine_spec.edges(spec):
        if edge.dst not in spec.states:
            problems.append(
                f"state {edge.src!r}: transition target {edge.dst!r} is not a declared state"
            )
    reachable = machine_spec.reachable_states(spec)
    for name in spec.states:
        if name not in reachable:
            problems.append(f"state {name!r} is unreachable from initial state {spec.initial!r}")
    return problems
