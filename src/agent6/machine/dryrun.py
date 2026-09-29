# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The author-time dry run behind `agent6 machine test`.

No jail, network, provider call or clock. Per state, the success fact the state would
emit is synthesized and pushed through the engine's `reduce`, so the capture binds and the
label routes to a declared state; per branch, every `when` clause is evaluated against the
declared defaults overlaid with the operator's fixture. A green run means the plumbing,
schemas, captures and routing are sound.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from agent6.machine import _semantics, engine, journal, template
from agent6.machine import predicate as machine_predicate
from agent6.machine import spec as machine_spec

__all__ = [
    "BranchCheck",
    "DryRunReport",
    "StateCheck",
    "dry_run",
    "synthesize_record",
]

_LIST_RE = re.compile(r"^list\[([a-z0-9_]+)\]$")
_SCALAR_EXAMPLES: dict[str, Any] = {"str": "", "int": 0, "float": 0.0, "bool": False}


@dataclasses.dataclass(frozen=True, slots=True)
class StateCheck:
    """One non-branch state's dry-run result.

    Attributes:
        name: The state's name.
        kind: The state's kind.
        ok: The capture bound and the label routed to a declared state.
        label: The label the synthesized fact produced, or None for a terminal.
        goto: The state the label routes to, or None.
        detail: The capture summary, the terminal's status, or the problem.
    """

    name: str
    kind: str
    ok: bool
    label: str | None
    goto: str | None
    detail: str


@dataclasses.dataclass(frozen=True, slots=True)
class BranchCheck:
    """One branch state's dry-run result.

    Attributes:
        name: The state's name.
        clause_index: The index of the clause that fired, or None on a predicate error.
        predicate: The clause's `if` text, or "else".
        goto: The state the clause routes to.
        ok: The target is a declared state and every predicate evaluated.
        detail: The problem, or "".
    """

    name: str
    clause_index: int | None
    predicate: str | None
    goto: str | None
    ok: bool
    detail: str


@dataclasses.dataclass(frozen=True, slots=True)
class DryRunReport:
    """The per-state and per-branch checks of one dry run."""

    states: tuple[StateCheck, ...]
    branches: tuple[BranchCheck, ...]

    @property
    def ok(self) -> bool:
        """Every check passed."""
        return all(s.ok for s in self.states) and all(b.ok for b in self.branches)


def synthesize_record(
    spec: machine_spec.MachineSpec, schema_name: str, _seen: tuple[str, ...] = ()
) -> Any:
    """Return a minimal schema-valid example object for a schema.

    Exactly the required fields: scalars zero, lists empty, enums their first member, nested
    records recursed. An optional field is omitted, the weakest state the capture gate
    permits, so an unguarded read of one fails offline as it would live.

    Args:
        spec: The machine.
        schema_name: The schema to synthesize.
        _seen: The schemas on the recursion path; a cycle yields {}.

    Returns:
        The example object.
    """
    fields = spec.schemas.get(schema_name)
    if fields is None:  # pragma: no cover - validate_semantics guarantees it exists
        return {}
    out: dict[str, Any] = {}
    for fname, field in fields.items():
        if field.optional:
            continue
        out[fname] = _synthesize_field(spec, field, (*_seen, schema_name))
    return out


def _synthesize_field(spec: machine_spec.MachineSpec, field: Any, seen: tuple[str, ...]) -> Any:
    """Return the example value for one field."""
    if field.enum:
        return field.enum[0]
    t: str = field.type
    if t in _SCALAR_EXAMPLES:
        return _SCALAR_EXAMPLES[t]
    if t == "json":
        return {}
    if _LIST_RE.match(t):
        return []
    if t in spec.schemas:  # record reference (guard cycles already rejected at load)
        return {} if t in seen else synthesize_record(spec, t, seen)
    return None  # pragma: no cover - unknown types already rejected at load


def _capture_summary(capture: Any) -> str:
    """Return the variables a state's capture binds, for the report."""
    if capture is None:
        return "no capture"
    if capture.stdout_json is not None:
        targets = [capture.stdout_json]
    elif capture.finish_json is not None:
        targets = [capture.finish_json]
    elif capture.set is not None:
        targets = sorted(capture.set)
    else:  # pragma: no cover - Capture validator guarantees one mode is set
        targets = []
    return f"captures {', '.join(targets)}" if targets else "no capture"


def _check_tool(
    spec: machine_spec.MachineSpec,
    name: str,
    state: machine_spec.ToolState,
    blackboard: dict[str, Any],
) -> StateCheck:
    """Return the check of a tool state's success path."""
    if state.output_schema is not None:
        stdout = json.dumps(synthesize_record(spec, state.output_schema))
    else:
        # A schema-less capture still requires one JSON value; null is the weakest valid one.
        stdout = "null" if state.capture is not None else ""
    fact = journal.ToolFact(exit_code=0, stdout=stdout, timed_out=False)
    engine.reduce(
        spec, state, fact, blackboard
    )  # exercises capture rendering; raises on a bad template
    goto = state.on["ok"]
    if goto not in spec.states:
        return StateCheck(name, "tool", False, "ok", goto, f"on.ok -> {goto!r} is not a state")
    return StateCheck(name, "tool", True, "ok", goto, _capture_summary(state.capture))


def _check_agent(
    spec: machine_spec.MachineSpec,
    name: str,
    state: machine_spec.AgentState,
    blackboard: dict[str, Any],
) -> StateCheck:
    """Return the check of an agent state's success path."""
    payload = synthesize_record(spec, state.output_schema)
    problems = _semantics.validate_record_payload(
        spec.schemas, state.output_schema, payload, where="finish_session payload"
    )
    if problems:  # pragma: no cover - synthesis is schema-valid by construction
        return StateCheck(name, "agent", False, "ok", None, "; ".join(problems))
    fact = journal.AgentFact(outcome="ok", reason="finish_session", payload=payload)
    engine.reduce(
        spec, state, fact, blackboard
    )  # exercises capture rendering; raises on a bad template
    goto = state.on["ok"]
    if goto not in spec.states:
        return StateCheck(name, "agent", False, "ok", goto, f"on.ok -> {goto!r} is not a state")
    return StateCheck(name, "agent", True, "ok", goto, _capture_summary(state.capture))


def _check_state(
    spec: machine_spec.MachineSpec, name: str, state: Any, blackboard: dict[str, Any]
) -> StateCheck:
    """Return the check of one non-branch state; a runtime error is the detail."""
    try:
        if isinstance(state, machine_spec.ToolState):
            return _check_tool(spec, name, state, blackboard)
        if isinstance(state, machine_spec.AgentState):
            return _check_agent(spec, name, state, blackboard)
        if isinstance(state, machine_spec.WaitState):
            # A wait with no timer parks until a poke (no `tick` edge).
            forever = state.every_secs is None and state.until is None
            label = "signal" if forever else "tick"
            goto = state.on[label]
            ok = goto in spec.states
            detail = f"{label} path" if ok else f"on.{label} -> {goto!r} is not a state"
            return StateCheck(name, "wait", ok, label, goto, detail)
        if isinstance(state, machine_spec.TerminalState):
            return StateCheck(name, "terminal", True, None, None, f"{state.status}: {state.reason}")
    except (engine.EngineError, template.TemplateError, machine_predicate.PredicateError) as exc:
        return StateCheck(name, getattr(state, "kind", "?"), False, None, None, str(exc))
    return StateCheck(name, getattr(state, "kind", "?"), True, None, None, "")  # pragma: no cover


def _check_branch(
    spec: machine_spec.MachineSpec,
    name: str,
    state: machine_spec.BranchState,
    blackboard: dict[str, Any],
) -> BranchCheck:
    """Return the check of a branch: the first clause that fires."""
    try:
        for index, clause in enumerate(state.when):
            if clause.else_ is not None:
                fired, label, goto = True, "else", clause.goto
            else:
                assert clause.if_ is not None
                fired, label, goto = (
                    machine_predicate.evaluate(
                        machine_predicate.parse_predicate(clause.if_), blackboard
                    ),
                    clause.if_,
                    clause.goto,
                )
            if fired:
                ok = goto in spec.states
                detail = "" if ok else f"goto {goto!r} is not a state"
                return BranchCheck(name, index, label, goto, ok, detail)
    except (machine_predicate.PredicateError, template.TemplateError) as exc:
        return BranchCheck(name, None, None, None, False, f"predicate error: {exc}")
    # validate_semantics guarantees a final else, so this is unreachable.
    return BranchCheck(name, None, None, None, False, "no clause matched")  # pragma: no cover


def dry_run(
    spec: machine_spec.MachineSpec, blackboard_fixture: dict[str, Any] | None = None
) -> DryRunReport:
    """Run the per-state and per-branch passes over a machine.

    Args:
        spec: The machine.
        blackboard_fixture: Values overlaid on the declared defaults (`--blackboard`).

    Returns:
        The report.
    """
    base = engine.initial_blackboard(spec)
    # A record var defaults to {}, which a branch reading `verdict.field` cannot evaluate against,
    # so it takes the schema's zero record (optional fields absent; `has()` is their guard).
    for name, var in (*spec.vars.code.items(), *spec.vars.agent.items()):
        if var.type in spec.schemas and base.get(name) == {}:
            base[name] = synthesize_record(spec, var.type)
    if blackboard_fixture:
        base.update(blackboard_fixture)
    states: list[StateCheck] = []
    branches: list[BranchCheck] = []
    for name, state in spec.states.items():
        if isinstance(state, machine_spec.BranchState):
            branches.append(_check_branch(spec, name, state, dict(base)))
        else:
            states.append(_check_state(spec, name, state, dict(base)))
    return DryRunReport(states=tuple(states), branches=tuple(branches))
