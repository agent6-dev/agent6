# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`{{ ... }}` interpolation: the single source of truth for both the
author-time validator (`agent6.machine.model`) and the runtime engine
(`agent6.machine.engine`).

The grammar is intentionally tiny (§4.4): an interpolation is exactly one
reference (§4.5) plus an optional single zero-argument filter (`len` or
`json`). There are no arbitrary expressions, no chained filters, and no
method calls — anything richer belongs in a `branch` predicate, which is
itself restricted (`agent6.machine.predicate`).

Parsing is pure and dependency-free; rendering navigates a *blackboard*
mapping as ordered dict lookups (never Python attribute access), mirroring
the predicate evaluator's data-navigation rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from agent6.machine.predicate import Reference

__all__ = [
    "FILTERS",
    "Interp",
    "Template",
    "TemplateError",
    "TemplateRuntimeError",
    "parse_template",
    "render_command",
    "render_string",
    "render_value",
    "resolve_reference",
]

FILTERS = frozenset({"len", "json"})

_REF_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_INTERP_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


class TemplateError(Exception):
    """Raised when a template string is malformed (a load-time error)."""


@dataclass(frozen=True, slots=True)
class Interp:
    ref: Reference
    filt: str | None


@dataclass(frozen=True, slots=True)
class Template:
    parts: tuple[str | Interp, ...]

    @property
    def is_lone_ref(self) -> bool:
        """True iff the template is exactly one filter-less interpolation."""
        return (
            len(self.parts) == 1
            and isinstance(self.parts[0], Interp)
            and self.parts[0].filt is None
        )


def parse_template(text: str) -> Template:
    parts: list[str | Interp] = []
    last = 0
    for match in _INTERP_RE.finditer(text):
        literal = text[last : match.start()]
        if "{{" in literal or "}}" in literal:
            raise TemplateError(f"unbalanced interpolation braces in {text!r}")
        if literal:
            parts.append(literal)
        parts.append(_parse_interp(match.group(1), text))
        last = match.end()
    tail = text[last:]
    if "{{" in tail or "}}" in tail:
        raise TemplateError(f"unbalanced interpolation braces in {text!r}")
    if tail:
        parts.append(tail)
    return Template(parts=tuple(parts))


def _parse_interp(body: str, whole: str) -> Interp:
    pieces = [piece.strip() for piece in body.split("|")]
    if len(pieces) == 1:
        ref_text, filt = pieces[0], None
    elif len(pieces) == 2:
        ref_text, filt = pieces[0], pieces[1]
        if filt not in FILTERS:
            raise TemplateError(f"unknown filter {filt!r} in {whole!r} (only {sorted(FILTERS)})")
    else:
        raise TemplateError(f"at most one filter is allowed per interpolation in {whole!r}")
    if not _REF_RE.match(ref_text):
        raise TemplateError(f"{ref_text!r} is not a valid reference in {whole!r}")
    segments = ref_text.split(".")
    return Interp(ref=Reference(root=segments[0], path=tuple(segments[1:])), filt=filt)


# ---------------------------------------------------------------------------
# Runtime rendering (engine side). The author-time validator in
# `agent6.machine.model` has already proven every reference resolves and
# every filter applies, so the failures here only fire on genuinely
# malformed blackboard data — which we surface loudly via TemplateError.
# ---------------------------------------------------------------------------


class TemplateRuntimeError(TemplateError):
    """Raised when a validated template cannot be rendered against actual data."""


def resolve_reference(ref: Reference, scope: Mapping[str, object]) -> object:
    """Resolve *ref* against *scope* as ordered dict navigation.

    Never uses ``getattr``: a record value is a ``Mapping`` and each path
    segment is a key lookup, exactly like the predicate evaluator.
    """
    if ref.root not in scope:
        raise TemplateRuntimeError(f"unknown reference {ref.root!r}")
    current: object = scope[ref.root]
    for key in ref.path:
        if not isinstance(current, Mapping):
            raise TemplateRuntimeError(f"cannot navigate into non-record value at {ref.dotted!r}")
        if key not in current:
            raise TemplateRuntimeError(f"record has no field {key!r} (in {ref.dotted!r})")
        current = current[key]
    return current


def _apply_filter(value: object, filt: str | None, where: str) -> object:
    if filt is None:
        return value
    if filt == "len":
        if not isinstance(value, (str, list, tuple, dict)):
            raise TemplateRuntimeError(f"{where}: `| len` has no length for {value!r}")
        return len(value)
    # filt == "json": compact, object keys sorted (deterministic).
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _scalar_str(value: object, where: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TemplateRuntimeError(f"{where}: value is not a scalar ({value!r})")


def render_string(template: Template, scope: Mapping[str, object], *, where: str) -> str:
    """Render *template* to a string. Every interpolation becomes text."""
    out: list[str] = []
    for part in template.parts:
        if isinstance(part, str):
            out.append(part)
            continue
        value = _apply_filter(resolve_reference(part.ref, scope), part.filt, where)
        if part.filt is None:
            out.append(_scalar_str(value, where))
        else:
            out.append(str(value))
    return "".join(out)


def render_value(template: Template, scope: Mapping[str, object], *, where: str) -> object:
    """Render *template* to a native value when it is a lone filter-less
    reference (the only way a non-string value reaches the blackboard);
    otherwise render to a string (§4.5)."""
    if template.is_lone_ref:
        interp = template.parts[0]
        assert isinstance(interp, Interp)
        return resolve_reference(interp.ref, scope)
    return render_string(template, scope, where=where)


def render_command(
    command: tuple[str, ...], scope: Mapping[str, object], *, where: str
) -> list[str]:
    """Render a `tool` state's argv, splicing a lone ``"{{ listvar }}"``
    element into one argument per list item (§4.4)."""
    argv: list[str] = []
    for index, element in enumerate(command):
        loc = f"{where}[{index}]"
        template = parse_template(element)
        if template.is_lone_ref:
            interp = template.parts[0]
            assert isinstance(interp, Interp)
            value = resolve_reference(interp.ref, scope)
            if isinstance(value, (list, tuple)):
                argv.extend(_scalar_str(item, loc) for item in value)
                continue
        argv.append(render_string(template, scope, where=loc))
    return argv
