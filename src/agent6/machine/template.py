# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `{{ ... }}` interpolation grammar, shared by the validator and the engine.

An interpolation is one reference plus at most one filter (`len` or `json`); anything richer
belongs in a `branch` predicate. Rendering navigates the blackboard as dict lookups, never
attribute access, as the predicate evaluator does.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping

from agent6.machine import predicate

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


@dataclasses.dataclass(frozen=True, slots=True)
class Interp:
    """One interpolation: a reference and its filter, if any."""

    ref: predicate.Reference
    filt: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class Template:
    """A parsed template: literal text and interpolations in order."""

    parts: tuple[str | Interp, ...]

    @property
    def is_lone_ref(self) -> bool:
        """The template is exactly one filter-less interpolation."""
        return (
            len(self.parts) == 1
            and isinstance(self.parts[0], Interp)
            and self.parts[0].filt is None
        )


def parse_template(text: str) -> Template:
    """Parse a template string.

    Args:
        text: The template.

    Returns:
        The parsed template.

    Raises:
        TemplateError: Unbalanced braces, an unknown filter or a malformed reference.
    """
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
    """Parse the inside of one `{{ ... }}`.

    Args:
        body: The text between the braces.
        whole: The whole template, named in the error.

    Returns:
        The interpolation.

    Raises:
        TemplateError: An unknown filter, more than one filter, or a malformed reference.
    """
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
        raise TemplateError(
            f"{ref_text!r} is not a valid reference in {whole!r}: an interpolation holds"
            " one dotted name (a variable, or result.<field>) plus at most one filter"
            f" ({', '.join(sorted(FILTERS))}); no operators or literals (branch on the"
            " captured value instead)"
        )
    segments = ref_text.split(".")
    return Interp(ref=predicate.Reference(root=segments[0], path=tuple(segments[1:])), filt=filt)


class TemplateRuntimeError(TemplateError):
    """A validated template cannot be rendered against the blackboard's actual data."""


def resolve_reference(ref: predicate.Reference, scope: Mapping[str, object]) -> object:
    """Resolve a reference against a scope by dict navigation, never `getattr`.

    Args:
        ref: The reference.
        scope: The blackboard.

    Returns:
        The value.

    Raises:
        TemplateRuntimeError: The root is unknown, a segment navigates into a non-record,
            or a field is missing.
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
    """Apply the filter: `len`, or `json` (compact, keys sorted).

    Returns:
        The filtered value, or the value itself with no filter.

    Raises:
        TemplateRuntimeError: `len` of a value with no length.
    """
    if filt is None:
        return value
    if filt == "len":
        if not isinstance(value, (str, list, tuple, dict)):
            raise TemplateRuntimeError(f"{where}: `| len` has no length for {value!r}")
        return len(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _scalar_str(value: object, where: str) -> str:
    """Render a scalar as text: TOML booleans, "" for None.

    Returns:
        The text.

    Raises:
        TemplateRuntimeError: The value is not a scalar.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TemplateRuntimeError(f"{where}: value is not a scalar ({value!r})")


def render_string(template: Template, scope: Mapping[str, object], *, where: str) -> str:
    """Render a template to a string; every interpolation becomes text.

    Args:
        template: The parsed template.
        scope: The blackboard.
        where: The location named in an error.

    Returns:
        The rendered text.
    """
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
    """Render a template to a native value when it is a lone reference, else to a string.

    A lone filter-less reference is the only way a non-string value reaches the blackboard.

    Args:
        template: The parsed template.
        scope: The blackboard.
        where: The location named in an error.

    Returns:
        The referenced value, or the rendered text.
    """
    if template.is_lone_ref:
        interp = template.parts[0]
        assert isinstance(interp, Interp)
        return resolve_reference(interp.ref, scope)
    return render_string(template, scope, where=where)


def render_command(
    command: tuple[str, ...], scope: Mapping[str, object], *, where: str
) -> list[str]:
    """Render a tool state's argv, splicing a lone list reference into one argument per item.

    Args:
        command: The declared argv templates.
        scope: The blackboard.
        where: The location named in an error.

    Returns:
        The argv.
    """
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
