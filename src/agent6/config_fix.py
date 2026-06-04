# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check-config --fix` — interactive repair for stale agent6.toml.

Security-sensitive fields in `agent6.config.Config` have no defaults
(every `sandbox.*`, `providers.*`, `models.*`, `budget.max_*_tokens`,
`git.allow_*`, and `workflow.verify_command` must be set explicitly).
A user whose config predates a release that added a new required field
gets a `field required` error with no hint about what to set.

This module bridges the gap. It compares the user's TOML to the canonical
starter template in `agent6.init` (`_STARTER_TOML`), produces a list of
concrete `Fix` proposals (each one a section or a single key with the
recommended value), and — only on the caller's confirmation — performs
line-level edits on the file. The starter template is the single source
of truth: a unit test asserts every recommendation matches it leaf-for-leaf.

Important invariants:

* Never round-trip the user's file through a TOML serializer. We only
  append blocks or insert single lines after a section header. Existing
  content (comments, ordering, formatting) is left untouched.
* `Fix` only produces explicit text to insert; it does NOT relax
  validation. Operational fields that already have a default in
  `Config` are not surfaced as missing.
* Only `missing` errors are addressable; any other `ValidationError`
  (e.g. wrong type, unknown extra key) is returned in `remaining_errors`
  for the caller to surface.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent6.config import Config
from agent6.init import _STARTER_TOML  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "Fix",
    "FixKind",
    "ProposalResult",
    "apply_fixes",
    "format_value",
    "propose_fixes",
    "starter_recommendations",
]


class FixKind(StrEnum):
    """How a `Fix` is applied to the user's TOML."""

    NEW_SECTION = "new_section"
    """Append a fresh `[section]` block plus its key=value lines."""

    INSERT_FIELD = "insert_field"
    """Insert a single `key = value` line right after the existing section header."""


@dataclass(frozen=True, slots=True)
class Fix:
    """A single concrete repair the caller may accept or skip.

    `section` is the dotted TOML header without brackets (e.g.
    `providers.anthropic`). For `NEW_SECTION`, `lines` is the list of
    `key = value` lines to append under that header. For `INSERT_FIELD`,
    `lines` is exactly one line; `key` names the field being inserted.
    """

    kind: FixKind
    section: str
    key: str | None
    lines: tuple[str, ...]
    description: str

    def render_preview(self) -> str:
        """Return the proposed addition as it would appear in the file."""
        if self.kind is FixKind.NEW_SECTION:
            return "\n".join((f"[{self.section}]", *self.lines))
        return self.lines[0]


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Result of `propose_fixes`."""

    fixes: tuple[Fix, ...]
    remaining_errors: tuple[str, ...]
    """Validation errors that `--fix` cannot address (wrong types, unknown
    keys, cross-field violations). Caller should surface these verbatim."""


# ---------------------------------------------------------------------------
# Recommendation table
# ---------------------------------------------------------------------------


def starter_recommendations() -> dict[str, dict[str, Any]]:
    """Parse `_STARTER_TOML` and return a `{section_path: {key: value}}` map.

    `section_path` is the dotted header (e.g. `providers.anthropic`,
    `models.worker`, `budget`). Values are native Python (str, int,
    bool, list[str]). This map is the canonical source of recommended
    values used by `propose_fixes`.

    A unit test asserts every required field in `Config` has a
    corresponding entry here, so the table stays in sync with the schema.
    """
    parsed = tomllib.loads(_STARTER_TOML)
    flat: dict[str, dict[str, Any]] = {}
    _walk_sections(parsed, prefix=(), out=flat)
    return flat


def _walk_sections(
    node: dict[str, Any], *, prefix: tuple[str, ...], out: dict[str, dict[str, Any]]
) -> None:
    """Walk a parsed-TOML dict and split scalars from sub-tables.

    Scalars at the current `prefix` form one entry in `out`. Sub-tables
    recurse with an extended prefix. Empty intermediate scalar dicts are
    skipped so we don't emit blank sections.
    """
    scalars: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            _walk_sections(value, prefix=(*prefix, key), out=out)
        else:
            scalars[key] = value
    if scalars:
        out[".".join(prefix)] = scalars


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


# Sections that are dynamic dicts in the schema (the user picks the inner
# key). For these, the loc reported by pydantic looks like
# ("providers", "anthropic", "api_key_env"): the FIRST two parts form the
# TOML section header.
_DICT_SECTIONS: frozenset[str] = frozenset({"providers", "models"})


def _section_for_loc(loc: tuple[str | int, ...]) -> tuple[str, str | None]:
    """Split a pydantic loc into `(section_path, field_or_None)`.

    For top-level sections like `("budget", "max_input_tokens")` returns
    `("budget", "max_input_tokens")`. For dict-keyed sections like
    `("providers", "anthropic", "api_key_env")` returns
    `("providers.anthropic", "api_key_env")`. For a loc pointing at an
    entire missing section (e.g. `("budget",)`) returns
    `("budget", None)`.
    """
    parts = [str(p) for p in loc]
    if not parts:
        return ("", None)
    if len(parts) == 1:
        return (parts[0], None)
    if parts[0] in _DICT_SECTIONS:
        if len(parts) == 2:
            return (f"{parts[0]}.{parts[1]}", None)
        return (f"{parts[0]}.{parts[1]}", ".".join(parts[2:]))
    return (parts[0], ".".join(parts[1:]))


def _existing_section_headers(raw_text: str) -> set[str]:
    """Return the set of section paths that already appear as `[header]`
    lines in `raw_text` (commented lines excluded).
    """
    headers: set[str] = set()
    for line in raw_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("[") or stripped.startswith("[["):
            continue
        if stripped.lstrip(" \t").startswith("#"):
            continue
        end = stripped.find("]")
        if end <= 1:
            continue
        headers.add(stripped[1:end].strip())
    return headers


def propose_fixes(config_path: Path) -> ProposalResult:
    """Inspect *config_path* and return the list of repairable fixes.

    Reads the file as TOML, runs strict pydantic validation, and
    interprets every `missing` error against the starter template.
    Other error kinds pass through as `remaining_errors` strings.
    """
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        raw = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        return ProposalResult(
            fixes=(),
            remaining_errors=(f"TOML parse error: {exc}",),
        )
    validation_error: ValidationError | None = None
    try:
        Config.model_validate(raw)
    except ValidationError as exc:
        validation_error = exc
    if validation_error is None:
        return ProposalResult(fixes=(), remaining_errors=())

    recs = starter_recommendations()
    headers_in_file = _existing_section_headers(raw_text)
    missing_errors: list[tuple[str, str | None]] = []
    remaining: list[str] = []
    for issue in validation_error.errors():
        if issue["type"] == "missing":
            section, field = _section_for_loc(issue["loc"])
            missing_errors.append((section, field))
        else:
            loc = ".".join(str(p) for p in issue["loc"]) or "<root>"
            remaining.append(f"{loc}: {issue['msg']} (type={issue['type']})")

    fixes: list[Fix] = []
    seen_whole_sections: set[str] = set()
    for section, field in missing_errors:
        if section in seen_whole_sections:
            continue
        if section not in recs:
            # The error points at a section we have no recommendation for
            # (e.g. user routed a role to a provider that isn't in the
            # starter). Defer to remaining_errors so the user sees it.
            remaining.append(
                f"{section}{'.' + field if field else ''}: no recommendation available"
            )
            continue
        if section not in headers_in_file:
            # Whole section absent: emit a single NEW_SECTION fix covering
            # every recommended key for it.
            lines = tuple(f"{k} = {format_value(v)}" for k, v in recs[section].items())
            fixes.append(
                Fix(
                    kind=FixKind.NEW_SECTION,
                    section=section,
                    key=None,
                    lines=lines,
                    description=f"Add missing section [{section}] with starter defaults",
                )
            )
            seen_whole_sections.add(section)
            continue
        # Section present, single field missing.
        if field is None or field not in recs[section]:
            remaining.append(
                f"{section}{'.' + field if field else ''}: no recommendation available"
            )
            continue
        value = recs[section][field]
        fixes.append(
            Fix(
                kind=FixKind.INSERT_FIELD,
                section=section,
                key=field,
                lines=(f"{field} = {format_value(value)}",),
                description=f"Add missing field {section}.{field}",
            )
        )

    return ProposalResult(fixes=tuple(fixes), remaining_errors=tuple(remaining))


# ---------------------------------------------------------------------------
# Value formatting (small, deliberately not a general TOML emitter)
# ---------------------------------------------------------------------------


def format_value(value: Any) -> str:
    """Format a Python scalar as a TOML right-hand side.

    Handles the value shapes that appear in `_STARTER_TOML`: bool, int,
    str, and list-of-str. Anything else raises `TypeError` — the starter
    is the only source so any new shape must be wired in explicitly.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list):
        items: list[str] = []
        for item in value:  # type: ignore[reportUnknownVariableType]
            if not isinstance(item, str):
                raise TypeError(f"unsupported list element type: {type(item).__name__}")
            items.append(_toml_string(item))
        return "[" + ", ".join(items) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def _toml_string(s: str) -> str:
    """Render *s* as a TOML basic string with conservative escaping.

    The starter template only ever ships printable-ASCII values, so a
    minimal escape table is sufficient. We avoid the literal-string form
    so that the same code can quote any future value containing a single
    quote without re-deriving rules.
    """
    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Application — line-level text edits, no TOML round-trip
# ---------------------------------------------------------------------------


def apply_fixes(config_path: Path, fixes: tuple[Fix, ...] | list[Fix]) -> None:
    """Apply *fixes* to the file at *config_path* in place.

    Strategy:
      * Read the file as text, splitlines preserving content.
      * For every `INSERT_FIELD` fix, locate the line containing
        `[section]` and insert the new line immediately after it.
      * After all field-inserts (which keep line numbers monotonically
        increasing relative to the original headers — we walk fixes in
        the same pass and offset by inserts already applied), append
        `NEW_SECTION` blocks to the end of the file, each preceded by a
        blank line.
    """
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)
    trailing_newline = text.endswith("\n")

    insert_fixes = [f for f in fixes if f.kind is FixKind.INSERT_FIELD]
    new_sections = [f for f in fixes if f.kind is FixKind.NEW_SECTION]

    # Plan field inserts as (original_index, new_line) and apply in
    # reverse order so earlier indices remain valid.
    insertions: list[tuple[int, str]] = []
    for fix in insert_fixes:
        header_index = _find_header_index(lines, fix.section)
        if header_index < 0:
            # Caller should not have requested this — `propose_fixes`
            # would have emitted NEW_SECTION instead. Skip defensively.
            continue
        insertions.append((header_index + 1, fix.lines[0]))
    insertions.sort(key=lambda x: x[0], reverse=True)
    for index, line in insertions:
        lines.insert(index, line)

    # Append new sections.
    for fix in new_sections:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(f"[{fix.section}]")
        lines.extend(fix.lines)

    out = "\n".join(lines)
    if trailing_newline or new_sections or insertions:
        out += "\n"
    config_path.write_text(out, encoding="utf-8")


def _find_header_index(lines: list[str], section: str) -> int:
    """Return the index of the `[section]` header line, or -1 if absent.

    Matches a non-commented line whose first `[..]` token equals *section*.
    """
    target = f"[{section}]"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == target:
            return i
    return -1
