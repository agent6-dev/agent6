# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Persistent agent memories under ``<state_dir>/memories/``.

Three scopes, each backed by a single markdown file:
  - facts.md       (immutable observations)
  - decisions.md   (policy / design choices the agent committed to)
  - preferences.md (user preferences and style guidance)

Entries are append-only. "Invalidation" is non-destructive: an extra
header line marks the entry as superseded but the body remains so the
audit trail stays intact.

File format — one entry per `### <id>` h3, deterministic so ripgrep
can search and we can parse without YAML:

    ## facts

    ### 01HX...
    created_at: 2026-01-01T00:00:00Z
    invalidated_at: 2026-02-01T00:00:00Z
    invalidation_reason: superseded by 01HY...

    body text, may span multiple lines and contain *markdown*.

    ### 01HY...
    created_at: 2026-02-01T00:00:00Z

    another entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from agent6.graph.ulid import new_ulid

MemoryScope = Literal["facts", "decisions", "preferences"]
_SCOPES: tuple[MemoryScope, ...] = ("facts", "decisions", "preferences")
_ID_RE = re.compile(r"^### ([0-9A-HJKMNP-TV-Z]{26})\s*$")
_KEY_RE = re.compile(r"^([a-z_]+):\s*(.+)$")


class MemoryError(Exception):
    """Memory-store operation failed."""


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    id: str
    scope: MemoryScope
    created_at: str
    body: str
    invalidated_at: str = ""
    invalidation_reason: str = ""

    @property
    def is_active(self) -> bool:
        return not self.invalidated_at


def _memories_dir(state_dir: Path) -> Path:
    return state_dir / "memories"


def _scope_path(state_dir: Path, scope: MemoryScope) -> Path:
    if scope not in _SCOPES:
        raise MemoryError(f"unknown memory scope: {scope!r} (want one of {_SCOPES})")
    return _memories_dir(state_dir) / f"{scope}.md"


def _now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_file(path: Path, scope: MemoryScope) -> list[MemoryEntry]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[MemoryEntry] = []
    current_id: str | None = None
    meta: dict[str, str] = {}
    body_lines: list[str] = []
    in_meta = True

    def _flush() -> None:
        if current_id is None:
            return
        body = "\n".join(body_lines).strip("\n")
        entries.append(
            MemoryEntry(
                id=current_id,
                scope=scope,
                created_at=meta.get("created_at", ""),
                invalidated_at=meta.get("invalidated_at", ""),
                invalidation_reason=meta.get("invalidation_reason", ""),
                body=body,
            )
        )

    for raw in lines:
        m = _ID_RE.match(raw)
        if m is not None:
            _flush()
            current_id = m.group(1)
            meta = {}
            body_lines = []
            in_meta = True
            continue
        if current_id is None:
            continue
        if in_meta:
            if raw.strip() == "":
                in_meta = False
                continue
            km = _KEY_RE.match(raw)
            if km is not None:
                meta[km.group(1)] = km.group(2)
                continue
            # not a meta line — treat the rest as body
            in_meta = False
            body_lines.append(raw)
        else:
            body_lines.append(raw)
    _flush()
    return entries


def _render_file(scope: MemoryScope, entries: list[MemoryEntry]) -> str:
    out: list[str] = [f"## {scope}", ""]
    for e in entries:
        out.append(f"### {e.id}")
        out.append(f"created_at: {e.created_at}")
        if e.invalidated_at:
            out.append(f"invalidated_at: {e.invalidated_at}")
        if e.invalidation_reason:
            out.append(f"invalidation_reason: {e.invalidation_reason}")
        out.append("")
        if e.body:
            out.append(e.body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def add(state_dir: Path, scope: MemoryScope, body: str) -> MemoryEntry:
    """Append a new entry. Returns the persisted entry (with assigned id)."""
    body = body.strip()
    if not body:
        raise MemoryError("memory body must be non-empty")
    path = _scope_path(state_dir, scope)
    entries = _parse_file(path, scope)
    entry = MemoryEntry(id=new_ulid(), scope=scope, created_at=_now(), body=body)
    entries.append(entry)
    _atomic_write(path, _render_file(scope, entries))
    return entry


def list_entries(state_dir: Path, scope: MemoryScope | None = None) -> tuple[MemoryEntry, ...]:
    scopes: tuple[MemoryScope, ...] = (scope,) if scope is not None else _SCOPES
    out: list[MemoryEntry] = []
    for s in scopes:
        out.extend(_parse_file(_scope_path(state_dir, s), s))
    return tuple(out)


def invalidate(state_dir: Path, memory_id: str, reason: str) -> MemoryEntry:
    """Mark `memory_id` invalidated. Body is preserved."""
    reason = reason.strip()
    if not reason:
        raise MemoryError("invalidation reason must be non-empty")
    for scope in _SCOPES:
        path = _scope_path(state_dir, scope)
        entries = _parse_file(path, scope)
        for i, e in enumerate(entries):
            if e.id != memory_id:
                continue
            if e.invalidated_at:
                raise MemoryError(f"memory {memory_id} already invalidated")
            updated = MemoryEntry(
                id=e.id,
                scope=scope,
                created_at=e.created_at,
                invalidated_at=_now(),
                invalidation_reason=reason,
                body=e.body,
            )
            entries[i] = updated
            _atomic_write(path, _render_file(scope, entries))
            return updated
    raise MemoryError(f"no memory with id {memory_id!r}")
