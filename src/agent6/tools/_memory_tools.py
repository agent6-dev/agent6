# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator-knowledge handlers: cross-run memory notes (add_memory /
invalidate_memory) and skill content lookup (use_skill). Both let the LLM
pull in curated context the dispatcher may not have wired for this run
(no state_dir / skills disabled), in which case the handler raises ToolError.

Memory writes go through trusted code (agent6.memory) to fixed markdown files
under <state_dir>/memories/, outside the workspace and the jail; the LLM
controls only the scope (schema-validated literal) and the note text, which
is inert data. Skill file reads stay inside the skill's own directory
(``resolve()`` collapses ``../``/symlinks before the containment check)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent6.memory import MemoryStoreError
from agent6.memory import add as memory_add
from agent6.memory import invalidate as memory_invalidate
from agent6.skills import ResolvedSkills
from agent6.tools.errors import ToolError
from agent6.tools.results import AddMemoryResult, InvalidateMemoryResult, SkillResult
from agent6.tools.schema import AddMemoryInput, InvalidateMemoryInput, UseSkillInput


def add_memory(state_dir: Path | None, raw: dict[str, Any]) -> AddMemoryResult:
    if state_dir is None:
        raise ToolError("add_memory: no memory store wired for this run")
    args = AddMemoryInput.model_validate(raw)
    try:
        entry = memory_add(state_dir, args.scope, args.body)
    except MemoryStoreError as exc:
        raise ToolError(f"add_memory: {exc}") from exc
    return AddMemoryResult(id=entry.id, scope=entry.scope, created_at=entry.created_at)


def invalidate_memory(state_dir: Path | None, raw: dict[str, Any]) -> InvalidateMemoryResult:
    if state_dir is None:
        raise ToolError("invalidate_memory: no memory store wired for this run")
    args = InvalidateMemoryInput.model_validate(raw)
    try:
        entry = memory_invalidate(state_dir, args.memory_id, args.reason)
    except MemoryStoreError as exc:
        raise ToolError(f"invalidate_memory: {exc}") from exc
    return InvalidateMemoryResult(id=entry.id, invalidated_at=entry.invalidated_at)


def use_skill(resolve_skills: Callable[[], ResolvedSkills], raw: dict[str, Any]) -> SkillResult:
    args = UseSkillInput.model_validate(raw)
    # Resolve after validation, exactly where the original handler did (the
    # first-use disk scan never happens for a rejected call).
    resolved = resolve_skills()
    by_name = {s.name: s for s in (*resolved.enabled, *resolved.always)}
    skill = by_name.get(args.name)
    if skill is None:
        raise ToolError(
            f"use_skill: unknown or disabled skill {args.name!r};"
            f" available: {', '.join(sorted(by_name)) or '(none)'}"
        )
    if args.file is None:
        return SkillResult(skill=skill.name, file="SKILL.md", content=skill.text)
    # Supplementary files stay inside the skill's own directory: resolve()
    # collapses ../ and symlinks BEFORE the containment check, so neither
    # traversal nor a symlink pointing out of the directory can escape.
    base = skill.dir.resolve()
    target = (base / args.file).resolve()
    if not target.is_relative_to(base):
        raise ToolError(f"use_skill: {args.file!r} escapes the skill directory")
    if not target.is_file():
        raise ToolError(f"use_skill: no such file in skill {skill.name!r}: {args.file!r}")
    if target.stat().st_size > 262_144:
        raise ToolError(f"use_skill: {args.file!r} exceeds the 256 KiB cap")
    content = target.read_text(encoding="utf-8", errors="replace")
    return SkillResult(skill=skill.name, file=args.file, content=content)
