# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skill content lookup (use_skill).

Reads stay inside the skill's own directory, through the same component-walked descriptor the
workspace tools use: no hop may be a symlink.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from agent6 import skills
from agent6.tools import _path_safety, errors, results, schema


def use_skill(
    resolve_skills: Callable[[], skills.ResolvedSkills], raw: dict[str, Any]
) -> results.SkillResult:
    """Return a skill's instructions, or one of its files, for the model's context.

    Args:
        resolve_skills: Loads the enabled skills; called only after the arguments validate.
        raw: The tool call's arguments.

    Returns:
        The skill's SKILL.md text, or the named file's content.

    Raises:
        ToolError: The skill is unknown or disabled, the file is missing, escapes the skill
            directory, is not a regular file, cannot be read, or exceeds 256 KiB.
    """
    args = schema.UseSkillInput.model_validate(raw)
    resolved = resolve_skills()
    by_name = {s.name: s for s in (*resolved.enabled, *resolved.always)}
    skill = by_name.get(args.name)
    if skill is None:
        raise errors.ToolError(
            f"unknown or disabled skill {args.name!r};"
            f" available: {', '.join(sorted(by_name)) or '(none)'}"
        )
    if args.file is None:
        return results.SkillResult(skill=skill.name, file="SKILL.md", content=skill.text)
    # Containment and the read are one lookup: `reference.md -> secrets.toml` serves a refusal.
    try:
        fd = _path_safety.open_contained(_path_safety.contain(skill.dir, args.file), os.O_RDONLY)
    except FileNotFoundError:
        raise errors.ToolError(f"no such file in skill {skill.name!r}: {args.file!r}") from None
    except _path_safety.NotRegularFileError:
        # A directory or a FIFO a hostile skill dir planted: refused by the open itself.
        raise errors.ToolError(f"no such file in skill {skill.name!r}: {args.file!r}") from None
    except errors.ToolError as exc:  # absolute, `..`, or a symlink component
        raise errors.ToolError(f"{args.file!r} escapes the skill directory") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(262_145)
    except OSError as exc:
        raise errors.ToolError(f"cannot read {args.file!r}: {exc}") from exc
    if len(data) > 262_144:
        raise errors.ToolError(f"{args.file!r} exceeds the 256 KiB cap")
    return results.SkillResult(
        skill=skill.name, file=args.file, content=data.decode("utf-8", errors="replace")
    )
