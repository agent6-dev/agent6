# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skill content lookup (use_skill).

A skill directory and its SKILL.md may be links (an operator's dotfiles tree); a file inside the
directory may not. Every read resolves the path first, refuses one that lands under the
workspace's hidden paths, and opens the resolved path through the component-walked descriptor
the workspace tools use.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from typing import Any

from agent6 import skills
from agent6.tools import _path_safety, errors, results, schema


def use_skill(
    ws: _path_safety.Workspace,
    resolve_skills: Callable[[], skills.ResolvedSkills],
    raw: dict[str, Any],
) -> results.SkillResult:
    """Return a skill's instructions, or one of its files, for the model's context.

    Args:
        ws: The run's file boundary; its hidden paths are refused here too.
        resolve_skills: Loads the enabled skills; called only after the arguments validate.
        raw: The tool call's arguments.

    Returns:
        The skill's SKILL.md text, or the named file's content, as the file reads now.

    Raises:
        ToolError: The skill is unknown or disabled, the file is missing, escapes the skill
            directory, lands under a hidden path, is not a regular file, cannot be read, or
            exceeds 256 KiB.
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
    name = args.file or "SKILL.md"
    try:
        if args.file is None:
            real = (skill.dir / name).resolve()
        else:
            real = _path_safety.contain(skill.dir.resolve(), args.file).abs_path
    except errors.ToolError as exc:  # absolute or `..`
        raise errors.ToolError(f"{name!r} escapes the skill directory") from exc
    if ws.is_denied(real):
        raise errors.ToolError(
            f"Path is hidden from this run: {name!r} of skill {skill.name!r} is under a path in"
            " sandbox.hide_paths or one of agent6's private directories"
        )
    # The check and the read are one lookup: the resolved path is walked from the filesystem
    # root with no symlink hop, so `reference.md -> secrets.toml` serves a refusal.
    anchor = pathlib.Path(real.anchor)
    try:
        fd = _path_safety.open_contained(
            _path_safety.contain(anchor, real.relative_to(anchor)), os.O_RDONLY
        )
    except FileNotFoundError:
        raise errors.ToolError(f"no such file in skill {skill.name!r}: {name!r}") from None
    except _path_safety.NotRegularFileError:
        # A directory or a FIFO a hostile skill dir planted: refused by the open itself.
        raise errors.ToolError(f"no such file in skill {skill.name!r}: {name!r}") from None
    except errors.ToolError as exc:  # a symlink component
        raise errors.ToolError(f"{name!r} escapes the skill directory") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(262_145)
    except OSError as exc:
        raise errors.ToolError(f"cannot read {name!r}: {exc}") from exc
    if len(data) > 262_144:
        raise errors.ToolError(f"{name!r} exceeds the 256 KiB cap")
    return results.SkillResult(
        skill=skill.name, file=name, content=data.decode("utf-8", errors="replace")
    )
