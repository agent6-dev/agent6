# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""use_skill serves supplementary files through one walked descriptor, with no symlink components.

The containment check and the read are the same lookup, so `reference.md -> secrets.toml`
serves a refusal even when the target sits inside the skill directory.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from agent6 import skills
from agent6.tools import (
    _skill_tools,  # pyright: ignore[reportPrivateUsage]
    errors,
)


def _resolver(tmp_path: pathlib.Path) -> tuple[Callable[[], skills.ResolvedSkills], pathlib.Path]:
    d = tmp_path / "skills" / "helper"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: helper\ndescription: Use when testing helper.\n---\n\nBODY\n",
        encoding="utf-8",
    )
    found, warns = skills.discover_skills([tmp_path / "skills"])
    assert not warns
    resolved = skills.resolve_states(found, {})
    return (lambda: resolved), d


def test_serves_a_real_file_and_names_a_missing_one(tmp_path: pathlib.Path) -> None:
    resolver, d = _resolver(tmp_path)
    sub = d / "notes"
    sub.mkdir()
    (sub / "tips.md").write_text("the tips\n", encoding="utf-8")
    out = _skill_tools.use_skill(resolver, {"name": "helper", "file": "notes/tips.md"})
    assert out.content == "the tips\n"
    with pytest.raises(errors.ToolError, match="no such file"):
        _skill_tools.use_skill(resolver, {"name": "helper", "file": "notes/absent.md"})
    with pytest.raises(errors.ToolError, match="no such file"):
        _skill_tools.use_skill(resolver, {"name": "helper", "file": "notes"})  # a directory


def test_any_symlink_component_is_refused(tmp_path: pathlib.Path) -> None:
    resolver, d = _resolver(tmp_path)
    secret = tmp_path / "secrets.toml"
    secret.write_text('api_key = "sk-OPERATOR"\n', encoding="utf-8")
    (d / "leak.md").symlink_to(secret)
    (d / "inside.md").symlink_to(d / "SKILL.md")  # target inside: still a link
    for name in ("leak.md", "inside.md"):
        with pytest.raises(errors.ToolError, match="escapes the skill directory"):
            _skill_tools.use_skill(resolver, {"name": "helper", "file": name})


def test_traversal_and_absolute_paths_are_refused(tmp_path: pathlib.Path) -> None:
    resolver, _d = _resolver(tmp_path)
    for path in ("../secrets.toml", "/etc/hostname"):
        with pytest.raises(errors.ToolError, match="escapes the skill directory"):
            _skill_tools.use_skill(resolver, {"name": "helper", "file": path})


def test_the_size_cap_holds(tmp_path: pathlib.Path) -> None:
    resolver, d = _resolver(tmp_path)
    (d / "big.md").write_text("x" * 262_145, encoding="utf-8")
    with pytest.raises(errors.ToolError, match="256 KiB cap"):
        _skill_tools.use_skill(resolver, {"name": "helper", "file": "big.md"})
