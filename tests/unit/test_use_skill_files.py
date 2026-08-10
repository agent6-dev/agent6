# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""use_skill serves supplementary files through one walked descriptor: the
containment check and the read are the same lookup, and no path component may
be a symlink -- a skill shipping `reference.md -> secrets.toml` serves a
refusal, not the operator's keys, even when the link's target sits inside the
skill directory (a check-then-open pair would race)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.skills import ResolvedSkills, discover_skills, resolve_states
from agent6.tools._memory_tools import use_skill  # pyright: ignore[reportPrivateUsage]
from agent6.tools.errors import ToolError


def _resolver(tmp_path: Path) -> tuple[Callable[[], ResolvedSkills], Path]:
    d = tmp_path / "skills" / "helper"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: helper\ndescription: Use when testing helper.\n---\n\nBODY\n",
        encoding="utf-8",
    )
    found, warns = discover_skills([tmp_path / "skills"])
    assert not warns
    resolved = resolve_states(found, {})
    return (lambda: resolved), d


def test_serves_a_real_file_and_names_a_missing_one(tmp_path: Path) -> None:
    resolver, d = _resolver(tmp_path)
    sub = d / "notes"
    sub.mkdir()
    (sub / "tips.md").write_text("the tips\n", encoding="utf-8")
    out = use_skill(resolver, {"name": "helper", "file": "notes/tips.md"})
    assert out.content == "the tips\n"
    with pytest.raises(ToolError, match="no such file"):
        use_skill(resolver, {"name": "helper", "file": "notes/absent.md"})
    with pytest.raises(ToolError, match="no such file"):
        use_skill(resolver, {"name": "helper", "file": "notes"})  # a directory


def test_any_symlink_component_is_refused(tmp_path: Path) -> None:
    resolver, d = _resolver(tmp_path)
    secret = tmp_path / "secrets.toml"
    secret.write_text('api_key = "sk-OPERATOR"\n', encoding="utf-8")
    (d / "leak.md").symlink_to(secret)
    (d / "inside.md").symlink_to(d / "SKILL.md")  # target inside: still a link
    for name in ("leak.md", "inside.md"):
        with pytest.raises(ToolError, match="escapes the skill directory"):
            use_skill(resolver, {"name": "helper", "file": name})


def test_traversal_and_absolute_paths_are_refused(tmp_path: Path) -> None:
    resolver, _d = _resolver(tmp_path)
    for path in ("../secrets.toml", "/etc/hostname"):
        with pytest.raises(ToolError, match="escapes the skill directory"):
            use_skill(resolver, {"name": "helper", "file": path})


def test_the_size_cap_holds(tmp_path: Path) -> None:
    resolver, d = _resolver(tmp_path)
    (d / "big.md").write_text("x" * 262_145, encoding="utf-8")
    with pytest.raises(ToolError, match="256 KiB cap"):
        use_skill(resolver, {"name": "helper", "file": "big.md"})
