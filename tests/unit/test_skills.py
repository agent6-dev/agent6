# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.skills (SKILL.md discovery, frontmatter parsing, states)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent6 import skills


def _write_skill(root: Path, dirname: str, text: str) -> Path:
    d = root / dirname
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d


PLAIN = """---
name: tidy
description: Use when output should be terse.
---

Body text here.
"""

FOLDED = """---
name: caveman
description: >
  Ultra-compressed communication mode. Cuts output tokens
  while keeping full technical accuracy.
extra_key: ignored-value
---

Respond terse like smart caveman.
"""


class TestParseFrontmatter:
    def test_plain_fields(self) -> None:
        fields, warnings = skills.parse_frontmatter(PLAIN)
        assert fields["name"] == "tidy"
        assert fields["description"] == "Use when output should be terse."
        assert warnings == []

    def test_folded_description_joins_lines(self) -> None:
        fields, warnings = skills.parse_frontmatter(FOLDED)
        assert fields["name"] == "caveman"
        assert (
            fields["description"] == "Ultra-compressed communication mode. Cuts output tokens "
            "while keeping full technical accuracy."
        )
        # unknown keys are surfaced, not errors (ecosystem skills carry extras)
        assert fields["extra_key"] == "ignored-value"
        assert warnings == []

    def test_quoted_value(self) -> None:
        text = '---\nname: x\ndescription: "Use when, quoted."\n---\n'
        fields, _ = skills.parse_frontmatter(text)
        assert fields["description"] == "Use when, quoted."

    def test_literal_block(self) -> None:
        text = "---\nname: x\ndescription: |\n  line one\n  line two\n---\n"
        fields, _ = skills.parse_frontmatter(text)
        assert fields["description"] == "line one\nline two"

    def test_indented_separator_inside_literal_description_is_content(self) -> None:
        text = (
            "---\nname: x\ndescription: |\n"
            "  Use when prose contains a divider.\n  ---\n  Keep this line too.\n---\nbody\n"
        )
        fields, warnings = skills.parse_frontmatter(text)
        assert fields["description"] == (
            "Use when prose contains a divider.\n---\nKeep this line too."
        )
        assert warnings == []

    def test_missing_frontmatter_warns(self) -> None:
        fields, warnings = skills.parse_frontmatter("no frontmatter at all\n")
        assert fields == {}
        assert warnings

    def test_unclosed_frontmatter_warns(self) -> None:
        fields, warnings = skills.parse_frontmatter("---\nname: x\n")
        assert fields == {}
        assert warnings


class TestDiscoverSkills:
    def test_discovers_and_reads(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "tidy", PLAIN)
        found, warnings = skills.discover_skills([tmp_path])
        assert warnings == ()
        assert [s.name for s in found] == ["tidy"]
        assert found[0].description == "Use when output should be terse."
        assert found[0].dir == tmp_path / "tidy"
        assert "Body text here." in found[0].text

    def test_frontmatter_name_wins_over_dirname(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "wrong-dir-name", PLAIN)
        found, warnings = skills.discover_skills([tmp_path])
        assert [s.name for s in found] == ["tidy"]
        assert any("wrong-dir-name" in w for w in warnings)

    def test_missing_required_fields_skips_with_warning(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "broken", "---\nname: broken\n---\nno description\n")
        found, warnings = skills.discover_skills([tmp_path])
        assert found == ()
        assert any("broken" in w for w in warnings)

    def test_invalid_name_skips(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "bad", "---\nname: bad name!\ndescription: Use when.\n---\n")
        found, warnings = skills.discover_skills([tmp_path])
        assert found == ()
        assert any("bad name!" in w for w in warnings)

    def test_first_dir_wins_duplicate_names(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        _write_skill(a, "tidy", PLAIN)
        _write_skill(b, "tidy", PLAIN.replace("terse", "verbose"))
        found, warnings = skills.discover_skills([a, b])
        assert len(found) == 1
        assert "terse" in found[0].description  # from dir a
        assert any("duplicate" in w for w in warnings)

    def test_single_skill_dir_direct(self, tmp_path: Path) -> None:
        # extra_dirs may point AT one skill dir (SKILL.md directly inside)
        d = _write_skill(tmp_path, "tidy", PLAIN)
        found, warnings = skills.discover_skills([d])
        assert [s.name for s in found] == ["tidy"]
        assert warnings == ()

    def test_dotfiles_and_plain_files_ignored(self, tmp_path: Path) -> None:
        # A dotted dir holding a VALID SKILL.md is the case the dot rule exists
        # for: without it a .backup/ copy would be discovered and its
        # instructions injected into every run's system prompt. An empty
        # .hidden/ proves nothing (the SKILL.md check already excludes it).
        _write_skill(tmp_path, "tidy", PLAIN)
        (tmp_path / "tidy" / ".origin.toml").write_text("url='x'\n")
        (tmp_path / "README.md").write_text("not a skill\n")
        _write_skill(tmp_path, ".backup", PLAIN)
        found, warnings = skills.discover_skills([tmp_path])
        assert [s.name for s in found] == ["tidy"]
        assert warnings == ()  # skipped silently, not reported as broken

    def test_missing_dir_is_fine(self, tmp_path: Path) -> None:
        found, warnings = skills.discover_skills([tmp_path / "nope"])
        assert found == ()
        assert warnings == ()


class TestResolveStates:
    def _skills(self, tmp_path: Path) -> tuple[skills.Skill, ...]:
        _write_skill(tmp_path, "a", PLAIN.replace("tidy", "a"))
        _write_skill(tmp_path, "b", PLAIN.replace("tidy", "b"))
        _write_skill(tmp_path, "c", PLAIN.replace("tidy", "c"))
        found, _ = skills.discover_skills([tmp_path])
        return found

    def test_default_all_enabled(self, tmp_path: Path) -> None:
        r = skills.resolve_states(self._skills(tmp_path), {})
        assert [s.name for s in r.enabled] == ["a", "b", "c"]
        assert r.always == ()
        assert r.warnings == ()

    def test_disabled_dropped_always_promoted(self, tmp_path: Path) -> None:
        r = skills.resolve_states(
            self._skills(tmp_path), {"a": "disabled", "b": "always", "c": "enabled"}
        )
        assert [s.name for s in r.enabled] == ["c"]
        assert [s.name for s in r.always] == ["b"]

    def test_unknown_name_warns(self, tmp_path: Path) -> None:
        r = skills.resolve_states(self._skills(tmp_path), {"ghost": "disabled"})
        assert any("ghost" in w for w in r.warnings)


def test_unreadable_skill_dir_warns_instead_of_crashing_every_run(tmp_path: Path) -> None:
    """A dir listed in extra_dirs (or the installed dir) that exists but cannot
    be listed (permission denied) crashed discovery via a bare `iterdir()`,
    same failure class as an unreadable SKILL.md: every run dies before a
    healthy sibling skill ever loads. It must degrade to a warning instead."""
    if os.geteuid() == 0:
        pytest.skip("root lists through a 000 mode")
    base = tmp_path / "extra"
    base.mkdir()
    base.chmod(0o000)
    good = tmp_path / "healthy"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: healthy\ndescription: works\n---\nbody\n", encoding="utf-8"
    )
    try:
        found, warnings = skills.discover_skills([base, good])
    finally:
        base.chmod(0o700)
    assert [s.name for s in found] == ["healthy"]
    assert any("extra" in w for w in warnings)
    # A dir whose PARENT denies search fails one step earlier, at the stat
    # that decides whether it is a dir: the same class, and it sat one line
    # outside the guard (and `Path.is_dir()` hides it on Python 3.13+).
    outer = tmp_path / "outer"
    inner = outer / "myskill"
    inner.mkdir(parents=True)
    outer.chmod(0o000)
    try:
        found, warnings = skills.discover_skills([inner, good])
    finally:
        outer.chmod(0o700)
    assert [s.name for s in found] == ["healthy"]
    assert any("myskill" in w for w in warnings)


def test_unreadable_skill_warns_instead_of_crashing_every_run(tmp_path: Path) -> None:
    """A SKILL.md with one non-UTF-8 byte (or an unreadable file) crashed
    discovery, and discovery runs at startup: every `agent6 run` then died with
    a bare UnicodeDecodeError naming no file, after session.start and before any
    session.end. A bad skill must degrade to a warning like every other malformed
    one, leaving the healthy skills usable."""
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "SKILL.md").write_bytes(b"---\nname: broken\ndescription: caf\xe9\n---\nbody\n")
    good = tmp_path / "healthy"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: healthy\ndescription: works\n---\nbody\n", encoding="utf-8"
    )

    found, warnings = skills.discover_skills([tmp_path])
    assert [s.name for s in found] == ["healthy"]
    assert any("broken" in w for w in warnings)


@pytest.mark.skipif(sys.version_info < (3, 13), reason="3.12's Path.is_dir raises here itself")
def test_a_skill_dir_the_operator_cannot_search_warns_instead_of_vanishing(
    tmp_path: Path,
) -> None:
    """A candidate under an extra dir with mode 0600 (readable, not
    searchable): `Path.is_dir()` and `is_file()` report it absent from Python
    3.13 on, so discovery dropped it with no warning; the explicit stats raise
    into the same warning as an unlistable dir."""
    if os.geteuid() == 0:
        pytest.skip("root searches through a 0600 dir")
    base = tmp_path / "extra"
    inner = base / "myskill"
    inner.mkdir(parents=True)
    (inner / "SKILL.md").write_text(
        "---\nname: myskill\ndescription: x\n---\nbody\n", encoding="utf-8"
    )
    inner.chmod(0o600)
    try:
        found, warnings = skills.discover_skills([base])
    finally:
        inner.chmod(0o700)
    assert not found
    assert any("myskill" in w for w in warnings)
