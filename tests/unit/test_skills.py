# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.skills (SKILL.md discovery, frontmatter parsing, states)."""

from __future__ import annotations

from pathlib import Path

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
        _write_skill(tmp_path, "tidy", PLAIN)
        (tmp_path / "tidy" / ".origin.toml").write_text("url='x'\n")
        (tmp_path / "README.md").write_text("not a skill\n")
        (tmp_path / ".hidden").mkdir()
        found, warnings = skills.discover_skills([tmp_path])
        assert [s.name for s in found] == ["tidy"]
        assert warnings == ()

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
