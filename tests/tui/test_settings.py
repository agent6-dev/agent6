# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI-only preferences store (ui.toml), separate from the agent config."""

from __future__ import annotations

import pathlib

import pytest

from agent6.ui.tui import settings


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    home = tmp_path / "agent6"
    home.mkdir()
    return home


def test_theme_roundtrips_in_own_file(cfg: pathlib.Path) -> None:
    assert settings.get_theme() == settings.DEFAULT_THEME  # default when ui.toml is absent
    settings.save_theme("nord")
    assert settings.get_theme() == "nord"
    # Lands in its own file (a sibling of config.toml), not the agent config.
    ui = cfg / "ui.toml"
    assert ui.is_file()
    assert "theme" in ui.read_text(encoding="utf-8")
    assert not (cfg / "config.toml").exists()


def test_corrupt_or_missing_file_degrades_to_default(cfg: pathlib.Path) -> None:
    (cfg / "ui.toml").write_text("this is [ not valid toml", encoding="utf-8")
    assert settings.load_ui_settings() == {}  # never raises
    assert settings.get_theme() == settings.DEFAULT_THEME


def test_save_preserves_other_keys(cfg: pathlib.Path) -> None:
    (cfg / "ui.toml").write_text('[ui]\ntheme = "nord"\nshow_x = true\n', encoding="utf-8")
    settings.save_theme("dracula")
    data = settings.load_ui_settings()["ui"]
    assert data["theme"] == "dracula"
    assert data["show_x"] is True  # unrelated keys survive the rewrite


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX symlinks")
def test_save_does_not_follow_a_planted_tmp_symlink(cfg: pathlib.Path) -> None:
    """A planted `ui.toml.tmp` symlink is ignored: the save writes through an unpredictable temp.

    The save chowns back to the real user, so under sudo a followed symlink was a truncate-as-root.
    """
    secret = cfg / "root_secret"
    secret.write_text("do-not-truncate", encoding="utf-8")
    (cfg / "ui.toml.tmp").symlink_to(secret)
    settings.save_theme("nord")
    assert secret.read_text(encoding="utf-8") == "do-not-truncate"  # untouched
    assert settings.get_theme() == "nord"  # the real save still landed


def test_a_control_character_in_a_ui_value_round_trips(tmp_path: pathlib.Path) -> None:
    """The settings writer escapes a newline in a value, so the written file parses on read."""
    import tomllib

    text = settings._render_ui_toml({"ui": {"copy_method": "osc52\nrogue", "theme": "dark"}})
    assert tomllib.loads(text)["ui"]["copy_method"] == "osc52\nrogue"
