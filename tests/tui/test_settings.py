# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI-only preferences store (ui.toml), separate from the agent config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.tui.settings import DEFAULT_THEME, get_theme, load_ui_settings, save_theme


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_theme_roundtrips_in_own_file(cfg: Path) -> None:
    assert get_theme() == DEFAULT_THEME  # default when ui.toml is absent
    save_theme("nord")
    assert get_theme() == "nord"
    # Lands in its own file (a sibling of config.toml), not the agent config.
    ui = cfg / "ui.toml"
    assert ui.is_file()
    assert "theme" in ui.read_text(encoding="utf-8")
    assert not (cfg / "config.toml").exists()


def test_corrupt_or_missing_file_degrades_to_default(cfg: Path) -> None:
    (cfg / "ui.toml").write_text("this is [ not valid toml", encoding="utf-8")
    assert load_ui_settings() == {}  # never raises
    assert get_theme() == DEFAULT_THEME


def test_save_preserves_other_keys(cfg: Path) -> None:
    (cfg / "ui.toml").write_text('[ui]\ntheme = "nord"\nshow_x = true\n', encoding="utf-8")
    save_theme("dracula")
    data = load_ui_settings()["ui"]
    assert data["theme"] == "dracula"
    assert data["show_x"] is True  # unrelated keys survive the rewrite
