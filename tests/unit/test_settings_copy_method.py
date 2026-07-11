# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The copy_method UI preference persists in ui.toml alongside the theme."""

from __future__ import annotations

from agent6.ui.tui import settings


def test_copy_method_defaults_to_auto() -> None:
    assert settings.get_copy_method() == "auto"


def test_copy_method_roundtrips() -> None:
    settings.save_copy_method("tmux-buffer")
    assert settings.get_copy_method() == "tmux-buffer"


def test_copy_method_coexists_with_theme() -> None:
    settings.save_theme("agent6-light")
    settings.save_copy_method("osc52")
    assert settings.get_theme() == "agent6-light"
    assert settings.get_copy_method() == "osc52"
