# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`ui.keymap.SCREEN_LETTERS` is the whole letter map, and it stays true.

A screen's own BINDINGS remain the thing textual reads; this asserts the table
matches them, so it cannot describe a keyboard we no longer ship. Reading it is
how a new shortcut is chosen with the existing ones in sight.
"""

from __future__ import annotations

from typing import Any

from agent6.ui.keymap import SCREEN_LETTERS
from agent6.ui.tui.config_page import ConfigScreen
from agent6.ui.tui.conversation import ConversationScreen
from agent6.ui.tui.dashboard import DashboardScreen
from agent6.ui.tui.home import HomeScreen
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.machines import MachineDetailScreen, MachinesScreen, MachineWatchScreen

SCREENS: dict[str, Any] = {
    "hub": HomeScreen,
    "conversation": ConversationScreen,
    "dashboard": DashboardScreen,
    "event log": LogScreen,
    "config": ConfigScreen,
    "machines": MachinesScreen,
    "machine": MachineDetailScreen,
    "machine watch": MachineWatchScreen,
}


def _letters(screen: Any) -> dict[str, str]:
    return {
        b.key: b.action
        for b in screen.BINDINGS
        if len(b.key) == 1 and b.key.isalpha()  # type: ignore[union-attr]
    }


def test_the_table_lists_every_screen_that_binds_a_letter() -> None:
    assert set(SCREEN_LETTERS) == set(SCREENS)


def test_every_screens_letters_match_the_table() -> None:
    for name, screen in SCREENS.items():
        assert _letters(screen) == SCREEN_LETTERS[name], f"{name} drifted from ui.keymap"


def test_one_letter_never_means_two_things_on_one_screen() -> None:
    """A dict cannot hold a duplicate, so the real check is that the screen's
    own list does not bind a letter twice (the last would silently win)."""
    for name, screen in SCREENS.items():
        keys = [b.key for b in screen.BINDINGS if len(b.key) == 1 and b.key.isalpha()]
        assert len(keys) == len(set(keys)), f"{name} binds a letter twice"
