# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The copy-method picker persists the chosen method to ui.toml on selection."""

from __future__ import annotations

import asyncio

from textual.app import App

from agent6.ui.tui.copy_method import CopyMethodPicker
from agent6.ui.tui.settings import get_copy_method, save_copy_method


def test_picker_persists_the_selected_method() -> None:
    save_copy_method("auto")  # start from the default

    class _Harness(App[None]):
        def on_mount(self) -> None:
            self.push_screen(CopyMethodPicker())

    async def drive() -> None:
        async with _Harness().run_test() as pilot:
            await pilot.pause()
            await pilot.press("down")  # highlight the second choice (osc52)
            await pilot.press("space")  # select it -> ChoiceField.Changed -> save
            await pilot.pause()

    asyncio.run(drive())
    assert get_copy_method() == "osc52"
