# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The copy-method picker persists the chosen method to ui.toml on selection."""

from __future__ import annotations

import asyncio

from textual import app

from agent6.ui.tui import copy_method, settings


def test_picker_persists_the_selected_method() -> None:
    settings.save_copy_method("auto")  # start from the default

    class _Harness(app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(copy_method.CopyMethodPicker())

    async def drive() -> None:
        async with _Harness().run_test() as pilot:
            await pilot.pause()
            await pilot.press("down")  # highlight the second choice (osc52)
            await pilot.press("space")  # select it -> ChoiceField.Changed -> save
            await pilot.pause()

    asyncio.run(drive())
    assert settings.get_copy_method() == "osc52"
