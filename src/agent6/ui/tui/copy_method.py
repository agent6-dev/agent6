# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The copy-method picker: a small View-menu chooser for how the TUI copies to the
clipboard, mirroring the theme picker.

The choice (``auto`` | ``osc52`` | ``osc52-tmux`` | ``tmux-buffer``) is a viewer
preference in ``ui.toml`` (never the agent config). ``auto`` resolves per
environment; the hint shows what it resolves to right now. Selecting persists
immediately, matching the theme picker.
"""

from __future__ import annotations

from typing import Any, ClassVar

try:
    from rich.text import Text
    from textual import events, on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The TUI needs textual: pip install 'agent6[tui]'") from e

from agent6.ui.tui import clipboard
from agent6.ui.tui.settings import get_copy_method, save_copy_method
from agent6.ui.tui.widgets import FORM_CSS, ChoiceField


def open_copy_method_picker(app: App[Any]) -> None:
    """Push the copy-method picker (the View>Copy method handler)."""
    app.push_screen(CopyMethodPicker())


class CopyMethodPicker(ModalScreen[None]):
    """Pick how copy reaches the clipboard. Selecting persists to ui.toml at once
    (like the theme picker); Enter or Esc close."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close"),
        Binding("enter", "confirm", "Use"),
    ]
    CSS = (
        FORM_CSS
        + """
    CopyMethodPicker { align: center middle; }
    #copy-box {
        width: 52; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #copy-title { text-style: bold; }
    #copy-scroll { height: auto; max-height: 12; scrollbar-size-vertical: 1; }
    #copy-hint { color: $text-muted; padding-top: 1; }
    """
    )

    def on_mount(self) -> None:
        self.query_one(ChoiceField).focus(scroll_visible=False)

    def compose(self) -> ComposeResult:
        choices = tuple(clipboard.COPY_METHODS)
        current = get_copy_method()
        if current not in choices:
            current = "auto"
        resolved = clipboard.resolve_method("auto")
        with Vertical(id="copy-box"):
            yield Static("Copy method", id="copy-title")
            with VerticalScroll(id="copy-scroll"):
                yield ChoiceField(choices, current, id="copy-list")
            yield Static(
                Text(
                    f"auto → {resolved} in this terminal · how `c` copies to your clipboard\n"
                    "↑↓ highlight · Space select (saved) · Esc closes",
                    style="dim",
                ),
                id="copy-hint",
            )

    @on(ChoiceField.Changed)
    def _save(self, event: ChoiceField.Changed) -> None:
        save_copy_method(event.field.value)  # persist immediately, like the theme picker

    def action_confirm(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:
            self.action_cancel()
