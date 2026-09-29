# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The copy-method picker, a View-menu chooser mirroring the theme picker.

The choice is a viewer preference in `ui.toml`, never the agent config; `auto`
resolves per environment and the hint shows what it resolves to. Selecting
persists at once.
"""

from __future__ import annotations

from typing import Any, ClassVar

try:
    import textual
    from rich import text
    from textual import app as textual_app
    from textual import binding, containers, events, screen
    from textual import widgets as textual_widgets
except ImportError as e:  # pragma: no cover
    raise SystemExit("The TUI needs textual: pip install 'agent6[tui]'") from e

from agent6.ui.tui import clipboard, forms, settings


def open_copy_method_picker(app: textual_app.App[Any]) -> None:
    """Push the copy-method picker (the View>Copy method handler)."""
    app.push_screen(CopyMethodPicker())


class CopyMethodPicker(screen.ModalScreen[None]):
    """Pick how copy reaches the clipboard; selecting persists, Enter or Esc close."""

    BINDINGS: ClassVar = [
        binding.Binding("escape", "cancel", "Close"),
        binding.Binding("enter", "confirm", "Use"),
    ]
    CSS = (
        forms.FORM_CSS
        + """
    CopyMethodPicker { align: center middle; }
    #copy-box {
        width: 64; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #copy-title { text-style: bold; }
    #copy-scroll { height: auto; max-height: 12; scrollbar-size-vertical: 1; }
    #copy-hint { color: $text-muted; padding-top: 1; }
    """
    )

    def on_mount(self) -> None:
        """Focus the choice list."""
        self.query_one(forms.ChoiceField).focus(scroll_visible=False)

    def compose(self) -> textual_app.ComposeResult:
        """Lay out the picker.

        Yields:
            The title, the choice list and the hint.
        """
        choices = tuple(clipboard.COPY_METHODS)
        current = settings.get_copy_method()
        if current not in choices:
            current = "auto"
        resolved = clipboard.resolve_method("auto")
        with containers.Vertical(id="copy-box"):
            yield textual_widgets.Static("Copy method", id="copy-title")
            with containers.VerticalScroll(id="copy-scroll"):
                yield forms.ChoiceField(choices, current, id="copy-list")
            # Split by hand: the box is 58 cells inside, so one line would wrap mid-phrase.
            yield textual_widgets.Static(
                text.Text(
                    "how the TUI copies to your clipboard\n"
                    f"auto → {resolved} in this terminal\n"
                    "↑↓ highlight · Space select (saved) · Esc closes",
                    style="dim",
                ),
                id="copy-hint",
            )

    @textual.on(forms.ChoiceField.Changed)
    def _save(self, event: forms.ChoiceField.Changed) -> None:
        settings.save_copy_method(event.field.value)

    def action_confirm(self) -> None:
        """Close the picker."""
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the picker."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Close on a click outside the box."""
        if event.widget is self:
            self.action_cancel()
