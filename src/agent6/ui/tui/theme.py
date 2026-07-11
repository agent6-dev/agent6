# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""TUI theming: two branded themes (``agent6-dark`` / ``agent6-light``), a
live-previewing picker (every registered theme, sorted alphabetically) reachable
from the View menu, and the wiring that loads the saved theme on startup and
persists any change.

Design: keep one quiet accent for focus and a calm, low-contrast resting state
(the lazygit/openapi-tui feel). All widget CSS across the TUI already uses
Textual theme variables ($primary, $accent, $surface, $panel, $text…), so
switching the theme re-skins everything for free — this module only chooses the
palettes and remembers the choice (in ``ui.toml``, never the agent config).
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
    from textual.theme import Theme
    from textual.widgets import Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The TUI theme support needs textual: pip install 'agent6[tui]'") from e

from agent6.ui.tui.settings import DEFAULT_THEME, get_theme, save_theme
from agent6.ui.tui.widgets import FORM_CSS, ChoiceField

# Branded defaults: a deep, low-saturation dark and a soft light, both with a
# green focus accent over a blue selection primary (mirrors lazygit's
# green-active-border / blue-selection split).
AGENT6_DARK = Theme(
    name="agent6-dark",
    primary="#7AA2F7",  # selection / cursor / resting card borders
    secondary="#9ECE6A",  # parks the old green accent -- unused today, kept for future
    accent="#06F5F3",  # focus borders, button/action text, key hints -- a vivid neon cyan
    foreground="#C0CAF5",
    # Near-black neutral charcoal (not a blue navy): screen < card < panel, so
    # tables/panels read as raised surfaces over an almost-black background.
    background="#161618",
    surface="#202023",
    panel="#2C2C30",
    success="#9ECE6A",
    warning="#E0AF68",
    error="#F7768E",
    dark=True,
    # The footer the baseline had: warm amber keys + neutral labels (reads more
    # "modern" than green keys on lavender text).
    variables={
        "footer-key-foreground": "#FFA62B",
        "footer-foreground": "#E0E0E0",
        "footer-description-foreground": "#E0E0E0",
        # Scrollbar tracks default to near-black (#000002); match the surface so the
        # track meshes with its panel (only the thumb shows) instead of a black gap.
        "scrollbar-background": "#202023",  # == surface
        "scrollbar-background-hover": "#202023",
        "scrollbar-background-active": "#202023",
        "scrollbar-corner-color": "#202023",
    },
)

AGENT6_LIGHT = Theme(
    name="agent6-light",
    primary="#2E5BA8",
    secondary="#1E6FA8",
    accent="#4C7A2F",
    foreground="#2A2E3F",
    background="#F4F5F8",
    surface="#EAECF2",
    panel="#DEE1EA",
    success="#4C7A2F",
    warning="#9A6E00",
    error="#C0392B",
    dark=False,
    variables={
        "footer-key-foreground": "#C2410C",  # warm orange keys, readable on light
        "scrollbar-background": "#EAECF2",  # == surface, so tracks mesh (see agent6-dark)
        "scrollbar-background-hover": "#EAECF2",
        "scrollbar-background-active": "#EAECF2",
        "scrollbar-corner-color": "#EAECF2",
    },
)

# Restyle textual's built-in command palette (Ctrl+P) to match our dialogs: a
# rounded $accent-framed $surface card, not the default flat $panel-darken box with
# black keyline borders. Add to an App's CSS (the palette is pushed on the App).
# Targets textual-internal ids (#--input etc.), so revisit if textual changes them.
PALETTE_CSS = """
CommandPalette > Vertical { background: $surface; border: round $accent; }
CommandPalette #--input { border: none; background: $panel; }
CommandPalette #--input.--list-visible { border: none; }
"""


def setup_theme(app: App[Any]) -> None:
    """Register the branded themes, apply the saved one, and persist changes.

    Call from ``App.on_mount``. Subscribing to ``theme_changed_signal`` means
    EVERY path that changes the theme — the View>Theme picker, the built-in
    Ctrl+P "change theme" palette — is remembered, with no extra wiring.
    """
    for theme in (AGENT6_DARK, AGENT6_LIGHT):
        if theme.name not in app.available_themes:
            app.register_theme(theme)
    wanted = get_theme()
    app.theme = wanted if wanted in app.available_themes else DEFAULT_THEME
    app.theme_changed_signal.subscribe(app, lambda theme: save_theme(theme.name))


def open_theme_picker(app: App[Any]) -> None:
    """Push the theme picker (the View>Theme handler)."""
    app.push_screen(ThemePicker())


class ThemePicker(ModalScreen[None]):
    """A small, live-previewing theme chooser: the same ``[x]``/``[ ]`` chooser
    the config dialogs use. Arrow through to preview; the previewed theme is kept
    on close (Enter or Esc both just dismiss). The choice is persisted by
    ``setup_theme``'s signal hook, so nothing here writes to disk directly."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Use theme"),
    ]
    CSS = (
        FORM_CSS
        + """
    ThemePicker { align: center middle; }
    #theme-box {
        width: 44; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #theme-title { text-style: bold; }
    /* The list scrolls (all themes) while the title + hint stay put. */
    #theme-scroll { height: auto; max-height: 16; scrollbar-size-vertical: 1; }
    #theme-hint { color: $text-muted; padding-top: 1; }
    """
    )

    def on_mount(self) -> None:
        # Focus without auto-scroll so the list opens at the top (focusing a list
        # taller than the dialog would otherwise scroll it).
        self.query_one(ChoiceField).focus(scroll_visible=False)

    def compose(self) -> ComposeResult:
        current = self.app.theme
        # Every registered theme, sorted alphabetically (includes ansi-dark/-light,
        # the "transparent" terminal-native options). The active one is guaranteed
        # present, so the chooser always opens on a real selection.
        names = sorted(self.app.available_themes)
        if current not in names:
            names.insert(0, current)
        with Vertical(id="theme-box"):
            yield Static("Theme", id="theme-title")
            # Just the scrollable list -- no button below (it added a cross-scroll
            # focus stop). Close with Esc or a click outside (handled below).
            with VerticalScroll(id="theme-scroll"):
                yield ChoiceField(tuple(names), current, id="theme-list")
            # Two balanced lines: the 44-wide box would wrap one line mid-phrase.
            yield Static(
                Text("↑↓ highlight · Space select\nEsc or click outside closes", style="dim"),
                id="theme-hint",
            )

    @on(ChoiceField.Changed)
    def _preview(self, event: ChoiceField.Changed) -> None:
        self.app.theme = event.field.value  # apply the selected theme (live)

    def action_confirm(self) -> None:
        self.dismiss(None)  # Enter: keep the applied theme + close

    def action_cancel(self) -> None:
        self.dismiss(None)  # Esc: just close, keeping whatever was previewed

    def on_click(self, event: events.Click) -> None:
        # Click on the backdrop (outside the dialog) = close, like Esc. A mouse +
        # key-swallowing-terminal alternative to Esc.
        if event.widget is self:
            self.action_cancel()
