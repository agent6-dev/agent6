# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A slim, universal menu bar for the TUI: classic ``File / Edit / View / Help``
titles with a mnemonic letter, each opening a dropdown of actions (with their
shortcut keys shown). One widget, reused on every screen — each screen just
passes its own :class:`Menu` list. Selecting an item posts
:class:`MenuBar.Selected`, which the host turns into ``action_<id>`` — so the
menu, the buttons, the key bindings, and the command palette all dispatch the
same handlers and never drift.

Every action is therefore reachable by mouse (click a title, click an item), by
keyboard (``Alt+<letter>`` opens a menu; arrows + Enter pick; Esc closes), and
by name in the command palette — nothing to memorize.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

try:
    from rich.text import Text
    from textual import events, on
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, VerticalScroll
    from textual.geometry import Offset
    from textual.message import Message
    from textual.screen import ModalScreen, Screen
    from textual.widget import Widget
    from textual.widgets import OptionList, Static
    from textual.widgets.option_list import Option
except ImportError as e:  # pragma: no cover
    raise SystemExit("The menu bar needs textual: pip install 'agent6[tui]'") from e


@dataclass(frozen=True, slots=True)
class MenuItem:
    label: str
    action: str  # dispatched as action_<action> on the host screen/app
    key: str | None = None  # shortcut shown next to the item (display only)


@dataclass(frozen=True, slots=True)
class Menu:
    title: str  # e.g. "File"; its first letter is the Alt mnemonic
    items: tuple[MenuItem, ...]

    @property
    def mnemonic(self) -> str:
        return self.title[0].lower()


def menu_bindings(menus: tuple[Menu, ...]) -> list[Binding]:
    """Keyboard openers for the menu bar, spread into a host's BINDINGS:
    Alt+<mnemonic> per menu, plus F10 to open the first one (the classic,
    terminal-robust menu key -- some terminals eat Alt+f as 'forward-word').
    Once open, Left/Right switch menus and arrows/Enter pick."""
    binds = [Binding(f"alt+{m.mnemonic}", f"menu('{m.mnemonic}')", show=False) for m in menus]
    if menus:
        # Shown in the footer: the discoverable, terminal-robust way to reach the
        # menus (Alt isn't bindable on its own, and some terminals eat Alt+f).
        binds.append(Binding("f10", f"menu('{menus[0].mnemonic}')", "Menu", show=True))
    return binds


_KEY_NAMES = {
    "question_mark": "?",
    "escape": "Esc",
    "enter": "Enter",
    "pageup": "PgUp",
    "pagedown": "PgDn",
    "home": "Home",
    "end": "End",
    "space": "Space",
    "tab": "Tab",
    "backtab": "⇧Tab",  # the terminal name for Shift+Tab
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "backspace": "Bksp",
    "delete": "Del",
}
_MODIFIERS = {"ctrl": "^", "shift": "⇧", "alt": "Alt+", "super": "Super+"}


def _key_label(key: str) -> str:
    """A compact display for one key, keeping the footer's casing so the menu, help,
    and footer match: n, ^c, ⇧Enter, Alt+f, PgDn, Esc. A bare capital letter stays
    capital (so g and G stay distinct)."""
    if key in _KEY_NAMES:
        return _KEY_NAMES[key]
    parts = key.split("+")
    prefix = "".join(_MODIFIERS.get(p, "") for p in parts[:-1])
    last = _KEY_NAMES.get(parts[-1], parts[-1])  # preserve case
    return f"{prefix}{last}"


def action_keys(source: object) -> dict[str, str]:
    """Map each bound ``action`` to its shortcut label(s) from the ACTIVE bindings --
    the single source of truth, so the menu bar, the help page, and the footer all
    show the same keys and can't drift from the actual key bindings. Multiple keys on
    one action (e.g. PageDown + Ctrl+End, or Shift+Enter + Ctrl+J) are joined:
    'PgDn / ^End'. ``source`` may be an App (its current screen is used) or a Screen."""
    screen = source if isinstance(source, Screen) else getattr(source, "screen", source)
    labels: dict[str, list[str]] = {}
    for key, active in getattr(screen, "active_bindings", {}).items():
        if "super" in key:  # Cmd on macOS; textual adds it beside Ctrl but it's noise on Linux
            continue
        label = _key_label(key)
        seen = labels.setdefault(active.binding.action, [])
        if label not in seen:
            seen.append(label)
    return {action: " / ".join(keys) for action, keys in labels.items()}


def _title_text(menu: Menu) -> Text:
    t = Text()
    t.append(menu.title[0], style="underline bold")
    t.append(menu.title[1:])
    return t


def _menu_options(items: tuple[MenuItem, ...], keys: dict[str, str]) -> list[Option]:
    """Dropdown rows with labels left-aligned and shortcut keys RIGHT-aligned to a
    common edge, so the keys line up in a column. The shortcut comes from the live
    key bindings (``keys`` = action -> label, possibly several joined), falling back
    to the item's own key hint for menu-only actions with no binding."""
    labels = [keys.get(it.action) or (_key_label(it.key) if it.key else "") for it in items]
    label_w = max((len(it.label) for it in items), default=0)
    key_w = max((len(k) for k in labels), default=0)
    width = label_w + 2 + key_w  # 2-space minimum gap between the two columns
    opts: list[Option] = []
    for it, key in zip(items, labels, strict=True):
        t = Text(it.label)
        if key:  # pad so the key's right edge lands at `width`
            t.pad_right(width - len(it.label) - len(key))
            t.append(key, style="dim")
        opts.append(Option(t, id=it.action))
    return opts


class HelpScreen(ModalScreen[None]):
    """A centered help/keys page generated from a screen's menus, so it is
    always complete and accurate. Replaces textual's right-docked keys panel
    with a single page you close with Esc. The Help menu opens this everywhere."""

    BINDINGS: ClassVar = [Binding("escape,q,question_mark,f1", "dismiss", "Close", show=False)]
    CSS = """
    HelpScreen { align: center middle; }
    #help-box {
        width: 64; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #help-title { text-style: bold; padding-bottom: 1; }
    .help-menu { text-style: bold; color: $accent; padding-top: 1; }
    """

    def __init__(
        self,
        menus: tuple[Menu, ...],
        keys: dict[str, str] | None = None,
        *,
        title: str = "Keys & actions",
    ) -> None:
        super().__init__()
        self._menus = menus
        self._keys = keys or {}  # action -> live shortcut(s); see action_keys()
        self._title = title

    def _shortcut(self, it: MenuItem) -> str:
        return self._keys.get(it.action) or (_key_label(it.key) if it.key else "")

    def compose(self) -> ComposeResult:
        # Keys right-aligned to a common edge across ALL sections (labels left),
        # matching the menu dropdowns. Width = indent + widest label + gap + key.
        items = [it for m in self._menus for it in m.items]
        label_w = max((len(it.label) for it in items), default=0)
        key_w = max((len(self._shortcut(it)) for it in items), default=0)
        right = 2 + label_w + 2 + key_w
        with VerticalScroll(id="help-box"):
            yield Static(self._title, id="help-title")
            for m in self._menus:
                # Underline the mnemonic letter, matching the top menu bar.
                yield Static(_title_text(m), classes="help-menu")
                for it in m.items:
                    line = Text(f"  {it.label}")
                    key = self._shortcut(it)
                    if key:  # pad so the key's right edge lands at `right`
                        line.pad_right(right - 2 - len(it.label) - len(key))
                        line.append(key, style="dim")
                    yield Static(line)
            yield Static("")
            yield Static(
                Text("F10 or Alt+<letter> opens a menu · Esc/q or click-out closes", style="dim"),
            )

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:  # click on the backdrop (outside the dialog) = close
            self.dismiss()


class _MenuTitle(Static):
    """One clickable title in the bar. Clicking opens (or toggles/switches) its
    menu; each title carries its own mnemonic because events.Click has no
    ``.widget`` to say which was hit. Titles are deliberately NOT focusable: a
    click on one then can't blur the open dropdown, so toggling is a race-free
    state check, and Tab moves to real content (closing any open menu) instead
    of hopping between titles. Keyboard opening is Alt+<letter> (menu_bindings)
    or the command palette; the open dropdown owns arrows/Enter/Left/Right."""

    def __init__(self, menu: Menu) -> None:
        super().__init__(_title_text(menu), classes="menu-title", id=f"menu-{menu.mnemonic}")
        self.mnemonic = menu.mnemonic

    def _bar(self) -> MenuBar:
        bar = self.parent
        assert isinstance(bar, MenuBar)
        return bar

    def on_click(self) -> None:
        # Titles aren't focusable, so the click didn't blur the open dropdown;
        # open() toggles (clicking the open title shuts it), switches, or opens.
        self._bar().open(self.mnemonic)


class _Dropdown(OptionList):
    """The open menu's item list; closes on Esc or focus loss. Carries the
    mnemonic of the menu it belongs to so the bar can toggle it, and a callback
    so a pick reaches the bar (it's mounted on the *screen*, not the bar, so its
    messages don't bubble through the bar).

    The styling lives HERE, not on MenuBar: the dropdown is mounted on the
    screen, outside MenuBar's subtree, so MenuBar's rules wouldn't beat
    OptionList's own defaults (full-width, tall border). `overlay: screen` lifts
    it out of the screen's layout so it sizes to its content and floats.
    """

    DEFAULT_CSS = """
    _Dropdown, _Dropdown:focus {
        layer: dropdown; overlay: screen; constrain: none inside;
        width: auto; height: auto; min-width: 20; max-width: 60; max-height: 16;
        border: round $accent; background: $surface; padding: 0 1;
    }
    """

    BINDINGS: ClassVar = [Binding("escape", "close", "Close", show=False)]

    def __init__(self, *options: Option, mnemonic: str, on_pick: Callable[[str], None]) -> None:
        super().__init__(*options)
        self.mnemonic = mnemonic
        self._on_pick = on_pick

    def _bar(self) -> MenuBar:
        return self.screen.query_one(MenuBar)

    def action_close(self) -> None:
        self._bar().close_menu()  # focus returns to the content underneath

    def on_blur(self) -> None:
        # Close only if I'm still the bar's open menu: a genuine dismiss (Tab to
        # content, click away). If a switch already replaced me (bar._open is now
        # another menu), I'm a stale dropdown being removed -- don't close the
        # new one. Routed through close_menu() so the -open highlight clears too.
        bar = self._bar()
        if bar.is_open(self.mnemonic):
            bar.close_menu()

    def on_key(self, event: events.Key) -> None:
        # Left/Right switch to the adjacent menu (classic menu-bar feel); the
        # OptionList itself only uses Up/Down/Enter, so these are free.
        if event.key in ("left", "right"):
            event.stop()
            self._bar().open_adjacent(self.mnemonic, 1 if event.key == "right" else -1)

    @on(OptionList.OptionSelected)
    def _picked(self, event: OptionList.OptionSelected) -> None:
        action = event.option.id
        if action:
            self._on_pick(action)
        self._bar().close_menu()  # clears the dropdown + the title's -open highlight


class MenuBar(Horizontal):
    """The single top row: the menu titles on the left, and the app title +
    context (``agent6 — <path>``) filling the rest on the right. Replaces a
    separate Header row entirely -- one row, no clock, no command-palette icon
    (the palette is in the Help menu, the footer, and Ctrl+P)."""

    DEFAULT_CSS = """
    MenuBar { height: 1; width: 1fr; background: $panel; color: $text; }
    MenuBar > .menu-title { height: 1; width: auto; padding: 0 1; }
    MenuBar > .menu-title:hover { background: $primary 30%; }  /* $boost is transparent */
    MenuBar > .menu-title.-open { background: $primary; text-style: bold; }
    MenuBar > .app-title {
        width: 1fr; height: 1; content-align: right middle; color: $text-muted;
        padding: 0 1;
    }
    """

    class Selected(Message):
        """An item was chosen; the host should run ``action_<action>``."""

        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def __init__(self, menus: tuple[Menu, ...]) -> None:
        super().__init__()
        self._menus = menus
        # The currently-open menu (or None). Tracking it in state -- rather than
        # inferring from focus/DOM -- lets a dropdown's on_blur tell "I'm being
        # dismissed" from "I'm being replaced by a switch", with no async race.
        self._open: str | None = None
        # The widget that had focus before the menu was opened, so closing the
        # dropdown returns focus there. Without this, removing the focused
        # dropdown lets textual's _reset_focus fall to the LAST focusable widget
        # in the chain -- which then auto-scrolls a scroll container (e.g. the
        # config #settings) to the bottom to reveal it.
        self._restore_focus: Widget | None = None

    def compose(self) -> ComposeResult:
        for m in self._menus:
            yield _MenuTitle(m)
        yield Static("", classes="app-title")  # app title + path, right-aligned

    def on_mount(self) -> None:
        # Mirror the app's title/sub_title into the bar's right side, live.
        self.watch(self.app, "title", self._refresh_title, init=False)
        self.watch(self.app, "sub_title", self._refresh_title, init=False)
        self._refresh_title()

    def _refresh_title(self, *_: object) -> None:
        app = self.app
        parts = [p for p in (app.title, app.sub_title) if p]
        self.query_one(".app-title", Static).update(" — ".join(parts))

    def open(self, mnemonic: str) -> None:
        """Open the menu *mnemonic* (a single letter). Opening the menu that is
        already open toggles it shut."""
        was_open = self._open
        # Tear down any open dropdown WITHOUT restoring focus yet (the dispatch
        # below decides). No menu open -> nothing to tear down anyway.
        self._teardown()
        if was_open is None:
            # Opening fresh from content: remember where focus was so closing
            # returns it there (a switch keeps the earlier-saved widget).
            focused = self.screen.focused
            if focused is not None and not isinstance(focused, _Dropdown):
                self._restore_focus = focused
        if was_open == mnemonic:
            self.close_menu()  # toggle: same menu was open -> close + restore focus
            return
        menu = next((m for m in self._menus if m.mnemonic == mnemonic), None)
        if menu is None:
            self.close_menu()  # unknown menu: nothing to open, restore focus
            return
        self._open = mnemonic
        # Float the dropdown on the screen, pinned one row below its title.
        # `overlay: screen` lifts it out of layout; absolute_offset places it.
        # (Mounting it in the 1-row bar clipped it to one row; mounting it in the
        # title Static suppressed the title's own text -- Static isn't a
        # container; mounting it on the screen with a plain offset anchored it at
        # the bottom.) No fixed id: remove() is async, so a re-open could mount a
        # second one before the first is gone (DuplicateIds).
        title = self.query_one(f"#menu-{mnemonic}", _MenuTitle)
        opts = _menu_options(menu.items, action_keys(self.screen))
        dd = _Dropdown(*opts, mnemonic=mnemonic, on_pick=self._dispatch)
        self.screen.mount(dd)
        dd.absolute_offset = Offset(title.region.x, title.region.y + 1)
        title.add_class("-open")  # keep the open menu's title highlighted
        dd.focus()

    def is_open(self, mnemonic: str) -> bool:
        """Whether *mnemonic*'s menu is the one currently open."""
        return self._open == mnemonic

    def open_adjacent(self, mnemonic: str, step: int) -> None:
        """Switch the open menu to the one *step* places left/right (wrapping)."""
        order = [m.mnemonic for m in self._menus]
        if mnemonic in order:
            self.open(order[(order.index(mnemonic) + step) % len(order)])

    def close_menu(self) -> None:
        """Close any open dropdown, return focus to the opener, and un-highlight
        all titles."""
        self._teardown()
        self._restore_focus = None

    def _teardown(self) -> None:
        """Remove any open dropdown (returning focus to the opener) and clear the
        open-title highlights -- but KEEP _restore_focus, so a menu *switch* can
        reuse it. Callers that are truly closing clear it themselves."""
        self._open = None
        # Move focus back to the opener BEFORE removing the dropdown: with the
        # dropdown no longer the focused widget, textual's _reset_focus on its
        # removal is a no-op -- it won't fall to the LAST focusable widget and
        # auto-scroll a scroll container (e.g. config #settings) to the bottom.
        # Use set_focus, not Widget.focus (which DEFERS via call_later, leaving
        # the dropdown focused at removal time).
        restore = self._restore_focus
        if restore is not None and restore.is_attached and self.screen.focused is not restore:
            self.screen.set_focus(restore, scroll_visible=False)
        self.screen.query(_Dropdown).remove()
        for t in self.query(_MenuTitle):
            t.remove_class("-open")

    def _dispatch(self, action: str) -> None:
        # Posted from the bar (in-tree) so it bubbles to the host screen's
        # @on(MenuBar.Selected); the dropdown itself lives on the screen.
        self.post_message(self.Selected(action))
