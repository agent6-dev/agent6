# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The menu bar every screen shares.

Titles with a mnemonic letter, each opening a dropdown of actions with their
shortcut keys. Selecting an item runs the host screen's `action_<id>`, else the
app's, so the menu, the key bindings and the command palette reach the same
handlers. Every action is reachable by mouse, by `Alt+<letter>` and the arrows,
and by name in the palette.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
from collections.abc import Callable
from typing import ClassVar

try:
    import textual
    from rich import text
    from textual import app as textual_app
    from textual import binding as textual_binding
    from textual import containers, events, geometry, message, widgets
    from textual import screen as textual_screen
    from textual.widget import Widget
    from textual.widgets.option_list import Option
except ImportError as e:  # pragma: no cover
    raise SystemExit("The menu bar needs textual, a required dependency; reinstall agent6.") from e


from agent6.ui import keymap


@dataclasses.dataclass(frozen=True, slots=True)
class MenuItem:
    """One menu row; its key, if any, is `ui.keymap.SCREEN_KEYS`' for the action.

    Attributes:
        label: The row's text.
        action: Dispatched as `action_<action>` on the host screen, else the app.
        priority: The binding fires before the focused widget, such as a composer.
    """

    label: str
    action: str
    priority: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Menu:
    """One menu; the title's first letter is its Alt mnemonic."""

    title: str
    items: tuple[MenuItem, ...]

    @property
    def mnemonic(self) -> str:
        """The title's first letter, lower case."""
        return self.title[0].lower()


# The View items of every screen that scrolls a long body; the run views bind them priority.
SCROLL_ITEMS: tuple[MenuItem, ...] = (
    MenuItem("Scroll ↑ a page", "page_up", priority=True),
    MenuItem("Scroll ↓ a page", "page_down", priority=True),
    MenuItem("Scroll → top", "scroll_top", priority=True),
    MenuItem("Scroll → end", "scroll_bottom", priority=True),
)


def menu_bindings(
    screen: str, menus: tuple[Menu, ...], *, footer: tuple[tuple[str, str], ...] = ()
) -> list[textual_binding.Binding]:
    """Return a screen's bindings from its menus and `SCREEN_KEYS[screen]`.

    A comma in the key table joins an action's aliases: the first key carries the
    footer entry, which names them all. The menu openers come last: `Alt+<mnemonic>`
    per menu, and F10 for the first, since some terminals eat Alt+f.

    Args:
        screen: The screen's name in the key table.
        menus: The screen's menus.
        footer: The actions the footer shows, in its order, with their labels; every
            other keyed action binds hidden.
    """
    keys = keymap.SCREEN_KEYS[screen]
    shown = dict(footer)
    items = {it.action: it for m in menus for it in m.items}
    binds: list[textual_binding.Binding] = []
    for action in shown:
        assert action in keys, f"{screen}: footer action {action!r} has no key"
    for action in (*shown, *(a for a in keys if a not in shown)):
        item = items.get(action)
        if item is None or action not in keys:
            continue
        first, *aliases = keys[action].split(",")
        display = "/".join(_key_label(k) for k in (first, *aliases))
        binds.append(
            textual_binding.Binding(
                first,
                action,
                shown.get(action, item.label),
                show=action in shown,
                key_display=display,
                priority=item.priority,
            )
        )
        binds.extend(
            textual_binding.Binding(alias, action, item.label, show=False, priority=item.priority)
            for alias in aliases
        )
    binds.extend(
        textual_binding.Binding(f"alt+{m.mnemonic}", f"menu('{m.mnemonic}')", show=False)
        for m in menus
    )
    if menus:
        binds.append(
            textual_binding.Binding("f10", f"menu('{menus[0].mnemonic}')", "Menu", show=True)
        )
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
    "backtab": "⇧Tab",
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "backspace": "Bksp",
    "delete": "Del",
}
_MODIFIERS = {"ctrl": "^", "shift": "⇧", "alt": "Alt+", "super": "Super+"}


def _key_label(key: str) -> str:
    """Return the compact display of one key, as the footer shows it: n, ^c, ⇧Enter, PgDn.

    A capital letter stays capital, so g and G stay distinct.
    """
    if key in _KEY_NAMES:
        return _KEY_NAMES[key]
    parts = key.split("+")
    prefix = "".join(_MODIFIERS.get(p, "") for p in parts[:-1])
    last = _KEY_NAMES.get(parts[-1], parts[-1])
    return f"{prefix}{last}"


def action_keys(source: object) -> dict[str, str]:
    """Return each bound action's shortcut labels from the active bindings.

    The one source the menu bar, the help page and the footer read, so they never
    drift; several keys on one action are joined, as in "PgDn / ^End".

    Args:
        source: A screen, or an app (its current screen is used).
    """
    screen = (
        source if isinstance(source, textual_screen.Screen) else getattr(source, "screen", source)
    )
    labels: dict[str, list[str]] = {}
    for key, active in getattr(screen, "active_bindings", {}).items():
        if "super" in key:  # textual adds Cmd beside Ctrl; noise on Linux
            continue
        label = _key_label(key)
        seen = labels.setdefault(active.binding.action, [])
        if label not in seen:
            seen.append(label)
    return {action: " / ".join(keys) for action, keys in labels.items()}


def _title_text(menu: Menu) -> text.Text:
    """Return the menu title with its mnemonic underlined."""
    t = text.Text()
    t.append(menu.title[0], style="underline bold")
    t.append(menu.title[1:])
    return t


def _menu_options(
    items: tuple[MenuItem, ...], keys: dict[str, str], screen: object
) -> list[Option]:
    """Return the dropdown rows, labels left-aligned and keys right-aligned to one edge.

    An item whose `check_action` reads False or None is disabled, like its key
    binding: the bar dispatches straight to the handler with no check of its own.

    Args:
        items: The menu's rows.
        keys: Each action's shortcut label, from the live bindings.
        screen: The host screen, asked `check_action`.
    """
    checker = getattr(screen, "check_action", None)
    labels = [keys.get(it.action, "") for it in items]
    label_w = max((len(it.label) for it in items), default=0)
    key_w = max((len(k) for k in labels), default=0)
    width = label_w + 2 + key_w
    opts: list[Option] = []
    for it, key in zip(items, labels, strict=True):
        t = text.Text(it.label)
        if key:
            t.pad_right(width - len(it.label) - len(key))
            t.append(key, style="dim")
        disabled = checker is not None and not checker(it.action, ())
        opts.append(Option(t, id=it.action, disabled=disabled))
    return opts


def _footer_only_rows(
    source: object, menus: tuple[Menu, ...], keys: dict[str, str]
) -> tuple[tuple[str, str], ...]:
    """Return a description and shortcut per visible footer binding no menu item covers.

    The menu openers are excluded: the help page's own footer line covers them.

    Args:
        source: A screen, or an app (its current screen is used).
        menus: The screen's menus.
        keys: Each action's shortcut label.
    """
    screen = (
        source if isinstance(source, textual_screen.Screen) else getattr(source, "screen", source)
    )
    covered = {it.action for m in menus for it in m.items}
    rows: list[tuple[str, str]] = []
    for _key, active in getattr(screen, "active_bindings", {}).items():
        binding = active.binding
        if not binding.show or binding.action in covered or binding.action.startswith("menu("):
            continue
        covered.add(binding.action)  # a multi-key action lands once
        rows.append((binding.description or binding.action, keys.get(binding.action, "")))
    return tuple(rows)


class HelpScreen(textual_screen.Screen[None]):
    """The keys and actions page, generated from a screen's menus and live bindings.

    Every menu action with its shortcut, every visible footer binding a menu does
    not cover, and the screen's hints, flowed into up to three centred columns that
    reflow on a resize.
    """

    BINDINGS: ClassVar = [
        textual_binding.Binding("escape,q,question_mark,f1", "dismiss", "Close", show=False)
    ]
    CSS = """
    HelpScreen { background: $surface; }
    #help-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #help-foot { dock: bottom; height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    /* The column block is centred as one auto-width unit. */
    #help-scroll { height: 1fr; padding: 1 2; align-horizontal: center; }
    #help-columns { width: auto; height: auto; }
    /* The Statics need width:auto too: their 1fr default collapses inside an auto-width parent. */
    .help-col { width: auto; height: auto; margin: 0 3; }
    .help-col Static { width: auto; pointer: text; }
    .help-menu { text-style: bold; color: $accent; padding-top: 1; }
    """

    def __init__(
        self,
        menus: tuple[Menu, ...],
        source: object,
        *,
        title: str = "Keys & actions",
        hints: tuple[str, ...] = (),
    ) -> None:
        """Build the page.

        Args:
            menus: The screen's menus.
            source: The screen, or app, whose live bindings the page reflects.
            title: The page's title line.
            hints: Interaction lines the bindings cannot express, such as widget keys.
        """
        super().__init__()
        self._menus = menus
        self._title = title
        self._hints = hints
        self._keys = action_keys(source)
        self._extra = _footer_only_rows(source, menus, self._keys)

    def _shortcut(self, it: MenuItem) -> str:
        """Return the item's shortcut label, or ""."""
        return self._keys.get(it.action, "")

    def _sections(self) -> list[tuple[text.Text, list[tuple[str, str]]]]:
        """Return a heading and rows per section: each menu, the footer-only keys, the hints."""
        sections = [
            (_title_text(m), [(it.label, self._shortcut(it)) for it in m.items])
            for m in self._menus
        ]
        if self._extra:
            sections.append((text.Text("Other keys"), list(self._extra)))
        if self._hints:
            sections.append((text.Text("Hints"), [(h, "") for h in self._hints]))
        return sections

    def _columns(self) -> list[list[widgets.Static]]:
        """Pack the sections whole into columns of roughly equal height, in reading order.

        Within a column the keys right-align to a shared edge, like the dropdowns.

        Returns:
            The rendered lines per column.
        """
        sections = self._sections()
        sizes = [len(rows) + 1 for _, rows in sections]  # +1 per heading
        total = sum(sizes)
        ncols = min(max(1, self.size.width // 50), 3, len(sections))
        prefix = list(itertools.accumulate(sizes))
        breaks = sorted(
            {
                min(range(1, len(sections)), key=lambda i: abs(prefix[i - 1] - total * k / ncols))
                for k in range(1, ncols)
            }
        )
        edges = [0, *breaks, len(sections)]
        packed = [sections[a:b] for a, b in itertools.pairwise(edges) if a < b]
        out: list[list[widgets.Static]] = []
        for col_sections in packed:
            rows = [r for _, section_rows in col_sections for r in section_rows]
            # Only keyed rows set the edge, so a long hint line cannot push the keys away.
            label_w = max((len(label) for label, key in rows if key), default=0)
            key_w = max((len(key) for _, key in rows), default=0)
            right = label_w + 2 + key_w
            lines: list[widgets.Static] = []
            for heading, section_rows in col_sections:
                lines.append(widgets.Static(heading, classes="help-menu"))
                for label, key in section_rows:
                    line = text.Text(label)
                    if key:
                        line.pad_right(right - len(label) - len(key))
                        line.append(key, style="dim")
                    lines.append(widgets.Static(line))
            out.append(lines)
        return out

    def compose(self) -> textual_app.ComposeResult:
        """Lay out the page.

        Yields:
            The title, the columns and the footer line.
        """
        yield widgets.Static(self._title, id="help-title")
        with containers.VerticalScroll(id="help-scroll"), containers.Horizontal(id="help-columns"):
            for column in self._columns():
                with containers.Vertical(classes="help-col"):
                    yield from column
        yield widgets.Static(
            text.Text("F10 or Alt+<letter> opens a menu · Esc/q closes this page", style="dim"),
            id="help-foot",
        )

    def _focus_scroll(self) -> None:
        """Focus the scroll container, so the page keys scroll at once."""
        self.query_one("#help-scroll", containers.VerticalScroll).focus()

    def on_mount(self) -> None:
        """Focus the scroll container."""
        self._focus_scroll()

    def on_resize(self) -> None:
        """Rebuild the columns for the new width, then refocus the new scroll container."""
        # Left on the detached old container, focus's binding chain would not reach this screen.
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_scroll)


class _MenuTitle(widgets.Static):
    """One clickable title in the bar; a click opens, toggles or switches its menu.

    Not focusable: a click then cannot blur the open dropdown, so toggling is a
    race-free state check, and Tab moves to real content instead of hopping
    between titles.
    """

    def __init__(self, menu: Menu) -> None:
        """Create the title for a menu."""
        super().__init__(_title_text(menu), classes="menu-title", id=f"menu-{menu.mnemonic}")
        self.mnemonic = menu.mnemonic

    def _bar(self) -> MenuBar:
        """Return the bar this title sits in."""
        bar = self.parent
        assert isinstance(bar, MenuBar)
        return bar

    def on_click(self) -> None:
        """Open, toggle or switch to this menu."""
        self._bar().open(self.mnemonic)


class _Dropdown(widgets.OptionList):
    """The open menu's item list; closes on Esc or focus loss.

    Mounted on the screen, not the bar, so a pick reaches the bar through a
    callback and the styling lives here, where it beats `OptionList`'s defaults.
    `overlay: screen` lifts it out of the layout so it sizes to its content.
    """

    DEFAULT_CSS = """
    _Dropdown, _Dropdown:focus {
        layer: dropdown; overlay: screen; constrain: none inside;
        width: auto; height: auto; min-width: 20; max-width: 60; max-height: 16;
        border: round $accent; background: $surface; padding: 0 1;
    }
    """

    BINDINGS: ClassVar = [textual_binding.Binding("escape", "close", "Close", show=False)]

    def __init__(self, *options: Option, mnemonic: str, on_pick: Callable[[str], None]) -> None:
        """Create the list for a menu, with the callback a pick reaches the bar through."""
        super().__init__(*options)
        self.mnemonic = mnemonic
        self._on_pick = on_pick

    def _bar(self) -> MenuBar:
        """Return the screen's menu bar."""
        return self.screen.query_one(MenuBar)

    def action_close(self) -> None:
        """Close the menu; focus returns to the content underneath."""
        self._bar().close_menu()

    def on_blur(self) -> None:
        """Close on a genuine dismiss, never when a switch already replaced this dropdown."""
        bar = self._bar()
        if bar.is_open(self.mnemonic):
            bar.close_menu()

    def on_key(self, event: events.Key) -> None:
        """Switch to the adjacent menu on Left or Right, keys the list does not use."""
        if event.key in ("left", "right"):
            event.stop()
            self._bar().open_adjacent(self.mnemonic, 1 if event.key == "right" else -1)

    @textual.on(widgets.OptionList.OptionSelected)
    def _picked(self, event: widgets.OptionList.OptionSelected) -> None:
        """Hand the pick to the bar and close."""
        action = event.option.id
        if action:
            self._on_pick(action)
        self._bar().close_menu()


class MenuBar(containers.Horizontal):
    """The top row: the menu titles on the left, the app title and context on the right."""

    DEFAULT_CSS = """
    MenuBar { height: 1; width: 1fr; background: $panel; color: $text; }
    MenuBar > .menu-title { height: 1; width: auto; padding: 0 1; }
    MenuBar > .menu-title:hover { background: $primary 30%; }
    MenuBar > .menu-title.-open { background: $primary; text-style: bold; }
    MenuBar > .app-title {
        width: 1fr; height: 1; content-align: right middle; color: $text-muted;
        padding: 0 1;
    }
    """

    class Selected(message.Message):
        """An item was chosen; the message hop lets the dropdown finish closing first."""

        def __init__(self, action: str) -> None:
            """Name the action."""
            self.action = action
            super().__init__()

    async def on_menu_bar_selected(self, event: Selected) -> None:
        """Run the host screen's handler for the action, else the app's."""
        event.stop()
        handler = getattr(self.screen, f"action_{event.action}", None) or getattr(
            self.app, f"action_{event.action}", None
        )
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    def __init__(self, menus: tuple[Menu, ...]) -> None:
        """Create the bar for a screen's menus."""
        super().__init__()
        self._menus = menus
        # Held as state, not inferred from focus, so a dropdown's blur can tell a dismiss from a
        # switch without a race.
        self._open: str | None = None
        # Closing returns focus here; otherwise textual's reset falls to the last focusable
        # widget and auto-scrolls its container to reveal it.
        self._restore_focus: Widget | None = None

    def compose(self) -> textual_app.ComposeResult:
        """Lay out the bar.

        Yields:
            A title per menu, then the app title.
        """
        for m in self._menus:
            yield _MenuTitle(m)
        yield widgets.Static("", classes="app-title")

    def on_mount(self) -> None:
        """Mirror the app's title and sub-title into the bar, live."""
        self.watch(self.app, "title", self._refresh_title, init=False)
        self.watch(self.app, "sub_title", self._refresh_title, init=False)
        self._refresh_title()

    def _refresh_title(self, *_: object) -> None:
        """Repaint the app title as text, never markup: the sub-title carries a typed task."""
        app = self.app
        parts = [p for p in (app.title, app.sub_title) if p]
        self.query_one(".app-title", widgets.Static).update(text.Text(" — ".join(parts)))

    def open(self, mnemonic: str) -> None:
        """Open a menu by mnemonic; opening the one already open toggles it shut."""
        was_open = self._open
        self._teardown()  # keeps the saved focus: a switch reuses it
        if was_open is None:
            focused = self.screen.focused
            if focused is not None and not isinstance(focused, _Dropdown):
                self._restore_focus = focused
        if was_open == mnemonic:
            self.close_menu()
            return
        menu = next((m for m in self._menus if m.mnemonic == mnemonic), None)
        if menu is None:
            self.close_menu()
            return
        self._open = mnemonic
        # Floated on the screen one row below its title: the one-row bar would clip it. No fixed
        # id, since remove() is async and a re-open could mount a second one first.
        title = self.query_one(f"#menu-{mnemonic}", _MenuTitle)
        opts = _menu_options(menu.items, action_keys(self.screen), self.screen)
        dd = _Dropdown(*opts, mnemonic=mnemonic, on_pick=self._dispatch)
        self.screen.mount(dd)
        dd.absolute_offset = geometry.Offset(title.region.x, title.region.y + 1)
        title.add_class("-open")
        dd.focus()

    def is_open(self, mnemonic: str) -> bool:
        """Return whether the menu with the mnemonic is the open one."""
        return self._open == mnemonic

    @property
    def opened(self) -> bool:
        """Whether any menu is open."""
        return self._open is not None

    def open_adjacent(self, mnemonic: str, step: int) -> None:
        """Switch the open menu to the one a number of places left or right, wrapping."""
        order = [m.mnemonic for m in self._menus]
        if mnemonic in order:
            self.open(order[(order.index(mnemonic) + step) % len(order)])

    def close_menu(self) -> None:
        """Close any open dropdown, return focus to the opener and clear the title highlights."""
        self._teardown()
        self._restore_focus = None

    def _teardown(self) -> None:
        """Remove any open dropdown and clear the highlights, keeping the saved focus."""
        self._open = None
        # Focus moves back before the removal, through set_focus (Widget.focus defers), so
        # textual's reset on the removal is a no-op.
        restore = self._restore_focus
        if restore is not None and restore.is_attached and self.screen.focused is not restore:
            self.screen.set_focus(restore, scroll_visible=False)
        self.screen.query(_Dropdown).remove()
        for t in self.query(_MenuTitle):
            t.remove_class("-open")

    def _dispatch(self, action: str) -> None:
        """Post the pick as a `Selected` message."""
        self.post_message(self.Selected(action))
