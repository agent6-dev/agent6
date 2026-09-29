# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared TUI form widgets.

The `[x]`/`[ ]` chooser, the type-to-narrow picker, the flat action label, the
one-row dropdown and the scroll pane, so every dialog uses the same
arrow-navigable controls.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

try:
    from rich.color import Color
    from rich.console import RenderableType
    from rich.text import Text
    from textual import events
    from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
    from textual.geometry import Region
    from textual.message import Message
    from textual.widget import Widget
    from textual.widgets import Input, Select, Static
    from textual.widgets._select import SelectCurrent, SelectOverlay
except ImportError as e:  # pragma: no cover
    raise SystemExit("The TUI widgets need textual: pip install 'agent6[tui]'") from e


def _scroll_row_into_view(widget: Widget, row: int) -> None:
    """Scroll the widget's nearest scrollable ancestor so its content row is visible.

    The widget is not itself scrollable, so the ancestor is driven directly.
    """
    for node in widget.ancestors:
        if isinstance(node, ScrollableContainer):
            content_y = (
                widget.content_region.y
                + row
                - node.scrollable_content_region.y
                + node.scroll_offset.y
            )
            node.scroll_to_region(
                Region(0, content_y, 1, 1), animate=False, force=True, x_axis=False
            )
            return


def focus_neighbor(widget: Widget, direction: int) -> None:
    """Move focus to the next or previous control in the dialog, never wrapping.

    Scroll containers are skipped, and a dialog's top and bottom are hard stops, so
    the arrows never strand on a focusable scroll box.

    Args:
        widget: The control that has the focus.
        direction: 1 for the next control, -1 for the previous.
    """
    kinds = (ChoiceField, TypeaheadField, Input, ActionItem)
    nav = [w for w in widget.screen.focus_chain if isinstance(w, kinds)]
    for i, w in enumerate(nav):
        if w is widget:
            j = i + direction
            if 0 <= j < len(nav):
                nav[j].focus()
            return


def _selection_bar(primary: str) -> str:
    """Return the style of a full-row selection bar, with its ink chosen by luminance."""
    rgb = Color.parse(primary).get_truecolor()
    lum = 0.299 * rgb.red + 0.587 * rgb.green + 0.114 * rgb.blue
    ink = "#11111b" if lum > 140 else "#f8f8f2"
    return f"bold {ink} on {primary}"


def _window(text: str, cursor: int, width: int) -> tuple[str, int]:
    """Return the slice of text a row shows with the cursor in view, and the cursor's column.

    The caret takes a cell, and a value longer than the row scrolls under it.

    Args:
        text: The whole value.
        cursor: The caret's index in it.
        width: The row's width in cells.
    """
    room = max(width - 1, 1)
    start = max(0, cursor - room + 1)
    return text[start : start + room], cursor - start


class ChoiceField(Widget, can_focus=True):
    """A vertical `[x]`/`[ ]` chooser.

    The arrows move a highlight the selection does not follow, and hand off focus
    at the edges, so a dialog reads as one arrow chain; Space or Enter selects the
    highlighted row, and Enter also bubbles so a dialog can confirm on it. With
    `allow_custom` the last row is an inline text field that typing selects.
    Posts `Changed` when the selection changes.
    """

    DEFAULT_CSS = """
    ChoiceField { height: auto; width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    class Changed(Message):
        """The selection changed."""

        def __init__(self, field: ChoiceField) -> None:
            """Name the field that changed."""
            self.field = field
            super().__init__()

    def __init__(
        self,
        options: tuple[str, ...],
        current: str,
        *,
        allow_custom: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Create the chooser with its options and the current value selected."""
        super().__init__(id=id, classes=classes)
        self._options = list(options)
        self._allow_custom = allow_custom
        in_list = current in self._options
        self._custom_text = "" if (in_list or not allow_custom) else current
        if in_list:
            sel = self._options.index(current)
        elif allow_custom:
            sel = len(self._options)  # the inline custom row
        else:
            sel = 0
        self._sel = sel  # the chosen ([x]) row
        self._cursor = sel  # the highlighted row
        self._hover = -1  # the mouse-hovered row (-1 = none), like a DataTable
        self._pos = len(self._custom_text)  # caret within the custom text

    @property
    def _row_count(self) -> int:
        """The options plus the custom row."""
        return len(self._options) + (1 if self._allow_custom else 0)

    @property
    def _custom_row(self) -> int:
        """The custom row's index; -1 without one."""
        return len(self._options) if self._allow_custom else -1

    @property
    def index(self) -> int:
        """The chosen row."""
        return self._sel

    @property
    def value(self) -> str:
        """The chosen option, or the custom text."""
        if self._sel == self._custom_row:
            return self._custom_text
        if 0 <= self._sel < len(self._options):
            return self._options[self._sel]
        return ""

    def select_value(self, value: str) -> None:
        """Select a fixed option without posting `Changed`; a no-op for any other value.

        Silent, so a dialog can prefill the field from a preset without retriggering
        its own change handlers.
        """
        if value in self._options:
            self._sel = self._cursor = self._options.index(value)
            self.refresh(layout=True)

    def render(self) -> Text:
        """Render the rows: the highlight, the hover bar and the chosen mark.

        Returns:
            The rows as one text.
        """
        focused = self.has_focus
        width = max(self.size.width, 1)
        try:
            bar = _selection_bar(self.app.current_theme.primary)
        except Exception:  # pragma: no cover
            bar = "bold reverse"
        # The hover bar is the resolved panel colour, weaker than the cursor bar ($boost is clear).
        hover_bg = ""
        if self._hover >= 0:
            try:
                hover_bg = f"on {self.app.get_css_variables()['panel']}"
            except Exception:  # pragma: no cover
                hover_bg = ""
        out = Text()
        for i in range(self._row_count):
            is_option = i < len(self._options)
            mark = "[x]" if i == self._sel else "[ ]"
            label = self._options[i] if is_option else (self._custom_text or "custom…")
            if (not is_option) and focused and i == self._cursor:
                pos = self._pos if self._custom_text else len(label)
                shown, col = _window(label, pos, width - len(mark) - 1)
                label = f"{shown[:col]}▌{shown[col:]}"
            line = Text(f"{mark} ")
            line.append(label, style="" if (is_option or self._custom_text) else "dim")
            line.pad_right(max(0, width - line.cell_len))
            if focused and i == self._cursor:
                line.stylize(bar)
            else:
                if hover_bg and i == self._hover:
                    line.stylize(hover_bg)
                if i == self._sel:
                    line.stylize("bold")
            out.append_text(line)
            if i < self._row_count - 1:
                out.append("\n")
        return out

    def on_key(self, event: events.Key) -> None:
        """Move the highlight, select, or edit the custom row."""
        key = event.key
        if key == "up":
            event.stop()
            if self._cursor <= 0:
                focus_neighbor(self, -1)
            else:
                self._cursor -= 1
                self._moved()
        elif key == "down":
            event.stop()
            if self._cursor >= self._row_count - 1:
                focus_neighbor(self, 1)
            else:
                self._cursor += 1
                self._moved()
        elif key == "space":
            event.stop()
            self._select()
        elif key == "enter":
            self._select()  # bubbles too, so a dialog may confirm on Enter
        elif self._cursor == self._custom_row:
            self._edit_custom(event)

    def _edit_custom(self, event: events.Key) -> None:
        """Edit the custom row's text; typing selects the row, and the arrows move the caret."""
        key = event.key
        if event.is_printable and event.character:
            event.stop()
            self._custom_text = (
                self._custom_text[: self._pos] + event.character + self._custom_text[self._pos :]
            )
            self._pos += 1
            self._sel = self._custom_row
            self._changed()
        elif key == "backspace" and self._pos > 0:
            event.stop()
            self._custom_text = self._custom_text[: self._pos - 1] + self._custom_text[self._pos :]
            self._pos -= 1
            self._sel = self._custom_row
            self._changed()
        elif key == "left":
            event.stop()
            self._pos = max(0, self._pos - 1)
            self.refresh()
        elif key == "right":
            event.stop()
            self._pos = min(len(self._custom_text), self._pos + 1)
            self.refresh()

    def _select(self) -> None:
        changed = self._sel != self._cursor
        self._sel = self._cursor
        if self._sel == self._custom_row:
            self._pos = len(self._custom_text)
        if changed:
            self._changed()
        else:
            self.refresh()

    def _changed(self) -> None:
        self.refresh(layout=True)
        self.post_message(self.Changed(self))

    def _moved(self) -> None:
        """Repaint the moved highlight and keep it in view."""
        self.refresh()
        _scroll_row_into_view(self, self._cursor)

    def on_click(self, event: events.Click) -> None:
        """Highlight and select the clicked row."""
        row = int(event.offset.y) - self.styles.padding.top  # the offset includes padding-top
        if 0 <= row < self._row_count:
            self._cursor = row
            self.focus()
            self._select()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Track the hovered row, repainting only on a change."""
        row = int(event.offset.y) - self.styles.padding.top
        row = row if 0 <= row < self._row_count else -1
        if row != self._hover:
            self._hover = row
            self.refresh()

    def on_leave(self, event: events.Leave) -> None:
        """Clear the hovered row."""
        if self._hover != -1:
            self._hover = -1
            self.refresh()


class TypeaheadField(Widget, can_focus=True):
    """A type-to-narrow picker for big lists such as model ids.

    An editable text line plus, while focused, the top matching suggestions; the
    down arrow highlights one, Enter saves the highlighted suggestion or the typed
    text. Hands off the arrows at its edges like `ChoiceField`; `set_suggestions`
    swaps the list in later, after a background fetch.
    """

    MAX_SHOWN = 8
    DEFAULT_CSS = """
    TypeaheadField {
        height: auto; width: 1fr; background: $panel;
        text-wrap: nowrap; text-overflow: ellipsis;
    }
    """

    class Changed(Message):
        """The value changed."""

        def __init__(self, field: TypeaheadField) -> None:
            """Name the field that changed."""
            self.field = field
            super().__init__()

    def __init__(
        self,
        current: str,
        suggestions: list[str],
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Create the picker with the current value and its suggestions."""
        super().__init__(id=id, classes=classes)
        self._text = current
        self._cursor = len(current)
        self._all = list(suggestions)
        self._index = -1  # -1 while editing the text, else the highlighted match
        # Fresh: the untouched current value lists every suggestion, and the first keystroke
        # replaces it.
        self._fresh = bool(current)

    def set_suggestions(self, suggestions: list[str]) -> None:
        """Replace the suggestions, keeping the highlighted one when it is still listed."""
        matches = self._matches
        highlighted = matches[self._index] if 0 <= self._index < len(matches) else None
        self._all = list(suggestions)
        if highlighted is not None:
            matches = self._matches
            self._index = matches.index(highlighted) if highlighted in matches else -1
        if self.is_mounted:
            self.refresh(layout=True)

    def on_focus(self) -> None:
        """Grow to show the suggestions."""
        self.refresh(layout=True)

    def on_blur(self) -> None:
        """Shrink back to the text line."""
        self.refresh(layout=True)

    @property
    def _matches(self) -> list[str]:
        """The suggestions matching the text, prefix matches first, capped."""
        q = "" if self._fresh else self._text.strip().lower()
        if not q:
            shown = self._all
        else:
            starts = [m for m in self._all if m.lower().startswith(q)]
            rest = [m for m in self._all if q in m.lower() and not m.lower().startswith(q)]
            shown = starts + rest
        return shown[: self.MAX_SHOWN]

    @property
    def value(self) -> str:
        """The highlighted suggestion, else the typed text."""
        matches = self._matches
        if 0 <= self._index < len(matches):
            return matches[self._index]
        return self._text

    def render(self) -> Text:
        """Render the text line and, while focused, the suggestions.

        Returns:
            The rows as one text.
        """
        focused = self.has_focus
        width = max(self.size.width, 1)
        try:
            bar = _selection_bar(self.app.current_theme.primary)
        except Exception:  # pragma: no cover
            bar = "bold reverse"
        out = Text()
        editing = focused and self._index < 0
        if self._text and editing:
            shown, col = _window(self._text, self._cursor, width)
            text = Text(f"{shown[:col]}▌{shown[col:]}")
        elif self._text:
            text = Text(self._text)
        else:
            text = Text("type to search…", style="dim")
            if editing:
                text.append("▌")
        out.append_text(text)
        if focused:
            matches = self._matches
            q = "" if self._fresh else self._text.strip().lower()
            total = sum(1 for m in self._all if q in m.lower())
            for i, m in enumerate(matches):
                out.append("\n")
                # One row per match however long: the value stays intact, only the display clips.
                row = Text(m, no_wrap=True, overflow="ellipsis")
                row.pad_right(max(0, width - row.cell_len))
                if i == self._index:
                    row.stylize(bar)
                out.append_text(row)
            if total > len(matches):
                out.append("\n")
                out.append(f"+{total - len(matches)} more, keep typing", style="dim")
            elif not matches:
                out.append("\n")
                out.append("(no matches; saved as typed)", style="dim")
        return out

    def _moved(self) -> None:
        """Repaint with the height re-measured, keep the highlight in view, post `Changed`."""
        self.refresh(layout=True)
        self.call_after_refresh(_scroll_row_into_view, self, max(self._index + 1, 0))
        self.post_message(self.Changed(self))

    def on_key(self, event: events.Key) -> None:
        """Move the highlight, accept it, or edit the text."""
        key = event.key
        if key == "down":
            event.stop()
            if self._index < len(self._matches) - 1:
                self._index += 1
                self._moved()
            else:
                focus_neighbor(self, 1)
        elif key == "up":
            event.stop()
            if self._index >= 0:
                self._index -= 1
                self._moved()
            else:
                focus_neighbor(self, -1)
        elif key == "space" and self._index >= 0:
            event.stop()
            self._accept()
        elif key == "enter":
            self._accept()  # bubbles too, so a dialog may confirm on Enter
        else:
            self._edit(event)

    def _accept(self) -> None:
        """Commit the highlighted match into the text line."""
        matches = self._matches
        if 0 <= self._index < len(matches):
            self._text = matches[self._index]
            self._cursor = len(self._text)
            self._fresh = False
            self._index = -1
            self.refresh(layout=True)
            self.post_message(self.Changed(self))

    def _edit(self, event: events.Key) -> None:
        """Edit the text line; the first keystroke on a fresh value replaces it."""
        key = event.key
        if key == "left":
            event.stop()
            self._fresh = False
            self._cursor = max(0, self._cursor - 1)
            self.refresh()
        elif key == "right":
            event.stop()
            self._fresh = False
            self._cursor = min(len(self._text), self._cursor + 1)
            self.refresh()
        elif key == "backspace":
            event.stop()
            self._fresh = False
            if self._cursor > 0:
                self._text = self._text[: self._cursor - 1] + self._text[self._cursor :]
                self._cursor -= 1
                self._index = -1
                self._moved()
        elif event.is_printable and event.character:
            event.stop()
            if self._fresh:
                self._text = ""
                self._cursor = 0
                self._fresh = False
            self._text = self._text[: self._cursor] + event.character + self._text[self._cursor :]
            self._cursor += 1
            self._index = -1
            self._moved()

    def on_click(self, event: events.Click) -> None:
        """Focus the text line, or highlight the clicked suggestion."""
        row = int(event.offset.y) - self.styles.padding.top
        matches = self._matches if self.has_focus else []
        if row == 0:
            self.focus()
        elif 1 <= row <= len(matches):
            self._index = row - 1
            self.focus()
            self._moved()


class ActionItem(Static):
    """A flat, focusable, clickable action label; Enter or a click activates it.

    Textual's `Button` assumes a three-row box and cannot render flat at height 1.
    """

    can_focus = True

    class Activated(Message):
        """The action was chosen."""

        def __init__(self, action: str) -> None:
            """Name the action."""
            self.action = action
            super().__init__()

    def __init__(self, label: str, action: str) -> None:
        """Create the label for an action."""
        super().__init__(label, classes="action")
        self._action = action

    def on_click(self) -> None:
        """Activate."""
        self.post_message(self.Activated(self._action))

    def on_key(self, event: events.Key) -> None:
        """Activate on Enter."""
        if event.key == "enter":
            event.stop()
            self.post_message(self.Activated(self._action))


class ScrollPane(VerticalScroll):
    """A scrollable pane that can be tabbed to and maximized; the host updates its child."""

    ALLOW_MAXIMIZE = True


_PICKER_ROWS = 10  # options a Picker's list shows before it scrolls


class PickerRow(Horizontal):
    """One line of labelled pickers; a `.picker-label` Static captions the picker after it."""

    DEFAULT_CSS = """
    PickerRow { height: 1; }
    PickerRow .picker-label { width: auto; padding: 0 1 0 0; color: $text-muted; }
    PickerRow Picker { margin-right: 2; }
    """


class Picker(Select[str]):
    """A one-row dropdown, sized to its value, whose list has the menus' round border.

    The list opens upward, so a row above a composer keeps both visible;
    `opens="down"` suits a row at the top of a pane. The last picker in a row takes
    what is left of it, and a value too long for that ends in an ellipsis.
    """

    DEFAULT_CSS = f"""
    Picker {{ width: auto; }}
    Picker:last-child {{ width: 1fr; }}
    Picker > SelectCurrent {{
        width: auto; max-width: 100%; border: none; padding: 0 1; background: $panel;
    }}
    Picker:focus > SelectCurrent {{ border: none; background: $primary 25%; }}
    Picker > SelectCurrent:hover {{ background: $primary 30%; }}
    Picker > SelectCurrent > Static#label {{
        width: auto; max-width: 100%; text-wrap: nowrap; text-overflow: ellipsis;
    }}
    Picker > SelectCurrent > .arrow {{ dock: right; }}
    Picker > SelectOverlay, Picker > SelectOverlay:focus {{
        width: auto; max-width: 60; max-height: {_PICKER_ROWS + 2}; constrain: inside inside;
        border: round $accent; padding: 0;
    }}
    """

    def __init__(
        self,
        options: Iterable[tuple[RenderableType, str]],
        *,
        opens: Literal["up", "down"] = "up",
        **kwargs: Any,
    ) -> None:
        """Create the picker with its options and the direction its list opens."""
        super().__init__(options, **kwargs)
        self._opens = opens

    def watch_expanded(self, expanded: bool) -> None:
        """Place the opened list so its values line up with the field's."""
        if expanded:
            field = self.query_one(SelectCurrent)
            overlay = self.query_one(SelectOverlay)
            rows = min(overlay.option_count, _PICKER_ROWS) + 2
            overlay.styles.min_width = field.outer_size.width + 2
            overlay.styles.offset = (-1, -(rows + 1) if self._opens == "up" else 0)

    def set_options(self, options: Iterable[tuple[RenderableType, str]]) -> None:
        """Replace the options, repainting an entry relabelled under the same value."""
        super().set_options(options)
        self.mutate_reactive(Picker.value)


# The CSS every form-style dialog includes for its actions, inputs and chooser.
FORM_CSS = """
.edit-gap { height: auto; padding-top: 1; }
.edit-label { color: $text-muted; padding-top: 1; }
/* $panel for resting fields: $boost resolves to transparent in every theme. */
.edit-input { border: none; background: $panel; height: 1; padding: 0 1; }
.edit-input:focus { background: $primary 25%; }
ChoiceField { padding: 0 1; }
.action {
    width: auto; height: 1; padding: 0 2; margin-right: 1;
    background: transparent; color: $accent;
}
.action:focus { background: $primary; color: $text; text-style: bold; }
.action:hover { background: $primary 30%; }
"""
