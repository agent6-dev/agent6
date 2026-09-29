# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config page: a renderer over the shared config view and the shared edit path.

The config logic lives in `viewmodel.config_view` and `config.write`, so this page
and the web editor never drift. One registry, `CONFIG_ACTIONS`, feeds the footer
labels and the palette descriptions; the keys are the keymap's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual import events, on
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.geometry import Region
    from textual.screen import ModalScreen, Screen
    from textual.widget import Widget
    from textual.widgets import (
        Collapsible,
        DataTable,
        Footer,
        Input,
        Static,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The config page needs textual: pip install 'agent6[tui]'") from e

from agent6.app.confine import resolved_config_values
from agent6.config import ConfigError
from agent6.config.io import ConfigLeafValue, format_toml_value
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.config.write import (
    PROVIDER_DEFAULTS,
    provider_choices,
    set_config_leaves,
    set_config_value,
    unset_config_value,
)
from agent6.errors import OperatorError
from agent6.models.cache import cached_models
from agent6.models.choices import config_value_choices, model_role_provider
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.screen_chrome import (
    MenuCommands,
    PaletteCommand,
    ScreenChrome,
    menu_palette_commands,
)
from agent6.ui.tui.widgets import (
    FORM_CSS,
    ActionItem,
    ChoiceField,
    TypeaheadField,
    focus_neighbor,
)
from agent6.viewmodel.config_view import (
    ConfigSetting,
    ConfigView,
    build_config_view,
    display_value,
    format_value,
    plain_description,
)


@dataclass(frozen=True, slots=True)
class Action:
    """A page action, reachable from the footer, the menus and the command palette.

    Attributes:
        id: The action name the handler `action_<id>` answers to.
        label: The footer word.
        description: The palette's help text.
    """

    id: str
    label: str
    description: str


# The one registry of the page's actions.
CONFIG_ACTIONS: tuple[Action, ...] = (
    Action("search", "Filter", "Filter settings by name"),
    Action("toggle_modified", "Modified only", "Show only settings a config layer set"),
    Action("edit", "Edit", "Edit the selected setting (dropdown for choices)"),
    Action("add_provider", "Add provider…", "Add a [providers.<name>] entry via a form"),
    # The labels match the home and run footers.
    Action("reset", "Unset", "Unset the selected setting in its source config layer"),
    Action("reload", "Refresh", "Re-read config from disk"),
    # Menu and palette only; the built-in "Theme" is filtered out app-wide.
    Action("choose_theme", "Theme…", "Choose a colour theme"),
    Action("help", "Help", "Show all actions and shortcuts"),
    Action("close", "Back", "Back to the hub"),
)


def _columns(settings: Iterable[ConfigSetting]) -> tuple[tuple[str, int], ...]:
    """Return the (label, width) columns the pinned header and every section table share."""
    setting = max((len(_leaf(s)) for s in settings), default=7)
    return (("setting", setting), ("value", 28), ("source", 12))


def _leaf(s: ConfigSetting) -> str:
    """Return the key under its section, as a row shows it."""
    return s.key.split(".", 1)[1] if "." in s.key else s.key


class _NavTable(DataTable[str]):
    """A settings table whose arrow keys hand off to the section headers at its edges.

    With the headers arrow-navigable too, the whole config reads as one list.
    """

    def action_cursor_down(self) -> None:
        """Move down, or hand off to the next section's header at the bottom row."""
        if self.cursor_row >= self.row_count - 1:
            self._hand_off(1)
        else:
            super().action_cursor_down()
            self._keep_in_view()

    def action_cursor_up(self) -> None:
        """Move up, or hand off to this section's header at the top row."""
        if self.cursor_row <= 0:
            self._hand_off(-1)
        else:
            super().action_cursor_up()
            self._keep_in_view()

    def _keep_in_view(self) -> None:
        """Scroll the page just enough to show the cursor row."""
        if isinstance(self.screen, ConfigScreen):
            self.screen.scroll_focused_into_view()

    def _hand_off(self, direction: int) -> None:
        """Pass the arrow at an edge to the screen's section navigation."""
        screen = self.screen
        section = self.id[4:] if self.id else ""
        if isinstance(screen, ConfigScreen):
            screen.nav_from_table(section, direction)


def _provider_preset_base_url(key: str) -> str:
    """Return the preset base_url for a known provider's `base_url` setting, else ""."""
    parts = key.split(".")
    if len(parts) == 3 and parts[0] == "providers" and parts[2] == "base_url":
        return PROVIDER_DEFAULTS.get(parts[1], {}).get("base_url", "")
    return ""


class _FormModal[ResultT](ModalScreen[ResultT]):
    """The form modals' shared keys: one vertical arrow chain, and left and right over the actions.

    An activated ActionItem dispatches to `action_<id>`.
    """

    def on_key(self, event: events.Key) -> None:
        """Move between fields and actions with the arrow keys."""
        focused = self.focused
        if event.key in ("left", "right") and isinstance(focused, ActionItem):
            actions = list(self.query(ActionItem))
            step = 1 if event.key == "right" else -1
            actions[(actions.index(focused) + step) % len(actions)].focus()
            event.stop()
        elif event.key == "up" and isinstance(focused, (Input, ActionItem)):
            focus_neighbor(focused, -1)
            event.stop()
        elif event.key == "down" and isinstance(focused, Input):
            focus_neighbor(focused, 1)
            event.stop()

    @on(ActionItem.Activated)
    def _action_activated(self, event: ActionItem.Activated) -> None:
        """Dispatch an activated action item to its handler."""
        getattr(self, f"action_{event.action}")()


class EditModal(_FormModal[tuple[str, str, bool] | None]):
    """Edit one setting: a chooser for choices and bools, a text box otherwise.

    The result is `(action, value, to_repo)` with the action "save" or "unset", or
    None on cancel.
    """

    # No Enter to save: Enter on a chooser selects the highlighted option.
    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel")]
    CSS = (
        FORM_CSS
        + """
    EditModal { align: center middle; }
    #edit-box {
        width: 70; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #edit-title { text-style: bold; }
    #edit-description { padding-top: 1; color: $text-muted; }
    #edit-actions { padding-top: 1; height: auto; }
    """
    )

    def __init__(
        self,
        setting: ConfigSetting,
        *,
        typeahead: list[str] | None = None,
        fetch: Callable[[], list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._setting = setting
        self._done = False
        # For a big open list (model ids): the cached suggestions now, a blocking fetch after.
        self._typeahead = typeahead
        self._fetch = fetch

    def on_mount(self) -> None:
        """Focus the value field and start the suggestion fetch."""
        self.query_one("#edit-value").focus()
        if self._fetch is not None:
            self.run_worker(self._fetch_worker, thread=True)

    def _fetch_worker(self) -> None:
        """Fetch the live suggestions off the UI thread."""
        models = self._fetch() if self._fetch else []
        if models:
            self.app.call_from_thread(self._apply_suggestions, models)

    def _apply_suggestions(self, models: list[str]) -> None:
        """Replace the typeahead's suggestions."""
        field = self.query("#edit-value").first()
        if isinstance(field, TypeaheadField):
            field.set_suggestions(models)

    def compose(self) -> ComposeResult:
        """Yield the title, the description, the value field, the target and the actions."""
        s = self._setting
        with VerticalScroll(id="edit-box"):
            yield Static(f"Edit {s.key}", id="edit-title")
            yield Static(
                Text(
                    f"type={s.py_type}  ·  default={format_value(s.default)}  ·  source={s.source}",
                    style="dim",
                )
            )
            if s.description:
                # Text, never markup: a description names `[git]`.
                yield Static(Text(plain_description(s.description)), id="edit-description")
            # The table's display form of a list is not valid TOML, so the box is prefilled with
            # the exact inverse of parse_cli_value; scalars stay bare.
            raw = s.value if s.value is not None else s.default
            current = (
                raw
                if isinstance(raw, str)
                else (
                    format_toml_value(raw)
                    if isinstance(raw, (list, tuple, dict))
                    else format_value(raw)
                )
            )
            if self._typeahead is not None:
                yield TypeaheadField(
                    "" if s.value is None else current,
                    self._typeahead,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.choices is not None:
                yield ChoiceField(
                    tuple(s.choices),
                    current,
                    allow_custom=True,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.py_type == "bool":
                yield ChoiceField(
                    ("true", "false"),
                    current if current in ("true", "false") else "false",
                    id="edit-value",
                    classes="edit-gap",
                )
            else:
                initial = "" if s.value is None else current
                # A known provider still on the generic default gets its preset host offered.
                if not s.modified and (preset_url := _provider_preset_base_url(s.key)):
                    initial = preset_url
                yield Input(
                    value=initial,
                    placeholder=str(format_value(s.default)),
                    id="edit-value",
                    classes="edit-input edit-gap",
                )
            yield Static("save to", classes="edit-label")
            target = "repo config" if s.source == "repo" else "global config"
            yield ChoiceField(("global config", "repo config"), target, id="edit-target")
            with Horizontal(id="edit-actions"):
                yield ActionItem("Save", "save")
                yield ActionItem("Unset override", "unset")
                yield ActionItem("Cancel", "cancel")
            yield Static(
                Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _new_value(self) -> str:
        """Return the field's value as the TOML text the writer takes."""
        field = self.query_one("#edit-value")
        value = field.value if isinstance(field, (ChoiceField, TypeaheadField, Input)) else ""
        return format_toml_value(value) if self._setting.py_type == "str" else value

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        """Advance from a text field on Enter, like Tab."""
        focus_neighbor(event.input, 1)

    def action_save(self) -> None:
        """Return the save, once; Enter may reach both an action item and Submitted."""
        if self._done:
            return
        self._done = True
        to_repo = self.query_one("#edit-target", ChoiceField).index == 1
        self.dismiss(("save", self._new_value(), to_repo))

    def action_unset(self) -> None:
        """Return the unset."""
        if not self._done:
            self._done = True
            self.dismiss(("unset", "", False))

    def action_cancel(self) -> None:
        """Return None for a cancel."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the dialog."""
        if event.widget is self:
            self.action_cancel()


class ProviderModal(_FormModal[None]):
    """Add a `[providers.<name>]` entry through a form.

    Choosers for the fixed-choice fields, inputs for the free ones; base_url and
    auth default from the format and deployment when blank. Add writes and
    validates, and an error stays in the form to fix.
    """

    # No Enter to add: Enter on a chooser selects the highlighted option.
    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel")]
    CSS = (
        FORM_CSS
        + """
    ProviderModal { align: center middle; }
    #prov-box {
        width: 80; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #prov-title { text-style: bold; }
    #prov-actions { padding: 1 0 0 0; height: auto; }
    """
    )

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self._repo = repo_root
        self._autofilled_baseurl = ""  # the last prefilled base_url; a typed one is never replaced

    def on_mount(self) -> None:
        """Focus the name field."""
        self.query_one("#prov-name", Input).focus()

    def compose(self) -> ComposeResult:
        """Yield the title, the fields, the target and the actions."""
        choices = provider_choices()
        with VerticalScroll(id="prov-box"):
            yield Static("Add provider", id="prov-title")
            # Split at the sentence: one line is wider than the box.
            yield Static(
                Text(
                    "A [providers.<name>] block.\n"
                    "base_url/auth default from the format + deployment when left blank.",
                    style="dim",
                )
            )
            yield Input(
                placeholder="name  (e.g. openrouter, my-azure)",
                id="prov-name",
                classes="edit-input edit-gap",
            )
            yield Static("api_format", classes="edit-label")
            yield ChoiceField(
                tuple(choices["api_format"]), choices["api_format"][0], id="prov-format"
            )
            yield Static("deployment", classes="edit-label")
            yield ChoiceField(
                tuple(choices["deployment"]), choices["deployment"][0], id="prov-deployment"
            )
            yield Static("base_url", classes="edit-label")
            yield Input(
                placeholder="blank = default for the format/deployment",
                id="prov-baseurl",
                classes="edit-input",
            )
            yield Static("api_key_env", classes="edit-label")
            yield Input(
                placeholder="blank = secrets.toml by provider name",
                id="prov-keyenv",
                classes="edit-input",
            )
            yield Static("save to", classes="edit-label")
            yield ChoiceField(("global config", "repo config"), "global config", id="prov-target")
            with Horizontal(id="prov-actions"):
                yield ActionItem("Add", "add")
                yield ActionItem("Cancel", "cancel")
            yield Static(
                Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _selected(self, widget_id: str, fallback: str) -> str:
        """Return a chooser's value, or the fallback when it has none."""
        field = self.query_one(widget_id, ChoiceField)
        return field.value or fallback

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        """Advance from a text field on Enter, like Tab."""
        focus_neighbor(event.input, 1)

    @on(Input.Changed, "#prov-name")
    def _prefill_from_preset(self, event: Input.Changed) -> None:
        """Prefill a known provider's api_format and base_url, as `agent6 connect` does.

        Only a blank or autofilled base_url is overwritten, never a typed one.
        """
        preset = PROVIDER_DEFAULTS.get(event.value.strip())
        baseurl = self.query_one("#prov-baseurl", Input)
        if preset is None:
            if self._autofilled_baseurl and baseurl.value == self._autofilled_baseurl:
                baseurl.value = ""
            self._autofilled_baseurl = ""
            return
        self.query_one("#prov-format", ChoiceField).select_value(preset["api_format"])
        if baseurl.value in ("", self._autofilled_baseurl):
            self._autofilled_baseurl = preset.get("base_url", "")
            baseurl.value = self._autofilled_baseurl

    def action_add(self) -> None:
        """Write and validate the entry; an error stays in the form."""
        name = self.query_one("#prov-name", Input).value.strip()
        if not name:
            self.notify("Enter a provider name.", severity="warning")
            return
        fields: dict[str, ConfigLeafValue] = {
            "api_format": self._selected("#prov-format", "anthropic")
        }
        dep = self._selected("#prov-deployment", "direct")
        if dep != "direct":
            fields["deployment"] = dep
        base = self.query_one("#prov-baseurl", Input).value.strip()
        if base:
            fields["base_url"] = base
        keyenv = self.query_one("#prov-keyenv", Input).value.strip()
        if keyenv:
            fields["api_key_env"] = keyenv
        to_repo = self.query_one("#prov-target", ChoiceField).index == 1
        try:
            err = set_config_leaves(self._repo, f"providers.{name}", fields, to_repo=to_repo)
        except OperatorError as exc:
            err = str(exc)  # an unwritable config file is a form error, not a crash
        if err:
            self.notify(f"Invalid: {err}", severity="error", timeout=8.0)
            return
        self.notify(f"Set provider '{name}'.")
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the form."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the dialog."""
        if event.widget is self:
            self.action_cancel()


class ConfigScreen(ScreenChrome, Screen[None]):
    """The config viewer and editor: per-section tables, a filter, provenance, edit and unset."""

    CSS = """
    ConfigScreen { layers: base dropdown; background: $surface; }
    /* One slim row: inline filter (left) + count/modified tag (right). */
    #topbar { height: 1; padding: 0 1; }
    #search { width: 1fr; height: 1; border: none; background: transparent; padding: 0; }
    #search:focus { background: $primary 25%; }  /* $boost is transparent */
    #status { width: auto; height: 1; color: $text-muted; content-align: right middle; }
    /* Sections sit on a surface card; the panel-coloured bars (the one pinned
       column header + each section's title row) carry the structure -- the same
       colour as the top menu bar and footer. No zebra. */
    /* The header + sections share ONE rounded card (matching the home runs table
       and the dashboard panels); the border tracks focus via :focus-within. */
    #config-card { height: 1fr; border: round $primary; background: $surface; }
    #config-card:focus-within { border: round $accent; }
    #settings { height: 1fr; background: $surface; }
    .section-table { height: auto; margin: 0; background: transparent; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    /* The ONE pinned column header (a panel bar) -- section tables hide theirs. */
    #col-header { height: 1; background: $panel; }
    #col-header > .datatable--header {
        background: $panel; color: $foreground; text-style: bold;
    }
    /* A selection bar marks only the focused table's row. */
    .section-table > .datatable--cursor { background: transparent; color: $foreground; }
    .section-table:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    /* Flat sections; each title is a panel bar (matching the column header) that
       separates the sections. The title gets its OWN focus bar so arrow nav onto
       a header stays visible (otherwise focus there looks "lost"). */
    ConfigScreen Collapsible {
        background: transparent; border: none; padding: 0; margin: 0;
    }
    ConfigScreen CollapsibleTitle {
        padding: 0 1; color: $foreground; text-style: bold; background: $panel;
    }
    ConfigScreen CollapsibleTitle:focus { background: $primary 40%; color: $text; }
    ConfigScreen Collapsible > Contents { padding: 0; }
    """
    MENUS: ClassVar = (
        Menu(
            "Config",
            (
                MenuItem("Refresh", "reload"),
                MenuItem("Back", "close"),
                MenuItem("Quit", "quit"),
            ),
        ),
        Menu(
            "Edit",
            (
                MenuItem("Edit setting…", "edit"),
                MenuItem("Add provider…", "add_provider"),
                MenuItem("Unset override", "reset"),
            ),
        ),
        Menu(
            "View",
            (
                MenuItem("Filter", "search"),
                MenuItem("Modified only", "toggle_modified"),
                MenuItem("Theme…", "choose_theme"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help"),
                MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    # Page actions first, then Help and Back, the order of the home and run footers.
    FOOTER: ClassVar = tuple(
        (a.id, a.label)
        for a in CONFIG_ACTIONS
        if a.id in {"search", "toggle_modified", "edit", "reload", "help", "close"}
    )
    BINDINGS: ClassVar = menu_bindings("config", MENUS, footer=FOOTER)
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 config — keys & actions"
    HELP_HINTS: ClassVar = ("Enter edits the selected setting",)

    def __init__(self, repo_root: Path, config_path: Path | None = None) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.config_path = config_path
        self._eff: EffectiveConfig | None = None
        self._view: ConfigView | None = None
        self._table_rows: dict[str, list[ConfigSetting]] = {}  # per section, in row order
        self._modified_only = False

    def palette_commands(self) -> Iterator[PaletteCommand]:
        """Yield the menu actions with the registry's descriptions as their help."""
        descriptions = {a.id: a.description for a in CONFIG_ACTIONS}
        for label, handler, menu_title in menu_palette_commands(self, self.MENUS):
            action = next((i.action for m in self.MENUS for i in m.items if i.label == label), "")
            yield label, handler, descriptions.get(action, menu_title)

    def compose(self) -> ComposeResult:
        """Yield the menu bar, the filter row, the pinned header and one table per section.

        The sections are fixed once the view loads; a reload only repopulates rows.
        """
        self._rebuild_view()
        yield MenuBar(self.MENUS)
        with Horizontal(id="topbar"):
            yield Input(placeholder="/  filter settings…", id="search")
            yield Static("", id="status")
        # One pinned column header; the section tables hide theirs and share its widths.
        with Vertical(id="config-card"):
            yield DataTable(id="col-header")
            with VerticalScroll(id="settings"):
                for section in self._sections():
                    table = _NavTable(id=f"tbl-{section}", classes="section-table")
                    yield Collapsible(
                        table, title=escape(f"[{section}]"), collapsed=False, id=f"sec-{section}"
                    )
        yield Footer()

    def _sections(self) -> tuple[str, ...]:
        """Return the view's sections, in order."""
        return self._view.sections if self._view is not None else ()

    def _rebuild_view(self) -> None:
        """Load the effective config and build the view over it."""
        eff = load_effective(self.repo_root, self.config_path)
        self._eff = eff
        self._view = build_config_view(eff, resolved=resolved_config_values(eff.config))

    def _reload(self) -> bool:
        """Re-read the config and repaint; a config invalid on disk keeps the last-good view.

        Returns:
            Whether the re-read succeeded.
        """
        try:
            self._rebuild_view()
        except ConfigError as exc:
            self.notify(
                f"config is invalid on disk; showing the last-loaded values."
                f" Run `agent6 config fix` (or edit the file). {exc}",
                severity="error",
                timeout=10.0,
            )
            return False
        self._refresh()
        return True

    def on_mount(self) -> None:
        """Set the shared columns, fill the rows and focus the first table."""
        header = self.query_one("#col-header", DataTable)
        header.show_cursor = False
        header.can_focus = False
        columns = _columns(self._view.settings if self._view is not None else ())
        for label, width in columns:
            header.add_column(label, width=width)
        for section in self._sections():
            table = self.query_one(f"#tbl-{section}", DataTable)
            table.cursor_type = "row"
            table.show_header = False
            for label, width in columns:
                table.add_column(label, width=width)
        self._refresh()
        self.app.sub_title = f"config · {self.repo_root.name}"
        tables = list(self.query(_NavTable))
        if tables:
            tables[0].focus()

    def _matches(self, s: ConfigSetting, query: str) -> bool:
        """Return whether a setting passes the filter box and the modified-only toggle."""
        if self._modified_only and not s.modified:
            return False
        return query in s.key.lower() if query else True

    def _refresh(self) -> None:
        """Repopulate every section's rows from the view through the filters."""
        if self._view is None:
            return
        focused = self.focused
        parent = getattr(focused, "parent", None)
        focused_section: str | None = None
        if isinstance(focused, _NavTable) and focused.id:
            focused_section = focused.id[4:]
        elif isinstance(parent, Collapsible) and parent.id and parent.id.startswith("sec-"):
            focused_section = parent.id[4:]
        query = self.query_one("#search", Input).value.strip().lower()
        by_section: dict[str, list[ConfigSetting]] = {}
        for s in self._view.settings:
            if self._matches(s, query):
                by_section.setdefault(s.section, []).append(s)
        shown = 0
        for section in self._view.sections:
            rows = by_section.get(section, [])
            self._table_rows[section] = rows
            table = self.query_one(f"#tbl-{section}", DataTable)
            table.clear()
            for s in rows:
                leaf = _leaf(s)
                src = s.source + (" *" if s.modified else "")
                # Text, never markup: a value or key may carry brackets.
                table.add_row(Text(leaf), Text(display_value(s)), Text(src), key=s.key)
            # Pinned to its row count so only #settings scrolls; height:auto clamps to the
            # viewport in a short window and gives a second scrollbar.
            table.styles.height = max(1, len(rows))
            self.query_one(f"#sec-{section}", Collapsible).display = bool(rows)
            shown += len(rows)
        flt = "   ·   modified only" if self._modified_only else ""
        self.query_one("#status", Static).update(f"{shown} setting{'' if shown == 1 else 's'}{flt}")
        if focused_section is not None and focused_section not in self._ordered_sections():
            self._focus_first_setting()

    def _current_setting(self) -> ConfigSetting | None:
        """Return the setting under the focused table's cursor."""
        focused = self.focused
        if isinstance(focused, DataTable) and focused.id and focused.id.startswith("tbl-"):
            section = focused.id[4:]
            rows = self._table_rows.get(section, [])
            row = focused.cursor_row
            if 0 <= row < len(rows):
                return rows[row]
        return None

    def _ordered_sections(self) -> list[str]:
        """Return the sections with a visible Collapsible, in display order."""
        return [s for s in self._sections() if self.query_one(f"#sec-{s}", Collapsible).display]

    def _section_has_rows(self, section: str) -> bool:
        """Return whether the section is expanded with rows to step into."""
        col = self.query_one(f"#sec-{section}", Collapsible)
        return not col.collapsed and self.query_one(f"#tbl-{section}", _NavTable).row_count > 0

    def _focus_title(self, section: str) -> None:
        """Focus a section's header, scrolling it into view by one row."""
        col = self.query_one(f"#sec-{section}", Collapsible)
        title = next(iter(col.query("CollapsibleTitle")), None)
        if title is not None:
            title.focus(scroll_visible=False)
            self.scroll_focused_into_view(title)  # passed: .focused updates async

    def _focus_table(self, section: str, *, top: bool) -> None:
        """Focus a section's table at its top or bottom row."""
        table = self.query_one(f"#tbl-{section}", _NavTable)
        table.move_cursor(row=0 if top else table.row_count - 1)
        table.focus(scroll_visible=False)
        self.scroll_focused_into_view(table)

    def scroll_focused_into_view(self, target: Widget | None = None) -> None:
        """Scroll the settings just enough to show one row, a table cursor or a section header.

        Textual's focus auto-scroll brings a whole section into view and jumps at the edges.

        Args:
            target: The row just focused; `focus()` updates `self.focused` asynchronously,
                so reading it here would scroll the old row.
        """
        settings = self.query_one("#settings", VerticalScroll)
        focused = target if target is not None else self.focused
        if focused is None:
            return
        if isinstance(focused, _NavTable):
            screen_y = focused.region.y + focused.cursor_row  # the header is hidden
        elif isinstance(focused.parent, Collapsible):
            # scroll_to_region leaves the topmost header a line off the top: pin to home.
            first = next((c for c in self.query("#settings Collapsible") if c.display), None)
            if focused.parent is first:
                settings.scroll_home(animate=False)
                return
            screen_y = focused.region.y
        else:
            return
        content_y = screen_y - settings.region.y + settings.scroll_offset.y
        settings.scroll_to_region(Region(0, content_y, 1, 1), animate=False)

    def nav_from_table(self, section: str, direction: int) -> None:
        """Take an arrow from a table's edge: Down to the next header, Up to this section's."""
        order = self._ordered_sections()
        if section not in order:
            return
        i = order.index(section)
        if direction > 0:
            if i + 1 < len(order):
                self._focus_title(order[i + 1])
        else:
            self._focus_title(section)

    def _nav_from_title(self, section: str, direction: int) -> None:
        """Take an arrow on a header: Down into its rows, Up to the previous section's."""
        order = self._ordered_sections()
        if section not in order:
            return
        i = order.index(section)
        if direction > 0:
            if self._section_has_rows(section):
                self._focus_table(section, top=True)
            elif i + 1 < len(order):
                self._focus_title(order[i + 1])
        elif i - 1 >= 0:
            prev = order[i - 1]
            if self._section_has_rows(prev):
                self._focus_table(prev, top=False)
            else:
                self._focus_title(prev)
        else:  # Up at the topmost header goes back to the filter box
            self.query_one("#search", Input).focus()

    def on_key(self, event: events.Key) -> None:
        """Step Down out of the filter box; on a header, arrows flow through and Space toggles."""
        focused = self.focused
        if isinstance(focused, Input) and focused.id == "search":
            if event.key == "down":
                self._focus_first_setting()
                event.stop()
            return
        parent = getattr(focused, "parent", None)
        if not (isinstance(parent, Collapsible) and parent.id and parent.id.startswith("sec-")):
            return
        if event.key in ("up", "down"):
            self._nav_from_title(parent.id[4:], 1 if event.key == "down" else -1)
            event.stop()
        elif event.key == "space":
            parent.collapsed = not parent.collapsed
            event.stop()

    def action_search(self) -> None:
        """Focus the filter box."""
        self.query_one("#search", Input).focus()

    def _focus_first_setting(self) -> None:
        """Focus the first visible section's first row, or its header."""
        order = self._ordered_sections()
        if order:
            first = order[0]
            if self._section_has_rows(first):
                self._focus_table(first, top=True)
            else:
                self._focus_title(first)
        else:
            self.query_one("#search", Input).focus()

    def _cancel_search(self) -> bool:
        """Clear an active filter and drop back to the settings.

        Returns:
            Whether there was a filter to back out of; False lets Esc close the page.
        """
        box = self.query_one("#search", Input)
        if not box.value and (self.focused is not box or not self._ordered_sections()):
            return False
        box.value = ""
        self._refresh()
        self._focus_first_setting()
        return True

    def action_toggle_modified(self) -> None:
        """Flip the modified-only filter."""
        self._modified_only = not self._modified_only
        self._refresh()

    def action_reload(self) -> None:
        """Re-read the config from disk."""
        if self._reload():
            self.notify("Config reloaded.")

    def action_quit(self) -> None:
        """Quit the app; `quit` is not built into a Screen."""
        self.app.exit()

    def action_close(self) -> None:
        """Back out of an active filter first, then leave the page."""
        if self._cancel_search():
            return
        self.dismiss(None)

    def action_add_provider(self) -> None:
        """Open the provider form and reload after it."""

        def reload_config(_: None) -> None:
            self._reload()

        self.app.push_screen(ProviderModal(self.repo_root), reload_config)

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        """Edit the row on Enter or a double click."""
        self.action_edit()

    def action_edit(self) -> None:
        """Open the editor on the selected setting and write its result."""
        setting = self._current_setting()
        if setting is None:
            self.notify(
                "Select a setting first (click a row or Tab into a table).", severity="warning"
            )
            return

        def _done(result: tuple[str, str, bool] | None) -> None:
            if result is None:
                return
            action, raw, to_repo = result
            if action == "unset":
                self._unset(setting)
                return
            try:
                err = set_config_value(self.repo_root, setting.key, raw, to_repo=to_repo)
            except OperatorError as exc:
                err = str(exc)
            if err:
                self.notify(err, severity="error", timeout=8.0)
            else:
                self.notify(f"Set {setting.key}")
                self._reload()

        # A model-id field gets a typeahead over the provider's models, cached now and live after.
        eff = self._eff
        provider = model_role_provider(eff, setting.key) if eff is not None else None
        if provider is not None and eff is not None:
            modal = EditModal(
                setting,
                typeahead=cached_models(provider),
                fetch=lambda: config_value_choices(eff, setting.key),
            )
        else:
            modal = EditModal(setting)
        self.app.push_screen(modal, _done)

    def action_reset(self) -> None:
        """Unset the selected setting."""
        setting = self._current_setting()
        if setting is None:
            self.notify("Select a setting first.", severity="warning")
            return
        self._unset(setting)

    def _unset(self, setting: ConfigSetting) -> None:
        """Remove the setting from the layer that set it, and say so.

        A leaf from a preset or a `--config` file is modified but outside the two
        files an unset writes.
        """
        if not setting.modified:
            self.notify(f"{setting.key} is already at its default.")
            return
        if setting.source not in ("global", "repo"):
            self.notify(
                f"{setting.key} comes from the {setting.source} layer;"
                " unset edits only the global and repo config.",
                severity="warning",
            )
            return
        try:
            err = unset_config_value(
                self.repo_root, setting.key, to_repo=setting.source == "repo"
            ).error
        except OperatorError as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
        else:
            self.notify(f"Unset {setting.key} from {setting.source} config")
            self._reload()

    @on(Input.Changed, "#search")
    def _on_search(self) -> None:
        """Refilter as the box changes."""
        self._refresh()

    @on(Input.Submitted, "#search")
    def _on_search_submit(self) -> None:
        """Step into the settings on Enter, keeping the filter."""
        self._focus_first_setting()
