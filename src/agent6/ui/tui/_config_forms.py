# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config page's form modals: edit one setting, add a named entry."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import ClassVar

try:
    import textual
    from rich import text
    from textual import app, binding, containers, events
    from textual import screen as textual_screen
    from textual import widgets as textual_widgets
except ImportError as e:  # pragma: no cover - clear runtime message
    raise SystemExit("The config page needs textual: pip install 'agent6[tui]'") from e

from agent6 import errors
from agent6.config import io, write
from agent6.ui.tui import forms
from agent6.viewmodel import config_view


def _provider_preset_base_url(key: str) -> str:
    """Return the preset base_url for a known provider's `base_url` setting, else ""."""
    parts = key.split(".")
    if len(parts) == 3 and parts[0] == "providers" and parts[2] == "base_url":
        return write.PROVIDER_DEFAULTS.get(parts[1], {}).get("base_url", "")
    return ""


class _FormModal[ResultT](textual_screen.ModalScreen[ResultT]):
    """The form modals' shared keys: one vertical arrow chain, and left and right over the actions.

    An activated ActionItem dispatches to `action_<id>`.
    """

    def on_key(self, event: events.Key) -> None:
        """Move between fields and actions with the arrow keys."""
        focused = self.focused
        if event.key in ("left", "right") and isinstance(focused, forms.ActionItem):
            actions = list(self.query(forms.ActionItem))
            step = 1 if event.key == "right" else -1
            actions[(actions.index(focused) + step) % len(actions)].focus()
            event.stop()
        elif event.key == "up" and isinstance(focused, (textual_widgets.Input, forms.ActionItem)):
            forms.focus_neighbor(focused, -1)
            event.stop()
        elif event.key == "down" and isinstance(focused, textual_widgets.Input):
            forms.focus_neighbor(focused, 1)
            event.stop()

    @textual.on(forms.ActionItem.Activated)
    def _action_activated(self, event: forms.ActionItem.Activated) -> None:
        """Dispatch an activated action item to its handler."""
        getattr(self, f"action_{event.action}")()


class EditModal(_FormModal[tuple[str, str, bool] | None]):
    """Edit one setting: a chooser for choices and bools, a text box otherwise.

    The result is `(action, value, to_repo)` with the action "save" or "unset", or
    None on cancel.
    """

    # No Enter to save: Enter on a chooser selects the highlighted option.
    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = (
        forms.FORM_CSS
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
        setting: config_view.ConfigSetting,
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
        if isinstance(field, forms.TypeaheadField):
            field.set_suggestions(models)

    def compose(self) -> app.ComposeResult:
        """Yield the title, the description, the value field, the target and the actions."""
        s = self._setting
        with containers.VerticalScroll(id="edit-box"):
            yield textual_widgets.Static(f"Edit {s.key}", id="edit-title")
            yield textual_widgets.Static(
                text.Text(
                    f"type={s.py_type}  ·  default={config_view.format_value(s.default)}  "
                    f"·  source={s.source}",
                    style="dim",
                )
            )
            if s.description:
                # Text, never markup: a description names `[git]`.
                yield textual_widgets.Static(
                    text.Text(config_view.plain_description(s.description)), id="edit-description"
                )
            # The table's display form of a list is not valid TOML, so the box is prefilled with
            # the exact inverse of parse_cli_value; scalars stay bare.
            raw = s.value if s.value is not None else s.default
            current = (
                raw
                if isinstance(raw, str)
                else (
                    io.format_toml_value(raw)
                    if isinstance(raw, (list, tuple, dict))
                    else config_view.format_value(raw)
                )
            )
            if self._typeahead is not None:
                yield forms.TypeaheadField(
                    "" if s.value is None else current,
                    self._typeahead,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.choices is not None:
                yield forms.ChoiceField(
                    tuple(s.choices),
                    current,
                    allow_custom=True,
                    id="edit-value",
                    classes="edit-gap",
                )
            elif s.py_type == "bool":
                yield forms.ChoiceField(
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
                yield textual_widgets.Input(
                    value=initial,
                    placeholder=str(config_view.format_value(s.default)),
                    id="edit-value",
                    classes="edit-input edit-gap",
                )
            yield textual_widgets.Static("save to", classes="edit-label")
            target = "repo config" if s.source == "repo" else "global config"
            yield forms.ChoiceField(("global config", "repo config"), target, id="edit-target")
            with containers.Horizontal(id="edit-actions"):
                yield forms.ActionItem("Save", "save")
                yield forms.ActionItem("Unset override", "unset")
                yield forms.ActionItem("Cancel", "cancel")
            yield textual_widgets.Static(
                text.Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _new_value(self) -> str:
        """Return the field's value as the TOML text the writer takes."""
        field = self.query_one("#edit-value")
        value = (
            field.value
            if isinstance(field, (forms.ChoiceField, forms.TypeaheadField, textual_widgets.Input))
            else ""
        )
        return io.format_toml_value(value) if self._setting.py_type == "str" else value

    @textual.on(textual_widgets.Input.Submitted)
    def _input_submitted(self, event: textual_widgets.Input.Submitted) -> None:
        """Advance from a text field on Enter, like Tab."""
        forms.focus_neighbor(event.input, 1)

    def action_save(self) -> None:
        """Return the save, once; Enter may reach both an action item and Submitted."""
        if self._done:
            return
        self._done = True
        to_repo = self.query_one("#edit-target", forms.ChoiceField).index == 1
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


class EntryModal(_FormModal[None]):
    """Add a named entry through a registry-driven form.

    Choosers for the fixed-choice fields, inputs for the free ones; a blank field
    writes nothing, so an existing entry keeps its value. Add writes and validates,
    and an error stays in the form to fix.
    """

    # No Enter to add: Enter on a chooser selects the highlighted option.
    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = (
        forms.FORM_CSS
        + """
    EntryModal { align: center middle; }
    #entry-box {
        width: 80; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #entry-title { text-style: bold; }
    #entry-actions { padding: 1 0 0 0; height: auto; }
    """
    )

    def __init__(self, repo_root: pathlib.Path, table: write.EntryTable) -> None:
        super().__init__()
        self._repo = repo_root
        self._table = table
        self._autofilled_baseurl = ""  # the last prefilled base_url; a typed one is never replaced

    def on_mount(self) -> None:
        """Focus the name field."""
        self.query_one("#entry-name", textual_widgets.Input).focus()

    def compose(self) -> app.ComposeResult:
        """Yield the title, the fields, the target and the actions."""
        choices = write.entry_choices(self._table)
        with containers.VerticalScroll(id="entry-box"):
            yield textual_widgets.Static(f"Add {self._table.label}", id="entry-title")
            yield textual_widgets.Static(
                text.Text(
                    f"{self._table.path}.<name>\n"
                    "A blank field keeps an existing value or uses the default.",
                    style="dim",
                )
            )
            yield textual_widgets.Input(
                placeholder="name", id="entry-name", classes="edit-input edit-gap"
            )
            for field in self._table.fields:
                yield textual_widgets.Static(field, classes="edit-label")
                description = config_view.plain_description(
                    write.entry_field_description(self._table, field)
                )
                if field in choices:
                    options = tuple(choices[field])
                    if field not in ("api_format", "state"):
                        options = ("blank = unchanged, or default", *options)
                    chooser = forms.ChoiceField(options, options[0], id=f"entry-{field}")
                    chooser.tooltip = description
                    yield chooser
                else:
                    yield textual_widgets.Input(
                        placeholder=description, id=f"entry-{field}", classes="edit-input"
                    )
            yield textual_widgets.Static("save to", classes="edit-label")
            yield forms.ChoiceField(
                ("global config", "repo config"), "global config", id="entry-target"
            )
            with containers.Horizontal(id="entry-actions"):
                yield forms.ActionItem("Add", "add")
                yield forms.ActionItem("Cancel", "cancel")
            yield textual_widgets.Static(
                text.Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    @textual.on(textual_widgets.Input.Submitted)
    def _input_submitted(self, event: textual_widgets.Input.Submitted) -> None:
        """Advance from a text field on Enter, like Tab."""
        forms.focus_neighbor(event.input, 1)

    @textual.on(textual_widgets.Input.Changed, "#entry-name")
    def _prefill_from_preset(self, event: textual_widgets.Input.Changed) -> None:
        """Prefill a known provider's api_format and base_url, as `agent6 connect` does.

        Only a blank or autofilled base_url is overwritten, never a typed one.

        Args:
            event: The edit to the provider name field.
        """
        if self._table.path != "providers":
            return
        preset = write.PROVIDER_DEFAULTS.get(event.value.strip())
        baseurl = self.query_one("#entry-base_url", textual_widgets.Input)
        if preset is None:
            if self._autofilled_baseurl and baseurl.value == self._autofilled_baseurl:
                baseurl.value = ""
            self._autofilled_baseurl = ""
            return
        self.query_one("#entry-api_format", forms.ChoiceField).select_value(preset["api_format"])
        if baseurl.value in ("", self._autofilled_baseurl):
            self._autofilled_baseurl = preset.get("base_url", "")
            baseurl.value = self._autofilled_baseurl

    def action_add(self) -> None:
        """Write and validate the entry; an error stays in the form."""
        name = self.query_one("#entry-name", textual_widgets.Input).value.strip()
        fields: dict[str, str] = {}
        choices = write.entry_choices(self._table)
        for field in self._table.fields:
            control = self.query_one(f"#entry-{field}")
            if isinstance(control, forms.ChoiceField):
                fields[field] = control.value if control.value in choices[field] else ""
            elif isinstance(control, textual_widgets.Input):
                fields[field] = control.value
        to_repo = self.query_one("#entry-target", forms.ChoiceField).index == 1
        try:
            err = write.add_entry(self._repo, self._table, name, fields, to_repo=to_repo)
        except errors.OperatorError as exc:
            err = str(exc)  # an unwritable config file is a form error, not a crash
        if err:
            self.notify(f"Invalid: {err}", severity="error", timeout=8.0)
            return
        self.notify(f"Set {self._table.label} '{name}'.")
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the form."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the dialog."""
        if event.widget is self:
            self.action_cancel()
