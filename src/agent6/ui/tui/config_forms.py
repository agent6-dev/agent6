# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config page's forms for settings, named entries and the reviewer roster."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import textual
from rich import text
from textual import app, binding, containers, events
from textual import screen as textual_screen
from textual import widgets as textual_widgets

from agent6 import errors
from agent6.config import ConfigError, io, notices, write
from agent6.ui.tui import forms, modals
from agent6.viewmodel import config_view


def _provider_preset_base_url(key: str) -> str:
    """Return the preset base_url for a known provider's `base_url` setting, else ""."""
    parts = io.split_key(key)
    if len(parts) == 3 and parts[0] == "providers" and parts[2] == "base_url":
        return write.PROVIDER_DEFAULTS.get(parts[1], {}).get("base_url", "")
    return ""


class FormModal[ResultT](textual_screen.ModalScreen[ResultT]):
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


class EditModal(FormModal[tuple[str, str, bool] | None]):
    """Edit one setting: a chooser for choices and bools, a text box otherwise.

    The result is `(action, value, to_repo)` with the action "save", "unset", "add_entry" or
    "remove_entry", or None on cancel. `entry_label` names the table the setting belongs to
    (a provider, an MCP server, a skill or the reviewer roster) and offers its Add and Remove.
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

    def __init__(self, setting: config_view.ConfigSetting, *, entry_label: str = "") -> None:
        super().__init__()
        self._setting = setting
        self._entry_label = entry_label
        self._done = False

    def on_mount(self) -> None:
        """Focus the value field."""
        self.query_one("#edit-value").focus()

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
            # Text controls edit strings without quotes; structured values use TOML.
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
            if s.choices is not None:
                yield forms.ChoiceField(
                    tuple(s.choices),
                    current,
                    allow_custom=not bool(s.choice_meanings),
                    meanings=s.choice_meanings,
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
                if self._entry_label:
                    yield forms.ActionItem(f"Add {self._entry_label}…", "add_entry")
                    yield forms.ActionItem(f"Remove {self._entry_label}", "remove_entry")
                yield forms.ActionItem("Cancel", "cancel")
            yield textual_widgets.Static(
                text.Text("↑↓ highlight · Space select · Tab field · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def _new_value(self) -> str:
        """Return the field's value as the TOML text the writer takes."""
        field = self.query_one("#edit-value")
        value = field.value if isinstance(field, (forms.ChoiceField, textual_widgets.Input)) else ""
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

    def action_add_entry(self) -> None:
        """Hand the page the add: it opens the table's form."""
        if not self._done:
            self._done = True
            self.dismiss(("add_entry", "", False))

    def action_remove_entry(self) -> None:
        """Hand the page the remove: it confirms, or picks the roster seat."""
        if not self._done:
            self._done = True
            self.dismiss(("remove_entry", "", False))

    def action_cancel(self) -> None:
        """Return None for a cancel."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the dialog."""
        if event.widget is self:
            self.action_cancel()


class ReviewerModal(FormModal[None]):
    """Append one reviewer through the validated list writer."""

    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = (
        forms.FORM_CSS
        + """
    ReviewerModal, RemoveReviewerModal { align: center middle; }
    .reviewer-box {
        width: 80; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    .reviewer-actions { padding-top: 1; height: auto; }
    """
    )

    def __init__(
        self,
        repo_root: pathlib.Path,
        config_paths: tuple[pathlib.Path, ...] = (),
        *,
        preset: str = "",
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_paths = config_paths
        self._preset = preset

    def compose(self) -> app.ComposeResult:
        """Yield the reviewer fields and the save target."""
        choices = write.reviewer_choices()
        with containers.VerticalScroll(classes="reviewer-box"):
            yield textual_widgets.Static("Add reviewer")
            yield textual_widgets.Static(
                "Focus is required. Set model to provider/model, or leave it blank.\n"
                "Blank optional fields inherit the route's settings."
            )
            for field in write.REVIEWER_FIELDS:
                yield textual_widgets.Static(field, classes="edit-label")
                if field in choices:
                    options = ("blank = inherit", *choices[field])
                    yield forms.ChoiceField(options, options[0], id=f"reviewer-{field}")
                else:
                    yield textual_widgets.Input(id=f"reviewer-{field}", classes="edit-input")
            yield textual_widgets.Static("save to", classes="edit-label")
            yield forms.ChoiceField(
                ("global config", "repo config"), "global config", id="reviewer-target"
            )
            with containers.Horizontal(classes="reviewer-actions"):
                yield forms.ActionItem("Add", "add")
                yield forms.ActionItem("Cancel", "cancel")

    def on_mount(self) -> None:
        """Focus the required field."""
        self.query_one("#reviewer-focus", textual_widgets.Input).focus()

    @textual.on(textual_widgets.Input.Submitted)
    def _input_submitted(self, event: textual_widgets.Input.Submitted) -> None:
        """Advance from a text field on Enter."""
        forms.focus_neighbor(event.input, 1)

    def action_add(self) -> None:
        """Append the reviewer, keeping an invalid form open."""
        choices = write.reviewer_choices()
        fields: dict[str, str] = {}
        for field in write.REVIEWER_FIELDS:
            control = self.query_one(f"#reviewer-{field}")
            if isinstance(control, forms.ChoiceField):
                fields[field] = control.value if control.value in choices[field] else ""
            elif isinstance(control, textual_widgets.Input):
                fields[field] = control.value
        to_repo = self.query_one("#reviewer-target", forms.ChoiceField).index == 1
        try:
            err = write.add_reviewer(self._repo, fields, to_repo=to_repo, preset=self._preset)
        except errors.OperatorError as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
            return
        self.notify(
            notices.write_message(
                self._repo,
                [write.preset_key(self._preset, "review.reviewers")],
                "Added reviewer.",
                to_repo=to_repo,
                config_paths=self._config_paths,
            )
        )
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the form."""
        self.dismiss(None)


class RemoveReviewerModal(FormModal[None]):
    """Choose a position and confirm removal from the displayed roster's source layer."""

    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = ReviewerModal.CSS

    def __init__(
        self,
        repo_root: pathlib.Path,
        focuses: tuple[str, ...],
        *,
        to_repo: bool,
        config_paths: tuple[pathlib.Path, ...] = (),
        preset: str = "",
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_paths = config_paths
        self._positions = tuple(f"{index + 1}: {focus}" for index, focus in enumerate(focuses))
        self._to_repo = to_repo
        self._preset = preset

    def compose(self) -> app.ComposeResult:
        """Yield the positional chooser and removal action."""
        with containers.VerticalScroll(classes="reviewer-box"):
            yield textual_widgets.Static("Remove reviewer")
            yield forms.ChoiceField(self._positions, self._positions[0], id="reviewer-position")
            with containers.Horizontal(classes="reviewer-actions"):
                yield forms.ActionItem("Remove…", "remove")
                yield forms.ActionItem("Cancel", "cancel")

    def action_remove(self) -> None:
        """Confirm the selected position before writing the remaining list."""
        index = self.query_one("#reviewer-position", forms.ChoiceField).index
        source = "repo" if self._to_repo else "global"

        def remove(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                err = write.remove_reviewer(
                    self._repo, index, to_repo=self._to_repo, preset=self._preset
                )
            except errors.OperatorError as exc:
                err = str(exc)
            if err:
                self.notify(err, severity="error", timeout=8.0)
                return
            self.notify(
                notices.write_message(
                    self._repo,
                    [write.preset_key(self._preset, "review.reviewers")],
                    f"Removed reviewer {index + 1} from {source} config.",
                    to_repo=self._to_repo,
                    config_paths=self._config_paths,
                )
            )
            self.dismiss(None)

        self.app.push_screen(
            modals.ConfirmModal(
                "Remove reviewer?",
                f"Remove {self._positions[index]} from {source} config.\n"
                "The remaining list is saved, including an empty list after the last reviewer.",
                confirm_label="Remove",
            ),
            remove,
        )

    def action_cancel(self) -> None:
        """Close the chooser."""
        self.dismiss(None)


class EntryModal(FormModal[None]):
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

    def __init__(
        self,
        repo_root: pathlib.Path,
        table: write.EntryTable,
        config_paths: tuple[pathlib.Path, ...] = (),
        *,
        preset: str = "",
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_paths = config_paths
        self._table = table
        self._preset = preset
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
                    if field not in ("backend", "state"):
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
        """Prefill a known provider's backend and base_url, as `agent6 connect` does.

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
        self.query_one("#entry-backend", forms.ChoiceField).select_value(preset["backend"])
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
            err = write.add_entry(
                self._repo, self._table, name, fields, to_repo=to_repo, preset=self._preset
            )
        except errors.OperatorError as exc:
            err = str(exc)  # an unwritable config file is a form error, not a crash
        if err:
            self.notify(f"Invalid: {err}", severity="error", timeout=8.0)
            return
        key = write.preset_key(self._preset, io.render_key((*io.split_key(self._table.path), name)))
        keys = (
            [key]
            if self._table.leaf
            else [
                io.render_key((*io.split_key(key), f)) for f, raw in fields.items() if raw.strip()
            ]
        )
        self.notify(
            notices.write_message(
                self._repo,
                keys,
                f"Set {self._table.label} '{name}'.",
                to_repo=to_repo,
                config_paths=self._config_paths,
            )
        )
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close the form."""
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the dialog."""
        if event.widget is self:
            self.action_cancel()


class AddPresetModal(FormModal[str | None]):
    """Create an empty `[presets.<name>]` table and select it; the name comes back on success."""

    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = (
        forms.FORM_CSS
        + """
    AddPresetModal { align: center middle; }
    #addpreset-box {
        width: 70; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #addpreset-title { text-style: bold; }
    #addpreset-actions { padding-top: 1; height: auto; }
    """
    )

    def __init__(
        self, repo_root: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_paths = config_paths

    def on_mount(self) -> None:
        """Focus the name field."""
        self.query_one("#addpreset-name", textual_widgets.Input).focus()

    def compose(self) -> app.ComposeResult:
        """Yield the title, the name field, the target and the actions."""
        with containers.VerticalScroll(id="addpreset-box"):
            yield textual_widgets.Static("Add preset", id="addpreset-title")
            yield textual_widgets.Static(
                text.Text(
                    "An empty [presets.<name>] table, selected as preset in the chosen layer;"
                    " edit the rows to fill it.",
                    style="dim",
                )
            )
            yield textual_widgets.Input(
                placeholder="name", id="addpreset-name", classes="edit-input edit-gap"
            )
            yield textual_widgets.Static("create in", classes="edit-label")
            yield forms.ChoiceField(
                ("global config", "repo config"), "global config", id="addpreset-target"
            )
            with containers.Horizontal(id="addpreset-actions"):
                yield forms.ActionItem("Add", "add")
                yield forms.ActionItem("Cancel", "cancel")

    @textual.on(textual_widgets.Input.Submitted, "#addpreset-name")
    def _input_submitted(self, event: textual_widgets.Input.Submitted) -> None:
        """Enter in the name field adds."""
        event.stop()
        self.action_add()

    def action_add(self) -> None:
        """Create the table and return its name."""
        name = self.query_one("#addpreset-name", textual_widgets.Input).value.strip()
        to_repo = self.query_one("#addpreset-target", forms.ChoiceField).index == 1
        try:
            err = write.add_preset(self._repo, name, to_repo=to_repo)
        except (ConfigError, errors.OperatorError) as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
            return
        layer = "repo" if to_repo else "global"
        self.notify(f"Created [presets.{name}] in the {layer} config and selected it there.")
        self.dismiss(name)

    def action_cancel(self) -> None:
        """Close the form."""
        self.dismiss(None)
