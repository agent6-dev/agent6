# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI config page (textual Pilot).

Covers the bits a human would otherwise have to eyeball: the page loads + renders
every section's settings, search narrows them, the modified-only filter narrows
to overridden settings, and Help opens — all over the shared config view-model.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest
from textual import app as textual_app
from textual import widgets

from agent6 import paths
from agent6.config import OpenAIProviderEntry, layer
from agent6.models import cache
from agent6.models import choices as models_choices
from agent6.ui.tui import config_page, menubar
from agent6.viewmodel import config_view

_GLOBAL = """\
[providers.anthropic]
api_format = "anthropic"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[sandbox]
run_commands = "yes"
"""


class _Host(textual_app.App[None]):
    def __init__(self, repo_root: pathlib.Path, config_path: pathlib.Path | None = None) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_path = config_path

    def on_mount(self) -> None:
        self.push_screen(config_page.ConfigScreen(self._repo, self._config_path))


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return repo_root


def _row_total(screen: config_page.ConfigScreen) -> int:
    return sum(t.row_count for t in screen.query(widgets.DataTable))


def test_config_page_view_search_filter_help(repo: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)

            # Every section renders; the whole effective config is shown.
            total = _row_total(screen)
            assert total > 10
            # run_commands is set in the (global) config -> present + sourced.
            sandbox = screen.query_one("#tbl-sandbox", widgets.DataTable)
            assert any(
                "run_commands" in str(sandbox.get_row_at(r)[0]) for r in range(sandbox.row_count)
            )

            # Search narrows to matching keys.
            screen.query_one("#search", widgets.Input).value = "run_commands"
            screen._refresh()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            narrowed = _row_total(screen)
            assert 0 < narrowed < total
            assert narrowed == 1  # one key matches, and the count says "1 setting"
            status = str(screen.query_one("#status", widgets.Static).render())
            assert status.startswith("1 setting") and "1 settings" not in status

            # Modified-only filter: clear search, show only overridden settings.
            screen.query_one("#search", widgets.Input).value = ""
            screen.action_toggle_modified()
            await pilot.pause()
            modified = _row_total(screen)
            assert 0 < modified < total  # fewer than everything, but >0 (run_commands)

            # Help overlay opens from its action (also a button + ? key).
            screen.action_help()
            await pilot.pause()
            assert isinstance(app.screen, menubar.HelpScreen)

    asyncio.run(scenario())


def test_cli_changes_reach_the_tui_and_web_with_their_layer(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli import cli_main
    from agent6.ui.web import model

    monkeypatch.chdir(repo)
    assert cli_main(["config", "set", "review.period", "9"]) == 0
    assert cli_main(["config", "set", "--repo", "review.period", "11"]) == 0

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-review", widgets.DataTable)
            row = next(
                table.get_row_at(i)
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]).strip() == "period"
            )
            assert str(row[1]).strip() == "11"
            assert "repo" in str(row[2])
            web = model.config_payload(repo)
            assert web["review.period"]["value"] == 11
            assert web["review.period"]["source"] == "repo"

            assert cli_main(["config", "unset", "--repo", "review.period"]) == 0
            screen.action_reload()
            await pilot.pause()
            table = screen.query_one("#tbl-review", widgets.DataTable)
            row = next(
                table.get_row_at(i)
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]).strip() == "period"
            )
            assert str(row[1]).strip() == "9"
            assert "global" in str(row[2])
            web = model.config_payload(repo)
            assert web["review.period"]["value"] == 9
            assert web["review.period"]["source"] == "global"

    asyncio.run(scenario())


def test_config_page_adaptive_value_shown(repo: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            ctx = screen.query_one("#tbl-context", widgets.DataTable)
            # Adaptive compaction shows its resolved number tagged "(adaptive)".
            cells = [str(ctx.get_row_at(r)[1]) for r in range(ctx.row_count)]
            assert any("(adaptive)" in c for c in cells)

    asyncio.run(scenario())


def test_config_page_edit_persists(repo: pathlib.Path) -> None:
    """Select a row, Edit, the chooser, a new value, Save: the whole edit ask end to end."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            # run_commands is an enum -> a [x]/[ ] chooser, focused, current "yes".
            field = modal.query_one("#edit-value", tui_widgets.ChoiceField)
            assert field.value == "yes"
            await pilot.press("down")  # highlight "no" (selection unchanged)
            await pilot.pause()
            assert field.value == "yes"  # arrows only highlight
            await pilot.press("space")  # select "no"
            await pilot.pause()
            assert field.value == "no"
            modal.action_save()  # equivalent to the Save action
            await pilot.pause()
            # Persisted through config_layer.set_config_value (global config).
            assert layer.load_effective(repo).config.sandbox.run_commands == "no"

    asyncio.run(scenario())


def test_edit_defaults_to_the_setting_source_layer(repo: pathlib.Path) -> None:
    """Editing a repo-sourced value targets the repo config by default.

    Otherwise the repo layer masks the global write and Save appears to do nothing.
    """
    from agent6.config import write

    assert write.set_config_value(repo, "sandbox.run_commands", "no", to_repo=True) is None

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-sandbox", widgets.DataTable)
            table.focus()
            row = next(
                i for i in range(table.row_count) if "run_commands" in str(table.get_row_at(i)[0])
            )
            table.move_cursor(row=row)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            assert modal.query_one("#edit-target", tui_widgets.ChoiceField).value == "repo config"
            modal.query_one("#edit-value", tui_widgets.ChoiceField).select_value("ask")
            modal.action_save()
            await pilot.pause()
            assert layer.load_effective(repo).config.sandbox.run_commands == "ask"

    asyncio.run(scenario())


def test_edit_unset_reverts_to_default(repo: pathlib.Path) -> None:
    """The edit modal's "Unset → default" removes the override rather than writing the default."""

    async def scenario() -> None:

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            # run_commands is set to "yes" in the (global) config fixture.
            eff = layer.load_effective(repo)
            assert layer.effective_leaf(eff, "sandbox.run_commands") == ("yes", "global")
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            modal.action_unset()
            await pilot.pause()
            # Override removed -> back to the default, sourced as default.
            assert layer.effective_leaf(layer.load_effective(repo), "sandbox.run_commands") == (
                "ask",
                "default",
            )

    asyncio.run(scenario())


def test_edit_custom_value_inline(repo: pathlib.Path) -> None:
    """A choice setting's last chooser row is an inline custom field, typed right there."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            field = modal.query_one("#edit-value", tui_widgets.ChoiceField)
            # Highlight down to the custom row, then type in place: typing selects it.
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()
            for ch in ("z", "z", "z"):
                await pilot.press(ch)
            await pilot.pause()
            assert field.value == "zzz"
            assert modal._new_value() == "zzz"  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_edit_action_arrows_navigate(repo: pathlib.Path) -> None:
    """Left and Right move between the focused flat actions (Save, Unset, Cancel), wrapping."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            items = list(modal.query(tui_widgets.ActionItem))
            assert len(items) == 3  # Save, Unset, Cancel
            items[0].focus()
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert modal.focused is items[1]
            await pilot.press("left")
            await pilot.pause()
            assert modal.focused is items[0]
            await pilot.press("left")  # wrap past the start
            await pilot.pause()
            assert modal.focused is items[2]

    asyncio.run(scenario())


def test_provider_field_is_a_picker_of_configured_providers(repo: pathlib.Path) -> None:
    """Editing models.<role>.provider shows a chooser of the configured provider names."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-models", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if "worker" in str(tbl.get_row_at(r)[0]) and "provider" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            # A ChoiceField, not a plain Input: the configured providers were injected as choices.
            field = modal.query_one("#edit-value", tui_widgets.ChoiceField)
            assert field.value == "anthropic"

    asyncio.run(scenario())


def test_model_field_is_a_typeahead_picker(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing models.<role>.model opens a type-to-narrow picker over the provider's models."""
    models = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]

    def _models(*_a: object, **_k: object) -> list[str]:
        return models

    monkeypatch.setattr(cache, "cached_models", _models)
    monkeypatch.setattr(models_choices, "config_value_choices", _models)  # mock the live fetch

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-models", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "worker.model"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            field = modal.query_one("#edit-value", tui_widgets.TypeaheadField)
            assert field.value == "claude-sonnet-4-5"  # the current model
            # First keystroke replaces + narrows; arrow highlights a match.
            await pilot.press("h")
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert field.value == "claude-haiku-4-5"

    asyncio.run(scenario())


def test_empty_preset_prefill_saves_back_unchanged(repo: pathlib.Path) -> None:
    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-preset", widgets.DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            assert modal.query_one("#edit-value", tui_widgets.ChoiceField).value == ""
            modal.action_save()
            await pilot.pause()
            assert layer.load_effective(repo).config.preset == ""

    asyncio.run(scenario())


def test_list_setting_prefill_saves_back_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A list-valued setting prefills the edit box as the exact inverse of parse_cli_value.

    The display formatter's `[uv, run, pytest]` is not TOML, so an untouched Save failed
    revalidation: there was no form in which the shown value saved.
    """
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        _GLOBAL + '\n[harness]\nverify_command = ["uv", "run", "pytest"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    async def scenario() -> None:
        app = _Host(repo_root)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-harness", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "verify_command"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            field = modal.query_one("#edit-value", widgets.Input)
            assert field.value == '["uv", "run", "pytest"]'  # round-trippable TOML
            modal.action_save()  # untouched Save must succeed, not error
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)

    asyncio.run(scenario())
    saved = layer.load_effective(repo_root, None).config.harness.verify_command
    assert saved == ("uv", "run", "pytest")  # unchanged, not corrupted to a str


def test_string_setting_saves_toml_like_text_as_a_string(repo: pathlib.Path) -> None:
    """A free-text field's schema, not TOML-looking text, determines its type.

    Otherwise entering `true` parses as a bool and the rejected save disappears.
    """
    from agent6.config import write

    assert write.set_config_value(repo, "git.commit.name", "Agent Six") is None

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-git", widgets.DataTable)
            table.focus()
            row = next(
                i
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]).strip() == "commit.name"
            )
            table.move_cursor(row=row)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            modal.query_one("#edit-value", widgets.Input).value = "true"
            modal.action_save()
            await pilot.pause()
            assert layer.load_effective(repo).config.git.commit.name == "true"

    asyncio.run(scenario())


def test_editing_a_model_survives_a_broken_secrets_file(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """secrets.toml with unsafe perms does not crash the TUI from the edit modal's model fetch.

    The thread worker's SecretsError hit textual's default exit_on_error; the fetch degrades to
    a keyless attempt instead, as models/validate.py does.
    """
    gdir = paths.global_config_dir()
    secrets = gdir / "secrets.toml"
    secrets.write_text('[anthropic]\napi_key = "sk-x"\n', encoding="utf-8")
    secrets.chmod(0o644)  # group/other-readable -> load_secrets raises

    models = ["claude-sonnet-4-5"]

    def _models(*_a: object, **_k: object) -> list[str]:
        return models

    monkeypatch.setattr(cache, "cached_models", _models)
    monkeypatch.setattr(cache, "list_models", _models)  # the live fetch, keyless here

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-models", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "worker.model"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            # Let the thread worker run; the app must survive it.
            for _ in range(6):
                await pilot.pause(0.05)
            assert isinstance(app.screen, config_page.EditModal)  # still open, app alive

    asyncio.run(scenario())


def test_edit_modal_up_at_top_is_a_hard_stop(repo: pathlib.Path) -> None:
    """Up at the top of the first chooser stays there, not escaping to the scroll container."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if "run_commands" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            field = modal.query_one("#edit-value", tui_widgets.ChoiceField)
            assert modal.focused is field and field._cursor == 0  # pyright: ignore[reportPrivateUsage]
            await pilot.press("up")  # at the top edge
            await pilot.pause()
            assert modal.focused is field  # stayed put (didn't escape to the scroll box)
            await pilot.press("down")  # highlight still moves afterwards
            await pilot.pause()
            assert modal.focused is field and field._cursor == 1  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_q_backs_out_from_config_but_types_in_search(repo: pathlib.Path) -> None:
    """Q backs out of the Config screen, yet types normally in the focused search box.

    Only the root hub quits on q; the menu's Quit (^Q) still exits the app.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            exits: list[int] = []
            orig = app.exit
            app.exit = lambda *a, **k: exits.append(1) or orig(*a, **k)  # type: ignore[assignment]

            # In the search Input, q types (no quit).
            screen.action_search()
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert screen.query_one("#search", widgets.Input).value == "q"
            assert not exits

            # Out on a table, q backs out (dismiss) instead of quitting.
            screen._cancel_search()  # pyright: ignore[reportPrivateUsage]
            screen.query_one("#tbl-sandbox", widgets.DataTable).focus()
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not exits  # q did NOT quit
            assert not isinstance(app.screen, config_page.ConfigScreen)  # it backed out

            # The menu's Quit (^Q) path still quits the app.
            screen.action_quit()
            await pilot.pause()
            assert exits

    asyncio.run(scenario())


def test_view_menu_opens_theme_picker(repo: pathlib.Path) -> None:
    """The View>Theme item (and action_choose_theme) opens the theme picker."""

    async def scenario() -> None:
        from agent6.ui.tui import theme

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_choose_theme()
            await pilot.pause()
            assert isinstance(app.screen, theme.ThemePicker)

    asyncio.run(scenario())


def test_menu_bar_opens_and_dispatches(repo: pathlib.Path) -> None:
    """Opening a menu by mouse, Alt or F-key shows its items, and a pick runs the bound action."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            mb = screen.query_one(menubar.MenuBar)

            # The View menu's items carry their action ids; the dropdown mounts on the screen.
            mb.open("v")
            await pilot.pause()
            dd = next(iter(screen.query(widgets.OptionList)))
            ids = [dd.get_option_at_index(i).id for i in range(dd.option_count)]
            assert "search" in ids and "toggle_modified" in ids

            # Pick "Modified only" and follow the whole dispatch chain to action_toggle_modified.
            assert screen._modified_only is False  # pyright: ignore[reportPrivateUsage]
            idx = next(
                i
                for i in range(dd.option_count)
                if dd.get_option_at_index(i).id == "toggle_modified"
            )
            dd.highlighted = idx
            await pilot.press("enter")
            await pilot.pause()
            assert screen._modified_only is True  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_menu_reopen_no_duplicate(repo: pathlib.Path) -> None:
    """Switching and re-opening menus raises no DuplicateIds and converges to one open menu."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            mb = screen.query_one(menubar.MenuBar)
            for m in ("v", "e", "v", "v", "c"):  # a DuplicateIds regression
                mb.open(m)
                await pilot.pause()
            assert len(list(screen.query(widgets.OptionList))) == 1  # exactly one menu open
            mb.open("c")  # opening the open menu toggles it shut
            await pilot.pause()
            assert len(list(screen.query(widgets.OptionList))) == 0

    asyncio.run(scenario())


def test_menu_opens_on_mouse_click(repo: pathlib.Path) -> None:
    """A mouse click on a title opens its menu, visible and not clipped by the 1-row bar.

    events.Click carries no .widget, so each title handles its own click.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            await pilot.click("#menu-v")  # click the View title
            await pilot.pause()
            dds = list(screen.query(widgets.OptionList))
            assert len(dds) == 1
            dd = dds[0]
            # Floated on the screen, so it shows below the bar at full height, not clipped.
            assert dd.region.height > 1
            assert dd.region.y >= 1

    asyncio.run(scenario())


def test_menu_toggle_switch_and_click_away(repo: pathlib.Path) -> None:
    """A title click toggles its menu, another title switches, and a body click closes."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)

            def n() -> int:
                return len(list(screen.query(widgets.OptionList)))

            await pilot.click("#menu-e")
            await pilot.pause()
            assert n() == 1
            await pilot.click("#menu-e")  # same title again -> toggle shut
            await pilot.pause()
            assert n() == 0
            await pilot.click("#menu-e")
            await pilot.pause()
            assert n() == 1
            await pilot.click("#menu-v")  # different title -> switch
            await pilot.pause()
            assert n() == 1
            assert screen.query_one("#menu-v").has_class("-open")
            # A click elsewhere closes the dropdown through focus loss, driven directly here.
            screen.query_one("#tbl-sandbox", widgets.DataTable).focus()
            await pilot.pause()
            assert n() == 0
            assert not screen.query_one("#menu-v").has_class("-open")

    asyncio.run(scenario())


def test_menu_left_right_switches_open_menu(repo: pathlib.Path) -> None:
    """Left/Right move between menus while one is open (classic menu-bar feel)."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.query_one(menubar.MenuBar).open("e")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")
            await pilot.press("right")
            await pilot.pause()
            assert screen.query_one("#menu-v").has_class("-open")
            await pilot.press("left")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")

    asyncio.run(scenario())


def test_open_menu_title_stays_highlighted(repo: pathlib.Path) -> None:
    """The open menu's title carries the -open class, reading as active, and drops it on close."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            mb = screen.query_one(menubar.MenuBar)
            mb.open("e")
            await pilot.pause()
            assert screen.query_one("#menu-e").has_class("-open")
            mb.close_menu()
            await pilot.pause()
            assert not screen.query_one("#menu-e").has_class("-open")

    asyncio.run(scenario())


def test_config_actions_in_command_palette(repo: pathlib.Path) -> None:
    """Every Config action is searchable in the Ctrl+P palette under its descriptive menu label."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            labels = [name for name, _, _ in screen.palette_commands()]
            for expected in (
                "Filter",
                "Modified only",
                "Edit setting…",
                "Unset override",
                "Refresh",
                "Keys & actions",
            ):
                assert expected in labels
            # the terse footer labels do NOT leak into the palette
            assert "Help" not in labels and "Edit" not in labels

    asyncio.run(scenario())


def test_enter_on_setting_row_opens_editor(repo: pathlib.Path) -> None:
    """Enter (or double-click) on a setting row opens the edit modal.

    The DataTable consumes Enter for its own RowSelected, so it's wired via that event, not the
    screen's `enter` binding.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, config_page.EditModal)

    asyncio.run(scenario())


def test_esc_clears_filter_before_closing(repo: pathlib.Path) -> None:
    """Esc backs out of an active filter first; a later Esc closes the page."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            search = screen.query_one("#search", widgets.Input)
            screen.action_search()  # / focuses the inline filter
            await pilot.pause()
            assert screen.focused is search
            search.value = "run_commands"
            screen._refresh()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)  # NOT closed
            assert search.value == ""  # filter cleared
            assert screen.focused is not search  # focus dropped into the settings

    asyncio.run(scenario())


def test_filter_arrow_in_and_out(repo: pathlib.Path) -> None:
    """Down and Enter step out of the filter into the settings; Up from the top header returns."""

    async def scenario() -> None:

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            search = screen.query_one("#search", widgets.Input)
            screen.action_search()
            await pilot.pause()
            await pilot.press("down")  # step out into the settings
            await pilot.pause()
            assert isinstance(screen.focused, config_page._NavTable)
            # Up from the first row -> header -> Up again returns to the filter.
            await pilot.press("up")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert screen.focused is search

    asyncio.run(scenario())


def test_modified_filter_moves_focus_out_of_a_hidden_section(repo: pathlib.Path) -> None:
    """Turning on the modified-only filter moves focus when it hides the selected section.

    Otherwise arrows and Edit stay trapped in an invisible table.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            hidden = screen.query_one("#tbl-agent6", widgets.DataTable)
            hidden.focus()
            await pilot.pause()
            screen.action_toggle_modified()
            await pilot.pause()
            assert screen.focused is screen.query_one("#tbl-providers", widgets.DataTable)

    asyncio.run(scenario())


def test_empty_modified_filter_keeps_focus_on_the_filter(repo: pathlib.Path) -> None:
    """With nothing modified, the modified-only view focuses its one remaining control."""
    paths.global_config_dir().joinpath("config.toml").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_toggle_modified()
            await pilot.pause()
            assert screen.focused is screen.query_one("#search", widgets.Input)

    asyncio.run(scenario())


def test_filter_down_stops_on_a_collapsed_first_section(repo: pathlib.Path) -> None:
    """Down from the filter lands on a visible header when the first section is collapsed."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            first = screen.query_one("#sec-agent6", widgets.Collapsible)
            first.collapsed = True
            screen.action_search()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert screen.focused is not None
            assert screen.focused.parent is first

    asyncio.run(scenario())


def test_arrows_flow_through_section_headers(repo: pathlib.Path) -> None:
    """Arrows flow as one list through the section headers; Enter on a header collapses it.

    Down at a section's last row lands on the next header, Down again enters its rows.
    """

    async def scenario() -> None:

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tables = [t for t in screen.query(config_page._NavTable) if t.row_count]

            def on_header() -> bool:
                p = getattr(app.focused, "parent", None)
                return isinstance(p, widgets.Collapsible) and bool(p.id and p.id.startswith("sec-"))

            tables[0].focus()
            tables[0].move_cursor(row=tables[0].row_count - 1)
            await pilot.press("down")
            await pilot.pause()
            assert on_header()  # landed on the next section's header
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is tables[1]  # then into its rows
            await pilot.press("up")
            await pilot.pause()
            assert on_header()  # back up onto the header
            # Enter on the header toggles its section.
            section = app.focused.parent.id[4:]  # type: ignore[union-attr]
            col = screen.query_one(f"#sec-{section}", widgets.Collapsible)
            was = col.collapsed
            await pilot.press("enter")
            await pilot.pause()
            assert col.collapsed is not was

    asyncio.run(scenario())


def test_add_provider_via_form_persists(repo: pathlib.Path) -> None:
    """The Add-provider form writes a validated [providers.<name>] block the page reflects."""

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.ProviderModal)
            modal.query_one("#prov-name", widgets.Input).value = "openrouter"
            fmt = modal.query_one("#prov-format", tui_widgets.ChoiceField)
            # A chooser round trip (up, down, Space), so the check holds however the union grows.
            fmt.focus()
            await pilot.pause()
            await pilot.press("up")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            assert fmt.value == "openai"
            modal.query_one("#prov-baseurl", widgets.Input).value = "https://openrouter.ai/api/v1"
            await pilot.pause()
            modal.action_add()  # equivalent to the Add action
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)  # closed on success
            cfg = layer.load_effective(repo).config
            assert "openrouter" in cfg.providers
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_add_provider_preserves_existing_provider_fields(repo: pathlib.Path) -> None:
    """Submitting an existing provider name keeps the fields the short form does not expose."""
    config_path = paths.global_config_dir() / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'api_format = "anthropic"',
            'api_format = "anthropic"\n# keep this tuning\nhttp_timeout_s = 120',
            1,
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:

        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.ProviderModal)
            modal.query_one("#prov-name", widgets.Input).value = "anthropic"
            await pilot.pause()
            modal.action_add()
            await pilot.pause()

    asyncio.run(scenario())
    text = config_path.read_text(encoding="utf-8")
    assert "# keep this tuning" in text
    assert "http_timeout_s = 120" in text


def test_add_provider_prefills_known_preset_base_url(repo: pathlib.Path) -> None:
    """Typing a known provider name in the Add-provider form prefills its preset URL.

    Submitting openrouter without a hand-typed URL lands on openrouter.ai, as `agent6 connect`
    does, not on the api.openai.com fallback.
    """

    async def scenario() -> None:
        from agent6.ui.tui import widgets as tui_widgets

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.ProviderModal)
            # Type only the name; leave api_format + base_url untouched.
            modal.query_one("#prov-name", widgets.Input).value = "openrouter"
            await pilot.pause()
            # Live prefill flipped the format dropdown and filled the URL field.
            assert modal.query_one("#prov-format", tui_widgets.ChoiceField).value == "openai"
            assert (
                modal.query_one("#prov-baseurl", widgets.Input).value
                == "https://openrouter.ai/api/v1"
            )
            modal.action_add()
            await pilot.pause()
            assert isinstance(
                app.screen, config_page.ConfigScreen
            )  # written + validated, modal closed
            cfg = layer.load_effective(repo).config
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_add_provider_prefill_keeps_user_typed_base_url(repo: pathlib.Path) -> None:
    """The name-based prefill never overwrites a base_url the user typed."""

    async def scenario() -> None:

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.ProviderModal)
            modal.query_one("#prov-baseurl", widgets.Input).value = "https://my.proxy/v1"
            await pilot.pause()
            modal.query_one("#prov-name", widgets.Input).value = "openrouter"
            await pilot.pause()
            # Their URL is preserved (only our own autofill, or a blank, is replaced).
            assert modal.query_one("#prov-baseurl", widgets.Input).value == "https://my.proxy/v1"

    asyncio.run(scenario())


def test_add_provider_clears_a_stale_preset_base_url(repo: pathlib.Path) -> None:
    """Changing a preset provider name to a custom one clears the URL that name autofilled."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.ProviderModal)
            name = modal.query_one("#prov-name", widgets.Input)
            base_url = modal.query_one("#prov-baseurl", widgets.Input)
            name.value = "openrouter"
            await pilot.pause()
            assert base_url.value == "https://openrouter.ai/api/v1"
            name.value = "custom"
            await pilot.pause()
            assert base_url.value == ""

    asyncio.run(scenario())


def test_edit_base_url_prefills_preset_for_known_provider(repo: pathlib.Path) -> None:
    """The base_url editor of a known provider still on the generic default offers its preset URL.

    Re-setting an unset openrouter is one Save, as in the Add form and `agent6 connect`.
    """

    async def scenario() -> None:
        from agent6.config import write

        # An openrouter provider with NO base_url -> effective default api.openai.com.
        assert (
            write.set_config_table(repo, "providers.openrouter", {"api_format": "openai"}) is None
        )

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-providers", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if "openrouter" in str(tbl.get_row_at(r)[0])
                and "base_url" in str(tbl.get_row_at(r)[0])
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            # Prefilled with the preset host, not the generic api.openai.com default.
            assert (
                modal.query_one("#edit-value", widgets.Input).value
                == "https://openrouter.ai/api/v1"
            )
            modal.action_save()
            await pilot.pause()
            cfg = layer.load_effective(repo).config
            entry = cfg.providers["openrouter"]
            assert isinstance(entry, OpenAIProviderEntry)
            assert entry.base_url == "https://openrouter.ai/api/v1"

    asyncio.run(scenario())


def test_up_off_first_setting_reveals_top_header_then_filter(repo: pathlib.Path) -> None:
    """In a short window, Up off the first setting focuses and reveals the first section's header.

    The smooth scroll left the top row a line off-screen, so Up looked like it skipped it.
    """
    from textual import containers
    from textual.widgets._collapsible import CollapsibleTitle

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 10)) as pilot:  # short: #settings scrolls
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            settings = screen.query_one("#settings", containers.VerticalScroll)
            screen._focus_first_setting()
            await pilot.pause()
            for _ in range(8):  # scroll down off the top
                await pilot.press("down")
                await pilot.pause()
            header = None
            for _ in range(20):  # walk back up to the [agent6] header
                await pilot.press("up")
                await pilot.pause()
                f = screen.focused
                if isinstance(f, CollapsibleTitle) and getattr(f.parent, "id", "") == "sec-agent6":
                    header = f
                    break
            assert header is not None, "never reached the [agent6] header going up"
            top, bottom = settings.region.y, settings.region.y + settings.region.height
            assert top <= header.region.y < bottom, "top header focused but scrolled off-screen"
            await pilot.press("up")
            await pilot.pause()
            assert isinstance(screen.focused, widgets.Input)  # Up off the top header -> filter

    asyncio.run(scenario())


def test_unset_names_the_layer_instead_of_claiming_the_default(repo: pathlib.Path) -> None:
    """Unsetting a repo override that reveals a global override says so in its notice."""
    from agent6.config import write

    assert write.set_config_value(repo, "sandbox.run_commands", "no", to_repo=True) is None

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            setting = next(
                s
                for s in config_view.build_config_view(layer.load_effective(repo, None)).settings
                if s.key == "sandbox.run_commands"
            )
            assert setting.source == "repo"
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            assert layer.load_effective(repo).config.sandbox.run_commands == "yes"
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes[-1] == "Unset sandbox.run_commands from repo config"

    asyncio.run(scenario())


def test_reset_on_a_profile_sourced_setting_tells_the_truth(repo: pathlib.Path) -> None:
    """A [presets.<name>] leaf renders modified with source "preset"; Reset says the preset owns it.

    No config-file unset can revert it.
    """
    gdir = paths.global_config_dir()
    (gdir / "config.toml").write_text(
        'preset = "fast"\n' + _GLOBAL + '\n[presets.fast.review]\ntrigger = "off"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            setting = next(
                s
                for s in config_view.build_config_view(layer.load_effective(repo, None)).settings
                if s.key == "review.trigger"
            )
            assert setting.source == "preset" and setting.modified
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes, "no notification fired"
            assert "already at its default" not in notes[-1]
            assert "preset" in notes[-1]

    asyncio.run(scenario())


def test_reset_on_a_flag_sourced_setting_names_the_flag_layer(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A leaf a `--config FILE` layer set is not a preset leaf: Reset names the layer."""
    overlay = tmp_path / "overlay.toml"
    overlay.write_text('[review]\ntrigger = "off"\n', encoding="utf-8")

    async def scenario() -> None:
        app = _Host(repo, overlay)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            setting = next(
                s
                for s in config_view.build_config_view(layer.load_effective(repo, overlay)).settings
                if s.key == "review.trigger"
            )
            assert setting.source == "flag" and setting.modified
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes, "no notification fired"
            assert "preset" not in notes[-1]
            assert "flag" in notes[-1]

    asyncio.run(scenario())


def test_reload_on_an_invalid_on_disk_config_keeps_the_last_good_view(repo: pathlib.Path) -> None:
    """A config made invalid in another terminal notifies on r and keeps the last-good table.

    The notice carries the `agent6 config fix` pointer; the action handler does not crash the TUI.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            baseline = _row_total(screen)
            assert baseline > 10
            gdir = paths.global_config_dir()
            (gdir / "config.toml").write_text(
                _GLOBAL + '\n[harness]\nplan = "yess"\n', encoding="utf-8"
            )
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)  # still alive
            assert _row_total(screen) == baseline  # last-good view retained
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("config fix" in m for m in notes), notes
            assert "Config reloaded." not in notes
            # Model suggestions use the last-good config too, never re-reading the invalid file.
            table = screen.query_one("#tbl-models", widgets.DataTable)
            table.focus()
            row = next(
                i
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]).strip() == "worker.model"
            )
            table.move_cursor(row=row)
            await pilot.pause()
            screen.action_edit()
            for _ in range(4):
                await pilot.pause(0.05)
            assert isinstance(app.screen, config_page.EditModal)

    asyncio.run(scenario())


def test_setting_description_lives_in_the_edit_modal_only(repo: pathlib.Path) -> None:
    """The edit modal explains the highlighted leaf; the page carries no detail pane."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            assert not screen.query("#detail")
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r
                for r in range(tbl.row_count)
                if str(tbl.get_row_at(r)[0]).strip() == "run_commands"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_page.EditModal)
            shown = str(modal.query_one("#edit-description", widgets.Static).render())
            assert "run_command" in shown and "**" not in shown

    asyncio.run(scenario())


def test_the_setting_column_fits_the_longest_key(repo: pathlib.Path) -> None:
    """The setting column takes the width the source column does not need.

    A fixed 26-cell column cut `token_command_ttl_s` and its siblings short.
    """

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            view = screen._view  # pyright: ignore[reportPrivateUsage]
            assert view is not None
            longest = max(len(s.key.split(".", 1)[1]) for s in view.settings if "." in s.key)
            assert longest > 26  # the fixture's provider keys make the old width bite
            header = screen.query_one("#col-header", widgets.DataTable)
            assert header.ordered_columns[0].width >= longest

    asyncio.run(scenario())
