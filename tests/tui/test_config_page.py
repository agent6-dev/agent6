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
from agent6.config import ClaudeCodeProviderEntry, OpenAIProviderEntry, io, layer
from agent6.models import cache
from agent6.models import choices as models_choices
from agent6.ui.tui import config_forms, config_model, config_page, forms, menubar, modals
from agent6.viewmodel import config_view

_GLOBAL = """\
[providers.anthropic]
backend = "anthropic"

[models.worker]
model = "anthropic/claude-sonnet-4-5"

[sandbox]
run_commands = "allow"
"""


class _Host(textual_app.App[None]):
    def __init__(
        self, repo_root: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._config_paths = config_paths

    def on_mount(self) -> None:
        self.push_screen(config_page.ConfigScreen(self._repo, self._config_paths))


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
    assert cli_main(["config", "set", "review.interval_turns", "9"]) == 0
    assert cli_main(["config", "set", "--repo", "review.interval_turns", "11"]) == 0

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
                if str(table.get_row_at(i)[0]).strip() == "interval_turns"
            )
            assert str(row[1]).strip() == "11"
            assert "repo" in str(row[2])
            web = model.config_payload(repo)
            assert web["review.interval_turns"]["value"] == 11
            assert web["review.interval_turns"]["source"] == "repo"

            assert cli_main(["config", "unset", "--repo", "review.interval_turns"]) == 0
            screen.action_reload()
            await pilot.pause()
            table = screen.query_one("#tbl-review", widgets.DataTable)
            row = next(
                table.get_row_at(i)
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]).strip() == "interval_turns"
            )
            assert str(row[1]).strip() == "9"
            assert "global" in str(row[2])
            web = model.config_payload(repo)
            assert web["review.interval_turns"]["value"] == 9
            assert web["review.interval_turns"]["source"] == "global"

    asyncio.run(scenario())


def test_config_page_adaptive_value_shown(repo: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            ctx = screen.query_one("#tbl-compaction", widgets.DataTable)
            # Adaptive compaction shows its resolved number tagged "(adaptive)".
            cells = [str(ctx.get_row_at(r)[1]) for r in range(ctx.row_count)]
            assert any("(adaptive)" in c for c in cells)

    asyncio.run(scenario())


def test_config_page_edit_persists(repo: pathlib.Path) -> None:
    """Select a row, Edit, the chooser, a new value, Save: the whole edit ask end to end."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            # Fixed choices have meanings and no free-text row.
            field = modal.query_one("#edit-value", forms.ChoiceField)
            assert field._custom_row == -1  # pyright: ignore[reportPrivateUsage]
            assert "Allow command tools and verification" in field.render().plain
            assert not modal.query(widgets.Input)
            assert field.value == "allow"
            await pilot.press("down")  # highlight "deny" (selection unchanged)
            await pilot.pause()
            assert field.value == "allow"  # arrows only highlight
            await pilot.press("space")  # select "deny"
            await pilot.pause()
            assert field.value == "deny"
            modal.action_save()  # equivalent to the Save action
            await pilot.pause()
            # Persisted through config_layer.set_config_value (global config).
            assert layer.load_effective(repo).config.sandbox.run_commands == "deny"

    asyncio.run(scenario())


def test_edit_defaults_to_the_setting_source_layer(repo: pathlib.Path) -> None:
    """Editing a repo-sourced value targets the repo config by default.

    Otherwise the repo layer masks the global write and Save appears to do nothing.
    """
    from agent6.config import write

    assert write.set_config_value(repo, "sandbox.run_commands", "deny", to_repo=True) is None

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            assert modal.query_one("#edit-target", forms.ChoiceField).value == "repo config"
            modal.query_one("#edit-value", forms.ChoiceField).select_value("ask")
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
            # run_commands is set to "allow" in the (global) config fixture.
            eff = layer.load_effective(repo)
            assert layer.effective_leaf(eff, "sandbox.run_commands") == ("allow", "global")
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
            assert isinstance(modal, config_forms.EditModal)
            modal.action_unset()
            await pilot.pause()
            # Override removed -> back to the default, sourced as default.
            assert layer.effective_leaf(layer.load_effective(repo), "sandbox.run_commands") == (
                "ask",
                "default",
            )

    asyncio.run(scenario())


def test_fixed_choice_has_no_custom_entry(repo: pathlib.Path) -> None:
    """Typing past the last finite option cannot create an unsupported value."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            field = modal.query_one("#edit-value", forms.ChoiceField)
            # The bottom row stays a choice rather than accepting custom text.
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause()
            for ch in ("z", "z", "z"):
                await pilot.press(ch)
            await pilot.pause()
            assert field.value == "allow"
            assert modal._new_value() == "allow"  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_edit_action_arrows_navigate(repo: pathlib.Path) -> None:
    """Left and Right move between the focused flat actions (Save, Unset, Cancel), wrapping."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            items = list(modal.query(forms.ActionItem))
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


def test_model_editor_has_one_qualified_route_field(repo: pathlib.Path) -> None:
    """The Model editor exposes one qualified route field, not a provider chooser."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-models", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if str(tbl.get_row_at(r)[0]) == "worker.model"
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_model.ModelModal)
            assert not modal.query("#model-provider")
            field = modal.query_one("#model-name", forms.TypeaheadField)
            assert field.value == "anthropic/claude-sonnet-4-5"

    asyncio.run(scenario())


def test_model_field_is_a_typeahead_picker(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing models.<role>.model opens a type-to-narrow picker over the provider's models."""
    models = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]

    def _models(*_a: object, **_k: object) -> list[str]:
        return models

    monkeypatch.setattr(cache, "cached_models", _models)
    monkeypatch.setattr(models_choices, "provider_model_choices", _models)  # mock the live fetch

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_model.ModelModal)
            field = modal.query_one("#model-name", forms.TypeaheadField)
            assert field.value == "anthropic/claude-sonnet-4-5"  # the current model
            field.focus()
            # An incomplete catalog must not prevent an unlisted model id.
            await pilot.press("z", "z", "z")
            assert field.value == "zzz"
            await pilot.press("backspace", "backspace", "backspace")
            # Typing narrows; an arrow highlights a match.
            await pilot.press("h")
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert field.value == "anthropic/claude-haiku-4-5"

    asyncio.run(scenario())


@pytest.mark.parametrize("to_repo", [False, True])
def test_scalar_model_editor_saves_one_key_and_preserves_siblings(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, to_repo: bool
) -> None:
    """A qualified route saves one key without replacing the sampling settings."""
    from agent6.config import io, write

    cfg = paths.global_config_path()
    cfg.write_text(
        cfg.read_text().replace(
            'model = "anthropic/claude-sonnet-4-5"',
            'model = "anthropic/claude-sonnet-4-5"\neffort = "high"\ntemperature = 0.4',
        )
        + '\n[providers.b]\nbackend = "openai"\nbase_url = "http://localhost:2"\n'
    )
    repo_cfg = paths.repo_config_path(repo)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text('[models.worker]\neffort = "low"\ntemperature = 0.8\n')

    def listing(provider: str, *_: object) -> list[str]:
        return [provider + "-live"]

    monkeypatch.setattr(cache, "list_models", listing)

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-models", widgets.DataTable)
            table.move_cursor(
                row=next(
                    r
                    for r in range(table.row_count)
                    if str(table.get_row_at(r)[0]) == "worker.model"
                )
            )
            table.focus()
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert not modal.query("#model-provider")
            field = modal.query_one("#model-name", forms.TypeaheadField)
            field.focus()
            await pilot.press(*["backspace"] * len(field.value))
            modal.query(forms.ActionItem).first().focus()
            await pilot.press("enter")
            assert app.screen is modal
            field.focus()
            await pilot.press(*"b/custom")
            target = modal.query_one("#model-target", forms.ChoiceField)
            target.select_value("repo config" if to_repo else "global config")
            before = (repo_cfg if to_repo else cfg).read_text()
            real_write = write.set_config_leaves

            def refused(*_: object, **__: object) -> str:
                return "model write refused for this test"

            monkeypatch.setattr(write, "set_config_leaves", refused)
            modal.query(forms.ActionItem).first().focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is modal
            assert (repo_cfg if to_repo else cfg).read_text() == before
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("model write refused for this test" in note for note in notes)
            monkeypatch.setattr(write, "set_config_leaves", real_write)
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is screen
            assert io.read_toml_file(repo_cfg if to_repo else cfg)["models"]["worker"] == {
                "model": "b/custom",
                "effort": "low" if to_repo else "high",
                "temperature": 0.8 if to_repo else 0.4,
            }

    asyncio.run(scenario())


def test_none_preset_prefill_saves_back_unchanged(repo: pathlib.Path) -> None:
    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            assert modal.query_one("#edit-value", forms.ChoiceField).value == "none"
            modal.action_save()
            await pilot.pause()
            assert layer.load_effective(repo).config.preset == "none"

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
            assert isinstance(modal, config_forms.EditModal)
            field = modal.query_one("#edit-value", widgets.Input)
            assert field.value == '["uv", "run", "pytest"]'  # round-trippable TOML
            modal.action_save()  # untouched Save must succeed, not error
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)

    asyncio.run(scenario())
    saved = layer.load_effective(repo_root, ()).config.harness.verify_command
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
            assert isinstance(modal, config_forms.EditModal)
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
            assert isinstance(app.screen, config_model.ModelModal)  # still open, app alive

    asyncio.run(scenario())


def test_edit_modal_up_at_top_is_a_hard_stop(repo: pathlib.Path) -> None:
    """Up at the top of the first chooser stays there, not escaping to the scroll container."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

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
            assert isinstance(modal, config_forms.EditModal)
            field = modal.query_one("#edit-value", forms.ChoiceField)
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
                "Add provider…",
                "Add MCP server…",
                "Add skill…",
                "Remove entry",
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
            assert isinstance(app.screen, config_forms.EditModal)

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
            first = screen.query_one("#sec-preset", widgets.Collapsible)
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


@pytest.mark.parametrize(
    ("action", "section", "leaf", "field", "value", "name"),
    [
        (
            "add_mcp_server",
            "mcp",
            "servers.openrouter.url",
            "url",
            "https://example.com/mcp",
            "openrouter",
        ),
        ("add_skill", "skills", "state.openrouter", "state", "always", "openrouter"),
        ("add_skill", "skills", 'state."org.some.skill"', "state", "always", "org.some.skill"),
    ],
)
def test_entry_form_and_remove_selected_entry(
    repo: pathlib.Path, action: str, section: str, leaf: str, field: str, value: str, name: str
) -> None:
    """Non-provider forms preserve dotted entry names through display and removal."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            handler = next(
                handler
                for label, handler, _ in screen.palette_commands()
                if label == ("Add skill…" if section == "skills" else "Add MCP server…")
            )
            handler()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = name
            control = modal.query_one(f"#entry-{field}")
            if isinstance(control, forms.ChoiceField):
                control.select_value(value)
            else:
                assert isinstance(control, widgets.Input)
                control.value = value
            if section == "mcp":
                modal.query_one("#entry-token_env", widgets.Input).value = "MCP_TOKEN"
                modal.query_one("#entry-enabled", forms.ChoiceField).select_value("false")
            modal.query_one("#entry-target", forms.ChoiceField).select_value("repo config")
            await pilot.pause()
            modal.action_add()
            await pilot.pause()
            assert app.screen is screen
            cfg = layer.load_effective(repo).config
            if section == "mcp":
                assert cfg.mcp.servers[name].url == value
                assert cfg.mcp.servers[name].token_env == "MCP_TOKEN"
                assert not cfg.mcp.servers[name].enabled
            else:
                assert cfg.skills.state[name] == value
            table = screen.query_one(f"#tbl-{section}", widgets.DataTable)
            row = next(i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == leaf)
            assert "repo" in str(table.get_row_at(row)[2])
            table.move_cursor(row=row)
            table.focus()
            await pilot.pause()
            screen.action_remove_entry()
            await pilot.pause()
            confirm = app.screen
            assert isinstance(confirm, modals.ConfirmModal)
            prompt = str(confirm.query_one("#confirm-box Static", widgets.Static).render())
            assert name in prompt and "repo" in prompt
            assert ("skills.state" if section == "skills" else "mcp.servers") in prompt
            await pilot.press("escape")
            await pilot.pause()
            assert name in paths.repo_config_path(repo).read_text()
            screen.action_remove_entry()
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert app.screen is screen
            assert name not in paths.repo_config_path(repo).read_text()
            assert not any(name in str(table.get_row_at(i)[0]) for i in range(table.row_count))
            footer_actions = {
                active.binding.action
                for active in screen.active_bindings.values()
                if active.binding.show
            }
            assert action in footer_actions
            assert "remove_entry" in footer_actions

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["bad.name", "bad name", "\u0430dmin"])
def test_provider_form_uses_the_schema_name_refusal(repo: pathlib.Path, name: str) -> None:
    """An invalid provider name stays in the form with the schema's refusal words."""
    before = paths.global_config_path().read_text()

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = name
            modal.query_one("#entry-backend", forms.ChoiceField).select_value("openai")
            modal.action_add()
            await pilot.pause()
            expected = (
                f"provider name {name!r} is not a valid TOML bare key"
                " (use only ASCII letters, digits, '-', '_')."
            )
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any(expected in note for note in notes), notes
            assert app.screen is modal
            assert paths.global_config_path().read_text() == before

    asyncio.run(scenario())


def test_reviewer_form_and_positional_removal(repo: pathlib.Path) -> None:
    """The roster form appends; the removal chooser distinguishes seats with a shared focus."""
    from agent6.config import write

    assert (
        write.set_config_value(
            repo,
            "review.reviewers",
            '[{ focus = "tests", effort = "off" }, { focus = "tests", effort = "low" }]',
        )
        is None
    )

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            handler = next(
                handler
                for label, handler, _ in screen.palette_commands()
                if label == "Add reviewer…"
            )
            handler()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.ReviewerModal)
            modal.query_one("#reviewer-focus", widgets.Input).value = "security"
            modal.query_one("#reviewer-model", widgets.Input).value = "anthropic/claude-sonnet-4-5"
            modal.query_one("#reviewer-effort", forms.ChoiceField).select_value("high")
            modal.query_one("#reviewer-temperature", widgets.Input).value = "0.5"
            modal.query_one("#reviewer-target", forms.ChoiceField).select_value("repo config")
            modal.action_add()
            await pilot.pause()
            assert app.screen is screen
            roster = layer.load_effective(repo).config.review.reviewers
            assert [seat.focus for seat in roster] == ["tests", "tests", "security"]
            assert roster[2].model == "anthropic/claude-sonnet-4-5"
            assert roster[2].effort == "high" and roster[2].temperature == 0.5
            table = screen.query_one("#tbl-review", widgets.DataTable)
            row = next(
                i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == "reviewers"
            )
            assert "security" in str(table.get_row_at(row)[1])
            table.move_cursor(row=row)
            table.focus()
            await pilot.pause()
            screen.action_remove_entry()  # one action: on the roster row it picks a seat
            await pilot.pause()
            chooser = app.screen
            assert isinstance(chooser, config_forms.RemoveReviewerModal)
            chooser.query_one("#reviewer-position", forms.ChoiceField).select_value("2: tests")
            chooser.action_remove()
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)
            prompt = str(app.screen.query_one("#confirm-box Static", widgets.Static).render())
            assert "2: tests" in prompt and "repo" in prompt
            await pilot.press("escape")
            await pilot.pause()
            assert len(layer.load_effective(repo).config.review.reviewers) == 3
            chooser.action_remove()
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert app.screen is screen
            roster = layer.load_effective(repo).config.review.reviewers
            assert [seat.focus for seat in roster] == ["tests", "security"]
            assert roster[0].effort == "off"
            footer_actions = {
                active.binding.action
                for active in screen.active_bindings.values()
                if active.binding.show
            }
            assert {"add_reviewer", "remove_entry"} <= footer_actions

    asyncio.run(scenario())


def test_remove_entry_from_a_default_leaf(repo: pathlib.Path) -> None:
    """A default-valued provider leaf still belongs to a removable entry."""
    from agent6.config import write

    assert write.set_config_leaves(repo, "providers.local", {"backend": "openai"}) is None

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            table = screen.query_one("#tbl-providers", widgets.DataTable)
            row = next(
                i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == "local.base_url"
            )
            assert str(table.get_row_at(row)[2]) == "default"
            table.move_cursor(row=row)
            table.focus()
            await pilot.pause()
            screen.action_remove_entry()
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)
            await pilot.press("y")
            await pilot.pause()
            assert "local" not in layer.load_effective(repo).config.providers
            assert "anthropic" in layer.load_effective(repo).config.providers

    asyncio.run(scenario())


def test_remove_entry_without_an_entry_row(repo: pathlib.Path) -> None:
    """Remove on an ordinary setting explains which rows it accepts without opening a dialog."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.query_one("#tbl-sandbox", widgets.DataTable).focus()
            await pilot.pause()
            screen.action_remove_entry()
            await pilot.pause()
            assert app.screen is screen
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes[-1] == (
                "Select a row of a provider, MCP server, skill or the reviewer roster first."
            )

    asyncio.run(scenario())


def test_skill_form_reports_a_repo_override_of_its_global_write(repo: pathlib.Path) -> None:
    """The skill form reports a masked write and the table retains the repo value."""
    from agent6.config import write

    assert write.set_config_value(repo, "skills.state.alpha", '"enabled"', to_repo=True) is None

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_skill()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = "alpha"
            modal.query_one("#entry-state", forms.ChoiceField).select_value("disabled")
            modal.query_one("#entry-target", forms.ChoiceField).select_value("global config")
            modal.action_add()
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("note: the repo config overrides skills.state.alpha;" in n for n in notes)
            table = screen.query_one("#tbl-skills", widgets.DataTable)
            row = next(
                table.get_row_at(i)
                for i in range(table.row_count)
                if str(table.get_row_at(i)[0]) == "state.alpha"
            )
            assert "enabled" in str(row[1]) and "repo" in str(row[2])

    asyncio.run(scenario())


def test_add_provider_via_form_persists(repo: pathlib.Path) -> None:
    """The Add-provider form writes a validated [providers.<name>] block the page reflects."""

    async def scenario() -> None:
        from agent6.ui.tui import forms

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = "openrouter"
            fmt = modal.query_one("#entry-backend", forms.ChoiceField)
            # A chooser round trip (up, down, Space), so the check holds however the union grows.
            fmt.focus()
            await pilot.pause()
            await pilot.press("up")
            await pilot.press("down")
            await pilot.press("space")
            await pilot.pause()
            assert fmt.value == "openai"
            modal.query_one("#entry-base_url", widgets.Input).value = "https://openrouter.ai/api/v1"
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
            'backend = "anthropic"',
            'backend = "anthropic"\n# keep this tuning\nhttp_timeout_s = 120',
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
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = "anthropic"
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
        from agent6.ui.tui import forms

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            # Type only the name; leave backend + base_url untouched.
            modal.query_one("#entry-name", widgets.Input).value = "openrouter"
            await pilot.pause()
            # Live prefill flipped the format dropdown and filled the URL field.
            assert modal.query_one("#entry-backend", forms.ChoiceField).value == "openai"
            assert (
                modal.query_one("#entry-base_url", widgets.Input).value
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
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-base_url", widgets.Input).value = "https://my.proxy/v1"
            await pilot.pause()
            modal.query_one("#entry-name", widgets.Input).value = "openrouter"
            await pilot.pause()
            # Their URL is preserved (only our own autofill, or a blank, is replaced).
            assert modal.query_one("#entry-base_url", widgets.Input).value == "https://my.proxy/v1"

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
            assert isinstance(modal, config_forms.EntryModal)
            name = modal.query_one("#entry-name", widgets.Input)
            base_url = modal.query_one("#entry-base_url", widgets.Input)
            name.value = "openrouter"
            await pilot.pause()
            assert base_url.value == "https://openrouter.ai/api/v1"
            name.value = "custom"
            await pilot.pause()
            assert base_url.value == ""

    asyncio.run(scenario())


def test_add_provider_writes_a_chosen_direct_over_an_existing_deployment(
    repo: pathlib.Path,
) -> None:
    """A chosen `direct` replaces `deployment = "azure"`; the form dropped it as a blank."""
    config_path = paths.global_config_dir() / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + '\n[providers.az]\nbackend = "openai"\ndeployment = "azure"\n'
        'base_url = "https://res.openai.azure.com"\n'
        'extra_query = { "api-version" = "2024-10-21" }\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        from agent6.ui.tui import forms

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = "az"
            modal.query_one("#entry-backend", forms.ChoiceField).select_value("openai")
            modal.query_one("#entry-deployment", forms.ChoiceField).select_value("direct")
            modal.query_one("#entry-base_url", widgets.Input).value = "https://api.openai.com/v1"
            await pilot.pause()
            modal.action_add()
            await pilot.pause()
            assert isinstance(app.screen, config_page.ConfigScreen)  # closed on success

    asyncio.run(scenario())
    entry = layer.load_effective(repo).config.providers["az"]
    assert isinstance(entry, OpenAIProviderEntry)
    assert entry.deployment == "direct"


def _add_provider_by_form(
    repo: pathlib.Path, name: str, backend: str, deployment: str | None
) -> bool:
    """Fill the add-provider form, leaving the deployment chooser on its blank row when None.

    Returns:
        Whether the form closed, which it does only after a write that landed.
    """
    closed = False

    async def scenario() -> None:
        nonlocal closed
        from agent6.ui.tui import forms

        app = _Host(repo)
        async with app.run_test(size=(110, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            screen.action_add_provider()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EntryModal)
            modal.query_one("#entry-name", widgets.Input).value = name
            modal.query_one("#entry-backend", forms.ChoiceField).select_value(backend)
            if deployment is not None:
                modal.query_one("#entry-deployment", forms.ChoiceField).select_value(deployment)
            await pilot.pause()
            modal.action_add()
            await pilot.pause()
            closed = isinstance(app.screen, config_page.ConfigScreen)

    asyncio.run(scenario())
    return closed


def test_add_provider_keeps_an_existing_deployment_when_left_blank(repo: pathlib.Path) -> None:
    """A re-add with the deployment chooser untouched keeps `deployment = "azure"`."""
    config_path = paths.global_config_dir() / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + '\n[providers.az]\nbackend = "openai"\ndeployment = "azure"\n'
        'base_url = "https://res.openai.azure.com"\n'
        'extra_query = { "api-version" = "2024-10-21" }\n',
        encoding="utf-8",
    )
    assert _add_provider_by_form(repo, "az", "openai", None)
    entry = layer.load_effective(repo).config.providers["az"]
    assert isinstance(entry, OpenAIProviderEntry)
    assert entry.deployment == "azure"


def test_add_provider_takes_claude_code_with_the_deployment_left_blank(
    repo: pathlib.Path,
) -> None:
    """A `claude_code` provider adds with the deployment chooser on its blank row."""
    assert _add_provider_by_form(repo, "cc", "claude_code", None)
    assert isinstance(layer.load_effective(repo).config.providers["cc"], ClaudeCodeProviderEntry)


def test_add_provider_refuses_a_deployment_for_claude_code(repo: pathlib.Path) -> None:
    """A deployment chosen for `claude_code` keeps the form open: the entry has no such field."""
    assert not _add_provider_by_form(repo, "cc", "claude_code", "direct")
    assert "cc" not in layer.load_effective(repo).config.providers


def test_edit_base_url_prefills_preset_for_known_provider(repo: pathlib.Path) -> None:
    """The base_url editor of a known provider still on the generic default offers its preset URL.

    Re-setting an unset openrouter is one Save, as in the Add form and `agent6 connect`.
    """

    async def scenario() -> None:
        from agent6.config import write

        # An openrouter provider with NO base_url -> effective default api.openai.com.
        assert write.set_config_table(repo, "providers.openrouter", {"backend": "openai"}) is None

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
            assert isinstance(modal, config_forms.EditModal)
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
            for _ in range(20):  # walk back up to the first section's header (`preset`)
                await pilot.press("up")
                await pilot.pause()
                f = screen.focused
                if isinstance(f, CollapsibleTitle) and getattr(f.parent, "id", "") == "sec-preset":
                    header = f
                    break
            assert header is not None, "never reached the first header going up"
            top, bottom = settings.region.y, settings.region.y + settings.region.height
            assert top <= header.region.y < bottom, "top header focused but scrolled off-screen"
            await pilot.press("up")
            await pilot.pause()
            assert isinstance(screen.focused, widgets.Input)  # Up off the top header -> filter

    asyncio.run(scenario())


def test_unset_names_the_layer_instead_of_claiming_the_default(repo: pathlib.Path) -> None:
    """Unsetting a repo override that reveals a global override says so in its notice."""
    from agent6.config import write

    assert write.set_config_value(repo, "sandbox.run_commands", "deny", to_repo=True) is None

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            setting = next(
                s
                for s in config_view.build_config_view(layer.load_effective(repo, ())).settings
                if s.key == "sandbox.run_commands"
            )
            assert setting.source == "repo"
            screen._current_setting = lambda: setting  # type: ignore[method-assign]
            screen.action_reset()
            await pilot.pause()
            assert layer.load_effective(repo).config.sandbox.run_commands == "allow"
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes[-1] == "Unset sandbox.run_commands from repo config"

    asyncio.run(scenario())


def test_reset_on_a_profile_sourced_setting_tells_the_truth(repo: pathlib.Path) -> None:
    """A [presets.<name>] leaf renders modified with source "preset"; Reset says the preset owns it.

    No config-file unset can revert it.
    """
    gdir = paths.global_config_dir()
    (gdir / "config.toml").write_text(
        'preset = "fast"\n' + _GLOBAL + '\n[presets.fast.review]\nwhen = "never"\n',
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
                for s in config_view.build_config_view(layer.load_effective(repo, ())).settings
                if s.key == "review.when"
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
    overlay.write_text('[review]\nwhen = "never"\n', encoding="utf-8")

    async def scenario() -> None:
        app = _Host(repo, (overlay,))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            setting = next(
                s
                for s in config_view.build_config_view(
                    layer.load_effective(repo, (overlay,))
                ).settings
                if s.key == "review.when"
            )
            assert setting.source == f"flag ({overlay})" and setting.modified
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
            assert isinstance(app.screen, config_model.ModelModal)

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
            assert isinstance(modal, config_forms.EditModal)
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


def test_reset_action_label_matches_the_menu() -> None:
    """The reset action has the same name in its registry and the menu."""
    action = next(action for action in config_page.CONFIG_ACTIONS if action.id == "reset")
    assert action.label == "Unset override"


@pytest.mark.parametrize("role", ["default", "helper"])
def test_an_unconfigured_role_opens_the_model_picker_too(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    """`models.default` and `models.helper` rows open the picker like the three roles once named."""

    def _models(*_a: object, **_k: object) -> list[str]:
        return ["x/y"]

    monkeypatch.setattr(models_choices, "provider_model_choices", _models)

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-models", widgets.DataTable)
            tbl.focus()
            ridx = next(
                r for r in range(tbl.row_count) if str(tbl.get_row_at(r)[0]).strip() == role
            )
            tbl.move_cursor(row=ridx)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            assert isinstance(app.screen, config_model.ModelModal)

    asyncio.run(scenario())


def test_the_edit_popup_offers_add_and_remove_for_an_entrys_table(repo: pathlib.Path) -> None:
    """A provider row's popup adds or removes a provider; a plain setting's popup offers neither."""

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            tbl = screen.query_one("#tbl-providers", widgets.DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EditModal)
            labels = [str(item.render()) for item in modal.query(forms.ActionItem)]
            assert "Add provider…" in labels and "Remove provider" in labels
            modal.action_add_entry()
            await pilot.pause()
            assert isinstance(app.screen, config_forms.EntryModal)
            await pilot.press("escape")
            await pilot.pause()
            tbl = screen.query_one("#tbl-sandbox", widgets.DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            plain = app.screen
            assert isinstance(plain, config_forms.EditModal)
            assert not [i for i in plain.query(forms.ActionItem) if "Add" in str(i.render())]

    asyncio.run(scenario())


def test_the_page_shows_and_edits_the_selected_preset(repo: pathlib.Path) -> None:
    """Preset… sets `preset`; the rows relayer; a save writes [presets.<name>]; Remove resets it."""
    cfg = paths.global_config_path()
    cfg.write_text(
        cfg.read_text(encoding="utf-8") + '\n[presets.fast.sandbox]\nrun_commands = "deny"\n'
    )

    def value_of(screen: config_page.ConfigScreen, section: str, leaf: str) -> str:
        table = screen.query_one(f"#tbl-{section}", widgets.DataTable)
        row = next(i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == leaf)
        return str(table.get_row_at(row)[1])

    async def scenario() -> None:
        app = _Host(repo)
        async with app.run_test(size=(100, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            assert value_of(screen, "sandbox", "run_commands") == '"allow"'
            screen.action_choose_preset()
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, config_forms.EditModal)
            picker.query_one("#edit-value", forms.ChoiceField).select_value("fast")
            picker.action_save()
            await pilot.pause()
            assert app.screen is screen
            assert io.read_toml_file(cfg)["preset"] == "fast"
            assert value_of(screen, "sandbox", "run_commands") == '"deny"'
            status = str(screen.query_one("#status", widgets.Static).render())
            assert "preset fast" in status
            table = screen.query_one("#tbl-sandbox", widgets.DataTable)
            row = next(
                i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == "network"
            )
            table.move_cursor(row=row)
            table.focus()
            await pilot.pause()
            screen.action_edit()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, config_forms.EditModal)
            modal.query_one("#edit-value", forms.ChoiceField).select_value("host")
            modal.action_save()
            await pilot.pause()
            written = io.read_toml_file(cfg)["presets"]["fast"]["sandbox"]
            assert written == {"run_commands": "deny", "network": "host"}
            plain = io.read_toml_file(cfg).get("sandbox", {})
            assert "network" not in plain  # the save went under the preset, not the plain table
            screen.action_remove_preset()
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)
            await pilot.press("y")
            await pilot.pause()
            saved = io.read_toml_file(cfg)
            assert "presets" not in saved and "preset" not in saved
            assert value_of(screen, "sandbox", "run_commands") == '"allow"'

    asyncio.run(scenario())
