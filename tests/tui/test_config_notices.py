from __future__ import annotations

import asyncio
import pathlib

import pytest
from textual import app as textual_app
from textual import widgets

from agent6 import paths
from agent6.config import io
from agent6.ui.tui import _config_forms, config_page, forms


class _Host(textual_app.App[None]):
    def __init__(self, repo: pathlib.Path, config_paths: tuple[pathlib.Path, ...]) -> None:
        super().__init__()
        self._repo = repo
        self._config_paths = config_paths

    def on_mount(self) -> None:
        self.push_screen(config_page.ConfigScreen(self._repo, self._config_paths))


@pytest.mark.parametrize("source", ["repo", "preset", "flag"])
@pytest.mark.parametrize(
    ("operation", "key"),
    [
        ("set", "sandbox.run_commands"),
        ("unset", "sandbox.run_commands"),
        ("add_entry", "providers.local.base_url"),
        ("remove_entry", "providers.local.base_url"),
        ("add_reviewer", "review.reviewers"),
        ("remove_reviewer", "review.reviewers"),
    ],
)
def test_config_write_notices_name_a_shadowing_layer(
    tmp_path: pathlib.Path, source: str, operation: str, key: str
) -> None:
    """A write reports a mask even if another editor added it after the page loaded."""
    target = paths.global_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    initial = (
        '[sandbox]\nrun_commands = "allow"\n'
        '[providers.local]\napi_format = "openai"\nbase_url = "https://old.test/v1"\n'
        '[review]\nreviewers = [{ focus = "old" }]\n'
    )
    target.write_text(initial)
    higher = paths.repo_config_path(tmp_path) if source == "repo" else tmp_path / "flag.toml"
    config_paths = (higher,) if source == "flag" else ()
    if source == "flag":
        higher.write_text("")
    described = {"preset": "the preset", "repo": "the repo config", "flag": f"--config {higher}"}[
        source
    ]

    async def scenario() -> None:
        app = _Host(tmp_path, config_paths)
        async with app.run_test(size=(120, 44)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            section, leaf = key.split(".", 1)
            table = screen.query_one(f"#tbl-{section}", widgets.DataTable)
            row = next(i for i in range(table.row_count) if str(table.get_row_at(i)[0]) == leaf)
            table.move_cursor(row=row)
            table.focus()
            await pilot.pause()
            # The page keeps its source while a second editor changes the on-disk layers.
            prefix = "presets.mask." if source == "preset" else ""
            overlay = (
                f'[{prefix}sandbox]\nrun_commands = "deny"\n'
                f'[{prefix}providers.local]\napi_format = "openai"\nbase_url = "https://mask.test/v1"\n'
                f'[{prefix}review]\nreviewers = [{{ focus = "mask" }}]\n'
            )
            if source == "preset":
                target.write_text('preset = "mask"\n' + initial + overlay)
            else:
                higher.parent.mkdir(parents=True, exist_ok=True)
                higher.write_text(overlay)
            if operation == "set":
                screen.action_edit()
                await pilot.pause()
                modal = app.screen
                assert isinstance(modal, _config_forms.EditModal)
                modal.query_one("#edit-value", forms.ChoiceField).select_value("ask")
                modal.action_save()
            elif operation == "unset":
                screen.action_reset()
            elif operation == "add_entry":
                screen.action_add_provider()
                await pilot.pause()
                modal = app.screen
                assert isinstance(modal, _config_forms.EntryModal)
                modal.query_one("#entry-name", widgets.Input).value = "local"
                modal.query_one("#entry-base_url", widgets.Input).value = "https://new.test/v1"
                modal.action_add()
            elif operation == "remove_entry":
                screen.action_remove_entry()
                await pilot.pause()
                await pilot.press("y")
            elif operation == "add_reviewer":
                screen.action_add_reviewer()
                await pilot.pause()
                modal = app.screen
                assert isinstance(modal, _config_forms.ReviewerModal)
                modal.query_one("#reviewer-focus", widgets.Input).value = "new"
                modal.action_add()
            else:
                screen.action_remove_reviewer()
                await pilot.pause()
                modal = app.screen
                assert isinstance(modal, _config_forms.RemoveReviewerModal)
                modal.action_remove()
                await pilot.pause()
                await pilot.press("y")
            await pilot.pause()
            assert target.read_text() != (
                'preset = "mask"\n' + initial + overlay if source == "preset" else initial
            )
            assert io.read_toml_file(target)
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            tail = (
                " while that flag is used." if source == "flag" else "; a run here reads its value."
            )
            assert any(f"note: {described} overrides {key}{tail}" in note for note in notes), notes

    asyncio.run(scenario())
