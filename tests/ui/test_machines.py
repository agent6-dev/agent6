# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` Machines page."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import DataTable, Input

from agent6.ui import machines as machmod
from agent6.ui.machines import (
    CreateMachineModal,
    MachineDetailScreen,
    MachinesScreen,
    find_machine_files,
    machine_detail_text,
)
from agent6.ui.modals import ConfirmModal

WAITER = """
machine = "waiter_demo"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write(path: Path, body: str = WAITER) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_find_machine_files_cwd_and_subdir(tmp_path: Path) -> None:
    _write(tmp_path / "a.asm.toml")
    (tmp_path / "machines").mkdir()
    _write(tmp_path / "machines" / "b.asm.toml")
    (tmp_path / "not-a-machine.toml").write_text("x = 1\n", encoding="utf-8")
    names = {p.name for p in find_machine_files(tmp_path)}
    assert names == {"a.asm.toml", "b.asm.toml"}


def test_machine_detail_text_parses_a_valid_machine(tmp_path: Path) -> None:
    text = machine_detail_text(_write(tmp_path / "m.asm.toml"))
    assert "machine: waiter_demo" in text
    assert "initial: poll" in text
    assert "validation: OK" in text
    assert "graph (mermaid):" in text


def test_machine_detail_text_reports_a_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.asm.toml"
    bad.write_text("this is not = valid [[[\n", encoding="utf-8")
    assert "failed to load bad.asm.toml" in machine_detail_text(bad)


class _Host(App[None]):
    def __init__(self, repo_cwd: Path) -> None:
        super().__init__()
        self._repo = repo_cwd

    def on_mount(self) -> None:
        self.push_screen(MachinesScreen(self._repo, self._repo / ".agent6"))


def test_machines_menu_items_all_resolve(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            for menu in screen.MENUS:
                for item in menu.items:
                    resolved = getattr(screen, f"action_{item.action}", None) or getattr(
                        app, f"action_{item.action}", None
                    )
                    assert resolved is not None, f"no handler for {item.action}"

    asyncio.run(scenario())


def test_create_opens_dashboard_on_the_draft(tmp_path: Path, monkeypatch: object) -> None:
    """Creating a machine spawns `machine create`, locates the draft it produces,
    and hands that dir to the dashboard via app.exit -- so it is watchable live,
    not fire-and-forget."""
    draft = tmp_path / "draft"
    draft.mkdir()

    def _fake_locate(*_a: object, **_k: object) -> tuple[Path, str]:
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            screen._on_create("make a greeter")  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
        assert app.return_value == draft  # handed the draft to the dashboard

    asyncio.run(scenario())


def test_machines_page_lists_and_views(tmp_path: Path) -> None:
    _write(tmp_path / "m.asm.toml")

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            table = screen.query_one("#machines", DataTable)
            assert table.row_count == 1
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("v")  # view -> parsed detail screen
            await pilot.pause()
            assert isinstance(app.screen, MachineDetailScreen)

    asyncio.run(scenario())


def test_machines_menu_bar_dispatches_an_item(tmp_path: Path) -> None:
    """Selecting an item from the menu bar (not just the key binding) runs its
    action -- exercises action_menu + on_menu_bar_selected, the dead-menu bug class."""
    from agent6.ui.menubar import MenuBar, _Dropdown

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachinesScreen)
            screen.query_one(MenuBar).open("m")  # the "Machines" menu
            await pilot.pause()
            dd = next(iter(screen.query(_Dropdown)))
            idx = next(
                i for i in range(dd.option_count) if dd.get_option_at_index(i).id == "create"
            )
            dd.highlighted = idx
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, CreateMachineModal)  # the menu actually fired

    asyncio.run(scenario())


def test_machine_run_confirms_then_spawns(tmp_path: Path, monkeypatch: object) -> None:
    path = _write(tmp_path / "m.asm.toml")
    captured: list[list[str]] = []

    def _fake_spawn(argv: list[str], cwd: Path) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_detached", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("r")  # run -> confirm modal
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert captured and captured[-1][-3:] == ["machine", "run", str(path)]

    asyncio.run(scenario())


def test_machine_create_spawns_with_task(tmp_path: Path, monkeypatch: object) -> None:
    """The create modal threads the typed task into `agent6 machine create <task>`
    (then the draft is located + handed to the dashboard)."""
    captured: list[list[str]] = []
    draft = tmp_path / "d"
    draft.mkdir()

    def _fake_locate(argv: list[str], cwd: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("c")  # create -> task modal
            await pilot.pause()
            assert isinstance(app.screen, CreateMachineModal)
            app.screen.query_one("#create-input", Input).value = "nightly sweep"
            await pilot.press("enter")  # submit
            await pilot.pause()
            assert captured and captured[-1][-2:] == ["create", "nightly sweep"]

    asyncio.run(scenario())
