# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` Machines page."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import DataTable, Input

from agent6.ui.tui import machines as machmod
from agent6.ui.tui.machines import (
    CreateMachineModal,
    MachineDetailScreen,
    MachinesScreen,
    MachineWatchScreen,
    find_machine_files,
    machine_detail_text,
)
from agent6.ui.tui.modals import ConfirmModal

# A no-I/O machine that reaches a terminal immediately (branch -> terminal), so a
# `machine run` produces a finished instance with no model/jail needed.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""

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


def test_watch_screen_shows_states_transitions_and_end(tmp_path: Path, monkeypatch: object) -> None:
    """The Machines watch screen renders the state overview (current marked `>`,
    visited `.`), the transition in the log, and the ended status -- the in-TUI
    equivalent of `agent6 attach`."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll or two run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            table = screen.query_one("#mw-states", DataTable)
            assert table.row_count == len(spec.states)
            assert table.get_cell("done", "mark") == ">"  # current (terminal) state
            assert table.get_cell("route", "mark") == "·"  # visited
            from textual.widgets import RichLog

            log = screen.query_one("#mw-log", RichLog)
            assert len(log.lines) >= 1  # the route->done transition was logged

    asyncio.run(scenario())


def test_watch_screen_disables_steer_and_message_when_ended(
    tmp_path: Path, monkeypatch: object
) -> None:
    """An ended machine takes no input: the watch screen dims Steer/Message (like
    the web disables both buttons) and their actions are no-ops, never dropping a
    steer marker into the dead per-state dir."""
    from agent6.config.layer import resolved_state_dir
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = resolved_state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)
    # A per-state dir so _state_dir() resolves -- the "dead dir" a steer would hit.
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll set _ended
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)
            assert screen._ended  # pyright: ignore[reportPrivateUsage]
            assert screen.check_action("steer", ()) is False
            assert screen.check_action("poke", ()) is False
            screen.action_steer()  # no-op when ended
            await pilot.pause()
            assert not (state / "steer.request").exists()  # nothing dropped in the dead dir

    asyncio.run(scenario())


def test_discrete_log_line_renders_tool_events_only() -> None:
    # The shared journal fold (current/visited/transitions) is tested in
    # tests/unit/test_viewmodel_machine_state.py; this covers the TUI-only
    # presentation helper for the per-state agent log.
    from agent6.ui.tui.machines import _discrete_log_line

    # A tool call renders compactly; a thinking delta is not a discrete line.
    assert _discrete_log_line({"type": "role.thinking_delta", "text": "hm"}) is None
    line = _discrete_log_line({"type": "tool.call", "name": "grep", "args": {"q": "x"}})
    assert line is not None and "grep" in line.plain


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
    from agent6.ui.tui.menubar import MenuBar, _Dropdown

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

    def _fake_spawn(argv: list[str], cwd: Path, **_k: object) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

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


def test_machine_run_refusal_notifies_and_skips_watch(tmp_path: Path, monkeypatch: object) -> None:
    """A `machine run` refusal (lock held, exit 2) must surface as an error
    notification, not open a watch screen on nothing."""
    _write(tmp_path / "m.asm.toml")

    def _fake_spawn(argv: list[str], cwd: Path, **_k: object) -> str:
        return "agent6 machine exited (1) before starting:\nERROR: lock held"

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("r")
            await pilot.pause()
            await pilot.press("y")  # confirm the run
            await pilot.pause()
            assert not isinstance(app.screen, MachineWatchScreen)  # no watch on nothing
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("lock held" in n for n in notes)

    asyncio.run(scenario())


def test_watch_screen_survives_corrupt_journal(tmp_path: Path) -> None:
    """A corrupt journal line must not crash the watch screen every poll tick;
    the header shows the corruption and polling continues."""
    from textual.widgets import Static

    from agent6.machine import load_machine

    f = _write(tmp_path / "m.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / ".agent6" / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text('{"type": "step", "bogus": 1}\n', encoding="utf-8")

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)  # still alive, not crashed
            head = screen.query_one("#mw-head", Static)
            assert "journal unreadable" in str(head.render())

    asyncio.run(scenario())


def test_watch_screen_tolerates_torn_utf8_state_log(tmp_path: Path) -> None:
    """A per-state agent log whose tail ends mid multibyte UTF-8 sequence (the
    writer flushes long lines in several syscalls) must not crash the poll; the
    complete prefix renders and the torn tail is picked up once completed."""
    import json as _json

    from textual.widgets import RichLog

    from agent6.machine import load_machine

    f = _write(tmp_path / "m.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / ".agent6" / "machines" / "tiny"
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (instance / "journal.jsonl").write_text("", encoding="utf-8")
    full = _json.dumps({"type": "tool.call", "name": "café", "args": {}}, ensure_ascii=False)
    raw = full.encode("utf-8")
    cut = raw.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é sequence
    (state / "logs.jsonl").write_bytes(
        _json.dumps({"type": "tool.call", "name": "grep", "args": {}}).encode() + b"\n" + raw[:cut]
    )

    class _WatchHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MachineWatchScreen)  # no UnicodeDecodeError crash
            log = screen.query_one("#mw-log", RichLog)
            assert any("grep" in line.text for line in log.lines)
            assert not any("café" in line.text for line in log.lines)  # torn line held back
            # Completing the line delivers it on a later poll.
            with (state / "logs.jsonl").open("ab") as fh:
                fh.write(raw[cut:] + b"\n")
            for _ in range(4):
                await pilot.pause()
            screen._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert any("café" in line.text for line in log.lines)

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
