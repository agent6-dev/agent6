# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` Machines page."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib

from textual import app as textual_app
from textual import widgets

from agent6 import paths
from agent6.machine import MachineSpec
from agent6.ui.tui import composer, modals
from agent6.ui.tui import machines as machmod
from agent6.viewmodel import machine_files, machine_state
from tests.tui._waits import answerable, focus_answers, wait_for

# A no-I/O machine that reaches a terminal at once, so `machine run` finishes with no model or jail.
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


def _write(path: pathlib.Path, body: str = WAITER) -> pathlib.Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_machine_files_cwd_and_subdir(tmp_path: pathlib.Path) -> None:
    _write(tmp_path / "a.asm.toml")
    (tmp_path / "machines").mkdir()
    _write(tmp_path / "machines" / "b.asm.toml")
    (tmp_path / "not-a-machine.toml").write_text("x = 1\n", encoding="utf-8")
    names = {p.name for p in machine_files(tmp_path)}
    assert names == {"a.asm.toml", "b.asm.toml"}


def test_machine_detail_text_parses_a_valid_machine(tmp_path: pathlib.Path) -> None:
    text = machmod.machine_detail_text(_write(tmp_path / "m.asm.toml"))
    assert "machine: waiter_demo" in text
    assert "initial: poll" in text
    # Named for what it ran: this view checks semantics, not the script bundle.
    assert "semantics: OK" in text
    assert "graph (mermaid):" in text
    # States read as the user's kind word, matching the watch screen and web, not the class name.
    assert "poll  (wait)" in text and "done  (terminal)" in text
    assert "State)" not in text


def test_machine_detail_text_reports_a_bad_file(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "bad.asm.toml"
    bad.write_text("this is not = valid [[[\n", encoding="utf-8")
    assert "failed to load bad.asm.toml" in machmod.machine_detail_text(bad)


async def _spawn_settled(app: textual_app.App[None]) -> None:
    """Wait for a spawn worker.

    One that hands its draft to `app.exit` cancels the worker group on the way out, so its
    completion reads as cancelled.
    """
    from textual.worker import WorkerCancelled

    with contextlib.suppress(WorkerCancelled):
        await app.workers.wait_for_complete()


class _Host(textual_app.App[None]):
    def __init__(self, repo_cwd: pathlib.Path) -> None:
        super().__init__()
        self._repo = repo_cwd

    def on_mount(self) -> None:
        self.push_screen(machmod.MachinesScreen(self._repo / ".agent6", self._repo))


def test_machines_menu_items_all_resolve(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            for menu in screen.MENUS:
                for item in menu.items:
                    resolved = getattr(screen, f"action_{item.action}", None) or getattr(
                        app, f"action_{item.action}", None
                    )
                    assert resolved is not None, f"no handler for {item.action}"

    asyncio.run(scenario())


def test_row_actions_are_dimmed_on_an_empty_machines_page(tmp_path: pathlib.Path) -> None:
    """View, Run and Watch need a row; with none they refuse instead of silently doing nothing."""

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            # None greys the key; False would hide it, reading as a missing capability.
            assert [screen.check_action(a, ()) for a in ("view", "run", "watch")] == [
                None,
                None,
                None,
            ]
            assert screen.check_action("create", ()) is True

    asyncio.run(scenario())


def test_watch_is_dimmed_for_an_authored_machine_that_has_not_run(tmp_path: pathlib.Path) -> None:
    """Watch is not offered for a file-only row.

    It attaches to an instance; offered, it created an empty instance dir and a stopped machine.
    """
    _write(tmp_path / "tiny.asm.toml", TINY)

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            assert screen.check_action("watch", ()) is None
            await pilot.press("w")
            await pilot.pause()
            assert isinstance(app.screen, machmod.MachinesScreen)
            assert not (tmp_path / ".agent6" / "machines" / "tiny").exists()

    asyncio.run(scenario())


def test_watch_screen_carries_the_menu_bar_and_its_items_resolve(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """The watch screen has every screen's chrome, and every menu item resolves to an action."""
    from agent6.machine import load_machine
    from agent6.ui.tui import menubar

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "instance"
    instance.mkdir()

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert screen.query(menubar.MenuBar) and screen.query(widgets.Footer)
            for menu in screen.MENUS:
                for item in menu.items:
                    assert getattr(screen, f"action_{item.action}", None) or getattr(
                        app, f"action_{item.action}", None
                    ), f"no handler for {item.action}"

    asyncio.run(scenario())


def test_watch_screen_shows_states_transitions_and_end(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """The watch screen renders the state overview, the log transition and the ended status."""
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = paths.state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll or two run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            table = screen.query_one("#mw-states", widgets.DataTable)
            assert table.row_count == len(spec.states)
            assert table.get_cell("done", "mark") == "▸"  # current (terminal) state
            assert table.get_cell("route", "mark") == "·"  # visited

            log = screen.query_one("#mw-log", widgets.RichLog)
            assert len(log.lines) >= 1  # the route->done transition was logged

    asyncio.run(scenario())


def test_watch_screen_does_not_reannounce_a_stale_end(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """Reviewing a machine that finished long ago pops no toast or desktop notification.

    The end flag seeds from the fold that seeds notification history; a machine ending while
    watched still announces.
    """
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main
    from agent6.ui.tui import machines as machines_mod

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    fired: list[tuple[str, str]] = []

    def fake_notify(title: str, body: str) -> None:
        fired.append((title, body))

    monkeypatch.setattr(machines_mod, "desktop_notify", fake_notify)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = paths.state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(4):  # let a few polls run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert screen._end_notified is True  # pyright: ignore[reportPrivateUsage]
            assert fired == []  # no desktop notification for the day-old end
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert not any(m == "ok" or m.endswith(" ok") for m in notes), notes

    asyncio.run(scenario())


def test_watch_screen_disables_steer_and_message_when_ended(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """An ended machine takes no input: Steer and Message dim and their actions are no-ops."""
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    instance = paths.state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)
    # A per-state dir so _current_state_dir() resolves: the dead dir a steer would hit.
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):  # let a poll set _ended
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert screen._ended  # pyright: ignore[reportPrivateUsage]
            assert screen.check_action("steer", ()) is None
            assert screen.check_action("poke", ()) is None
            screen.action_steer()  # no-op when ended
            await pilot.pause()
            assert not (state / "steer.request").exists()  # nothing dropped in the dead dir
            screen.action_poke()  # a direct call: the action itself answers
            await pilot.pause()
            toasts = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            # The CLI's refusal, word for word.
            assert (
                machine_state.machine_verb_refusal(instance, "tiny", "poke"),
                "warning",
            ) in toasts, toasts

    asyncio.run(scenario())


def test_watch_screen_suppresses_phantom_thinking_on_an_ended_machine(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """An ended machine's log ending on a role.call renders no live thinking line."""
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0  # terminates at once -> ended
    instance = paths.state_dir(tmp_path) / "machines" / "tiny"
    spec = load_machine(f)
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text(
        '{"type": "role.call", "role": "worker", "model": "kimi"}\n', encoding="utf-8"
    )

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert screen._ended  # pyright: ignore[reportPrivateUsage]
            log = screen.query_one("#mw-log", widgets.RichLog)
            assert not any("effort" in strip.text for strip in log.lines)

    asyncio.run(scenario())


def test_discrete_log_line_renders_tool_events_only() -> None:
    # The journal fold is covered in test_viewmodel_machine_state; this is the TUI helper.

    # A tool call renders compactly; a thinking delta is not a discrete line.
    assert machmod._discrete_log_line({"type": "role.effort_delta", "text": "hm"}) is None
    line = machmod._discrete_log_line({"type": "tool.call", "name": "grep", "args": {"q": "x"}})
    assert line is not None and "grep" in line.plain
    # The verdict goes through tool_result_ok, never bool(): "False" is not a green tick.
    bad = machmod._discrete_log_line({"type": "tool.result", "ok": "False", "summary": "boom"})
    assert bad is not None and "✗" in bad.plain
    good = machmod._discrete_log_line({"type": "tool.result", "ok": "True", "summary": "fine"})
    assert good is not None and "✓" in good.plain


def test_create_opens_dashboard_on_the_draft(tmp_path: pathlib.Path, monkeypatch: object) -> None:
    """Creating a machine spawns `machine create`, locates the draft and hands it to the dashboard.

    The machine is watchable live, not fire-and-forget.
    """
    draft = tmp_path / "draft"
    draft.mkdir()

    def _fake_locate(*_a: object, **_k: object) -> tuple[pathlib.Path, str]:
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen._on_create("make a greeter")  # pyright: ignore[reportPrivateUsage]
            await _spawn_settled(app)
            await pilot.pause()
        assert app.return_value == draft  # handed the draft to the dashboard

    asyncio.run(scenario())


def test_create_spawns_off_the_ui_thread(tmp_path: pathlib.Path, monkeypatch: object) -> None:
    """The create spawns run off the event loop, so the TUI does not freeze for the locate.

    The handler returns at once; the spawn's answer lands from a worker thread.
    """
    import threading

    draft = tmp_path / "draft"
    draft.mkdir()
    entered = threading.Event()
    gate = threading.Event()
    ran_on: list[int] = []

    def _slow_locate(*_a: object, **_k: object) -> tuple[pathlib.Path | None, str]:
        ran_on.append(threading.get_ident())
        entered.set()
        if not gate.wait(timeout=5.0):
            return None, "the test never released the spawn"
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _slow_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen._on_create("make a greeter")  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()  # the worker starts from the loop
            assert entered.wait(2.0), "the spawn never ran"
            # The loop runs this line while the locate blocks on another thread.
            assert ran_on != [threading.get_ident()], "the spawn ran on the event loop"
            gate.set()
            await _spawn_settled(app)
            await pilot.pause()
        assert app.return_value == draft

    asyncio.run(scenario())


def test_machines_page_lists_and_views(tmp_path: pathlib.Path) -> None:
    _write(tmp_path / "m.asm.toml")

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            table = screen.query_one("#machines", widgets.DataTable)
            assert table.row_count == 1
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("v")  # view -> parsed detail screen
            await pilot.pause()
            assert isinstance(app.screen, machmod.MachineDetailScreen)

    asyncio.run(scenario())


def test_machines_page_title_counts_or_names_the_empty_case(tmp_path: pathlib.Path) -> None:
    """An empty machines table says so, and the title carries the count, like the hub's."""

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.sub_title.endswith("· no machines yet (c creates one)")
            _write(tmp_path / "m.asm.toml")
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen.action_refresh()
            await pilot.pause()
            assert app.sub_title.endswith("· 1 machine")

    asyncio.run(scenario())


def test_machines_menu_bar_dispatches_an_item(tmp_path: pathlib.Path) -> None:
    """Selecting an item from the menu bar runs its action, not only the key binding."""
    from agent6.ui.tui import menubar

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen.query_one(menubar.MenuBar).open("m")  # the "Machines" menu
            await pilot.pause()
            dd = next(iter(screen.query(menubar._Dropdown)))
            idx = next(
                i for i in range(dd.option_count) if dd.get_option_at_index(i).id == "create"
            )
            dd.highlighted = idx
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, machmod.CreateMachineModal)  # the menu actually fired

    asyncio.run(scenario())


def test_machine_run_confirms_then_spawns(tmp_path: pathlib.Path, monkeypatch: object) -> None:
    path = _write(tmp_path / "m.asm.toml")
    captured: list[list[str]] = []

    def _fake_spawn(argv: list[str], cwd: pathlib.Path, **_k: object) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", widgets.DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("R")  # run -> confirm modal (r refreshes, as everywhere)
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)
            await pilot.press("y")  # confirm
            await app.workers.wait_for_complete()
            assert captured and captured[-1][-3:] == ["machine", "run", str(path)]

    asyncio.run(scenario())


def test_machine_run_refusal_notifies_and_skips_watch(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """A `machine run` refusal surfaces as an error notification, not a watch screen on nothing."""
    _write(tmp_path / "m.asm.toml")

    def _fake_spawn(argv: list[str], cwd: pathlib.Path, **_k: object) -> str:
        return "agent6 machine exited (1) before starting:\nERROR: lock held"

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", widgets.DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("R")
            await pilot.pause()
            await pilot.press("y")  # confirm the run
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, machmod.MachineWatchScreen)  # no watch on nothing
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("lock held" in n for n in notes)

    asyncio.run(scenario())


def test_watch_screen_survives_corrupt_journal(tmp_path: pathlib.Path) -> None:
    """A corrupt journal line does not crash the watch screen.

    The header shows the corruption and polling continues.
    """
    from agent6.machine import load_machine

    f = _write(tmp_path / "m.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / ".agent6" / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text('{"type": "step", "bogus": 1}\n', encoding="utf-8")

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)  # still alive, not crashed
            head = screen.query_one("#mw-head", widgets.Static)
            assert "journal unreadable" in str(head.render())

    asyncio.run(scenario())


def test_watch_screen_tolerates_torn_utf8_state_log(tmp_path: pathlib.Path) -> None:
    """A state log whose tail ends mid multibyte sequence renders its complete prefix.

    The writer flushes long lines in several syscalls; the torn tail is picked up once complete.
    """
    import json as _json

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

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)  # no UnicodeDecodeError crash
            log = screen.query_one("#mw-log", widgets.RichLog)
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


def test_machine_create_spawns_with_task(tmp_path: pathlib.Path, monkeypatch: object) -> None:
    """The create modal threads the typed task into `agent6 machine create <task>`."""
    captured: list[list[str]] = []
    draft = tmp_path / "d"
    draft.mkdir()

    def _fake_locate(argv: list[str], cwd: pathlib.Path, **_k: object) -> tuple[pathlib.Path, str]:
        captured.append(list(argv))
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _fake_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("c")  # create -> task modal
            await pilot.pause()
            assert isinstance(app.screen, machmod.CreateMachineModal)
            app.screen.query_one("#create-input", composer.SteerInput).load_text("nightly sweep")
            await pilot.press("enter")  # submit
            await _spawn_settled(app)
            assert captured and captured[-1][-3:] == ["create", "--", "nightly sweep"]

    asyncio.run(scenario())


def test_watch_screen_refuses_a_steer_no_state_would_read(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """Steering a parked machine is refused with a reason, as the web refuses it.

    Its worker is gone and its newest state dir is a finished agent state, so nothing polls
    the marker; writing it reported success and dropped the course-correction.
    """
    from agent6.machine import load_machine

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "parked"
    instance.mkdir(parents=True)
    # A begun-but-not-ended journal: not `_ended`, and no worker.pid -> dead.
    (instance / "journal.jsonl").write_text(
        '{"kind": "machine.begin", "ts": "t", "machine": "tiny", "version": 1}\n',
        encoding="utf-8",
    )
    state = instance / "states" / "0001-work"
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert not screen._ended  # pyright: ignore[reportPrivateUsage]
            assert screen.check_action("steer", ()) is None
            screen.action_steer()
            await pilot.pause()
            assert not (state / "steer.request").exists()
            # A refusal is a warning, not a success-coloured toast.
            notes = app._notifications  # pyright: ignore[reportPrivateUsage]
            assert [n.severity for n in notes] == ["warning"]

    asyncio.run(scenario())


def test_watch_header_reads_a_corrupt_wait_as_waiting(tmp_path: pathlib.Path) -> None:
    """A corrupt pending-wait file counts as parked, so the watch header says "waiting"."""
    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    (instance / "wait.json").write_text("{ not json", encoding="utf-8")
    (instance / "worker.pid").write_text("999999999", encoding="utf-8")  # dead

    class _WaitHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WaitHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            head = str(screen.query_one("#mw-head", widgets.Static).render())
            assert "waiting" in head
            assert "stopped" not in head

    asyncio.run(scenario())


def test_watch_footer_steer_key_follows_liveness(tmp_path: pathlib.Path) -> None:
    """The footer's Steer key follows steerability, refreshing when it flips.

    Refreshed only on the ended edge, a killed worker kept Steer lit for a machine nobody can steer.
    """
    import os

    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (instance / "journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    log = instance / "states" / "0000-route" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"type":"session.start","mode":"run","user_task":"t"}\n', encoding="utf-8")
    (instance / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    class _LiveHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _LiveHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert screen.check_action("steer", ()) is True  # live: the key is real
            (instance / "worker.pid").write_text("999999999", encoding="utf-8")  # dies
            for _ in range(4):  # let the 0.5s poll observe the flip
                await pilot.pause(0.3)
            assert all(screen.check_action(a, ()) is None for a in ("steer", "poke", "stop"))

    asyncio.run(scenario())


def _blocked_machine(tmp_path: pathlib.Path, *, alive: bool) -> tuple[pathlib.Path, MachineSpec]:
    """A machine instance whose newest agent state is blocked on an unanswered approval."""
    import json
    import os

    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    instance.joinpath("machine.asm.toml").write_text(TINY, encoding="utf-8")
    instance.joinpath("journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    instance.joinpath("worker.pid").write_text(
        str(os.getpid()) if alive else "999999999", encoding="utf-8"
    )
    (state / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "approval.prompt", "id": "ap1", "prompt": "Allow rm -rf"})
        + "\n",
        encoding="utf-8",
    )
    return instance, spec


def test_watch_screen_offers_no_approval_on_a_dead_machine(tmp_path: pathlib.Path) -> None:
    """Allow/Deny is not offered over a machine nobody can answer.

    The fold keeps an unanswered prompt past a worker death, so the watch screen wrote the
    answer into a per-state dir whose loop has exited: the machine twin of the run views' gate.
    """
    instance, spec = _blocked_machine(tmp_path, alive=False)

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(4):  # let several polls run
                await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            assert not screen.query(composer.ApprovalRow), "docked a row on a dead machine"
            assert "stopped" in str(screen.query_one("#mw-head", widgets.Static).render())

    asyncio.run(scenario())


def test_watch_screen_docks_the_approval_on_a_live_machine(tmp_path: pathlib.Path) -> None:
    # The converse: gating on liveness must not cost a RUNNING machine its row.
    instance, spec = _blocked_machine(tmp_path, alive=True)

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            await wait_for(pilot, lambda: answerable(screen), "the approval row")

    asyncio.run(scenario())


def test_machines_page_lists_instances_with_their_files(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """The page shows the rows `agent6 machine` lists: instances with their file, then bare files.

    An instance's status and current state join its authored file; a file no instance ran has
    a blank status.
    """
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    (tmp_path / "other.asm.toml").write_text(TINY.replace('"tiny"', '"other"'), encoding="utf-8")
    assert cli_main(["machine", "run", str(f)]) == 0
    state = paths.state_dir(tmp_path)

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachinesScreen(state, tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            table = screen.query_one("#machines", widgets.DataTable)
            assert table.row_count == 2
            rows = [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]
            ran = next(r for r in rows if r[0] == "tiny")
            assert ran[1] == "ok" and ran[2] == "done" and ran[5] == "valid"
            never = next(r for r in rows if r[0] == "other")
            assert never[1] == "-" and never[2] == "-" and never[3] == "-"

    asyncio.run(scenario())


def test_a_create_whose_screen_was_left_still_opens_its_draft(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """The create worker reaches the app it was handed on the UI thread.

    Reading `self.app` from the thread died once the screen was popped, so the draft never opened.
    """
    import threading

    draft = tmp_path / "draft"
    draft.mkdir()
    entered = threading.Event()
    gate = threading.Event()

    def _slow_locate(*_a: object, **_k: object) -> tuple[pathlib.Path | None, str]:
        entered.set()
        if not gate.wait(timeout=5.0):
            return None, "the test never released the spawn"
        return draft, ""

    monkeypatch.setattr(machmod, "spawn_and_locate", _slow_locate)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen._on_create("make a greeter")  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert entered.wait(2.0), "the spawn never ran"
            app.pop_screen()  # Esc while the create is still locating
            await pilot.pause()
            gate.set()
            await _spawn_settled(app)
            await pilot.pause()
        assert app.return_value == draft

    asyncio.run(scenario())


def test_a_run_whose_screen_was_left_still_opens_its_watch(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """The run worker reaches the app it was handed on the UI thread.

    Reading `self.app` from the thread died once the page was popped, so the watch never opened.
    """
    import threading

    path = _write(tmp_path / "m.asm.toml")
    entered = threading.Event()
    gate = threading.Event()

    def _slow_confirm(*_a: object, **_k: object) -> str:
        entered.set()
        if not gate.wait(timeout=5.0):
            return "the test never released the spawn"
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _slow_confirm)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachinesScreen)
            screen._on_run_confirm(path)(True)  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert entered.wait(2.0), "the spawn never ran"
            app.pop_screen()  # Esc while the run is still starting
            await pilot.pause()
            gate.set()
            await _spawn_settled(app)
            await pilot.pause()
            assert isinstance(app.screen, machmod.MachineWatchScreen)

    asyncio.run(scenario())


def test_watch_screen_stop_on_a_parked_machine_says_why(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    """`x` on a machine with no live worker prints the CLI's refusal.

    Disabling the binding made it a silent no-op that Help still listed.
    """
    from agent6.machine import load_machine
    from agent6.ui.cli import main as cli_main

    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    f = _write(tmp_path / "waiter.asm.toml")
    assert cli_main(["machine", "run", str(f), "--exit-on-wait"]) == 0  # parks, worker gone
    instance = paths.state_dir(tmp_path) / "machines" / "waiter_demo"
    spec = load_machine(f)

    class _WatchHost(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _WatchHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            toasts = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            # Nothing to stop is the note, not a warning: the CLI's answer.
            assert any("not running" in m and s == "information" for m, s in toasts), toasts
            assert not (instance / "stop").exists()

    asyncio.run(scenario())


def test_machine_run_confirm_backs_out_on_q(tmp_path: pathlib.Path, monkeypatch: object) -> None:
    """The footer under the confirm reads "Esc/q Back": q backs out like Esc."""
    _write(tmp_path / "m.asm.toml")
    captured: list[list[str]] = []

    def _fake_spawn(argv: list[str], cwd: pathlib.Path, **_k: object) -> str:
        captured.append(list(argv))
        return ""

    monkeypatch.setattr(machmod, "spawn_and_confirm", _fake_spawn)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = _Host(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.screen.query_one("#machines", widgets.DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("R")
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, modals.ConfirmModal), "q did not back out"
            assert captured == []

    asyncio.run(scenario())


def test_a_live_machine_marks_only_the_turn_in_flight(tmp_path: pathlib.Path) -> None:
    """Replaying the state log marks only an open `role.call` as thinking, and only when live.

    Three finished turns and one in flight read as four live markers.
    """
    import json
    import os
    from typing import Any

    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine

    root = tmp_path / "hunt"
    root.mkdir()
    (root / "machine.asm.toml").write_text(
        """machine = "hunt"
version = 1
initial = "work"

[budget]
max_usd = 1.0
max_transitions = 100

[schemas.verdict]
approved = "bool"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.work]
kind = "agent"
model = "m1"
prompt = "do the thing"
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
""",
        encoding="utf-8",
    )
    j = machine_journal.MachineJournal(root)
    j.ensure_dirs()
    j.begin(machine="hunt", version=1)
    sd = root / "states" / "0000-work"
    sd.mkdir(parents=True)
    evs: list[dict[str, Any]] = [{"type": "session.start", "mode": "run", "user_task": "t"}]
    for i in (1, 2, 3):  # three COMPLETED turns
        evs += [
            {"type": "role.call", "role": "worker", "model": "m1"},
            {"type": "role.thinking_delta", "text": f"turn {i} reasoning. "},
            {"type": "role.result", "role": "worker", "tokens_in": 10, "tokens_out": 5},
        ]
    evs.append({"type": "role.call", "role": "worker", "model": "m1"})  # in flight
    (sd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (root / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    class WatchApp(textual_app.App[int]):
        def compose(self) -> textual_app.ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(
                machmod.MachineWatchScreen(root, load_machine(root / "machine.asm.toml"))
            )

    out: list[str] = []

    async def scenario() -> None:
        app = WatchApp()
        async with app.run_test(size=(140, 40)) as pilot:
            for _ in range(14):
                await pilot.pause(0.25)
            log = app.screen.query_one("#mw-log", widgets.RichLog)
            out.append(
                "\n".join(
                    "".join(seg.text for seg in line._segments)  # pyright: ignore[reportPrivateUsage]
                    for line in log.lines
                )
            )

    asyncio.run(scenario())
    assert out[0].count("thinking…") == 1, out[0]


def test_watch_footer_keeps_every_machine_verb_visible_and_gates_it(tmp_path: pathlib.Path) -> None:
    """An ended machine's footer shows Steer, Message and Stop all as unavailable."""
    from textual.widgets._footer import FooterKey

    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine

    f = _write(tmp_path / "tiny.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = machine_journal.MachineJournal(instance)
    journal.begin(machine="tiny", version=1)
    journal.append(
        machine_journal.MachineEnd(ts="t", status="ok", reason="done", state="done", transitions=1)
    )

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            keys = {key.description: key for key in app.screen.query(FooterKey)}
            assert {"Steer", "Message", "Stop"} <= keys.keys()
            assert all(keys[label].has_class("-disabled") for label in ("Steer", "Message", "Stop"))
            # A click on a dimmed key shows the refusal the key shows, not only the bell.
            await pilot.click(keys["Steer"])
            await pilot.pause()
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert (
                machine_state.machine_verb_refusal(instance, "tiny", "steer"),
                "warning",
            ) in notes

    asyncio.run(scenario())


def test_watch_footer_refreshes_when_only_poke_availability_changes(tmp_path: pathlib.Path) -> None:
    """Consuming an armed wait disables Message even though steerability stays false."""
    from textual.widgets._footer import FooterKey

    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine

    f = _write(tmp_path / "tiny.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = machine_journal.MachineJournal(instance)
    journal.begin(machine="tiny", version=1)
    journal.write_pending_wait(machine_journal.PendingWait(state="route", wake_epoch=None))

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            message = next(k for k in app.screen.query(FooterKey) if k.description == "Message")
            assert not message.has_class("-disabled")
            journal.clear_pending_wait()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            screen._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            message = next(k for k in app.screen.query(FooterKey) if k.description == "Message")
            assert message.has_class("-disabled")

    asyncio.run(scenario())


def test_a_steer_refused_while_its_modal_is_open_writes_nothing(tmp_path: pathlib.Path) -> None:
    """A steer submitted after the worker died re-checks the gate and shows its exact refusal."""
    import json
    import os

    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine

    f = _write(tmp_path / "tiny.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    state = instance / "states" / "0000-route"
    state.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    machine_journal.MachineJournal(instance).begin(machine="tiny", version=1)
    (instance / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (state / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "role.call", "role": "worker", "model": "m"})
        + "\n",
        encoding="utf-8",
    )

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, modals.SteerModal)
            app.screen.query_one(widgets.TextArea).insert("turn left")
            (instance / "worker.pid").write_text("999999999", encoding="utf-8")
            refusal = machine_state.machine_verb_refusal(instance, "tiny", "steer")
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert not (state / "steer.answer").exists()
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert (refusal, "warning") in notes

    asyncio.run(scenario())


def test_a_poke_refused_while_its_modal_is_open_writes_nothing(tmp_path: pathlib.Path) -> None:
    """A message submitted after the armed wait closed re-checks the gate.

    Otherwise it leaves an unconsumable signal.
    """
    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine

    f = _write(tmp_path / "tiny.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    journal = machine_journal.MachineJournal(instance)
    journal.begin(machine="tiny", version=1)
    journal.write_pending_wait(machine_journal.PendingWait(state="route", wake_epoch=None))

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, modals.TextInputModal)
            journal.clear_pending_wait()
            refusal = machine_state.machine_verb_refusal(instance, "tiny", "poke")
            app.screen.query_one(widgets.Input).value = "wake up"
            await pilot.press("enter")
            await pilot.pause()
            assert not journal.signal_path.exists()
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert (refusal, "warning") in notes

    asyncio.run(scenario())


def test_an_answer_submitted_after_the_worker_died_writes_nothing(tmp_path: pathlib.Path) -> None:
    """The answer gate is re-read at submit, as a steer's and a poke's are.

    Read from the poll's cache, an approval submitted after the worker died landed in a dead dir.
    """
    instance, spec = _blocked_machine(tmp_path, alive=True)
    state = instance / "states" / "0000-route"

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            await wait_for(pilot, lambda: answerable(screen), "the approval row")
            await focus_answers(screen, pilot)
            # The next poll would withdraw the dead machine's row; the answer lands first.
            screen._poll_timer.pause()  # pyright: ignore[reportPrivateUsage]
            (instance / "worker.pid").write_text("999999999", encoding="utf-8")  # dies
            await pilot.press("y")
            await pilot.pause()
            assert not (state / "approvals").exists()
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert (machmod._ANSWER_LOST, "warning") in notes

    asyncio.run(scenario())


def test_the_watch_poll_folds_the_machine_and_its_execution_once(tmp_path: pathlib.Path) -> None:
    """One poll folds the journal once and reads the newest state log incrementally."""
    import os

    import agent6.ui.tui.machines as tui_mod
    import agent6.viewmodel.machine_state as vm_mod
    from agent6.machine import load_machine

    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (instance / "journal.jsonl").write_text("", encoding="utf-8")  # started, not ended
    log = instance / "states" / "0000-route" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"type":"session.start","mode":"run","user_task":"t"}\n', encoding="utf-8")
    (instance / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live
    counts = {"fold": 0, "execution": 0}
    real_fold, real_execution = vm_mod.fold_machine, vm_mod.newest_agent_execution

    def counting_fold(*args: object, **kwargs: object) -> object:
        counts["fold"] += 1
        return real_fold(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    def counting_execution(*args: object, **kwargs: object) -> object:
        counts["execution"] += 1
        return real_execution(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            tui_mod.fold_machine = counting_fold  # type: ignore[assignment]
            vm_mod.fold_machine = counting_fold  # type: ignore[assignment]
            vm_mod.newest_agent_execution = counting_execution  # type: ignore[assignment]
            try:
                screen._poll()  # pyright: ignore[reportPrivateUsage]
            finally:
                tui_mod.fold_machine = real_fold
                vm_mod.fold_machine = real_fold
                vm_mod.newest_agent_execution = real_execution
            assert counts == {"fold": 1, "execution": 0}

    asyncio.run(scenario())


def test_a_click_on_a_stale_lit_verb_explains_itself_once(tmp_path: pathlib.Path) -> None:
    """A footer click on a refused key explains it once.

    An unfocused terminal keeps the footer as painted, so a click can land on a refused key; the
    footer simulates the key, which on_key explains.
    """
    import os

    from textual.widgets._footer import FooterKey

    from agent6.machine import journal as machine_journal
    from agent6.machine import load_machine
    from agent6.sessions import ipc

    f = _write(tmp_path / "tiny.asm.toml", TINY)
    spec = load_machine(f)
    instance = tmp_path / "machines" / "tiny"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    machine_journal.MachineJournal(instance).begin(machine="tiny", version=1)
    ipc.write_worker_pid(instance, os.getpid())

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(machmod.MachineWatchScreen(instance, spec))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, machmod.MachineWatchScreen)
            app.app_focus = False  # the operator clicked away to another window
            await pilot.pause()
            (instance / "worker.pid").unlink()  # the worker exits
            for _ in range(8):
                await pilot.pause(0.3)
                if screen.check_action("stop", ()) is None:
                    break
            stop = next(k for k in screen.query(FooterKey) if k.description == "Stop")
            assert not stop.has_class("-disabled"), "the blurred footer keeps its paint"
            app._notifications.clear()  # pyright: ignore[reportPrivateUsage]
            await pilot.click(stop)
            await pilot.pause()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert notes == [machine_state.machine_verb_refusal(instance, "tiny", "stop")]

    asyncio.run(scenario())
