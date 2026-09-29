# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every menu item on every screen resolves to a real action handler.

An unawaited coroutine made Quit do nothing; the quit tests below cover the dispatch.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
from collections.abc import Callable
from typing import cast

from rich import text
from textual import app as textual_app

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import config_page, dashboard, home, menubar


def _resolve(host: object, action: str) -> Callable[..., object] | None:
    # Exactly the real dispatcher: getattr on the screen, then the app, no dotted-namespace parsing.
    app = getattr(host, "app", host)
    return getattr(host, f"action_{action}", None) or getattr(app, f"action_{action}", None)


def _assert_all_items_resolve(host: object, menus: tuple[menubar.Menu, ...]) -> None:
    missing = [
        item.action for menu in menus for item in menu.items if _resolve(host, item.action) is None
    ]
    assert not missing, f"menu items with no action handler: {missing}"


def test_home_menu_items_all_resolve() -> None:
    adir, repo = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = home.Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            _assert_all_items_resolve(app.screen, app.screen.MENUS)  # type: ignore[attr-defined]

    asyncio.run(scenario())


def test_config_menu_items_all_resolve(tmp_path: pathlib.Path) -> None:
    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(config_page.ConfigScreen(tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, config_page.ConfigScreen)
            _assert_all_items_resolve(screen, screen.MENUS)

    asyncio.run(scenario())


def test_dashboard_menu_items_all_resolve(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "x"}) + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Both sibling views of the run app: the dashboard's menus resolve even while covered.
            _assert_all_items_resolve(app._conv, app._conv.MENUS)
            assert isinstance(app._dash, dashboard.DashboardScreen)
            _assert_all_items_resolve(app._dash, app._dash.MENUS)

    asyncio.run(scenario())


def test_quit_from_menu_exits_home() -> None:
    """Selecting Quit runs the whole chain to an awaited action_quit and the app exits."""
    adir, repo = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())

    async def scenario() -> None:

        app = home.Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            mb = app.screen.query_one(menubar.MenuBar)
            mb.open("f")
            await pilot.pause()
            dd = next(iter(app.screen.query(menubar._Dropdown)))
            qi = next(i for i in range(dd.option_count) if dd.get_option_at_index(i).id == "quit")
            dd.highlighted = qi
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app._running is False  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_f10_opens_menu_bar(tmp_path: pathlib.Path) -> None:
    """F10 opens the menu bar (terminal-robust: some terminals eat Alt+f)."""

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(config_page.ConfigScreen(tmp_path))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("f10")
            await pilot.pause()
            assert len(list(app.screen.query(menubar._Dropdown))) == 1

    asyncio.run(scenario())


def test_q_key_quits_home() -> None:
    """The footer's 'q Quit' must actually quit.

    A Screen doesn't inherit the App's built-in action_quit and the binding doesn't bubble to it, so
    HomeScreen defines its own, else only Ctrl+Q (an app default) would work.
    """
    adir, repo = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = home.Agent6HomeApp(adir, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert app._running is False  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_menu_dropdown_keys_right_align_to_common_edge() -> None:
    """Dropdown shortcut keys share a right edge, lining up in a column."""
    items = (
        menubar.MenuItem("New run/plan/ask", "a"),
        menubar.MenuItem("Open selected", "b"),
        menubar.MenuItem("Theme…", "c"),  # keyless
        menubar.MenuItem("Quit", "d"),
    )
    keys = {"a": "n", "b": "Enter", "d": "q"}  # the live bindings' labels
    opts = {o.id: cast(text.Text, o.prompt).plain for o in menubar._menu_options(items, keys, None)}
    keyed = [opts["a"], opts["b"], opts["d"]]
    assert len({len(r) for r in keyed}) == 1  # all padded to one width => shared right edge
    assert opts["a"].endswith(" n") and opts["b"].endswith("Enter") and opts["d"].endswith(" q")
    assert opts["c"] == "Theme…"  # keyless row is just the label


def test_help_screen_closes_after_resize_reflow() -> None:
    """The help page reflows on resize via recompose, which replaces the focused #help-scroll.

    Focus moves to the new instance: left on the detached old one, its binding chain cannot reach
    the screen and Esc, q and ? stop closing the page (ttyd and vhs resize right after mount).
    """
    adir, repo = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())

    async def scenario() -> None:
        app = home.Agent6HomeApp(adir, repo)
        async with app.run_test(size=(190, 50)) as pilot:
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            await pilot.resize_terminal(150, 40)  # triggers the reflow recompose
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert type(app.screen).__name__ == "HomeScreen"

    asyncio.run(scenario())


def test_a_menu_item_the_footer_greys_out_is_not_clickable_either() -> None:
    """A dropdown option is disabled exactly when its key binding is.

    The File dropdown offered Merge on a live run the footer had greyed.
    """
    import os

    a6, repo = pathlib.Path(tempfile.mkdtemp()), pathlib.Path(tempfile.mkdtemp())
    live = a6 / "sessions" / "runs" / "r-live"
    live.mkdir(parents=True)
    (live / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "x"}) + "\n",
        encoding="utf-8",
    )
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(a6, repo)
        async with app.run_test() as pilot:
            await pilot.pause()
            mb = app.screen.query_one(menubar.MenuBar)
            mb.open("f")
            await pilot.pause()
            dd = next(iter(app.screen.query(menubar._Dropdown)))
            ids = [dd.get_option_at_index(i).id for i in range(dd.option_count)]
            idx = ids.index("merge_selected")
            assert dd.get_option_at_index(idx).disabled, (
                "the menu offers a merge the footer's key refuses"
            )

    asyncio.run(scenario())
