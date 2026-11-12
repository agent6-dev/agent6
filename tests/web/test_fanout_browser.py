from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.sessions import layout
from tests.unit import test_sessions_show_fanout
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
fan_out = test_sessions_show_fanout.fan_out
ordinary_parent = test_sessions_show_fanout.ordinary_parent
browser = test_web_browser.browser
open_page = test_web_browser.open_page


@pytest.fixture
def fan_port(fan_out: pathlib.Path) -> Iterator[int]:
    yield from test_web_browser._serve(fan_out)


def test_group_lanes_refresh_and_link(
    fan_out: pathlib.Path,
    fan_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/session/fan")
    block = page.locator(".card-lanes")
    expect(block).to_be_visible()
    expect(block.locator("h2")).to_have_text("fan-out: 2 lanes")
    expect(block.locator("tr")).to_have_count(3)
    expect(block).to_contain_text("openai/lane-1")
    lane = layout.SessionLayout(state_dir=paths.state_dir(fan_out), session_id="fan-l1")
    with lane.logs_path.open("a") as log:
        log.write('{"type":"budget.update","usd_total":1.25}\n')
        log.write('{"type":"session.stop","reason":"steer_abort"}\n')
    expect(block).to_contain_text("$1.25", timeout=10000)
    expect(block).to_contain_text("stopped")
    page.set_viewport_size({"width": 390, "height": 844})
    expect(block).to_be_visible()
    block.get_by_role("link", name="fan-l1", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{fan_port}/#/session/fan-l1")
    expect(block).to_be_hidden()
    assert not errors


@pytest.fixture
def parent_port(ordinary_parent: pathlib.Path) -> Iterator[int]:
    yield from test_web_browser._serve(ordinary_parent)


def test_an_ordinary_parent_shows_its_lanes_and_the_receipt_links_them(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A `/parallel` run's page lists its lanes with their joins and keeps its own actions."""
    import json

    from playwright.sync_api import expect

    page, errors = open_page(parent_port)
    page.goto(f"http://127.0.0.1:{parent_port}/#/session/co")
    block = page.locator(".card-lanes")
    expect(block).to_be_visible()
    expect(block.locator("h2")).to_have_text("lanes: 2")
    expect(block).to_contain_text("joined at bbbbbbbbbbbb")
    expect(block).to_contain_text("no changes")
    expect(page.get_by_role("button", name="Fork", exact=True)).to_be_enabled()
    # A third lane appears on disk while the page is open: the card follows it.
    third = layout.SessionLayout(state_dir=paths.state_dir(ordinary_parent), session_id="co-p2-l1")
    third.ensure()
    third.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "co-p2-l1",
                "mode": "run",
                "models": {"driver": {"provider": "openai", "model": "lane-3"}},
                "parallel": {"group": "co-p2", "lane": 1, "coordinator": "co"},
            }
        ),
        encoding="utf-8",
    )
    third.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "more"}) + "\n",
        encoding="utf-8",
    )
    expect(block.locator("h2")).to_have_text("lanes: 3", timeout=20000)
    expect(block).to_contain_text("openai/lane-3")
    # The receipt names each lane and links it through the session navigation.
    page.get_by_role("button", name="Receipt", exact=True).click()
    receipt = page.locator(".receipt")
    expect(receipt).to_contain_text("Lanes")
    expect(receipt).to_contain_text("lane 1 co-p1-l1: joined at bbbbbbbbbbbb, $0.50")
    receipt.get_by_role("link", name="co-p1-l1", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{parent_port}/#/session/co-p1-l1")
    assert not errors
