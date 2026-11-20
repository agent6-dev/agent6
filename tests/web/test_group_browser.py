from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.sessions import child_dispatch, layout
from tests.unit import test_sessions_show_group
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
group = test_sessions_show_group.group
ordinary_parent = test_sessions_show_group.ordinary_parent
browser = test_web_browser.browser
open_page = test_web_browser.open_page


@pytest.fixture
def fan_port(group: pathlib.Path) -> Iterator[int]:
    yield from test_web_browser._serve(group)


def test_group_children_refresh_and_link(
    group: pathlib.Path,
    fan_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/session/fan")
    block = page.locator(".card-children")
    expect(block).to_be_visible()
    summary = block.locator(":scope > summary")
    expect(summary).to_have_text("children: 2 finished")
    summary.focus()
    summary.press("Enter")
    expect(block.locator("details")).to_have_count(2)
    expect(block).to_contain_text("openai/child-1")
    child = layout.SessionLayout(state_dir=paths.state_dir(group), session_id="fan-l1")
    with child.logs_path.open("a") as log:
        log.write('{"type":"budget.update","usd_total":1.25}\n')
        log.write('{"type":"session.stop","reason":"steer_abort"}\n')
    expect(block).to_contain_text("$1.25", timeout=10000)
    expect(block).to_contain_text("stopped")
    expect(summary).to_be_focused()
    block.locator("details > summary").first.press("Space")
    block.get_by_role("link", name="fan-l1", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{fan_port}/#/session/fan-l1")
    expect(block).to_be_hidden()
    assert not errors


@pytest.fixture
def parent_port(ordinary_parent: pathlib.Path) -> Iterator[int]:
    yield from test_web_browser._serve(ordinary_parent)


def test_an_ordinary_parent_shows_its_children_and_the_receipt_links_them(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A `/parallel` run's page lists its children with their joins and keeps its own actions."""
    from playwright.sync_api import expect

    page, errors = open_page(parent_port)
    page.goto(f"http://127.0.0.1:{parent_port}/#/session/co")
    block = page.locator(".card-children")
    expect(block).to_be_visible()
    summary = block.locator(":scope > summary")
    expect(summary).to_have_text("children: 1 joined, 1 finished")
    summary.click()
    expect(block).to_contain_text("joined co's chain at bbbbbbbbbbbb")
    expect(block).to_contain_text("no changes")
    expect(page.get_by_role("button", name="Fork", exact=True)).to_be_enabled()
    parent = layout.SessionLayout(state_dir=paths.state_dir(ordinary_parent), session_id="co")
    failed = child_dispatch.ChildRecord(session_id="co-p2-l1", number=1, task="Check dates")
    record = child_dispatch.DispatchRecord(
        identity="co-p2",
        source="model",
        owner="exec",
        turn=3,
        base="a" * 40,
        children=(failed,),
        state="complete",
        message="no model route",
    )
    child_dispatch.write_record(parent.session_dir, record, now=2.0)
    expect(summary).to_have_text("children: 1 joined, 1 finished, 1 failed")
    last = block.locator("details").last
    last.locator("summary").click()
    expect(last).to_contain_text("never started")
    expect(last).to_contain_text("no model route")
    expect(last.get_by_role("link")).to_have_count(0)
    expect(last.get_by_role("button")).to_have_count(0)
    # The receipt names each child and links it through the session navigation.
    page.get_by_role("button", name="Receipt", exact=True).click()
    receipt = page.locator(".receipt")
    expect(receipt).to_contain_text("Children")
    expect(receipt).to_contain_text("child 1 co-p1-l1: joined co's chain at bbbbbbbbbbbb, $0.50")
    receipt.get_by_role("link", name="co-p1-l1", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{parent_port}/#/session/co-p1-l1")
    assert not errors
