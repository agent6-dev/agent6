from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.sessions import child_dispatch, layout
from tests.unit import test_git_history, test_sessions_show_group
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
    expect(summary).to_have_text("children: 2 passed")
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


@pytest.mark.parametrize("width, height", [(1440, 900), (390, 844)])
def test_the_footer_distinguishes_own_checkpoints_from_a_merged_childs_commits(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
    width: int,
    height: int,
) -> None:
    """A parent's zero counts its own checkpoints even when its child contributed a commit."""
    from playwright import sync_api

    child = layout.SessionLayout(state_dir=paths.state_dir(ordinary_parent), session_id="co-p1-l1")
    subject = "List the last 2 commits"
    git = test_git_history._git
    git(ordinary_parent, "init", "-q")
    (ordinary_parent / "log.txt").write_text("log\n", encoding="utf-8")
    git(ordinary_parent, "add", "log.txt")
    git(ordinary_parent, "commit", "-qm", subject)
    sha = git(ordinary_parent, "rev-parse", "HEAD")
    events = [
        {"type": "session.start", "mode": "run"},
        {"type": "checkpoint.committed", "sha": sha, "subject": subject},
        {"type": "diff.updated", "sha": sha, "patch": "+log"},
        {"type": "session.stop", "reason": "finish_session"},
    ]
    child.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    page, errors = open_page(parent_port)
    page.set_viewport_size({"width": width, "height": height})
    for sid, count in (
        ("co", "0 checkpoint commits from this execution"),
        (child.session_id, f"1 checkpoint commit from this execution · {subject}"),
    ):
        page.goto(f"http://127.0.0.1:{parent_port}/#/session/{sid}")
        sync_api.expect(page.locator(".card-conv .s-done-detail")).to_have_text(
            f"$0 · 0 tools · {count}"
        )
    assert not errors


def test_an_ordinary_parent_shows_its_children_and_the_receipt_links_them(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A `/parallel` run's page lists its children with their merges and keeps its own actions."""
    from playwright.sync_api import expect

    page, errors = open_page(parent_port)
    page.goto(f"http://127.0.0.1:{parent_port}/#/session/co")
    block = page.locator(".card-children")
    expect(block).to_be_visible()
    summary = block.locator(":scope > summary")
    expect(summary).to_have_text("children: 1 merged, 1 no changes")
    summary.click()
    expect(block).to_contain_text("passed · merged into co's chain at bbbbbbbbbbbb")
    expect(block).to_contain_text("passed · no changes")
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
    expect(summary).to_have_text("children: 1 merged, 1 no changes, 1 failed")
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
    expect(receipt).to_contain_text(
        "co-p1 attempt 1: co-p1-l1: passed · merged into co's chain at bbbbbbbbbbbb, $0.50"
    )
    receipt.get_by_role("link", name="co-p1-l1", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{parent_port}/#/session/co-p1-l1")
    assert not errors


def test_a_conflict_counts_the_parent_outcome_not_the_child_gate(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A child whose gate passed counts as a conflict if its work did not merge."""
    from playwright.sync_api import expect

    parent = layout.SessionLayout(state_dir=paths.state_dir(ordinary_parent), session_id="co")
    (record,) = child_dispatch.read_records(parent.session_dir)
    first, second = record.children
    conflict = child_dispatch.MergeRecord(
        status="conflict", branch="agent6/work", detail="dates.py"
    )
    second = second.model_copy(update={"merge": conflict})
    child_dispatch.write_record(
        parent.session_dir, record.model_copy(update={"children": (first, second)}), now=2.0
    )
    page, errors = open_page(parent_port)
    page.goto(f"http://127.0.0.1:{parent_port}/#/session/co")
    card = page.locator(".card-children")
    fold = card.locator(":scope > summary")
    expect(fold).to_have_text("children: 1 merged, 1 conflict")
    fold.click()
    expect(card.locator(".child-row").last.locator("summary")).to_contain_text("passed · conflict")
    expect(card).to_contain_text("dates.py")
    assert not errors


def test_merge_returns_only_when_the_child_advances(
    ordinary_parent: pathlib.Path,
    parent_port: int,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The child page offers no merge for work already in the parent's chain."""
    from playwright.sync_api import expect

    repo = ordinary_parent
    git = test_git_history._git
    git(repo, "init", "-q")
    git(repo, "commit", "--allow-empty", "-qm", "Child answer")
    sha = git(repo, "rev-parse", "HEAD")
    parent = layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="co")
    (record,) = child_dispatch.read_records(parent.session_dir)
    first, second = record.children
    assert first.merge
    ref = f"refs/agent6/{first.session_id}/head"
    git(repo, "update-ref", ref, sha)
    first = first.model_copy(update={"merge": first.merge.model_copy(update={"sha": sha})})
    child_dispatch.write_record(
        parent.session_dir, record.model_copy(update={"children": (first, second)}), now=2.0
    )
    page, errors = open_page(parent_port)
    page.goto(f"http://127.0.0.1:{parent_port}/#/session/{first.session_id}")
    expect(page.locator(".card-head")).to_contain_text("merged into co's chain")
    button = page.get_by_role("button", name="Merge", exact=True)
    expect(button).to_be_hidden()
    git(repo, "commit", "--allow-empty", "-qm", "Extend answer")
    git(repo, "update-ref", ref, "HEAD")
    page.reload()
    expect(button).to_be_enabled()
    assert not errors
