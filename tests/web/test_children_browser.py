# SPDX-License-Identifier: Apache-2.0
"""Drive the Children card against a live clone link and imported session records."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from agent6.sessions import child_dispatch, completion, ipc, layout, process
from tests.web import captures, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
PARENT = "csv-reader"
CHILD = "csv-reader-p1-l1"
PROMPT = (
    "Check quoted CSV fields.\nKeep the comma inside a quoted field.\nCheck the empty field too."
)


def append(session: pathlib.Path, *events: dict[str, Any]) -> None:
    """Append complete journal records for a running worker.

    Args:
        session: The worker's session directory.
        events: Records in journal order.
    """
    with (session / "logs.jsonl").open("a", encoding="utf-8") as log:
        log.writelines(json.dumps(event) + "\n" for event in events)


def seed_family(repo: pathlib.Path, prompt: str) -> pathlib.Path:
    """Write a parent, a linked child and the records of a stopped child and a reviewer.

    Args:
        repo: A throwaway Git repository.
        prompt: The live child's task.

    Returns:
        The live child's session directory in its clone.
    """
    runs = layout.bucket_dir(paths.state_dir(repo), "runs")
    parent = runs / PARENT
    parent.mkdir(parents=True)
    (parent / "manifest.json").write_text(json.dumps({"session_id": PARENT, "mode": "run"}))
    append(
        parent,
        {"type": "session.start", "mode": "run", "user_task": "Read CSV records"},
        {"type": "model.result", "text": "I will check field boundaries and line endings."},
    )
    process.write_worker_pid(parent, os.getpid())
    clone = repo.parent / "child-clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    live = layout.bucket_dir(paths.state_dir(clone), "runs") / CHILD
    live.mkdir(parents=True)
    (runs / CHILD).symlink_to(live, target_is_directory=True)
    start = time.time() - 14
    sid = "csv-reader-p2-l1"
    reviewer = "csv-reader-r1-l1"
    stopped = runs / sid
    stopped.mkdir()
    review = layout.bucket_dir(paths.state_dir(repo), layout.EVIDENCE_BUCKET) / reviewer
    review.mkdir(parents=True)
    groups = [
        (live, "csv-reader-p1", "run", prompt),
        (stopped, "csv-reader-p2", "run", "Check line endings"),
        (review, "csv-reader-r1", "reviewer", "Review the CSV attempts"),
    ]
    for session, group, mode, task in groups:
        manifest = {
            "version": 3,
            "session_id": session.name,
            "mode": mode,
            "user_task": task,
            "parallel": {"parent": PARENT, "group": group, "number": 1},
            "models": {"driver": {"provider": "local", "model": "reader"}},
        }
        (session / "manifest.json").write_text(json.dumps(manifest))
        append(session, {"type": "session.start", "mode": mode, "user_task": task, "ts": start})
        child = child_dispatch.ChildRecord(
            session_id=session.name,
            number=1,
            task=task,
            model="worker" if session == live else None,
        )
        if session == live:
            process.write_worker_pid(live, os.getpid())
            append(
                live,
                {
                    "type": "budget.update",
                    "input_total": 100,
                    "output_total": 24,
                    "usd_total": 0.02,
                },
                {"type": "model.result", "text": "The quoted comma stays in one field."},
                {
                    "type": "tool.call",
                    "name": "read_file",
                    "args_preview": {"path": "csv.py"},
                    "turn": 2,
                },
            )
        elif session == stopped:
            merge = child_dispatch.MergeRecord(
                status="stopped",
                branch="agent6/line-endings",
                commits=2,
                detail="stopped after the newline check",
            )
            child = child.model_copy(update={"merge": merge})
            append(stopped, {"type": "session.stop", "reason": "steer_abort", "ts": start + 10})
        else:
            append(review, {"type": "session.stop", "reason": "answered", "ts": start + 8})
            completion.write_completion(
                review,
                completion.CompletionRecord(
                    text=json.dumps(
                        {
                            "order": [CHILD, sid],
                            "summary": "The quoted comma test covers both fields.",
                        }
                    ),
                    stop_reason="answered",
                    output_tokens=15,
                    calls=1,
                ),
            )
        record = child_dispatch.DispatchRecord(
            identity=group,
            owner="exec",
            turn=1,
            base="a" * 40,
            source="panel" if mode == "reviewer" else "model",
            state="running" if session == live else "complete",
            node_id="0002" if session == live else None,
            children=(child,),
        )
        child_dispatch.write_record(parent, record, now=start)
    return live


@pytest.fixture
def family(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[tuple[int, pathlib.Path]]:
    """Serve the family from its own checkout with the suite's throwaway XDG homes."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    child = seed_family(repo, getattr(request, "param", PROMPT))
    for port in test_web_browser._serve(repo):
        yield port, child


_LONG_PROMPT = ("Check the quoted commas in the CSV file.\n" * 50)[:2000]


@pytest.mark.parametrize("family", [_LONG_PROMPT], indirect=True, ids=["long-prompt"])
@pytest.mark.parametrize("width, height", [(390, 844), (1440, 900)], ids=["phone", "desktop"])
def test_a_long_prompt_leaves_the_conversation_and_controls_in_view(
    open_page: Callable[[int], tuple[Page, list[str]]],
    family: tuple[int, pathlib.Path],
    width: int,
    height: int,
) -> None:
    """An opened row keeps a long prompt behind its own disclosure."""
    from playwright import sync_api

    port, _ = family
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/session/{PARENT}")
    if width == 390:
        page.get_by_role("button", name="Widgets").click()
        page.get_by_role("button", name="Children", exact=True).click()
    card = page.locator(".drawer .card-children")
    card.locator(":scope > summary").click()
    row = card.locator(".child-row").filter(
        has=page.get_by_role("link", name=CHILD, include_hidden=True)
    )
    row.locator(":scope > summary").click()
    for element in (
        row.get_by_text("The quoted comma stays in one field.", exact=False),
        row.get_by_role("textbox", name="Message"),
        row.get_by_role("button", name="Stop", exact=True),
    ):
        sync_api.expect(element).to_be_in_viewport(ratio=1)
    disclosure = row.locator("details")
    assert not disclosure.evaluate("e => e.open")
    sync_api.expect(row.get_by_text(_LONG_PROMPT, exact=True)).to_be_hidden()
    preview = row.locator(".child-body pre:visible").first
    assert (preview.text_content() or "").startswith("\n".join(_LONG_PROMPT.splitlines()[:2]))
    sync_api.expect(preview).to_be_in_viewport(ratio=1)
    disclosure.locator("summary").press("Enter")
    sync_api.expect(disclosure.get_by_text(_LONG_PROMPT, exact=True)).to_be_visible()
    assert not errors


@pytest.mark.parametrize("width", [390, 1440], ids=["phone", "desktop"])
def test_read_and_message_a_child_in_place(
    open_page: Callable[[int], tuple[Page, list[str]]], family: tuple[int, pathlib.Path], width: int
) -> None:
    """Refreshing a child row must not fold it or take the cursor out of its message."""
    from playwright import sync_api

    port, child = family
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844 if width == 390 else 900})
    if width == 390:
        page.context.new_cdp_session(page).send(
            "Emulation.setTouchEmulationEnabled", {"enabled": True}
        )
    url = f"http://127.0.0.1:{port}/#/session/{PARENT}"
    page.goto(url)
    card = page.locator(".drawer details.card-children")
    if width == 390:
        widgets = page.get_by_role("button", name="Widgets")
        widgets.click()
        page.get_by_role("button", name="Overview", exact=True).click()
        sync_api.expect(card).to_be_hidden()
        widgets.click()
        page.get_by_role("button", name="Children", exact=True).click()
        sync_api.expect(page.locator(".run-app .card:visible")).to_have_count(1)
    summary = card.locator(":scope > summary")
    sync_api.expect(summary).to_have_text("children: 1 answered, 1 running, 1 stopped")
    assert not card.evaluate("e => e.open")
    assert summary.evaluate("e => e.clientHeight <= parseFloat(getComputedStyle(e).lineHeight) + 1")
    assert summary.locator("span").evaluate("e => e.scrollWidth <= e.clientWidth")
    captures.save(page, "children-folded")
    summary.press("Enter")
    reviewer, row, held = card.locator("details").all()
    sync_api.expect(row.locator("summary")).to_contain_text("holds task 0002")
    sync_api.expect(held).to_contain_text("2 commits beyond start")
    sync_api.expect(held).to_contain_text("agent6/line-endings")
    sync_api.expect(held).to_contain_text("stopped after the newline check")
    sync_api.expect(reviewer).to_contain_text("reviewer 1")
    sync_api.expect(reviewer).to_contain_text("The quoted comma test covers both fields.")
    sync_api.expect(reviewer).to_contain_text(f"{CHILD}, csv-reader-p2-l1")
    captures.save(page, "children-rows")
    reviewer.locator("summary").click()
    sync_api.expect(reviewer.get_by_text("Recent conversation")).to_be_visible()
    sync_api.expect(reviewer.get_by_role("button")).to_have_count(0)
    reviewer.locator("summary").click()
    row.locator("summary").press("Space")
    sync_api.expect(row.get_by_text(PROMPT, exact=True)).to_be_visible()
    for fact in (
        "attempt 1",
        "local/reader",
        "turn 2",
        "124 tokens",
        "$0.02",
        "elapsed",
        "read_file",
    ):
        sync_api.expect(row.locator("summary")).to_contain_text(fact)
    sync_api.expect(row).to_contain_text("The quoted comma stays in one field.")
    entry = row.get_by_role("textbox", name="Message")
    entry.fill("Check doubled quotes too.")
    entry.dispatch_event("keydown", {"key": "Enter", "isComposing": True})
    append(
        child,
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "Read csv.py"},
        {"type": "model.call", "actor": "worker", "model": "reader", "turn": 3},
        {"type": "model.text_delta", "text": "Checking doubled quotes."},
    )
    sync_api.expect(row).to_contain_text("Checking doubled quotes.")
    sync_api.expect(row.locator("summary")).to_contain_text("turn 3")
    sync_api.expect(entry).to_be_focused()
    sync_api.expect(entry).to_have_value("Check doubled quotes too.")
    assert not (child / ipc.STEER_ANSWER_FILE).exists()
    assert page.evaluate("document.scrollingElement.scrollWidth") == width
    row.locator("summary").evaluate("e => e.scrollIntoView({block:'start'})")
    captures.save(page, "children-open")
    with page.expect_response(lambda r: r.url.endswith(f"/{CHILD}/steer")) as sent:
        if width == 390:
            row.get_by_role("button", name="Send", exact=True).click()
        else:
            entry.press("Enter")
    assert sent.value.ok
    assert (child / ipc.STEER_ANSWER_FILE).read_text().strip() == "Check doubled quotes too."
    sync_api.expect(entry).to_have_value("")
    row.get_by_role("button", name="Stop", exact=True).click()
    dialog = page.get_by_role("alertdialog")
    sync_api.expect(dialog).to_contain_text(CHILD)
    with page.expect_response(lambda r: r.url.endswith(f"/{CHILD}/stop")) as stopped:
        dialog.get_by_role("button", name="Stop now").click()
        deadline = time.monotonic() + 4
        while not ipc.stop_request_is_now(child):
            assert time.monotonic() < deadline, "the stop did not reach the child"
            time.sleep(0.01)
        append(child, {"type": "session.stop", "reason": "steer_abort", "ts": time.time()})
    assert stopped.value.ok
    sync_api.expect(entry).to_be_hidden()
    sync_api.expect(row.get_by_role("button", name="Stop", exact=True)).to_be_hidden()
    sync_api.expect(summary).to_have_text("children: 1 answered, 2 stopped")
    sync_api.expect(page).to_have_url(url)
    assert not errors


def test_a_live_reviewer_can_receive_a_message_from_its_parent(
    open_page: Callable[[int], tuple[Page, list[str]]], family: tuple[int, pathlib.Path]
) -> None:
    """Reviewers have operator controls in their own sessions."""
    from playwright import sync_api

    port, _ = family
    reviewer = "csv-reader-r1-l1"
    directory = (
        layout.bucket_dir(paths.state_dir(pathlib.Path.cwd()), layout.EVIDENCE_BUCKET) / reviewer
    )
    append(directory, {"type": "session.start", "mode": "reviewer", "user_task": "Review CSV"})
    process.write_worker_pid(directory, os.getpid())
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{PARENT}")
    card = page.locator(".drawer details.card-children")
    card.locator(":scope > summary").click()
    row = card.locator("details").filter(has_text=reviewer)
    row.locator("summary").click()
    entry = row.get_by_role("textbox", name="Message")
    sync_api.expect(entry).to_be_visible()
    entry.fill("Check both attempts.")
    entry.press("Enter")
    sync_api.expect(entry).to_have_value("")
    assert (directory / ipc.STEER_ANSWER_FILE).read_text().strip() == "Check both attempts."
    row.get_by_role("button", name="Stop", exact=True).click()
    sync_api.expect(page.get_by_role("alertdialog")).to_contain_text(reviewer)
    assert not errors
