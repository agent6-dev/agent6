from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.unit import test_git_history
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


def test_commit_picker_requests_the_selected_commit(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The picker selects history through commit= and restores the live details."""
    from playwright.sync_api import expect

    base = test_git_history._git(tmp_path, "rev-parse", "HEAD")
    source = tmp_path / "history.txt"
    source.write_text("first\n")
    first = test_git_history._commit(tmp_path, "first change")
    source.write_text("second\n")
    second = test_git_history._commit(tmp_path, "second change")
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "commit-picker"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "base_sha": base, "git_control": "agent6"})
    )
    events = [
        {"type": "session.start", "mode": "run", "user_task": "Select history"},
        {"type": "loop.auto_commit", "turn": 1, "sha": first, "subject": "first change"},
        {"type": "loop.auto_commit", "turn": 2, "sha": second, "subject": "second change"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/session/commit-picker")
    picker = page.locator(".card-diff select")
    expect(picker.locator("option")).to_have_count(3)
    with page.expect_response(lambda response: response.url.endswith(f"?commit={first}")) as read:
        picker.select_option(first)
    assert read.value.status == 200
    assert read.value.json()["as_of"] == {"turn": 1, "sha": first}
    expect(page.locator(".card-budget")).to_contain_text("as of turn 1")
    picker.select_option("")
    expect(page.locator(".card-budget")).not_to_contain_text("as of turn")
    assert errors == [], errors


@pytest.mark.parametrize("control", ["model", "agent6"])
@pytest.mark.parametrize("live", [True, False])
def test_git_control_gates_session_actions(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    control: str,
    live: bool,
) -> None:
    """Fork and Merge do not offer an agent6 chain for a model-controlled run."""
    import os

    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "git-control"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "git_control": control, "run_branch": "main"})
    )
    events = [{"type": "session.start", "mode": "run", "user_task": "Own git"}]
    if live:
        (session / "worker.pid").write_text(str(os.getpid()))
    else:
        events.append({"type": "session.end", "reason": "finish_session"})
    (session / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/session/git-control")
    fork = page.get_by_role("button", name="Fork", exact=True)
    merge = page.get_by_role("button", name="Merge", exact=True)
    if control == "model":
        for button in (fork, merge):
            expect(button).to_be_disabled()
            expect(button).to_have_attribute(
                "title", "the model owns git in this run: no agent6 commit chain"
            )
    else:
        expect(fork).to_be_enabled()
        if live:
            expect(merge).to_be_disabled()
        else:
            expect(merge).to_be_enabled()
    composer = page.locator(".composer textarea")
    composer.fill("/u")
    if control == "model":
        expect(page.locator(".ac-pop .ac-item")).to_have_count(0)
    else:
        expect(page.locator(".ac-pop .ac-item")).to_contain_text(["/undo"])
    assert errors == []


@pytest.mark.parametrize("live", [False, True])
def test_fanout_actions(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    live: bool,
) -> None:
    """A group has a reviewable record, but no loop or commit chain."""
    import os

    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "fanout"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "mode": "run",
                "fanout": {"lanes": 2, "spec": "2"},
            }
        )
    )
    events = [{"type": "session.start", "mode": "run", "user_task": "Compare lanes"}]
    if live:
        (session / "worker.pid").write_text(str(os.getpid()))
    else:
        events.append({"type": "session.stop", "reason": "no_lane_passed"})
    (session / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/session/fanout")
    review = page.get_by_role("button", name="Review run", exact=True)
    expect(review).to_be_visible()
    for name in ("Compact context", "Fork", "Merge"):
        expect(page.get_by_role("button", name=name, exact=True)).to_be_hidden()
    if live:
        expect(review).to_be_disabled()
        expect(review).to_have_attribute(
            "title", "the session is still live; its record is not complete"
        )
    else:
        expect(review).to_be_enabled()
    assert errors == []
