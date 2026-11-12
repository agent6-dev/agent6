# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A click on a diff line's number quotes the line into a composer."""

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


def test_a_session_diff_line_is_quoted_into_the_sessions_composer(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The quote names the path, the side and the commit, then the line, for the steer beneath."""
    from playwright.sync_api import expect

    base = test_git_history._git(tmp_path, "rev-parse", "HEAD")
    source = tmp_path / "history.txt"
    source.write_text("first\n")
    first = test_git_history._commit(tmp_path, "first change")
    source.write_text("second\n")
    second = test_git_history._commit(tmp_path, "second change")
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "quote-line"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "base_sha": base, "git_control": "agent6"})
    )
    events = [
        {"type": "session.start", "mode": "run", "user_task": "Quote a line"},
        {"type": "checkpoint.committed", "turn": 1, "sha": first, "subject": "first change"},
        {"type": "checkpoint.committed", "turn": 2, "sha": second, "subject": "second change"},
        {"type": "session.stop", "reason": "finish_session", "all_passed": True},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/session/quote-line")
    composer = page.locator(".composer textarea")
    composer.fill("keep the wording")
    page.locator(".card-diff .diff .dl.add .g").nth(1).click()
    expect(composer).to_have_value(
        f"keep the wording\n\nhistory.txt:1 (head side, commit {second[:12]}):\n> second\n"
    )
    page.locator(".card-diff .diff .dl.del .g").first.click()
    expect(composer).to_have_value(
        f"keep the wording\n\nhistory.txt:1 (head side, commit {second[:12]}):\n> second\n\n"
        f"history.txt:1 (base side, the parent of commit {second[:12]}):\n> first\n"
    )
    assert errors == [], errors


def test_a_git_page_diff_line_is_quoted_for_the_new_work_composer(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """Quoted lines queue up on the Git page and open with the Sessions composer."""
    from playwright.sync_api import expect

    (tmp_path / "code.py").write_text("def old():\n    pass\n")
    base = test_git_history._commit(tmp_path, "Python base")
    (tmp_path / "code.py").write_text("class New:\n    pass\n")
    head = test_git_history._commit(tmp_path, "Python head")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{head}?path=code.py")
    page.locator(".git-diff .dl.add .g").nth(1).click()
    assert errors == [], errors
    expect(page.locator(".toast").last).to_contain_text("1 line quoted")
    page.locator(".git-diff .dl.del .g").first.click()
    expect(page.locator(".toast").last).to_contain_text("2 lines quoted")
    page.evaluate("location.hash = '#/'")
    expect(page.locator(".composer.dock-fixed textarea")).to_have_value(
        f"code.py:1 (head side, commit {head[:12]}):\n> class New:\n\n"
        f"code.py:1 (base side, commit {base[:12]}):\n> def old():\n\n"
    )
    assert errors == [], errors
