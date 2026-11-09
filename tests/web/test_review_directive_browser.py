from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.sessions import process
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("mode", ["new", "live", "resume"])
def test_review_completion_is_available_before_and_during_a_session(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    mode: str,
) -> None:
    """Tab completes review without submitting it or mistaking its ref for a model name."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "review-completion"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "session_id": session.name}), encoding="utf-8"
    )
    rows = [{"type": "session.start", "mode": "run", "user_task": "inspect"}]
    if mode == "live":
        process.write_worker_pid(session, os.getpid())
    else:
        rows.append({"type": "session.stop", "reason": "finish_session"})
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    route = "/" if mode == "new" else f"/session/{session.name}"
    page.goto(f"http://127.0.0.1:{port}/#{route}")
    composer = page.locator(".composer textarea")
    composer.fill("/rev")
    expect(page.locator(".ac-pop")).to_contain_text("/review")
    expect(page.locator(".ac-pop")).to_contain_text("[TARGET]")
    composer.press("Tab")
    expect(composer).to_have_value("/review ")
    composer.fill("/review main...HEAD")
    expect(page.locator(".ac-pop")).to_have_count(0)
    assert errors == [], errors


def test_review_refusal_renders_in_the_conversation(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A refused live review shows its new-session command without a model answer."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "asks" / "review-refusal"
    session.mkdir(parents=True)
    feedback = (
        "/review feature-x needs a different tree; start a new session: "
        "agent6 ask '/review feature-x'"
    )
    rows = [
        {"type": "session.start", "mode": "ask", "user_task": "inspect"},
        {"type": "loop.steer.feedback", "text": feedback},
        {"type": "session.stop", "reason": "silent_finish"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    expect(page.locator(".ci").filter(has_text="needs a different tree")).to_contain_text(feedback)
    assert errors == [], errors
