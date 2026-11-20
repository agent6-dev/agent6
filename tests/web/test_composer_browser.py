# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The docked composers driven in a browser: a session's send button."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from tests.web import session_logs, signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, FloatRect, Locator, Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

OpenPage = Callable[[int], tuple["Page", list[str]]]
TouchPage = Callable[[int, int, int, bool], "Page"]
# The agent finished the run, so a resume needs an instruction; a run out of turns resumes bare.
STOP = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
OUT_OF_TURNS = {"type": "session.stop", "reason": "max_turns", "verification": "failed"}


def _box(locator: Locator) -> FloatRect:
    box = locator.bounding_box()
    assert box is not None
    return box


@pytest.fixture
def touch_page(browser: Browser) -> Iterator[TouchPage]:
    """Open a signed-in page in a context with or without a touch screen, failing on page errors."""
    contexts: list[BrowserContext] = []
    errors: list[str] = []

    def _open(port: int, width: int, height: int, touch: bool) -> Page:
        context = browser.new_context(
            viewport={"width": width, "height": height},
            has_touch=touch,
            is_mobile=touch and width <= 480,
        )
        contexts.append(context)
        context.add_init_script(
            f"localStorage.setItem('a6-token', {json.dumps(signin.token(port))})"
        )
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        return page

    try:
        yield _open
        assert not errors, errors
    finally:
        for context in contexts:
            context.close()


@pytest.mark.parametrize("width", [1440, 390])
def test_a_sessions_button_sits_right_of_its_entry_and_sends_what_enter_sends(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, width: int
) -> None:
    """A session's composer had no button: Enter was its only way to send."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "live-AAA111", [], live=True)
    session_logs.write_run(tmp_path, "ended-AAA111", [STOP])
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.route("**/api/session/*/steer", lambda route: route.fulfill(json={"message": "sent"}))
    page.route("**/api/session/*/resume", lambda route: route.fulfill(json={}))
    entry = page.locator(".composer textarea")
    steer = {"text": "check the tests"}
    resume = {"text": "check the tests", "preset": "", "model": ""}
    for session, name, verb, body in (
        ("live-AAA111", "Send", "steer", steer),
        ("ended-AAA111", "Resume", "resume", resume),
    ):
        page.goto(f"http://127.0.0.1:{port}/#/session/{session}")
        button = page.get_by_role("button", name=name, exact=True)
        sent: list[object] = []
        for press in (lambda: entry.press("Enter"), button.click):
            page.reload()  # a resume holds the composer while it waits for the run to come live
            entry.fill("check the tests")
            expect(button).to_be_enabled()
            field, key = _box(entry), _box(button)
            assert key["x"] >= field["x"] + field["width"]
            assert key["y"] + key["height"] == pytest.approx(field["y"] + field["height"], abs=0.5)
            with page.expect_request(f"**/api/session/{session}/{verb}") as request:
                press()
            sent.append(request.value.post_data_json)
        assert sent == [body, body]
    # A typed directive runs from the button as from Enter.
    page.reload()
    entry.fill("/restate")
    page.get_by_role("button", name="Resume", exact=True).click()
    expect(page.get_by_role("dialog", name="Since your last message")).to_be_visible()
    assert not errors


def test_a_phones_session_button_is_a_named_icon_that_leaves_the_entry_the_row(
    touch_page: TouchPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A phone's session composer had no button; a worded one would crowd its entry."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "ended-AAA111", [STOP])
    page = touch_page(port, 390, 844, True)
    page.goto(f"http://127.0.0.1:{port}/#/session/ended-AAA111")
    button = page.get_by_role("button", name="Resume", exact=True)
    expect(button).to_have_attribute("title", "Resume")
    expect(button.locator("svg")).to_be_visible()
    expect(button.locator("span")).to_be_hidden()
    entry = page.locator(".composer textarea")
    row, field, key = _box(entry.locator("..")), _box(entry), _box(button)
    assert key["width"] == key["height"] == field["height"]
    assert key["x"] >= field["x"] + field["width"] and key["y"] == field["y"]
    assert field["width"] > 0.8 * row["width"]


def test_a_sessions_button_waits_for_something_to_send(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An empty Send does nothing, and an empty Resume of a run the agent finished is refused."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "live-AAA111", [], live=True)
    session_logs.write_run(tmp_path, "ended-AAA111", [STOP])
    session_logs.write_run(tmp_path, "short-AAA111", [OUT_OF_TURNS])
    page, errors = open_page(port)
    entry = page.locator(".composer textarea")
    for session, name, placeholder, bare in (
        ("live-AAA111", "Send", "steer this session… (type / for directives)", False),
        ("ended-AAA111", "Resume", "what should it do next…", False),
        ("short-AAA111", "Resume", "continue this session…", True),
    ):
        page.goto(f"http://127.0.0.1:{port}/#/session/{session}")
        expect(entry).to_have_attribute("placeholder", placeholder)  # the first frame
        button = page.get_by_role("button", name=name, exact=True)
        entry.fill("check the tests")
        expect(button).to_be_enabled()
        entry.fill("  ")
        if bare:
            expect(button).to_be_enabled()
        else:
            expect(button).to_be_disabled()
    assert not errors


@pytest.mark.parametrize("font", [16, 20])
@pytest.mark.parametrize("width", [1440, 1024, 820, 390])
def test_a_sessions_button_keeps_its_entrys_line_height_and_bottom_edge(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, width: int, font: int
) -> None:
    """The button is one line of the entry tall, as wide as its word or a square icon, at any font.

    It stays on the entry's bottom edge while the entry grows.
    """
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "ended-AAA111", [STOP])
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.context.new_cdp_session(page).send("Page.setFontSizes", {"fontSizes": {"standard": font}})
    page.goto(f"http://127.0.0.1:{port}/#/session/ended-AAA111")
    entry = page.locator(".composer textarea")
    button = page.get_by_role("button", name="Resume", exact=True)
    expect(entry).to_have_attribute("placeholder", "what should it do next…")  # the first frame
    field, key = _box(entry), _box(button)
    assert field["height"] == pytest.approx(2 * font, abs=0.1)  # --ctl-h, 2rem
    assert key["height"] == pytest.approx(field["height"], abs=0.1)
    assert key["y"] + key["height"] == pytest.approx(field["y"] + field["height"], abs=0.5)
    if width <= 48.75 * font:  # the phone layout: a square icon
        expect(button.locator("svg")).to_be_visible()
        assert key["width"] == pytest.approx(key["height"], abs=0.1)
    else:  # the word, in the standard padding and no wider
        word = _box(button.locator("span"))
        padding = page.evaluate(
            "parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.75"
        )
        assert key["width"] == pytest.approx(word["width"] + 2 * padding + 2, abs=0.5)
    entry.fill("one\ntwo\nthree")
    field, key = _box(entry), _box(button)
    assert field["height"] > 2 * key["height"]
    assert key["y"] + key["height"] == pytest.approx(field["y"] + field["height"], abs=0.5)
    assert not errors
