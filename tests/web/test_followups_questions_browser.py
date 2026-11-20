from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_question_options_follow_the_answer_field(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """A picked answer has a visible pressed state that follows typed changes too."""
    from playwright.sync_api import expect

    session_logs.write_run(
        tmp_path,
        "question-AAA111",
        [
            {
                "type": "question.prompt",
                "id": "q1",
                "questions": [{"question": "Which colour?", "options": ["Blue", "Green"]}],
            }
        ],
        live=True,
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/question-AAA111")
    blue = page.get_by_role("button", name="Blue", exact=True)
    green = page.get_by_role("button", name="Green", exact=True)
    blue.click()
    expect(blue).to_have_attribute("aria-pressed", "true")
    expect(green).to_have_attribute("aria-pressed", "false")
    assert blue.evaluate("e => getComputedStyle(e).backgroundColor") != green.evaluate(
        "e => getComputedStyle(e).backgroundColor"
    )
    field = page.get_by_role("textbox", name="Which colour?")
    field.fill("Other")
    expect(blue).to_have_attribute("aria-pressed", "false")
    field.fill("Green")
    expect(green).to_have_attribute("aria-pressed", "true")
    captures.save(page, "question-selected")
    assert not errors
