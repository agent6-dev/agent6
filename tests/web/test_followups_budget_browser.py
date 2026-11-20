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
@pytest.mark.parametrize("cap", [None, -1, 0, 2])
def test_budget_has_a_labelled_meter_only_with_a_cap(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    cap: int | None,
) -> None:
    """Capped usage has meter semantics; uncapped usage has no empty track."""
    from playwright.sync_api import expect

    session_logs.write_run(
        tmp_path,
        "meter-AAA111",
        [
            {"type": "budget.update", "usd_total": 0.5, "usd_cap": cap},
        ],
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/meter-AAA111")
    if width == 390:
        page.get_by_role("button", name="Widgets", exact=True).click()
        page.locator(".wmenu").get_by_role("button", name="Budget", exact=True).click()
    budget = page.locator(".card-budget")
    expect(budget).to_contain_text("$0.50")
    expect(budget.locator(".bar")).to_have_count(0)
    if cap is None or cap <= 0:
        expect(budget.get_by_role("meter")).to_have_count(0)
    else:
        meter = budget.get_by_role("meter", name="usage:", exact=False)
        expect(meter).to_have_attribute("value", "0.5")
        expect(meter).to_have_attribute("max", "2")
    captures.save(page, "budget-" + ("capped" if cap and cap > 0 else "uncapped"))
    assert not errors
