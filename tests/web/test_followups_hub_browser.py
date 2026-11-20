from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_machine_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_machine_files_have_names_paths_and_aligned_run_buttons(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Machine files read as rows, not a set of differently sized Run buttons."""
    from playwright.sync_api import expect

    for name in ("loop", "nightly_dependency_upgrade_and_security_advisory_sweep_for_all_services"):
        (tmp_path / f"{name}.asm.toml").write_text(test_machine_browser.LOOP, encoding="utf-8")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    rows = page.locator(".machine-files .item")
    expect(rows).to_have_count(2)
    expect(rows.first.locator(".sub")).to_contain_text(str(tmp_path))
    assert (
        len(
            set(
                rows.locator("button").evaluate_all(
                    "bs => bs.map(b => b.getBoundingClientRect().right)"
                )
            )
        )
        == 1
    )
    assert rows.locator(".title").evaluate_all(
        "names => names.every(n => n.scrollWidth <= n.clientWidth)"
    )
    captures.save(page, "machine-files-list")
    rows.first.get_by_role("button").click()
    expect(page.get_by_role("alertdialog")).to_contain_text("paid machine run")
    assert not errors
