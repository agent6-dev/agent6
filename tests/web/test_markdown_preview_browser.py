from __future__ import annotations

import pathlib
import threading
from collections.abc import Callable, Iterator
from http import server
from typing import TYPE_CHECKING

import pytest

from tests.web import (
    captures,
    test_files_edit,
    test_files_preview,
    test_files_preview_browser,
    test_web_browser,
    test_web_git,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


@pytest.fixture
def outside() -> Iterator[tuple[str, list[tuple[str, str]]]]:
    """Count actual network reads at a separate image server."""
    reads: list[tuple[str, str]] = []

    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reads.append((self.path, self.headers.get("Referer", "")))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(test_web_git.PNG_RED_1X1)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as listener:
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{listener.server_port}", reads
        finally:
            listener.shutdown()
            thread.join()


def test_theme_changes_keep_the_markdown_scroll_position(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """Changing the frame's theme must not reset the section being read."""
    from playwright import sync_api

    repo, port = served
    (repo / "plan.md").write_text("# Plan\n\n" + "A paragraph.\n\n" * 80 + "## Last part\n")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md?anchor=plan")
    frame = page.frame_locator(".files-preview iframe")
    last = frame.get_by_role("heading", name="Last part")
    last.scroll_into_view_if_needed()
    sync_api.expect(last).to_be_in_viewport()
    old = frame.locator("body").evaluate("() => scrollY")
    assert old > 0
    page.locator("aside.rail .theme-toggle").click()
    expected = page.locator("html").get_attribute("class")
    sync_api.expect(frame.locator("html")).to_have_attribute("class", expected or "")
    sync_api.expect(last).to_be_in_viewport()
    assert frame.locator("body").evaluate("() => scrollY") == old


def test_a_failed_revision_load_revokes_the_replaced_preview(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """A failed Files route must not keep renewing the frame it removed."""
    from playwright import sync_api

    repo, port = served
    (repo / "plan.md").write_text("# Plan\n")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    frame = page.frame_locator(".files-preview iframe")
    sync_api.expect(frame.get_by_role("heading", name="Plan")).to_be_visible()
    url = page.locator(".files-preview iframe").get_attribute("src")
    assert url
    page.route(
        "**/api/git/refs",
        lambda route: route.fulfill(status=503, json={"error": "Repository is unavailable."}),
    )
    with page.expect_response("**/api/files/preview/close", timeout=5000) as closing:
        page.evaluate("location.hash = '#/files/HEAD/plan.md'")
        sync_api.expect(
            page.get_by_role("heading", name="Could not load this page")
        ).to_be_visible()
    assert closing.value.ok
    assert test_files_preview.get(port, url, "iframe")[0] == 403


@pytest.mark.parametrize("width", [390, 1440])
def test_remote_images_wait_for_ui_confirmation_and_forget_on_close(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    outside: tuple[str, list[tuple[str, str]]],
    width: int,
) -> None:
    """The frame cannot grant its own image hosts or retain them after it closes."""
    from playwright import sync_api

    repo, port = served
    base, reads = outside
    addresses = [base + "/diagram.png?text=the-entire-address-is-visible", base + "/detail.png"]
    (repo / "plan.md").write_text(
        "# Project plan\n\nRepository images load without a prompt.\n\n![Local](logo.png)\n\n"
        + "\n\n".join(f"![Remote {i}]({url})" for i, url in enumerate(addresses))
        + '\n\n<script>document.body.dataset.executed = "yes"</script>\n'
    )
    page, _errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    frame = page.frame_locator(".files-preview iframe")
    sync_api.expect(frame.get_by_role("heading", name="Project plan")).to_be_visible(timeout=3000)
    sync_api.expect(page.locator(".files-preview iframe")).to_have_attribute(
        "sandbox", "allow-scripts"
    )
    sync_api.expect(frame.locator("body")).not_to_have_attribute("data-executed", "yes")
    for address in addresses:
        sync_api.expect(frame.get_by_text(address, exact=True)).to_be_visible()
    sync_api.expect(frame.get_by_role("img", name="Local", exact=True)).to_have_js_property(
        "naturalWidth", 1
    )
    assert reads == []
    assert page.context.browser
    captures.save(page, "files-markdown-blocked-" + page.context.browser.browser_type.name)
    page.evaluate(
        "async () => { window.postMessage({type: 'preview-images'}, '*');"
        " await new Promise(r => setTimeout(r, 50)); }"
    )
    sync_api.expect(page.get_by_role("dialog")).to_have_count(0)
    frame.get_by_role("button", name="Load", exact=True).click()
    dialog = page.get_by_role("dialog", name="Load remote images?")
    sync_api.expect(dialog).to_be_visible()
    assert reads == []
    for address in addresses:
        sync_api.expect(dialog).to_contain_text(address)
    captures.save(page, "files-markdown-confirm-" + page.context.browser.browser_type.name)
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert reads == []
    frame.get_by_role("button", name="Load", exact=True).click()
    dialog.get_by_role("button", name="Load", exact=True).click()
    for index in range(2):
        sync_api.expect(
            frame.get_by_role("img", name=f"Remote {index}", exact=True)
        ).to_have_js_property("naturalWidth", 1)
    assert sorted(reads) == sorted((url.removeprefix(base), "") for url in addresses)
    sync_api.expect(frame.get_by_role("button", name="Load", exact=True)).to_have_count(0)
    url = page.locator(".files-preview iframe").get_attribute("src")
    assert url
    page.get_by_role("button", name="Source", exact=True).click()
    assert test_files_preview.get(port, url, "iframe")[0] == 403
    page.get_by_role("button", name="Rendered", exact=True).click()
    sync_api.expect(frame.get_by_role("button", name="Load", exact=True)).to_be_visible()
    assert len(reads) == 2
    assert not _errors
