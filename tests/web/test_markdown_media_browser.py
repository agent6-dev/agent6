from __future__ import annotations

import pathlib
import threading
from collections.abc import Callable, Iterator
from http import server
from typing import TYPE_CHECKING, override

import pytest

from tests.web import captures, media, test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


@pytest.fixture
def outside_media(tmp_path: pathlib.Path) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    """Serve real media from a separate listener and record every request."""
    media.write(tmp_path)
    reads: list[tuple[str, str]] = []
    clips = {
        "/clip.webm": (media.VIDEO, "video/webm"),
        "/sound.wav": ((tmp_path / "sound.wav").read_bytes(), "audio/wav"),
    }

    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reads.append((self.path, self.headers.get("Referer", "")))
            data, content_type = clips[self.path.partition("?")[0]]
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        @override
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


@pytest.mark.parametrize("width", [390, 1440])
def test_remote_players_wait_for_confirmation_and_forget_on_close(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    outside_media: tuple[str, list[tuple[str, str]]],
    width: int,
) -> None:
    """No remote media leaves the browser until signed-in consent, which ends on close."""
    from playwright import sync_api

    repo, port = served
    base, reads = outside_media
    urls = [base + "/clip.webm?detail=the-full-address", base + "/sound.wav"]
    (repo / "media.md").write_text(
        "# Remote media\n\n" + "\n\n".join(f"![Clip]({url})" for url in urls)
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/media.md")
    frame = page.frame_locator(".files-preview iframe")
    for tag in ("video", "audio"):
        sync_api.expect(frame.locator(tag)).to_have_count(1)
        assert frame.locator(tag).get_attribute("src") is None
    for url in urls:
        sync_api.expect(frame.get_by_text(url, exact=True)).to_be_visible()
    assert reads == []
    frame.get_by_role("button", name="Load", exact=True).click()
    dialog = page.get_by_role("dialog", name="Load remote files?")
    sync_api.expect(dialog).to_be_visible()
    for url in urls:
        sync_api.expect(dialog).to_contain_text(url)
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert reads == []
    frame.get_by_role("button", name="Load", exact=True).click()
    dialog.get_by_role("button", name="Load", exact=True).click()
    for tag in ("video", "audio"):
        player = frame.locator(tag)
        sync_api.expect(player).to_have_js_property("duration", 3)
        sync_api.expect(player).to_have_js_property("paused", True)
        player.press("Space")
        sync_api.expect(player).to_have_js_property("paused", False)
        player.evaluate(
            "e => new Promise(r => { if (e.currentTime > 0) r();"
            " else e.addEventListener('timeupdate', r, {once:true}); })"
        )
        assert player.evaluate("e => e.currentTime") > 0
        player.press("Space")
    assert {path for path, _ in reads} == {url.removeprefix(base) for url in urls}
    assert all(not referrer for _, referrer in reads)
    count = len(reads)
    page.get_by_role("button", name="Source", exact=True).click()
    page.get_by_role("button", name="Rendered", exact=True).click()
    sync_api.expect(frame.get_by_role("button", name="Load", exact=True)).to_be_visible()
    assert len(reads) == count
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_repository_media_plays_with_native_controls(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    width: int,
) -> None:
    """Both native players load repository bytes and play without a custom control path."""
    from playwright import sync_api

    repo, port = served
    media.write(repo)
    (repo / "media.md").write_text("# Media\n\n![Video](clip.webm)\n\n![Audio](sound.wav)\n")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/media.md")
    frame = page.frame_locator(".files-preview iframe")
    for tag in ("video", "audio"):
        player = frame.locator(tag)
        sync_api.expect(player).to_be_visible()
        sync_api.expect(player).to_have_js_property("duration", 3)
        sync_api.expect(player).to_have_js_property("controls", True)
        sync_api.expect(player).to_have_js_property("paused", True)
        bounds = player.bounding_box()
        assert bounds and bounds["width"] <= width
        player.press("Space")
        sync_api.expect(player).to_have_js_property("paused", False)
        player.evaluate(
            "e => new Promise(resolve => {"
            " if (e.currentTime > 0) resolve();"
            " else e.addEventListener('timeupdate', resolve, {once: true}); })"
        )
        assert player.evaluate("e => e.currentTime") > 0
        player.evaluate("e => e.pause()")
    assert page.context.browser
    captures.save(page, "files-markdown-media-" + page.context.browser.browser_type.name)
    assert not errors
