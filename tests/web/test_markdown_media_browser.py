from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, media, test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


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
