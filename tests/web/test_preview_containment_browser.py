"""Check the network and origin boundaries of previews opened from Files."""

from __future__ import annotations

import pathlib
import socket
import socketserver
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING
from urllib import parse

import pytest

from tests.web import test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


@pytest.fixture
def outside_listener() -> Iterator[tuple[str, threading.Event]]:
    """Count even a connection that sends no HTTP request."""
    connected = threading.Event()

    class Connection(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            connected.set()

    with socketserver.TCPServer(("127.0.0.1", 0), Connection) as listener:
        address = ("127.0.0.1", int(listener.server_address[1]))
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{address[1]}", connected
            with socket.create_connection(address, timeout=2):
                assert connected.wait(2), "listener did not see its control connection"
        finally:
            listener.shutdown()
            thread.join()


@pytest.mark.parametrize("name", ["index.html", "readme.md"])
def test_preview_network_boundary(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    outside_listener: tuple[str, threading.Event],
    name: str,
) -> None:
    """Opening repository markup must not contact another server or run its scripts."""
    from playwright import sync_api

    repo, port = served
    base, connected = outside_listener
    markup = f'''<link rel="prefetch" href="{base}/next">
<link rel="preconnect" href="{base}">
<link rel="stylesheet" href="{base}/style.css">
<style>@font-face {{font-family: demo; src: url({base}/font.woff2)}}
h1 {{font-family: demo}}</style>
<h1>Fixture heading</h1>
<img src="{base}/picture.png" alt="Fixture picture">
<script src="{base}/page.js"></script>
<script>
  document.documentElement.dataset.repositoryScript = "ran";
  fetch("{base}/fetch");
  const xhr = new XMLHttpRequest(); xhr.open("GET", "{base}/xhr"); xhr.send();
  new WebSocket("{base.replace("http:", "ws:")}/socket");
  new RTCPeerConnection({{iceServers: []}}).close();
  fetch("/api/git/refs?preview-fixture");
</script>
<form action="{base}/submit"><button>Submit form</button></form>
<a target="_top" href="{base}/top">Top link</a>
<a target="_blank" href="{base}/tab">Tab link</a>
<a href="{base}/self">Frame link</a>
'''
    is_html = name.endswith(".html")
    text = (
        "<!doctype html>" if is_html else f"# Fixture heading\n\n![Remote]({base}/remote.png)\n\n"
    )
    (repo / name).write_text(text + markup, encoding="utf-8")
    page, _errors = open_page(port)
    api_calls: list[str] = []
    page.context.on(
        "request",
        lambda request: (
            api_calls.append(request.url)
            if request.url.endswith("/api/git/refs?preview-fixture")
            else None
        ),
    )
    ui = f"http://127.0.0.1:{port}/#/files/WORKTREE/{name}"
    page.goto(ui)
    if is_html:
        page.get_by_role("button", name="Preview", exact=True).click()
    iframe = page.locator(".files-preview iframe")
    frame = iframe.content_frame
    sync_api.expect(frame.get_by_role("heading", name="Fixture heading")).to_be_visible()
    sync_api.expect(frame.locator("html")).not_to_have_attribute("data-repository-script", "ran")
    sync_api.expect(iframe).to_have_attribute(
        "sandbox", "allow-forms" if is_html else "allow-scripts"
    )
    assert iframe.evaluate("element => element.contentDocument === null")
    assert frame.locator("html").evaluate("() => window.origin") == "null"
    storage = "() => { try { void localStorage; return 'available' } catch (e) { return e.name } }"
    assert frame.locator("html").evaluate(storage) == "SecurityError"
    url = iframe.get_attribute("src")
    assert url
    direct = page.context.new_page()
    response = direct.goto(parse.urljoin(ui, url))
    assert response and response.status == 403
    assert response.headers["content-type"].startswith("text/plain")
    assert "sandbox" in response.headers["content-security-policy"].split("; ")
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "Fixture heading" not in response.text()
    direct.close()
    if is_html:
        frame.get_by_role("button", name="Submit form").click()
        assert not connected.wait(1), "outside form action opened a TCP connection"
        # A blocked form navigation can replace its frame with a browser error page.
        page.get_by_role("button", name="Source", exact=True).click()
        page.get_by_role("button", name="Preview", exact=True).click()
        for label in ("Top link", "Tab link", "Frame link"):
            frame.get_by_role("link", name=label).click()
    else:
        sync_api.expect(frame.get_by_role("button", name="Load", exact=True)).to_be_visible()
        sync_api.expect(frame.locator("form, link, script[src]")).to_have_count(0)
    assert not connected.wait(1), "preview opened an outside TCP connection"
    assert not api_calls
    assert page.url == ui
    assert page.context.pages == [page]
