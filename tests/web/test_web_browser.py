# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The rendered web UI driven in a browser: the forms and views the HTTP tests cannot reach.

Marked `browser`: Playwright's Chromium runs them, and they skip where it is not installed.
"""

from __future__ import annotations

import json
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.ui.web import server as web_server
from tests.unit.test_git_history import build_history
from tests.web import signin

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One Chromium for the module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


def _serve(cwd: pathlib.Path) -> Iterator[int]:
    srv = web_server.WebServer(("127.0.0.1", 0), cwd, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server on an ephemeral port, serving from an empty directory."""
    yield from _serve(tmp_path)


@pytest.fixture
def git_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server reading a repository with a merge, a rename, a binary file and tags."""
    build_history(tmp_path)
    yield from _serve(tmp_path)


@pytest.fixture
def open_page(browser: Browser) -> Iterator[Callable[[int], tuple[Page, list[str]]]]:
    """Open a signed-in desktop page on a server's port, collecting console and page errors."""
    contexts: list[BrowserContext] = []

    def _open(port: int) -> tuple[Page, list[str]]:
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        contexts.append(context)
        token = json.dumps(signin.token(port))
        context.add_init_script(f"localStorage.setItem('a6-token', {token})")
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        return page, errors

    try:
        yield _open
    finally:
        for context in contexts:
            context.close()


def test_the_provider_form_writes_the_chosen_deployment(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The rendered add-provider form posts the deployment the operator picked.

    The HTTP tests post hand-written payloads; this drives the form the browser builds from
    the entry-table registry, so a field the form drops or misnames shows here.
    """
    cfg_path = paths.global_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        '[providers.az]\napi_format = "openai"\ndeployment = "direct"\n'
        'base_url = "https://res.openai.azure.com"\n'
        'extra_query = { "api-version" = "2024-10-21" }\n',
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name=re.compile(r"^Add provider", re.I)).click()
    dialog = page.get_by_role("dialog")
    dialog.locator("input[name='name']").fill("az")
    dialog.locator("select[name='api_format']").select_option("openai")
    dialog.locator("select[name='deployment']").select_option("azure")
    dialog.get_by_role("button", name="Add", exact=True).click()
    page.wait_for_function("() => !document.querySelector('[role=dialog]')")
    assert errors == [], errors
    cfg = cfg_path.read_text(encoding="utf-8")
    assert 'deployment = "azure"' in cfg and "api-version" in cfg, cfg


def test_a_files_diff_fills_the_git_tab(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """A click on a changed file opens its patch over the whole tab; the back button returns.

    The graph and the side panel hide while the diff shows; the HTTP tests see only the JSON.
    """
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.click()
    page.locator(".gf").first.click()
    page.wait_for_selector(".git-diff .diff")
    body = page.locator(".git-body")
    assert "diffing" in (body.get_attribute("class") or "")
    assert not page.locator(".git-graph").is_visible()
    assert page.locator(".git-diff-pos").inner_text().startswith("1 / ")
    page.locator(".git-diff-head button").first.click()  # the "files" back button
    assert "diffing" not in (body.get_attribute("class") or "")
    assert page.locator(".git-graph").is_visible()
    assert errors == [], errors
