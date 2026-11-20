from __future__ import annotations

import pathlib
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from tests import browser_env
from tests.unit import test_git_history
from tests.web import captures, test_files_edit, test_files_preview, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
open_page = test_web_browser.open_page


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def browser(request: pytest.FixtureRequest) -> Iterator[Browser]:
    """Exercise the same preview in both supported browsers."""
    from playwright import sync_api

    with sync_api.sync_playwright() as pw:
        engine = getattr(pw, request.param)
        browser_env.require_installed(engine)
        instance = engine.launch()
        try:
            yield instance
        finally:
            instance.close()


@pytest.mark.parametrize("width", [390, 1440])
def test_html_preview_is_static_and_closes_with_its_view(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    width: int,
) -> None:
    """HTML loads its repository CSS and images, but not scripts or standalone documents."""
    from playwright import sync_api

    repo, port = served
    (repo / "index.html").write_text(
        '<!doctype html><meta name="viewport" content="width=device-width">'
        '<link rel="stylesheet" href="site.css"><h1>Project preview</h1>'
        '<p>Repository HTML with scripts off.</p><img src="picture.svg" alt="Example">'
        '<script>document.body.dataset.executed = "yes"</script>'
    )
    (repo / "site.css").write_text(
        "body { font: 18px system-ui; padding: 24px; background: #f5f8ff; color: #18334a }"
        "h1 { color: rgb(24, 51, 74) } img { width: 160px }"
    )
    (repo / "picture.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90">'
        '<rect width="160" height="90" fill="#4060b0"/></svg>'
    )
    page, _errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/index.html")
    page.get_by_role("button", name="Preview", exact=True).click(timeout=3000)
    frame = page.frame_locator('iframe[title="Preview of index.html"]')
    sync_api.expect(frame.get_by_role("heading", name="Project preview")).to_be_visible()
    sync_api.expect(frame.locator("h1")).to_have_css("color", "rgb(24, 51, 74)")
    sync_api.expect(frame.locator("body")).not_to_have_attribute("data-executed", "yes")
    sync_api.expect(page.get_by_text("Scripts are off for this page.", exact=True)).to_be_visible()
    iframe = page.locator(".files-preview iframe")
    sync_api.expect(iframe).to_have_attribute("sandbox", "allow-forms")
    assert frame.locator("img").evaluate("img => img.complete && img.naturalWidth > 0")
    assert page.context.browser
    captures.save(page, "files-html-" + page.context.browser.browser_type.name)
    url = iframe.get_attribute("src")
    assert url
    direct = page.context.new_page()
    response = direct.goto(f"http://127.0.0.1:{port}" + url)
    assert response and response.status == 403
    sync_api.expect(direct.locator("body")).to_contain_text("Previews open inside the Files page")
    direct.close()
    with page.expect_response("**/api/files/preview/close") as closing:
        page.get_by_role("button", name="Source", exact=True).click()
    assert closing.value.ok
    assert test_files_preview.get(port, url, "iframe")[0] == 403
    page.get_by_role("button", name="Preview", exact=True).click()
    sync_api.expect(iframe).to_be_visible()
    with page.expect_response("**/api/files/preview/close") as closing:
        page.locator('nav.tabs a[data-tab="git"]').click() if width == 390 else page.locator(
            'aside.rail a[data-tab="git"]'
        ).click()
    assert closing.value.ok


def test_html_scripts_require_a_new_confirmation_each_time(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """Cancel and reopening the preview keep scripts off in both sandbox layers."""
    from playwright import sync_api

    repo, port = served
    guide = repo / "guide"
    guide.mkdir()
    (guide / "index.html").write_text(
        '<h1>Guide</h1><output>Static</output><script src="app.js"></script>'
    )
    (guide / "app.js").write_text(
        'fetch("title.txt").then(r => r.text()).then(t => '
        'document.querySelector("output").textContent = t)'
    )
    (guide / "title.txt").write_text("Interactive")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/guide/index.html")
    preview = page.get_by_role("button", name="Preview", exact=True)
    preview.click()
    iframe = page.locator(".files-preview iframe")
    result = iframe.content_frame.locator("output")
    run = page.get_by_role("button", name="Run scripts", exact=True)
    sync_api.expect(result).to_have_text("Static")
    run.click(timeout=3000)
    dialog = page.get_by_role("alertdialog")
    sync_api.expect(dialog).to_contain_text(
        "This page's scripts can read every file under guide/ and send them to any server. "
        "The browser cannot stop that."
    )
    dialog.get_by_role("button", name="Cancel").click()
    sync_api.expect(result).to_have_text("Static")
    run.click()
    dialog.get_by_role("button", name="Run scripts", exact=True).click()
    sync_api.expect(iframe).to_have_attribute("sandbox", "allow-forms allow-scripts")
    sync_api.expect(result).to_have_text("Interactive")
    url = iframe.get_attribute("src")
    assert url
    with page.expect_response("**/api/files/preview/close") as closing:
        page.get_by_role("button", name="Source", exact=True).click()
    assert closing.value.ok
    assert test_files_preview.get(port, url, "iframe")[0] == 403
    preview.click()
    sync_api.expect(result).to_have_text("Static")
    sync_api.expect(run).to_be_visible()


def test_the_script_warning_names_the_whole_repository_for_a_top_level_page(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """The warning said "its folder" for a top-level page, whose grant covers every shown file."""
    from playwright import sync_api

    repo, port = served
    (repo / "top.html").write_text("<h1>Top</h1>")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/top.html")
    page.get_by_role("button", name="Preview", exact=True).click()
    page.get_by_role("button", name="Run scripts", exact=True).click()
    sync_api.expect(page.get_by_role("alertdialog")).to_contain_text(
        "This page's scripts can read every file Files shows in this repository"
        " and send them to any server. The browser cannot stop that."
    )


def test_html_preview_uses_the_listed_commit(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """A moving ref cannot change the revision of the Files preview."""
    from playwright import sync_api

    repo, port = served
    (repo / "index.html").write_text("<h1>Committed page</h1>")
    test_git_history._git(repo, "add", "index.html")
    test_git_history._git(repo, "commit", "-qm", "page")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/index.html")
    button = page.get_by_role("button", name="Preview", exact=True)
    sync_api.expect(button).to_be_enabled(timeout=3000)
    (repo / "index.html").write_text("<h1>Later page</h1>")
    test_git_history._git(repo, "commit", "-qam", "changed page")
    button.click()
    sync_api.expect(page.frame_locator(".files-preview iframe").locator("h1")).to_have_text(
        "Committed page"
    )
    sync_api.expect(page.get_by_role("button", name="Edit", exact=True)).to_have_count(0)


def test_preview_refuses_an_insecure_remote_address(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """An HTTP address without Fetch Metadata cannot offer a working preview."""
    from playwright import sync_api

    repo, port = served
    (repo / "index.html").write_text("<h1>Preview</h1>")
    page, _errors = open_page(port)
    created: list[str] = []

    def forward(route: sync_api.Route) -> None:
        path = route.request.url.removeprefix("http://insecure.test")
        if path == "/api/files/preview":
            created.append(path)
        route.fulfill(response=route.fetch(url=f"http://127.0.0.1:{port}" + path))

    page.route("http://insecure.test/**", forward)
    page.goto("http://insecure.test/#/files/WORKTREE/index.html")
    assert page.evaluate("window.isSecureContext") is False
    page.get_by_role("button", name="Preview", exact=True).click()
    sync_api.expect(
        page.get_by_text("Previews need HTTPS or localhost.", exact=True)
    ).to_be_visible()
    sync_api.expect(page.locator(".files-preview iframe")).to_have_count(0)
    assert created == []


@pytest.mark.parametrize("exit_action", ["close", "navigate"])
def test_closing_the_tab_revokes_its_preview(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    exit_action: str,
) -> None:
    """Pagehide revokes the capability through a signed-in keepalive request."""
    from playwright import sync_api

    repo, port = served
    (repo / "index.html").write_text("<h1>Preview</h1>")
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/index.html")
    page.get_by_role("button", name="Preview", exact=True).click()
    sync_api.expect(page.frame_locator(".files-preview iframe").locator("h1")).to_be_visible()
    url = page.locator(".files-preview iframe").get_attribute("src")
    assert url
    assert test_files_preview.get(port, url, "iframe")[0] == 200
    if exit_action == "close":
        page.close()
    else:
        page.goto("about:blank")
    # A keepalive response can outlive the browser target that would report it.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if test_files_preview.get(port, url, "iframe")[0] == 403:
            break
        time.sleep(0.05)
    else:
        pytest.fail("closing the page did not revoke its preview before the lease expired")
