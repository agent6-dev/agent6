from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.unit import test_git_history
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = [pytest.mark.browser, pytest.mark.needs_grammar(language="python")]
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


@pytest.fixture
def python_commit(tmp_path: pathlib.Path, git_port: int) -> tuple[int, str, str]:
    """Commit different Python keywords on each side of a file change."""
    (tmp_path / "code.py").write_text("def old():\n    pass\n")
    base = test_git_history._commit(tmp_path, "Python base")
    (tmp_path / "code.py").write_text("class New:\n    pass\n")
    return git_port, base, test_git_history._commit(tmp_path, "Python head")


@pytest.mark.parametrize("mode", ["inline", "split"])
def test_git_commit_diff_highlights_both_sides(
    python_commit: tuple[int, str, str],
    open_page: Callable[[int], tuple[Page, list[str]]],
    mode: str,
) -> None:
    """Inline and split commit views highlight source without changing blame rows."""
    from playwright.sync_api import expect

    port, _base, head = python_commit
    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-diff-mode', '{mode}')")
    page.goto(f"http://127.0.0.1:{port}/#/git/commit/{head}?path=code.py")
    expect(page.locator(".git-diff .del span.hl-keyword")).to_have_text("def")
    expect(page.locator(".git-diff .add span.hl-keyword")).to_have_text("class")
    page.locator(".diff-blame").click()
    expect(page.locator(".git-diff .blame-cell").first).to_be_visible()
    expect(page.locator(".git-diff .add span.hl-keyword")).to_have_text("class")
    if mode == "inline":
        expect(page.locator(".git-diff .add .diff-marker")).to_have_text("+")
    assert errors == [], errors


def test_run_latest_commit_highlights_and_reuses_the_commit_payload(
    python_commit: tuple[int, str, str],
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The default commit pane fetches spans by SHA and reuses them across live frames."""
    from playwright.sync_api import expect

    port, base, head = python_commit
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "highlight-run"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "base_sha": base, "git_control": "agent6"})
    )
    events = [
        {"type": "session.start", "mode": "run", "user_task": "Highlight Python"},
        {"type": "loop.auto_commit", "sha": head, "turn": 1},
        {"type": "diff.updated", "sha": head, "patch": "plain fallback"},
    ]
    log = session / "logs.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    page, errors = open_page(port)
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url) if "/diff?" in r.url else None)
    page.goto(f"http://127.0.0.1:{port}/#/session/highlight-run")
    expect(page.locator(".card-diff .del span.hl-keyword")).to_have_text("def")
    expect(page.locator(".card-diff .add span.hl-keyword")).to_have_text("class")
    with log.open("a") as stream:
        stream.write(json.dumps({"type": "loop.resume.start", "steer": "repaint marker"}) + "\n")
    expect(page.locator(".card-head")).to_contain_text("repaint marker")
    expect(page.locator(".card-diff .add span.hl-keyword")).to_have_text("class")
    assert len(requests) == 1
    assert f"sha={head}" in requests[0]
    assert errors == [], errors
