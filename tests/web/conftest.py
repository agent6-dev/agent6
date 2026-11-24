from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING
from urllib import parse

import pytest
from docs.screenshots import seed, workflow_repo

from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, ConsoleMessage, Page, Route

browser = test_web_browser.browser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _homes(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Point this process and the server it starts at throwaway homes, with no provider key."""
    root = pathlib.Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "PYTHONPATH": str(root / "src")}
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        env[name] = str(tmp_path / name.lower())
        monkeypatch.setenv(name, env[name])
    env["AGENT6_WEB_SECRET"] = signin.SECRET
    return env


# What a seeded tour may write: a preview, a file save, a stage and a commit.
_TOUR_WRITES = frozenset(
    {
        "/api/files/preview",
        "/api/files/preview/answers",
        "/api/files/preview/close",
        "/api/files/preview/renew",
        "/api/files/text",
        "/api/git/commit",
        "/api/git/stage",
    }
)


def _tour_request(method: str, path: str) -> bool:
    return method == "GET" or (method == "POST" and path in _TOUR_WRITES)


def _blocked_preview_script(text: str, _url: str) -> bool:
    """Chromium reports each script an HTML preview's sandbox refuses as a console error."""
    return text.startswith("Blocked script execution in ")


def _live_child_diff(text: str, url: str) -> bool:
    """A live child's commit is still in its clone, so its view's diff request answers 422."""
    return "422" in text and "/diff?sha=" in url


def _signed_in(
    repo: pathlib.Path,
    env: dict[str, str],
    tmp_path: pathlib.Path,
    browser: Browser,
    allowed: Callable[[str, str], bool],
    expected: Callable[[str, str], bool],
) -> Iterator[Page]:
    """Serve the repository through the real CLI and yield a signed-in page.

    Args:
        repo: The checkout `agent6 web` starts in.
        env: The server's whole environment.
        tmp_path: Where `server.log` goes.
        browser: The module's browser.
        allowed: Whether a request to the server, by method and path, may pass.
        expected: Whether a console error, by its text and source address, is the tour's own.

    Yields:
        The page; a refused request, a page error or any other console error fails the test
        at its end.
    """
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "server.log"
    with log.open("w") as output:
        server = subprocess.Popen(
            [
                sys.executable,
                str(pathlib.Path(sys.executable).with_name("agent6")),
                "--allow-root",
                "web",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                assert server.poll() is None, log.read_text()
                try:
                    token = signin.token(port)
                    break
                except ConnectionRefusedError:
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(0.1)
            with browser.new_context(viewport={"width": 1440, "height": 1000}) as context:
                context.add_init_script(
                    "if (window === top && location.pathname === '/') "
                    f"localStorage.setItem('a6-token', {json.dumps(token)})"
                )
                errors: list[str] = []

                def filtered(route: Route) -> None:
                    request = route.request
                    url = parse.urlsplit(request.url)
                    if url.netloc == f"127.0.0.1:{port}" and allowed(request.method, url.path):
                        route.continue_()
                    else:
                        errors.append(f"Unexpected request: {request.method} {request.url}")
                        route.abort()

                context.route("**/*", filtered)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))

                def console(message: ConsoleMessage) -> None:
                    address = str(message.location.get("url", ""))
                    if message.type == "error" and not expected(message.text, address):
                        errors.append(message.text)

                page.on("console", console)
                page.goto(base)
                yield page
                assert not errors, errors
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


@pytest.fixture
def tour_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, browser: Browser
) -> Iterator[Page]:
    """Serve the docs sessions and bundled Git history through the real CLI, without keys.

    The browser may save a file, stage it and commit it, as the recorded tour does; every
    other write is refused, so no tour starts or resumes a run.
    """
    seeds = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "seed"
    repo = tmp_path / "repo"
    bundle = seeds / "stats-demo.bundle"
    subprocess.run(["git", "clone", "--quiet", str(bundle), str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", str(bundle), "+refs/agent6/*:refs/agent6/*"],
        check=True,
    )
    for key, value in (("user.name", "agent6 demo"), ("user.email", "demo@agent6.dev")):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    for page_file in (seeds / "pages").iterdir():
        shutil.copy(page_file, repo)
    env = _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT6_DEMO_REPO", str(repo))
    seed.main()
    yield from _signed_in(repo, env, tmp_path, browser, _tour_request, _blocked_preview_script)


@pytest.fixture
def workflow_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, browser: Browser
) -> Iterator[Page]:
    """Serve the workflow tour's repository with its replay proxy, so the children really run.

    The model answers come from the committed temperature cassette, one replay per child;
    every request passes, because the tour starts, steers, stops and merges.
    """
    screenshots = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots"
    repo = tmp_path / "repo"
    env = _homes(tmp_path, monkeypatch) | {"AGENT6_FORCE_STREAM": "1"}
    monkeypatch.setenv("AGENT6_DEMO_REPO", str(repo))
    proxy_port = _free_port()
    workflow_repo.main(proxy_port)
    proxy = subprocess.Popen(
        [sys.executable, str(screenshots / "llm_proxy.py")],
        env={
            "AGENT6_PROXY_MODE": "replay",
            "AGENT6_PROXY_CASSETTE": str(screenshots / "seed" / "temps-cassette.jsonl"),
            "AGENT6_PROXY_PORT": str(proxy_port),
            "AGENT6_PROXY_PER_CONVERSATION": "1",
            "AGENT6_PROXY_CHUNK_MS": "100",
        },
    )
    try:
        yield from _signed_in(
            repo, env, tmp_path, browser, lambda _method, _path: True, _live_child_diff
        )
    finally:
        # A tour that failed midway leaves its run and children live.
        agent6 = str(pathlib.Path(sys.executable).with_name("agent6"))
        subprocess.run(
            [sys.executable, agent6, "--allow-root", "stop", "--all"],
            cwd=repo,
            env=env,
            capture_output=True,
            check=False,
        )
        proxy.terminate()
        proxy.wait(timeout=10)
