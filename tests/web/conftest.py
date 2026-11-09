from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING
from urllib import parse

import pytest
from docs.screenshots import seed

from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Route

browser = test_web_browser.browser


@pytest.fixture
def tour_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, browser: Browser
) -> Iterator[Page]:
    """Serve the docs sessions and bundled Git history through the real CLI, without keys."""
    root = pathlib.Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    bundle = root / "docs" / "screenshots" / "seed" / "stats-demo.bundle"
    subprocess.run(["git", "clone", "--quiet", str(bundle), str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", str(bundle), "+refs/agent6/*:refs/agent6/*"],
        check=True,
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "PYTHONPATH": str(root / "src")}
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        env[name] = str(tmp_path / name.lower())
        monkeypatch.setenv(name, env[name])
    monkeypatch.setenv("AGENT6_DEMO_REPO", str(repo))
    seed.main()
    env["AGENT6_WEB_SECRET"] = signin.SECRET
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
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
                context.add_init_script(f"localStorage.setItem('a6-token', {json.dumps(token)})")
                errors: list[str] = []

                def read_only(route: Route) -> None:
                    request = route.request
                    if (
                        parse.urlsplit(request.url).netloc == f"127.0.0.1:{port}"
                        and request.method == "GET"
                    ):
                        route.continue_()
                    else:
                        errors.append(f"Unexpected request: {request.method} {request.url}")
                        route.abort()

                context.route("**/*", read_only)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "console", lambda msg: errors.append(msg.text) if msg.type == "error" else None
                )
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
