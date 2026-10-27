# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The opt-in scrape route accepts only the web secret as a bearer token."""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Iterator
from http import client
from typing import Any

import pytest

from agent6 import paths
from agent6.config import layer
from agent6.ui.telemetry import export
from agent6.ui.web import server
from tests.unit import test_telemetry_machine as machine_tests
from tests.web import signin


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    """Serve a repository with an explicit config file on an ephemeral loopback port."""
    config = tmp_path / "web.toml"
    config.write_text("", encoding="utf-8")
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", (config,), secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def _get(port: int, headers: dict[str, str], path: str = "/metrics") -> tuple[int, str, str]:
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read().decode()
    finally:
        conn.close()


def _configure(
    repo: pathlib.Path, web: str = 'metrics_route = "on"', metrics: str = "prometheus"
) -> None:
    (repo / "web.toml").write_text(
        f'[web]\n{web}\n[telemetry]\nmetrics = "{metrics}"\n'
        f"prometheus_file = {json.dumps(str(repo / 'agent6.prom'))}\n"
        'endpoint = "http://127.0.0.1:4318"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("web", "metrics"),
    [
        ("", "prometheus"),
        ('metrics_route = "off"', "prometheus"),
        ('metrics_route = "on"', "none"),
        ('metrics_route = "on"', "otlp"),
    ],
)
def test_metrics_is_404_unless_both_settings_enable_it(
    port: int, tmp_path: pathlib.Path, web: str, metrics: str
) -> None:
    """Neither a selected exporter nor the web opt-in alone exposes the route."""
    _configure(tmp_path, web, metrics)
    for headers in ({}, {"Authorization": f"Bearer {signin.SECRET}"}):
        assert _get(port, headers)[0] == 404


def test_metrics_requires_the_secret_not_a_cookie_or_login_token(
    port: int, tmp_path: pathlib.Path
) -> None:
    """A scrape authenticates directly with the secret, never with browser credentials."""
    _configure(tmp_path)
    assert _get(port, {"Authorization": f"Bearer {signin.SECRET}"})[0] == 200
    token = signin.token(port)
    for headers in (
        {},
        {"Authorization": "Bearer wrong-secret"},
        {"Authorization": f"Basic {signin.SECRET}"},
        {"Authorization": f"Bearer {token}"},
        {"Cookie": f"agent6_web_{port}={token}"},
        {"Cookie": f"agent6_web_{port}={signin.SECRET}"},
    ):
        status, _ctype, body = _get(port, headers)
        assert status == 401
        assert "bearer" in body.lower() and "secret" in body.lower()
        assert signin.SECRET not in body and token not in body
    assert _get(port, {}, f"/metrics?secret={signin.SECRET}")[0] == 401
    assert _get(port, {"Authorization": f"Bearer {signin.SECRET}"}, "/api/hub")[0] == 401


def _journal(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_metrics_is_a_fresh_read_of_the_exporters_sessions(
    port: int, tmp_path: pathlib.Path
) -> None:
    """Scrapes fold hub journals and machine workers without reading or writing the textfile."""
    _configure(tmp_path)
    state = paths.state_dir(tmp_path)
    for bucket in ("runs", "plans", "asks", "machines"):
        _journal(
            state / "sessions" / bucket / f"{bucket}-one" / "logs.jsonl", machine_tests.worker()
        )
    machine = state / "machines" / "demo"
    _journal(machine / "journal.jsonl", machine_tests.journal())
    _journal(machine / "states" / "4-review" / "logs.jsonl", machine_tests.worker())
    target = tmp_path / "agent6.prom"
    target.write_text("not the scrape\n", encoding="utf-8")
    status, ctype, text = _get(port, {"Authorization": f"Bearer {signin.SECRET}"})
    assert status == 200
    assert ctype == "text/plain; version=0.0.4; charset=utf-8"
    assert "agent6_sessions_total 6\n" in text
    assert target.read_text() == "not the scrape\n"
    assert not (state / "telemetry").exists()
    config = layer.load_effective(tmp_path, (tmp_path / "web.toml",)).config.telemetry
    export.export(tmp_path, config)
    assert text == target.read_text()

    _journal(state / "sessions" / "runs" / "new-run" / "logs.jsonl", machine_tests.worker())
    status, _ctype, fresh = _get(port, {"Authorization": f"Bearer {signin.SECRET}"})
    assert status == 200 and fresh != text
    assert "agent6_sessions_total 7\n" in fresh
    export.export(tmp_path, config)
    assert fresh == target.read_text()
