# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration tests for the `agent6 web` server.

Starts the stdlib server on an ephemeral loopback port and drives it with
`http.client`, asserting the JSON endpoints emit the same wire form as
`agent6 watch --json` and that SSE streams a folded snapshot. No browser."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path

import pytest

from agent6.cli import main
from agent6.config_layer import resolved_state_dir
from agent6.web.server import WebServer, _create_web_server  # pyright: ignore[reportPrivateUsage]

TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _make_run(cwd: Path, run_id: str, events: list[dict[str, object]]) -> None:
    runs = resolved_state_dir(cwd) / "runs" / run_id
    runs.mkdir(parents=True)
    body = "".join(json.dumps(e) + "\n" for e in events)
    (runs / "logs.jsonl").write_text(body, encoding="utf-8")


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[WebServer, int]]:
    """A WebServer bound to an ephemeral loopback port, serving from tmp_path."""
    srv = WebServer(("127.0.0.1", 0), tmp_path, "")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv, port
    finally:
        srv.shutdown()
        srv.server_close()


def _get(port: int, path: str) -> tuple[int, bytes, str]:
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read(), resp.getheader("Content-Type", "")
    finally:
        conn.close()


def _post(port: int, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps(body).encode()
        conn.request("POST", path, payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_page_served(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, ctype = _get(port, "/")
    assert status == 200
    assert "text/html" in ctype
    assert b"<title>agent6</title>" in body


@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_ipv6_loopback_bind_uses_ipv6_socket(tmp_path: Path, host: str) -> None:
    srv = _create_web_server(host, 0, tmp_path, "")  # pyright: ignore[reportPrivateUsage]
    try:
        assert srv.address_family == socket.AF_INET6
    finally:
        srv.server_close()


def test_run_snapshot_matches_watch_json(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _srv, port = server
    _make_run(
        tmp_path,
        "willing-glen-001",
        [
            {"type": "run.start", "user_task": "demo"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
            {"type": "tool.result", "name": "grep", "ok": True, "summary": "1 hit"},
        ],
    )
    # The web GET must equal `agent6 watch <id> --json` byte-for-byte in content.
    status, body, ctype = _get(port, "/api/run/willing-glen-001")
    assert status == 200
    assert "application/json" in ctype
    from_web = json.loads(body)

    monkeypatch.chdir(tmp_path)
    assert main(["watch", "willing-glen-001", "--json"]) == 0
    from_cli = json.loads(capsys.readouterr().out)
    assert from_web == from_cli
    assert from_web["tool_calls"][0]["name"] == "grep"


def test_hub_lists_runs(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(tmp_path, "run-a", [{"type": "run.start", "mode": "run", "user_task": "task a"}])
    _make_run(
        tmp_path,
        "run-b",
        [
            {"type": "run.start", "mode": "run", "user_task": "task b"},
            {"type": "run.end", "all_passed": True},
        ],
    )
    status, body, _ = _get(port, "/api/hub")
    assert status == 200
    hub = json.loads(body)
    ids = {r["id"] for r in hub["runs"]}
    assert ids == {"run-a", "run-b"}
    by_id = {r["id"]: r for r in hub["runs"]}
    assert by_id["run-b"]["status"] == "ok"
    assert by_id["run-b"]["task"] == "task b"


def test_machine_snapshot_matches_watch_json(
    server: tuple[WebServer, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _srv, port = server
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.asm.toml").write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(tmp_path / "tiny.asm.toml")]) == 0
    capsys.readouterr()

    status, body, _ = _get(port, "/api/machine/tiny")
    assert status == 200
    from_web = json.loads(body)

    assert main(["watch", "tiny", "--json"]) == 0
    from_cli = json.loads(capsys.readouterr().out)
    assert from_web == from_cli
    assert from_web["machine"] == "tiny"
    assert from_web["ended"]["status"] == "ok"


def test_unknown_run_is_404(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, _ = _get(port, "/api/run/nope")
    assert status == 404
    assert "no run" in json.loads(body)["error"]


def test_config_endpoint(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body, _ = _get(port, "/api/config")
    assert status == 200
    cfg = json.loads(body)
    # A per-leaf view keyed by dotted key, each carrying provenance.
    assert any(k.startswith("sandbox.") for k in cfg)
    sample = next(iter(cfg.values()))
    assert {"value", "effective", "default", "source", "modified"} <= set(sample)


def test_approve_writes_answer_file(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    run_dir = resolved_state_dir(tmp_path) / "runs" / "appr-run"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(port, "/api/run/appr-run/approve", {"id": "p1", "approved": True})
    assert status == 200
    assert body["ok"] is True
    assert (run_dir / "approvals" / "p1.answer").read_text(encoding="utf-8") == "yes"


def test_answer_writes_question_file(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    run_dir = resolved_state_dir(tmp_path) / "runs" / "q-run"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(port, "/api/run/q-run/answer", {"id": "q1", "answer": "option B"})
    assert status == 200 and body["ok"] is True
    assert (run_dir / "questions" / "q1.answer").read_text(encoding="utf-8") == "option B"


def test_steer_writes_answer_and_request(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    run_dir = resolved_state_dir(tmp_path) / "runs" / "steer-run"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(port, "/api/run/steer-run/steer", {"text": "focus on tests"})
    assert status == 200 and body["ok"] is True
    assert (run_dir / "steer.answer").read_text(encoding="utf-8") == "focus on tests"
    assert (run_dir / "steer.request").exists()


def test_approve_id_traversal_is_contained(server: tuple[WebServer, int], tmp_path: Path) -> None:
    # A malicious answer id must not escape the run's approvals/ dir.
    _srv, port = server
    run_dir = resolved_state_dir(tmp_path) / "runs" / "trav-run"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    escape = tmp_path / "pwned.answer"
    status, _ = _post(
        port, "/api/run/trav-run/approve", {"id": "../../../../pwned", "approved": True}
    )
    assert status != 200
    assert not escape.exists()
    # a normal id still works
    ok_status, ok_body = _post(port, "/api/run/trav-run/approve", {"id": "p1", "approved": True})
    assert ok_status == 200 and ok_body["ok"] is True


def test_run_id_traversal_is_404(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, _body, _ = _get(port, "/api/run/..")
    assert status == 404


def test_extra_api_path_segments_are_404(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(tmp_path, "seg-run", [{"type": "run.start", "user_task": "x"}])
    machine = resolved_state_dir(tmp_path) / "machines" / "tiny"
    machine.mkdir(parents=True)
    (machine / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    draft = resolved_state_dir(tmp_path) / "machine-drafts" / "drafty"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text('{"type": "run.start"}\n', encoding="utf-8")

    assert _get(port, "/api/run/seg-run/events/extra")[0] == 404
    assert _get(port, "/api/machine/tiny/events/extra")[0] == 404
    assert _get(port, "/api/draft/drafty/events/extra")[0] == 404


def test_draft_snapshot_folds_the_draft_log(server: tuple[WebServer, int], tmp_path: Path) -> None:
    # A machine-create draft is watched through the run endpoints.
    _srv, port = server
    draft = resolved_state_dir(tmp_path) / "machine-drafts" / "brave-otter"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text(
        '{"type": "run.start", "user_task": "author a fixer machine"}\n', encoding="utf-8"
    )
    status, body, _ = _get(port, "/api/draft/brave-otter")
    assert status == 200
    assert json.loads(body)["user_task"] == "author a fixer machine"
    # traversal rejected
    assert _get(port, "/api/draft/..")[0] == 404


def test_web_refuses_non_loopback_host_without_optin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `--host 0.0.0.0` must be refused before binding unless opted in.
    monkeypatch.chdir(tmp_path)
    assert main(["web", "--host", "0.0.0.0"]) == 2
    assert "refusing to bind non-loopback" in capsys.readouterr().err


def test_new_work_empty_task_rejected(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body = _post(port, "/api/new", {"mode": "run", "task": "   "})
    assert status == 422
    assert body["ok"] is False


def test_machine_run_rejects_unknown_file(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    status, body = _post(port, "/api/machine/run", {"file": "/etc/passwd"})
    assert status == 422
    assert "unknown machine file" in str(body["error"])


def test_bad_post_body_is_400(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    # extra="forbid": an unknown field fails validation loudly.
    status, body = _post(port, "/api/new", {"mode": "run", "task": "x", "bogus": 1})
    assert status == 400
    assert "bad request" in str(body["error"])


def test_sse_run_streams_snapshot(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_run(
        tmp_path,
        "stream-run",
        [
            {"type": "run.start", "user_task": "streamed"},
            {"type": "run.end", "all_passed": True},
        ],
    )
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/run/stream-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type", "")
        # The tailer emits a snapshot per event then a final one and closes the
        # stream (stop_when_finished). Drain to EOF and check the last data frame.
        seen = b""
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            seen += chunk
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert frames, "expected at least one SSE data frame"
        snap = json.loads(frames[-1][len(b"data:") :].strip())
        assert snap["user_task"] == "streamed"
        assert snap["finished"] is True
    finally:
        conn.close()
