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


def _post_raw(
    port: int, path: str, body: bytes, headers: dict[str, str]
) -> tuple[int, dict[str, object]]:
    """POST with caller-controlled headers (for the CSRF checks)."""
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body, headers)
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


def _make_machine_with_state(cwd: Path, name: str, seq_state: str) -> tuple[Path, Path]:
    """A machine instance dir + one per-state agent-log dir. Returns (instance, state)."""
    inst = resolved_state_dir(cwd) / "machines" / name
    inst.mkdir(parents=True)
    (inst / "machine.asm.toml").write_text(TINY, encoding="utf-8")
    (inst / "journal.jsonl").write_text("", encoding="utf-8")
    state = inst / "states" / seq_state
    state.mkdir(parents=True)
    (state / "logs.jsonl").write_text("", encoding="utf-8")
    return inst, state


def test_machine_poke_writes_signal(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "pokable", "0000-review")
    status, body = _post(port, "/api/machine/pokable/poke", {"message": "reload"})
    assert status == 200 and body["ok"] is True
    assert json.loads((inst / "signal").read_text(encoding="utf-8")) == "reload"


def test_machine_poke_json_data_payload(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "datapoke", "0000-review")
    status, body = _post(port, "/api/machine/datapoke/poke", {"data": {"cmd": "go", "n": 2}})
    assert status == 200 and body["ok"] is True
    assert json.loads((inst / "signal").read_text(encoding="utf-8")) == {"cmd": "go", "n": 2}


def test_machine_answer_writes_to_per_state_dir(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    _inst, state = _make_machine_with_state(tmp_path, "asker", "0003-classify")
    status, body = _post(port, "/api/machine/asker/answer", {"id": "question-1", "answer": "yes"})
    assert status == 200 and body["ok"] is True
    assert (state / "questions" / "question-1.answer").read_text(encoding="utf-8") == "yes"


def test_machine_approve_and_steer_target_per_state_dir(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    _inst, state = _make_machine_with_state(tmp_path, "acter", "0001-work")
    _post(port, "/api/machine/acter/approve", {"id": "approval-1", "approved": False})
    assert (state / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "no"
    _post(port, "/api/machine/acter/steer", {"text": "focus"})
    assert (state / "steer.answer").read_text(encoding="utf-8") == "focus"
    assert (state / "steer.request").exists()


def test_machine_answer_id_traversal_is_contained(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    _inst, _state = _make_machine_with_state(tmp_path, "travm", "0000-review")
    escape = tmp_path / "pwned.answer"
    status, _ = _post(port, "/api/machine/travm/answer", {"id": "../../pwned", "answer": "x"})
    assert status != 200
    assert not escape.exists()


def test_pwa_assets_served(server: tuple[WebServer, int]) -> None:
    _srv, port = server
    st, body, ctype = _get(port, "/manifest.webmanifest")
    assert st == 200 and "manifest" in ctype
    assert json.loads(body)["name"] == "agent6"
    assert _get(port, "/sw.js")[0] == 200
    assert _get(port, "/icon.svg")[0] == 200


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


# --- a corrupt journal degrades, never 500s / kills the stream ---------------


def _corrupt_journal(inst: Path) -> None:
    (inst / "journal.jsonl").write_text('{"type": "step", "bogus": true}\n', encoding="utf-8")


def test_corrupt_journal_hub_shows_unreadable(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # One corrupt journal line must not 500 the whole landing page; the entry
    # stays listed with an unreadable status.
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick", "0000-review")
    _corrupt_journal(inst)
    _make_run(tmp_path, "healthy-run", [{"type": "run.start", "user_task": "x"}])
    status, body, _ = _get(port, "/api/hub")
    assert status == 200
    hub = json.loads(body)
    (entry,) = [m for m in hub["machines"] if m["name"] == "sick"]
    assert entry["status"] == "unreadable"


def test_corrupt_journal_machine_snapshot_is_422(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick2", "0000-review")
    _corrupt_journal(inst)
    status, body, _ = _get(port, "/api/machine/sick2")
    assert status == 422
    assert "corrupt journal" in json.loads(body)["error"]


def test_corrupt_journal_machine_sse_sends_error_frame(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # The SSE stream must emit an in-band error frame and close, never write a
    # second HTTP status line into the open stream.
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "sick3", "0000-review")
    _corrupt_journal(inst)
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/machine/sick3/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # stream closes after the error frame
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert len(frames) == 1
        assert "corrupt journal" in json.loads(frames[0][len(b"data:") :].strip())["error"]
        assert b"HTTP/1" not in seen  # no second status line inside the stream
    finally:
        conn.close()


# --- SSE catch-up folds history into one frame --------------------------------


def test_sse_run_catchup_folds_history_into_few_frames(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # Connecting to a run with a long history must not emit one full RunState
    # frame per historical event (13 MB probed on a 502-event run): the backlog
    # folds into (almost) one snapshot.
    _srv, port = server
    events: list[dict[str, object]] = [{"type": "run.start", "user_task": "big"}]
    for i in range(150):
        events.append({"type": "tool.call", "name": f"t{i}", "args": {}})
        events.append({"type": "tool.result", "name": f"t{i}", "ok": True, "summary": "ok"})
    events.append({"type": "run.end", "all_passed": True})
    _make_run(tmp_path, "big-run", events)
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/run/big-run/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert 1 <= len(frames) <= 5  # was ~1 per historical event
        snap = json.loads(frames[-1][len(b"data:") :].strip())
        assert snap["finished"] is True
        assert snap["log_count"] == len(events)
    finally:
        conn.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_sse_run_closes_even_if_tailer_dies(
    server: tuple[WebServer, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tail thread must ALWAYS enqueue its None sentinel: if it raises (the
    # injected raise below is intentionally unhandled in that thread), the
    # stream sends the folded snapshot and closes instead of hanging until the
    # client gives up.
    import agent6.web.server as server_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("tailer died")

    monkeypatch.setattr(server_mod, "tail_events", _boom)
    _srv, port = server
    _make_run(tmp_path, "dead-tail", [{"type": "run.start", "user_task": "x"}])
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/run/dead-tail/events")
        resp = conn.getresponse()
        assert resp.status == 200
        seen = resp.read()  # must reach EOF, not time out
        frames = [f for f in seen.split(b"\n\n") if f.startswith(b"data:")]
        assert len(frames) == 1  # the final (initial-state) snapshot
    finally:
        conn.close()


# --- POST hardening -----------------------------------------------------------


def test_oversize_post_body_is_413(server: tuple[WebServer, int]) -> None:
    # Headers only: the server refuses on Content-Length alone, before any body
    # bytes arrive (actually streaming 1 MiB races the server's early close).
    _srv, port = server
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("POST", "/api/new")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str((1 << 20) + 100))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        assert "body larger" in json.loads(resp.read())["error"]
    finally:
        conn.close()


def test_prune_body_is_drained_so_keepalive_is_not_poisoned(
    server: tuple[WebServer, int],
) -> None:
    # The client posts `{}` to prune. If the route does not read that body, the
    # 2 bytes sit on the keep-alive socket and the next pipelined request line is
    # parsed with them prepended -> 400 Bad Request. Pipeline prune + a GET on a
    # single socket and require the GET to be answered cleanly.
    _srv, port = server
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        body = b"{}"
        prune = (
            b"POST /api/runs/prune HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        follow = b"GET /api/hub HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        sock.sendall(prune + follow)
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
        raw = b"".join(chunks)
    finally:
        sock.close()
    # Both requests were answered (prune then GET), the GET returned the hub
    # payload, and nothing was a 400 framing error: the prune body was drained.
    # Undrained, the GET line would parse as `{}GET /api/hub...` -> 400 and no
    # hub JSON.
    assert raw.count(b"HTTP/1.1 ") == 2, raw
    assert b" 400 " not in raw, raw
    assert b'"runs":' in raw, raw  # the GET /api/hub payload came back intact


def test_negative_content_length_is_rejected(server: tuple[WebServer, int]) -> None:
    # A negative Content-Length must not reach rfile.read(n) (which would read to
    # EOF and park the worker); reject it up front.
    _srv, port = server
    status, body = _post_raw(
        port,
        "/api/new",
        b"",
        {"Content-Type": "application/json", "Content-Length": "-1"},
    )
    assert status == 400
    assert "Content-Length" in str(body["error"])


def test_chunked_post_body_is_refused(server: tuple[WebServer, int]) -> None:
    # Only Content-Length bodies are read; a chunked body would sit unread on
    # the connection exactly like an undrained early-error body.
    _srv, port = server
    status, body = _post_raw(
        port,
        "/api/runs/prune",
        b"",
        {"Transfer-Encoding": "chunked", "Content-Type": "application/json"},
    )
    assert status == 411
    assert "chunked" in str(body["error"])


def test_unknown_post_verb_does_not_poison_keepalive(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    # A 404 that leaves the body undrained poisoned the keep-alive connection:
    # the next request parsed the leftover body as its request line (probed
    # garbage 400). The server now closes; the client reconnects cleanly.
    _srv, port = server
    _make_run(tmp_path, "ka-run", [{"type": "run.start", "user_task": "x"}])
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps({"text": "hello"}).encode()
        conn.request(
            "POST", "/api/run/ka-run/bogusverb", payload, {"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        # Same client object again: must yield a clean 200, not body-garbage 400.
        conn.request("GET", "/api/hub")
        resp2 = conn.getresponse()
        assert resp2.status == 200
        assert json.loads(resp2.read())["runs"]
    finally:
        conn.close()


# --- CSRF: cross-site state-changing POSTs are refused -----------------------


def test_cross_origin_post_refused(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    _make_machine_with_state(tmp_path, "csrf1", "0000-review")
    status, body = _post_raw(
        port,
        "/api/machine/csrf1/poke",
        json.dumps({"message": "x"}).encode(),
        {
            "Content-Type": "application/json",
            "Host": f"127.0.0.1:{port}",
            "Origin": "https://evil.example",
        },
    )
    assert status == 403
    assert "cross-origin" in str(body.get("error", ""))


def test_non_json_content_type_post_refused(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "csrf2", "0000-review")
    # A JSON body smuggled in as a CORS-simple text/plain request is refused,
    # and the signal file is NOT written.
    status, _ = _post_raw(
        port,
        "/api/machine/csrf2/poke",
        json.dumps({"message": "x"}).encode(),
        {"Content-Type": "text/plain", "Host": f"127.0.0.1:{port}"},
    )
    assert status == 403
    assert not (inst / "signal").exists()


def test_same_origin_post_allowed(server: tuple[WebServer, int], tmp_path: Path) -> None:
    _srv, port = server
    inst, _ = _make_machine_with_state(tmp_path, "csrf3", "0000-review")
    status, body = _post_raw(
        port,
        "/api/machine/csrf3/poke",
        json.dumps({"message": "ok"}).encode(),
        {
            "Content-Type": "application/json",
            "Host": f"127.0.0.1:{port}",
            "Origin": f"http://127.0.0.1:{port}",
        },
    )
    assert status == 200 and body["ok"] is True
    assert (inst / "signal").exists()


# --- machine answers route to the rendered state, not the newest -------------


def test_machine_answer_routes_to_named_state_not_newest(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    # Two agent states, each with its own approval-1. The operator was shown the
    # OLDER state's prompt; the machine has since advanced to a newer state.
    inst, old_state = _make_machine_with_state(tmp_path, "adv", "0001-work")
    new_state = inst / "states" / "0002-review"
    new_state.mkdir(parents=True)
    (new_state / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(
        port,
        "/api/machine/adv/approve",
        {"id": "approval-1", "approved": True, "state": "0001-work"},
    )
    assert status == 200 and body["ok"] is True
    # The answer landed in the state the prompt was rendered from, NOT the newest.
    assert (old_state / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "yes"
    assert not (new_state / "approvals" / "approval-1.answer").exists()


def test_machine_answer_defaults_to_newest_state_without_hint(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    inst, _old = _make_machine_with_state(tmp_path, "adv2", "0001-work")
    new_state = inst / "states" / "0002-review"
    new_state.mkdir(parents=True)
    (new_state / "logs.jsonl").write_text("", encoding="utf-8")
    status, body = _post(port, "/api/machine/adv2/answer", {"id": "question-1", "answer": "hi"})
    assert status == 200 and body["ok"] is True
    assert (new_state / "questions" / "question-1.answer").read_text(encoding="utf-8") == "hi"


def test_machine_answer_state_hint_traversal_is_contained(
    server: tuple[WebServer, int], tmp_path: Path
) -> None:
    _srv, port = server
    _make_machine_with_state(tmp_path, "adv3", "0001-work")
    escape = tmp_path / "pwned.answer"
    status, _ = _post(
        port,
        "/api/machine/adv3/approve",
        {"id": "approval-1", "approved": True, "state": "../../../../pwned"},
    )
    assert status != 200
    assert not escape.exists()
