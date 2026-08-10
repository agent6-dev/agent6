# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Streamable-HTTP session ids: captured from initialize, echoed thereafter,
dropped on the 404 that means the server expired the session; a stateless
server never grows the header."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent6.tools.mcp_http import HttpTransport, MCPSessionExpired


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    issue_session: str = ""  # "" = stateless: never send the header
    expire_after: int = -1  # request # that answers 404 (spec: session expired)
    seen_headers: list[str]
    count: int = 0


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        srv = self.server
        assert isinstance(srv, _Server)
        srv.count += 1
        srv.seen_headers.append(self.headers.get("mcp-session-id", ""))
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        if srv.expire_after == srv.count:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        self.send_response(200)
        if srv.issue_session:
            self.send_header("mcp-session-id", srv.issue_session)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.seen_headers = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _transport(srv: _Server) -> HttpTransport:
    return HttpTransport(name="t", url=f"http://127.0.0.1:{srv.server_address[1]}/mcp")


def test_session_id_is_captured_and_echoed(server: _Server) -> None:
    server.issue_session = "sess-1"
    t = _transport(server)
    t.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, timeout_s=5)
    t.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, timeout_s=5)
    assert server.seen_headers == ["", "sess-1"]
    assert t.session_id == "sess-1"


def test_a_stateless_server_never_grows_the_header(server: _Server) -> None:
    t = _transport(server)
    t.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, timeout_s=5)
    t.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, timeout_s=5)
    assert server.seen_headers == ["", ""]
    assert t.session_id == ""


def test_expiry_404_drops_the_session_and_raises_its_own_type(server: _Server) -> None:
    server.issue_session = "sess-1"
    server.expire_after = 2
    t = _transport(server)
    t.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, timeout_s=5)
    with pytest.raises(MCPSessionExpired):
        t.send({"jsonrpc": "2.0", "id": 2, "method": "tools/call"}, timeout_s=5)
    assert t.session_id == "", "the expired id must not be re-echoed"
    # The next send starts clean, ready for the caller's fresh handshake.
    t.send({"jsonrpc": "2.0", "id": 3, "method": "initialize"}, timeout_s=5)
    assert server.seen_headers[-1] == ""


def test_a_bare_404_with_no_session_is_an_ordinary_error(server: _Server) -> None:
    from agent6.tools.mcp_http import MCPHttpError

    server.expire_after = 1
    t = _transport(server)
    with pytest.raises(MCPHttpError) as exc:
        t.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, timeout_s=5)
    assert not isinstance(exc.value, MCPSessionExpired)


def test_a_malformed_session_id_is_dropped_not_echoed(server: _Server) -> None:
    """A server-assigned id with a control or non-ASCII byte is untrusted: it
    is dropped (treated as stateless), never echoed into a header, so it cannot
    crash a later send or smuggle a header. Mirrors the token guard in _auth."""
    server.issue_session = "abc\x9f\r\nX-Evil: 1"  # non-ASCII + a CRLF payload
    t = _transport(server)
    t.send({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, timeout_s=5)
    assert t.session_id == "", "a malformed session id must not be stored"
    # The next send carries no session header and does not raise.
    t.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, timeout_s=5)
    assert server.seen_headers[-1] == ""
