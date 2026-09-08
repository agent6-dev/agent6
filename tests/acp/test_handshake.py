# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The ACP transport and handshake, driven the way an editor drives it."""

from __future__ import annotations

import io
import json
from typing import Any

from agent6.ui.acp.rpc import INVALID_PARAMS, PARSE_ERROR
from agent6.ui.acp.server import (
    INVALID_REQUEST,
    MAX_LINE_BYTES,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    ACPServer,
    capabilities_from,
)


def _exchange(*messages: object, raw: bytes = b"") -> list[dict[str, Any]]:
    """Feed messages in, return whatever came back out."""
    payload = raw or b"".join(json.dumps(m).encode() + b"\n" for m in messages)
    out = io.BytesIO()
    ACPServer(stdin=io.BytesIO(payload), stdout=out).serve()
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _init(**client_caps: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": 1, "clientCapabilities": client_caps},
    }


def test_the_handshake_answers_with_what_agent6_can_do() -> None:
    (reply,) = _exchange(_init())
    assert reply["id"] == 1
    result = reply["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["agentInfo"]["name"] == "agent6"


def test_session_load_is_reported_absent_rather_than_half_answered() -> None:
    """It is exactly what ACP v2 reorganises, and resume is where agent6 has
    the most of its own semantics."""
    (reply,) = _exchange(_init())
    assert reply["result"]["agentCapabilities"]["loadSession"] is False


def test_the_clients_capabilities_become_the_frontend_seam() -> None:
    """The whole reason FrontendCapabilities went in first: ACP's handshake IS
    a capability exchange, so it maps rather than needing new plumbing."""
    bare = capabilities_from({})
    assert bare.can_ask is True, "every ACP client must answer session/request_permission"


def test_an_unknown_method_is_an_error_not_a_crash() -> None:
    (reply,) = _exchange({"jsonrpc": "2.0", "id": 7, "method": "session/load", "params": {}})
    assert reply["error"]["code"] == METHOD_NOT_FOUND
    assert "session/load" in reply["error"]["message"]


def test_a_notification_is_acted_on_and_not_answered() -> None:
    """JSON-RPC: no id means no reply. Answering one desynchronises a client
    that is not waiting for anything."""
    assert _exchange({"jsonrpc": "2.0", "method": "initialize", "params": {}}) == []


def test_a_request_with_no_method_is_refused_by_id() -> None:
    (reply,) = _exchange({"jsonrpc": "2.0", "id": 3})
    assert reply["error"]["code"] == INVALID_REQUEST


def test_garbage_gets_a_parse_error_without_killing_the_connection() -> None:
    """JSON-RPC requires a null-id parse error, then the next valid request
    must still work on the same connection."""
    replies = _exchange(raw=b"not json\n" + json.dumps(_init()).encode() + b"\n")
    assert replies[0]["id"] is None
    assert replies[0]["error"]["code"] == PARSE_ERROR
    assert "invalid JSON" in replies[0]["error"]["message"]
    assert replies[1]["id"] == 1


def test_a_wrong_jsonrpc_version_names_the_invalid_envelope() -> None:
    (reply,) = _exchange({**_init(), "jsonrpc": "1.0"})
    assert reply["error"]["code"] == INVALID_REQUEST
    assert "jsonrpc" in reply["error"]["message"] and "2.0" in reply["error"]["message"]


def test_non_object_params_name_the_supported_method_they_malformed() -> None:
    (reply,) = _exchange({"jsonrpc": "2.0", "id": 8, "method": "initialize", "params": []})
    assert reply["error"]["code"] == INVALID_PARAMS
    assert "initialize" in reply["error"]["message"] and "object" in reply["error"]["message"]


def test_initialize_requires_a_numeric_protocol_version() -> None:
    for params in (
        {},
        {"protocolVersion": "one"},
        {"protocolVersion": True},
        {"protocolVersion": -1},
        {"protocolVersion": 65_536},
    ):
        (reply,) = _exchange({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": params})
        assert reply["error"]["code"] == INVALID_PARAMS
        assert "protocolVersion" in reply["error"]["message"]


def test_an_invalid_request_id_is_refused_with_a_null_protocol_id() -> None:
    for bad_id in ([], True, 1.5):
        (reply,) = _exchange({**_init(), "id": bad_id})
        assert reply["id"] is None
        assert reply["error"]["code"] == INVALID_REQUEST
        assert "id" in reply["error"]["message"]


def test_a_non_object_message_is_an_invalid_request() -> None:
    (reply,) = _exchange(["initialize"])
    assert reply["id"] is None
    assert reply["error"]["code"] == INVALID_REQUEST
    assert "object" in reply["error"]["message"]


def test_an_oversized_line_is_refused_not_buffered() -> None:
    """An unbounded readline buffers the whole line BEFORE any size check, so
    a runaway client could exhaust memory before the cap could refuse it. The
    refusal carries no id (the id is in the dropped bytes) and the next
    request still works."""
    huge = b'{"jsonrpc":"2.0","id":9,"method":"initialize","params":{"x":"'
    huge += b"A" * (MAX_LINE_BYTES + 64) + b'"}}\n'
    replies = _exchange(raw=huge + json.dumps(_init()).encode() + b"\n")
    assert [r["id"] for r in replies] == [None, 1]
    assert replies[0]["error"]["code"] == INVALID_REQUEST
    assert str(MAX_LINE_BYTES) in replies[0]["error"]["message"]


def test_text_that_cannot_encode_does_not_desynchronise_the_stream() -> None:
    """A lone surrogate in model-emitted text would otherwise raise mid-write,
    leaving a half-written line an editor cannot parse."""
    out = io.BytesIO()
    server = ACPServer(stdin=io.BytesIO(b""), stdout=out)
    server.notify_raw({"jsonrpc": "2.0", "method": "x", "params": {"t": "ok \ud83d tail"}})
    line = out.getvalue()
    assert line.endswith(b"\n")
    assert json.loads(line)["params"]["t"].startswith("ok ")


def test_a_clients_answer_is_delivered_before_its_envelope_is_judged() -> None:
    """An answer to session/request_permission that omits `jsonrpc` was
    refused by the envelope check before the reply path saw it: the worker
    waited out the permission timeout and denied, and the error frame named
    the id agent6 had minted, answering agent6's own request. The slot
    waiting on the answer vouches for it; a malformed answer to a minted id
    is refused under a null id."""
    import threading

    answer = {"id": "agent6-1", "result": {"outcome": {"outcome": "selected", "optionId": "0"}}}
    malformed = {"jsonrpc": "2.0", "id": "agent6-2"}
    payload = (json.dumps(answer) + "\n" + json.dumps(malformed) + "\n").encode()
    out = io.BytesIO()
    server = ACPServer(stdin=io.BytesIO(payload), stdout=out)
    got: list[dict[str, Any]] = []

    def ask() -> None:
        got.append(server.request("session/request_permission", {}, timeout_s=5.0))

    asker = threading.Thread(target=ask, daemon=True)
    asker.start()
    while "agent6-1" not in server._pending:  # pyright: ignore[reportPrivateUsage]
        pass
    server.serve()
    asker.join(timeout=5.0)
    assert got == [answer["result"]]
    frames = [json.loads(line) for line in out.getvalue().decode().splitlines() if line]
    errors = [f for f in frames if "error" in f]
    assert [f["id"] for f in errors] == [None]
    assert "no method" in errors[0]["error"]["message"]
