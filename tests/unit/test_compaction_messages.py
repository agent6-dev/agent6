from __future__ import annotations

import http.server
import io
import json
import socket
import time
from unittest import mock

import pytest
from bench import compaction_messages

BODY = b'{"text": "test reply"}'


@pytest.fixture
def scripted_handler() -> type[compaction_messages.Messages]:
    with (
        mock.patch.object(http.server, "HTTPServer") as server,
        compaction_messages.scripted(lambda body: body["text"]),
    ):
        return server.call_args.args[1]


def _head(length: str | None) -> bytes:
    header = f"Content-Length: {length}\r\n" if length is not None else ""
    return f"POST /v1/messages HTTP/1.0\r\n{header}\r\n".encode()


def _post(
    monkeypatch: pytest.MonkeyPatch,
    handler: type[compaction_messages.Messages],
    pieces: list[bytes],
    pace: float = 0.0,
) -> tuple[bytes, float]:
    """Deliver one piece per `pace` simulated seconds; return the reply and the time it took."""
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    sock = mock.Mock(spec=socket.socket)
    # The reader `socket.makefile` builds: a handler without its own still reads these pieces.
    sock.makefile.return_value = io.BufferedReader(socket.SocketIO(sock, "rb"))

    def recv_into(buffer: memoryview) -> int:
        timeout = sock.settimeout.call_args.args[0] if sock.settimeout.called else float("inf")
        if timeout < pace:
            clock[0] += timeout
            raise TimeoutError
        clock[0] += pace
        assert pieces, "read past the request"
        piece = pieces.pop(0)
        buffer[: len(piece)] = piece
        return len(piece)

    sock.recv_into.side_effect = recv_into
    reply = bytearray()
    sock.sendall.side_effect = reply.extend
    handler(sock, ("127.0.0.1", 1), mock.Mock())
    return bytes(reply), clock[0]


@pytest.mark.parametrize("scripted", [True, False], ids=["scripted", "replay"])
@pytest.mark.parametrize(("length", "status"), [(None, 411), ("-1", 400), ("8388609", 413)])
def test_rejected_lengths_never_reach_the_body_reader(
    monkeypatch: pytest.MonkeyPatch,
    scripted_handler: type[compaction_messages.Messages],
    scripted: bool,
    length: str | None,
    status: int,
) -> None:
    """Neither handler reads a body with a missing, negative or excessive length."""
    handler = scripted_handler if scripted else compaction_messages.Messages
    reply, _ = _post(monkeypatch, handler, [_head(length)])
    assert reply.split()[1] == str(status).encode()


@pytest.mark.parametrize(("pace", "seconds", "served"), [(2.0, 8.0, True), (3.0, 10.0, False)])
def test_the_reads_of_one_request_share_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
    scripted_handler: type[compaction_messages.Messages],
    pace: float,
    seconds: float,
    served: bool,
) -> None:
    """A request whose every read beats the socket timeout still ends ten seconds in."""
    head = _head(str(len(BODY)))
    pieces = [head[:20], head[20:], BODY[:10], BODY[10:]]
    reply, elapsed = _post(monkeypatch, scripted_handler, pieces, pace)
    assert (elapsed, bool(reply)) == (seconds, served)
    if served:
        headers, data = reply.split(b"\r\n\r\n", 1)
        assert headers.split()[1] == b"200"
        assert json.loads(data)["content"] == [{"type": "text", "text": "test reply"}]
