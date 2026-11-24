from __future__ import annotations

import http.server
import io
import json
import socket
from unittest import mock

import pytest
from bench import compaction_messages


@pytest.fixture
def scripted_handler() -> type[compaction_messages.Messages]:
    with (
        mock.patch.object(http.server, "HTTPServer") as server,
        compaction_messages.scripted(lambda body: body["text"]),
    ):
        return server.call_args.args[1]


def _post(
    handler: type[compaction_messages.Messages], sock: mock.Mock, length: str | None
) -> bytes:
    header = f"Content-Length: {length}\r\n" if length is not None else ""
    headers = io.BytesIO(f"POST /v1/messages HTTP/1.0\r\n{header}\r\n".encode())
    sock.makefile.return_value.readline.side_effect = headers.readline
    response = bytearray()
    sock.sendall.side_effect = response.extend
    handler(sock, ("127.0.0.1", 1), mock.Mock())
    return bytes(response)


@pytest.mark.parametrize("scripted", [True, False], ids=["scripted", "replay"])
@pytest.mark.parametrize(("length", "status"), [(None, 411), ("-1", 400), ("8388609", 413)])
def test_rejected_lengths_never_reach_the_body_reader(
    scripted_handler: type[compaction_messages.Messages],
    scripted: bool,
    length: str | None,
    status: int,
) -> None:
    """Neither handler reads a body with a missing, negative or excessive length."""
    sock = mock.Mock(spec=socket.socket)
    sock.makefile.return_value.read.side_effect = AssertionError("read with a rejected length")
    reply = _post(scripted_handler if scripted else compaction_messages.Messages, sock, length)
    assert reply.split()[1] == str(status).encode()


def test_a_scripted_post_reads_the_declared_bytes_with_a_socket_timeout(
    scripted_handler: type[compaction_messages.Messages],
) -> None:
    """An ordinary request reaches the script with a timeout already on its socket."""
    body = b'{"text": "test reply"}'
    sock = mock.Mock(spec=socket.socket)

    def read(size: int) -> bytes:
        assert size == len(body)
        sock.settimeout.assert_called_once()
        assert 0 < sock.settimeout.call_args.args[0] <= 60
        return body

    sock.makefile.return_value.read.side_effect = read
    reply = _post(scripted_handler, sock, str(len(body)))
    headers, data = reply.split(b"\r\n\r\n", 1)
    assert headers.split()[1] == b"200"
    assert json.loads(data)["content"] == [{"type": "text", "text": "test reply"}]
