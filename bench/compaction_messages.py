"""Serve a loopback Messages endpoint for the size and idle compaction smokes."""

from __future__ import annotations

import argparse
import contextlib
import copy
import http.server
import io
import itertools
import json
import pathlib
import socket
import threading
import time
from collections.abc import Buffer, Callable, Generator
from typing import Any, ClassVar

MAX_BODY_BYTES = 8 * 1024 * 1024

SAMPLES = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "tests/unit/fixtures/anthropic_compaction.json"
    ).read_text()
)


class Deadline(io.RawIOBase):
    """Give each socket read the time left until one deadline."""

    def __init__(self, sock: socket.socket, seconds: float) -> None:
        super().__init__()
        self.sock = sock
        self.end = time.monotonic() + seconds

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        left = self.end - time.monotonic()
        if left <= 0:
            raise TimeoutError("the request passed its deadline")
        self.sock.settimeout(left)
        return self.sock.recv_into(buffer)


class Messages(http.server.BaseHTTPRequestHandler):
    """Check the replay before serving the next captured reply."""

    timeout = 10.0
    idle: ClassVar[bool]
    previous: ClassVar[dict[str, Any]] = {}
    reply: ClassVar[list[dict[str, Any]]] = []
    turns: ClassVar[int] = 0

    def setup(self) -> None:
        """Read the request line, the headers and the body within `timeout` seconds."""
        super().setup()
        self.rfile = io.BufferedReader(Deadline(self.connection, self.timeout))

    def do_GET(self) -> None:
        """Return the local model listing."""
        self._send({"data": [{"id": "claude-haiku-5-5", "type": "model"}]})

    def do_POST(self) -> None:
        """Read one size-limited Messages request."""
        lengths = self.headers.get_all("Content-Length")
        if not lengths:
            self.send_error(411, "Content-Length is required")
            return
        try:
            (raw,) = lengths
            raw = raw.strip(" \t")
            size = int(raw) if raw.isascii() and raw.isdecimal() else -1
        except ValueError:
            size = -1
        if size < 0:
            self.send_error(400, "Invalid Content-Length")
            return
        if size > MAX_BODY_BYTES:
            self.send_error(413, f"Request exceeds {MAX_BODY_BYTES} bytes")
            return
        self._answer(json.loads(self.rfile.read(size)))

    def _answer(self, body: dict[str, Any]) -> None:
        """Accept a summary only for the preceding worker request."""
        betas = self.headers["anthropic-beta"].split(",")
        messages = body["messages"]
        print(
            f"turns={Messages.turns} messages={len(messages)} clearing={'context_management' in body} summary={'compaction' in body}",
            flush=True,
        )
        assert body["thinking"] == SAMPLES["own-restart-drop_block"]["thinking"]
        if "compaction" in body:
            assert all(
                body[key] == Messages.previous[key] for key in ("messages", "system", "tools")
            )
            assert "context_management" not in body and "compact-2026-09-04" in betas
            assert 0 < len(body["compaction"]["instructions"]) <= 16384
            self._send(SAMPLES["compact"]["response"])
            return
        response = copy.deepcopy(SAMPLES["turn3"]["response"])
        call = response["content"][-1]
        if messages[0]["role"] == "assistant":
            assert "compact-2026-09-04" in betas
            assert messages[0]["content"] == SAMPLES["compact"]["response"]["content"]
            assert messages[1]["content"] == Messages.reply
            result = messages[2]["content"][0]
            assert result["tool_use_id"] == Messages.reply[-1]["id"]
            if Messages.idle:
                assert json.loads(result["content"])["returncode"] == 0
            assert "TASK (the goal, verbatim)" in str(messages[2:])
            call.update(name="finish_session", input={"summary": "Checked the signed restart."})
        elif Messages.turns and Messages.idle:
            call.update(name="run_command", input={"argv": ["sleep", "4"]})
        else:
            call.update(
                name="read_file", input={"path": "a.txt" if not Messages.turns else "b.txt"}
            )
        call["id"] = f"tool-{Messages.turns}"
        if "context_management" in body:
            assert body["context_management"]["edits"][0]["exclude_tools"] == ["ask_user"]
            response["context_management"] = SAMPLES["clearing-strict"]["response"][
                "context_management"
            ]
        Messages.turns += 1
        Messages.previous = body
        Messages.reply = response["content"]
        self._send(response)

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextlib.contextmanager
def scripted(reply: Callable[[dict[str, Any]], str | tuple[str, dict[str, Any]]]) -> Generator[str]:
    """Serve scripted local-compaction turns using the captured Messages response shape."""
    ids = itertools.count(1)

    class Local(Messages):
        def do_GET(self) -> None:
            self._send({"data": [{"id": "claude-sonnet-4-5", "type": "model"}]})

        def _answer(self, body: dict[str, Any]) -> None:
            answer = reply(body)
            response = copy.deepcopy(SAMPLES["turn3"]["response"])
            if isinstance(answer, str):
                response.update(content=[{"type": "text", "text": answer}], stop_reason="end_turn")
            else:
                response["content"][-1].update(
                    id=f"script-{next(ids)}", name=answer[0], input=answer[1]
                )
            self._send(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with http.server.HTTPServer(("127.0.0.1", 0), Local) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["size", "idle"])
    Messages.idle = parser.parse_args().mode == "idle"
    with http.server.HTTPServer(("127.0.0.1", 0), Messages) as server:
        print(f"http://127.0.0.1:{server.server_port}/v1", flush=True)
        server.serve_forever()
