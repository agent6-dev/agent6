# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve a provider reply over loopback and keep the requests it received."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from collections.abc import Generator
from typing import Any


@contextlib.contextmanager
def serve(
    reply: dict[str, Any] | list[dict[str, Any]], *, status: int = 200
) -> Generator[tuple[str, list[dict[str, Any]]]]:
    """Serve one JSON object or a sequence of SSE data frames on every request."""
    calls: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            calls.append({})
            self.answer()

        def do_POST(self) -> None:
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.answer()

        def answer(self) -> None:
            self.send_response(status)
            stream = isinstance(reply, list)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.end_headers()
            body = (
                "".join(f"data: {json.dumps(frame)}\n\n" for frame in reply)
                if stream
                else json.dumps(reply)
            )
            self.wfile.write(body.encode())

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join()
