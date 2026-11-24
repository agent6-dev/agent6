# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The recording proxy under `docs/screenshots` replays a cassette to a client that may leave."""

from __future__ import annotations

import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import time

_PROXY = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "llm_proxy.py"


def test_a_dropped_call_prints_no_traceback(tmp_path: pathlib.Path) -> None:
    """A client that left mid-stream got a second write, whose BrokenPipe printed a traceback."""
    events = "".join(f'data: {{"n": {i}}}\n\n' for i in range(200))
    entry = {
        "method": "POST",
        "path": "/v1/chat/completions",
        "status": 200,
        "content_type": "text/event-stream",
        "stream": True,
        "body": events,
    }
    cassette = tmp_path / "cassette.jsonl"
    cassette.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        **os.environ,
        "AGENT6_PROXY_MODE": "replay",
        "AGENT6_PROXY_CASSETTE": str(cassette),
        "AGENT6_PROXY_PORT": str(port),
        "AGENT6_PROXY_CHUNK_MS": "5",
    }
    proxy = subprocess.Popen(
        [sys.executable, str(_PROXY)], env=env, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                client = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                assert time.monotonic() < deadline, "the proxy never listened"
                time.sleep(0.05)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        )
        assert client.recv(64)
        # Reset, not a clean close, so the proxy's next write fails at once.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        time.sleep(0.5)
    finally:
        proxy.terminate()
        _, stderr = proxy.communicate(timeout=10)
    assert "Traceback" not in stderr, stderr
