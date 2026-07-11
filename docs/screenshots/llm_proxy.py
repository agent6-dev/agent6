#!/usr/bin/env python3
"""Record/replay LLM proxy: a fake "local model" for deterministic demos + tests.

agent6 already talks to local models over plain HTTP (`base_url =
http://127.0.0.1:PORT`, `sandbox.agent_network = "local"`). This is such an
endpoint, with two modes:

  record   forward each request to the real upstream (relaying the caller's auth
           header), stream the response back while capturing it, and append the
           exchange to a cassette. Run a real agent6 run through this once to
           capture a real trajectory (needs real API keys).

  replay   serve the recorded responses from the cassette, in order per
           (method, path). No upstream, no key, fully deterministic -- so a real
           agent6 run (real loop, real tools, real dashboard) reproduces exactly,
           which is what the demo videos record and what a CI integration test
           could assert against.

It relays/replays raw bytes, so it is api-format-agnostic (openai or anthropic).
Nothing here imports agent6; agent6 is unmodified, just pointed at this URL.

  AGENT6_PROXY_MODE      record | replay   (default replay)
  AGENT6_PROXY_CASSETTE  path to the JSONL cassette
  AGENT6_PROXY_UPSTREAM  real base origin for record, e.g. https://openrouter.ai
  AGENT6_PROXY_PORT      default 8900
  AGENT6_PROXY_CHUNK_MS  per-SSE-event delay on replay (default 12), for a live
                         streaming feel in the recording
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODE = os.environ.get("AGENT6_PROXY_MODE", "replay")
CASSETTE = os.environ.get("AGENT6_PROXY_CASSETTE", "cassette.jsonl")
UPSTREAM = os.environ.get("AGENT6_PROXY_UPSTREAM", "").rstrip("/")
PORT = int(os.environ.get("AGENT6_PROXY_PORT", "8900"))
CHUNK_MS = int(os.environ.get("AGENT6_PROXY_CHUNK_MS", "12"))

_LOCK = threading.Lock()
# replay: (method, path) -> list of recorded exchanges, consumed in order.
_REPLAY: dict[tuple[str, str], list[dict]] = defaultdict(list)
_REPLAY_IDX: dict[tuple[str, str], int] = defaultdict(int)


def _load_cassette() -> None:
    if MODE != "replay":
        return
    with Path(CASSETTE).open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            e = json.loads(line)
            _REPLAY[(e["method"], e["path"])].append(e)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: object) -> None:  # quiet
        pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:
        self._handle("GET", self._body())

    def do_POST(self) -> None:
        self._handle("POST", self._body())

    # -- record: forward upstream, relay incrementally, capture -------------
    def _record(self, method: str, body: bytes) -> None:
        url = UPSTREAM + self.path
        # Drop accept-encoding so the upstream replies uncompressed: the cassette
        # stays plain text and we never have to decompress to relay.
        fwd = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "connection", "accept-encoding")
        }
        req = Request(url, data=body or None, headers=fwd, method=method)  # noqa: S310
        try:
            resp = urlopen(req, timeout=300)  # noqa: S310 - operator-fixed upstream, host process
            status, ctype = resp.status, resp.headers.get("Content-Type", "")
        except HTTPError as exc:
            self._save(method, exc.code, exc.headers.get("Content-Type", ""), exc.read())
            self._send(exc.code, exc.headers.get("Content-Type", ""), exc.read(), False)
            return
        # Relay chunks as they arrive (so agent6's idle watchdog sees bytes) and
        # capture the same bytes for the cassette.
        self.send_response(status)
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Connection", "close")
        self.end_headers()
        captured = bytearray()
        while chunk := resp.read(512):
            captured += chunk
            self.wfile.write(chunk)
            self.wfile.flush()
        self._save(method, status, ctype, bytes(captured))

    def _save(self, method: str, status: int, ctype: str, raw: bytes) -> None:
        entry = {
            "method": method,
            "path": self.path,
            "status": status,
            "content_type": ctype,
            "stream": "text/event-stream" in ctype,
            "body": raw.decode("utf-8", errors="replace"),
        }
        with _LOCK, Path(CASSETTE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # -- replay: serve the next recorded exchange for this (method, path) -----
    def _replay(self, method: str) -> None:
        key = (method, self.path)
        with _LOCK:
            i = _REPLAY_IDX[key]
            entries = _REPLAY.get(key, [])
            entry = entries[i] if i < len(entries) else None
            if entry is not None:
                _REPLAY_IDX[key] = i + 1
        if entry is None:
            self._send(500, "text/plain", b"no recorded response for this call", False)
            return
        self._send(
            entry["status"],
            entry["content_type"],
            entry["body"].encode("utf-8"),
            entry["stream"],
        )

    def _handle(self, method: str, body: bytes) -> None:
        try:
            if MODE == "record":
                self._record(method, body)
            else:
                self._replay(method)
        except Exception as exc:
            self._send(500, "text/plain", f"proxy error: {exc}".encode(), False)

    def _send(self, status: int, ctype: str, body: bytes, stream: bool) -> None:
        self.send_response(status)
        if ctype:
            self.send_header("Content-Type", ctype)
        if stream:
            # SSE: flush per event for a live streaming feel; close-framed.
            self.send_header("Connection", "close")
            self.end_headers()
            stall_ms = int(os.environ.get("AGENT6_PROXY_STALL_MS", "0"))
            stall_at = int(os.environ.get("AGENT6_PROXY_STALL_AT", "3"))
            for i, event in enumerate(body.split(b"\n\n")):
                if not event:
                    continue
                self.wfile.write(event + b"\n\n")
                self.wfile.flush()
                # Simulate a wedged upstream: pause once, partway through a
                # streamed response (AGENT6_PROXY_STALL_MS, off by default).
                if stall_ms and i == stall_at:
                    time.sleep(stall_ms / 1000.0)
                if CHUNK_MS:
                    time.sleep(CHUNK_MS / 1000.0)
        else:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    if MODE == "replay":
        _load_cassette()
        total = sum(len(v) for v in _REPLAY.values())
        print(f"[llm_proxy] replay :{PORT} <- {CASSETTE} ({total} exchanges)", file=sys.stderr)
    else:
        if not UPSTREAM:
            sys.exit("record mode needs AGENT6_PROXY_UPSTREAM (e.g. https://openrouter.ai)")
        Path(CASSETTE).write_text("", encoding="utf-8")  # fresh cassette
        print(f"[llm_proxy] record :{PORT} -> {UPSTREAM} ; cassette {CASSETTE}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
