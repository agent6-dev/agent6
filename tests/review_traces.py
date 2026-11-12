from __future__ import annotations

import contextlib
import http.server
import json
import threading
from collections.abc import Generator
from typing import Any

PASS = '{"verdict":"pass","summary":"ok","findings":[]}'
BLOCK = json.dumps(
    {
        "verdict": "block",
        "summary": "bug",
        "findings": [
            {
                "category": "correctness",
                "severity": "block",
                "location": {"path": "README.md", "side": "head", "start_line": 2, "end_line": 2},
                "title": "bug",
                "detail": "changed",
            }
        ],
    }
)
CASES = (
    "pass",
    "block",
    "risk",
    "risk-cut",
    "junk",
    "invalid",
    "truncated",
    "empty",
    "provider-error",
    "tools",
)


def response(case: str, call: int) -> tuple[str, str, list[dict[str, Any]]]:
    text = {
        "pass": PASS,
        "block": BLOCK,
        "risk": json.dumps({**json.loads(PASS), "risk": "an unsafe boundary"}),
        "risk-cut": json.dumps({**json.loads(PASS), "risk": "x" * 1100}),
        "junk": "not JSON",
        "invalid": '{"verdict":"maybe"}',
        "truncated": '{"verdict":"blo',
        "empty": "",
        "tools": PASS if call == 3 else "",
    }.get(case, "")
    tools = (
        [
            {
                "index": 0,
                "id": f"read-{call}",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ]
        if case == "tools"
        else []
    )
    return text, "length" if case == "truncated" else "tool_calls" if tools else "stop", tools


def edits(body: dict[str, Any]) -> bool:
    """Return whether a request offers an edit tool: a worker's does, a reviewer's never."""
    return any(t.get("function", {}).get("name") == "apply_edit" for t in body.get("tools", []))


@contextlib.contextmanager
def server(case: str) -> Generator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if case == "provider-error":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"scripted refusal"}}')
                return
            if case == "worker" and edits(body):
                worker_calls = sum(edits(c) for c in calls)
                name, args = (
                    (
                        "apply_edit",
                        {"path": "README.md", "old_string": "base", "new_string": "changed"},
                    )
                    if worker_calls == 1
                    else ("finish_session", {"summary": "Changed README"})
                )
                text, stop, tools = (
                    "",
                    "tool_calls",
                    [
                        {
                            "index": 0,
                            "id": f"worker-{worker_calls}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                )
            else:
                text, stop, tools = response("pass" if case == "worker" else case, len(calls))
            delta: dict[str, Any] = {"role": "assistant", "content": text}
            if tools:
                delta["tool_calls"] = tools
            chunk = {
                "id": "trace",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": stop}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cost": 0.001,
                },
            }
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                chunk["choices"] = [{"index": 0, "message": delta, "finish_reason": stop}]
                self.wfile.write(json.dumps(chunk).encode())

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
