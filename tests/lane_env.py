# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mark the test process as a lane: write a launch file and point `AGENT6_LANE` at it."""

from __future__ import annotations

import contextlib
import http.server
import json
import pathlib
import threading
from collections.abc import Callable, Generator, Sequence
from typing import Any

import pytest

from agent6 import paths
from agent6.app import lane_launch
from agent6.sessions import manifest

FINISH: tuple[str, dict[str, Any]] = ("finish_session", {"summary": "Local lane completed."})


@contextlib.contextmanager
def scripted_provider(
    replies: Sequence[tuple[str, dict[str, Any]] | str] = (FINISH,),
    *,
    cost_usd: float | None = None,
    before_reply: Callable[[dict[str, Any]], None] | None = None,
) -> Generator[tuple[str, list[dict[str, Any]]]]:
    """Serve an OpenAI-shaped endpoint that answers each call with the next scripted reply.

    The last reply repeats once the script runs out. Every request body is kept so a test can
    read what the child sent.

    Args:
        replies: `(tool name, arguments)` or an answer's text per call, in order.
        cost_usd: The reported cost per response; None exercises fallback-token accounting.
        before_reply: An optional barrier after recording the request and before answering.

    Yields:
        The endpoint's `base_url` and the recorded request bodies.
    """
    calls: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if before_reply is not None:
                before_reply(body)
            reply = replies[min(len(calls) - 1, len(replies) - 1)]
            message: dict[str, Any] = {"role": "assistant", "content": reply}
            if not isinstance(reply, str):
                name, args = reply
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-{len(calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk: dict[str, Any] = {
                "id": f"local-{len(calls)}",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": message,
                        "finish_reason": "stop" if isinstance(reply, str) else "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            if cost_usd is not None:
                chunk["usage"]["cost"] = cost_usd
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def mark_lane(
    monkeypatch: pytest.MonkeyPatch,
    where: pathlib.Path,
    *,
    config: pathlib.Path | None = None,
    parent: str = "co",
    group: str = "fan",
    lane: int = 1,
    gate_source: manifest.GateSource = "",
    notes: tuple[str, ...] = (),
    operator_task: str | None = None,
    record_state: pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a lane launch file under `where` and set `AGENT6_LANE` and `AGENT6_CHILD`.

    Args:
        monkeypatch: The test's monkeypatch.
        where: A directory for the launch file.
        config: The snapshot the launch names; a placeholder path when the test loads none.
        parent: The parent session id.
        group: The fan-out id.
        lane: The lane number.
        gate_source: The gate's origin.
        notes: The snapshot's loss notes.
        operator_task: The operator's task before composition.
        record_state: Where the lane's record goes; the lane's own state when None.

    Returns:
        The launch file's path.
    """
    path = lane_launch.write(
        where,
        lane_launch.LaneLaunch(
            worktree=where / f"lane-{lane}",
            record_state=record_state or paths.state_dir(where / f"lane-{lane}"),
            session_id=f"{group}-l{lane}",
            task="task",
            config=config if config is not None else where / f"lane-{lane}-config.toml",
            notes=notes,
            parent=parent,
            group=group,
            lane=lane,
            gate_source=gate_source,
            operator_task=operator_task,
        ),
    )
    monkeypatch.setenv(lane_launch.ENV, str(path))
    monkeypatch.setenv("AGENT6_CHILD", "1")
    return path
