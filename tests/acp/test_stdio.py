# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive the ACP entry point with a local recorded provider response."""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import select
import subprocess
import sys
import threading
from typing import Any

import pytest

from agent6 import paths
from agent6.viewmodel import transcript


class _Finisher(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "finish",
                                "type": "function",
                                "function": {
                                    "name": "finish_session",
                                    "arguments": json.dumps(
                                        {"summary": "The recorded model summary."}
                                    ),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object, **kwargs: object) -> None:
        pass


def test_console_script_splits_the_model_summary_from_the_run_status(
    tmp_path: pathlib.Path,
) -> None:
    """The console script sends the model summary separately from the labelled run status."""
    exe = pathlib.Path(sys.executable).with_name("agent6")
    if not exe.is_file():
        pytest.skip("agent6 console script is not installed beside the test interpreter")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "seed",
        ],
        check=True,
    )
    provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Finisher)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config.toml"
    config.write_text(
        '[providers.recorded]\napi_format = "openai"\nauth_style = "none"\n'
        f'base_url = "http://127.0.0.1:{provider.server_port}/v1"\n'
        '[models.worker]\nprovider = "recorded"\nmodel = "recorded"\n'
        '[sandbox]\nisolation = "none"\nrun_commands = "no"\n',
        encoding="utf-8",
    )
    root_flags = ["--allow-root"] if os.geteuid() == 0 else []
    with (tmp_path / "stderr").open("wb") as stderr:
        proc = subprocess.Popen(
            [str(exe), *root_flags, "--config", str(config), "acp"],
            cwd=repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            bufsize=0,
            env={**os.environ},
        )
        assert proc.stdin is not None and proc.stdout is not None

        def send(req_id: int, method: str, params: dict[str, Any]) -> None:
            assert proc.stdin is not None
            proc.stdin.write(
                json.dumps(
                    {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
                ).encode()
                + b"\n"
            )

        def recv() -> dict[str, Any]:
            assert proc.stdout is not None
            assert select.select([proc.stdout], [], [], 30)[0], (tmp_path / "stderr").read_text()
            line = proc.stdout.readline()
            assert line, (tmp_path / "stderr").read_text()
            return json.loads(line)

        try:
            send(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
            assert "result" in recv()
            send(2, "session/new", {"cwd": str(repo), "mcpServers": []})
            session_id = recv()["result"]["sessionId"]
            send(
                3,
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "Finish."}]},
            )
            texts: list[str] = []
            while True:
                message = recv()
                if message.get("id") == 3:
                    assert message["result"]["stopReason"] == "end_turn", texts
                    break
                update = message.get("params", {}).get("update", {})
                if update.get("sessionUpdate") == "agent_message_chunk":
                    texts.append(update["content"]["text"])
            journals = list((paths.state_dir(repo) / "sessions" / "runs").glob("*/logs.jsonl"))
            assert len(journals) == 1
            fold = transcript.TranscriptFold()
            items = [
                item
                for line in journals[0].read_text().splitlines()
                for item in fold.feed(json.loads(line))
            ]
            done = next(item for item in items if item.kind == "done")
            assert done.body == "The recorded model summary."
            assert done.body + "\n\n" in texts
            assert any(text.startswith(f"[agent6] Session {done.name}") for text in texts), texts
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)
