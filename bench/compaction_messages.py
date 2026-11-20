"""Serve a loopback Messages endpoint for the size and idle compaction smokes."""

from __future__ import annotations

import argparse
import copy
import http.server
import json
import pathlib
from typing import Any, ClassVar

SAMPLES = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "tests/unit/fixtures/anthropic_compaction.json"
    ).read_text()
)


class Messages(http.server.BaseHTTPRequestHandler):
    """Check the replay before serving the next captured reply."""

    idle: ClassVar[bool]
    previous: ClassVar[dict[str, Any]] = {}
    reply: ClassVar[list[dict[str, Any]]] = []
    turns: ClassVar[int] = 0

    def do_GET(self) -> None:
        """Return the local model listing."""
        self._send({"data": [{"id": "claude-haiku-5-5", "type": "model"}]})

    def do_POST(self) -> None:
        """Accept a summary only for the preceding worker request."""
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["size", "idle"])
    Messages.idle = parser.parse_args().mode == "idle"
    with http.server.HTTPServer(("127.0.0.1", 0), Messages) as server:
        print(f"http://127.0.0.1:{server.server_port}/v1", flush=True)
        server.serve_forever()
