"""The real detached launcher carries lane snapshots and their diagnostic metadata."""

from __future__ import annotations

import http.server
import json
import pathlib
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from agent6 import kinds, paths
from agent6.app import lane_watch, lanes, stop
from agent6.config import Config, RoleModel
from agent6.harness import subrun
from agent6.sessions import manifest, process
from agent6.ui.cli import parallel
from agent6.viewmodel import session_log


@pytest.fixture
def local_provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Answer a real child's model call with finish_session, without a paid provider."""
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
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "finish-1",
                        "type": "function",
                        "function": {
                            "name": "finish_session",
                            "arguments": json.dumps({"summary": "Local lane completed."}),
                        },
                    }
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "local-1",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [{"index": 0, "delta": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
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


def test_detached_lane_loads_snapshot_and_records_notes(
    short_repo: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    local_provider: tuple[str, list[dict[str, Any]]],
) -> None:
    """Dropping either snapshot marker at the launcher changes the child's durable facts."""
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    subprocess.run(["git", "init", "-q", "-b", "main", str(short_repo)], check=True)
    (short_repo / "README.md").write_text("Local lane smoke.\n")
    subprocess.run(["git", "-C", str(short_repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(short_repo), "commit", "-q", "-m", "init"], check=True)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text(
        '[models.default]\nmodel="p/base"\neffort="high"\ntemperature=0.7\n'
        '[models.worker]\neffort="max"\n'
    )
    base_url, calls = local_provider
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": base_url}},
            "models": {"default": {"model": "p/base", "effort": "high", "temperature": 0.7}},
            "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "deny"},
            "harness": {"verify_when": "never", "max_turns": 3},
            "review": {"trigger": "off"},
        }
    )
    cfg = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"worker": RoleModel(effort=None, temperature=None)}
            )
        }
    )
    spec = subrun.LaneSpec(
        lane=1,
        session_id="detached-l1",
        workdir=tmp_path / "lane-1",
        route=kinds.ModelRoute("p", "local-reasoning"),
    )
    session_dir = paths.state_dir(spec.workdir) / "sessions" / "runs" / spec.session_id
    try:
        result = lanes.bridge_spawner(
            spec,
            "Finish immediately without edits.",
            cfg=cfg,
            explicit_leaves=frozenset(
                {"sandbox.isolation", "sandbox.protect_git", "sandbox.run_commands"}
            ),
            origin=short_repo,
            max_usd=None,
            fanout_id="detached",
            coordinator="detached",
            runtime=parallel.lane_runtime(),
        )
        assert result.ok, result.error
        deadline = time.monotonic() + 30
        assert lane_watch.await_lane(
            result, poll_interval_s=0.05, should_stop=lambda: time.monotonic() >= deadline
        ), "detached lane did not end before the deadline"
        log = session_dir / "logs.jsonl"
        assert session_log.scan_session_log(log).stop_reason == "finish_session", log.read_text()
        record = manifest.read_manifest(session_dir)
        assert record.models.driver is not None
        assert record.models.driver.model == "local-reasoning"
        assert record.models.driver.effort == "low"
        assert record.models.driver.effort_source == "backend default"
        assert record.parallel is not None
        assert record.parallel.config_notes == (
            "models.worker.temperature: selected None; TOML key omitted; reload uses 0.0.",
        )
        assert calls and all(
            call["model"] == "local-reasoning"
            and call["temperature"] == 0.0
            and call["reasoning"] == {"effort": "low"}
            for call in calls
        )
    finally:
        if process.worker_is_alive(session_dir):
            stop.stop_session(session_dir, wait_s=1, grace_s=1)
