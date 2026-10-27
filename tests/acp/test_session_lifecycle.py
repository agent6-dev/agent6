# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ACP `session/list`, `session/resume` and `session/close`: an editor reopens its threads."""

from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import threading
import time
from typing import Any

import pytest

from agent6 import paths
from agent6.sessions import process
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as acp_session
from tests import review_helpers


@pytest.fixture(autouse=True)
def _state_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _sessions(run: Any) -> acp_session.Sessions:
    return acp_session.Sessions(run=run, state_dir_for=paths.state_dir)


def _drive(payload: bytes, sessions: acp_session.Sessions) -> list[dict[str, Any]]:
    out = io.BytesIO()
    acp_server.ACPServer(stdin=io.BytesIO(payload), stdout=out, sessions=sessions).serve()
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _msg(req_id: int, method: str, **params: Any) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}).encode()


def _record(
    cwd: pathlib.Path,
    session_id: str,
    task: str,
    *,
    bucket: str = "runs",
    mtime: float,
    checkpoint: bool = True,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Record one ended session the way a run leaves it: a manifest, a journal, a snapshot."""
    d = paths.state_dir(cwd) / "sessions" / bucket / session_id
    d.mkdir(parents=True)
    mode = bucket[:-1]
    manifest: dict[str, Any] = {
        "version": 2,
        "session_id": session_id,
        "mode": mode,
        "user_task": task,
        **(extra or {}),
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {"type": "session.start", "mode": mode, "user_task": task},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if checkpoint:
        (d / "loop_state.json").write_text("{}", encoding="utf-8")
    for path in (d, d / "logs.jsonl"):
        os.utime(path, (mtime, mtime))
    return d


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_the_handshake_advertises_the_session_lifecycle() -> None:
    (reply,) = _drive(
        _msg(1, "initialize", protocolVersion=1, clientCapabilities={}) + b"\n", _sessions(None)
    )
    capabilities = reply["result"]["agentCapabilities"]
    assert capabilities["sessionCapabilities"] == {"list": {}, "resume": {}, "close": {}}
    assert capabilities["loadSession"] is True


def test_list_names_the_repositorys_resumable_runs_newest_first(tmp_path: pathlib.Path) -> None:
    """The listing is the runs an editor can reopen: agent6's ids, the tasks, the last activity."""
    cwd = _repo(tmp_path)
    _record(cwd, "older-run-AAAAAA", "first task", mtime=1_700_000_000.0)
    _record(cwd, "newer-run-BBBBBB", "second task\nwith a second line", mtime=1_700_000_600.0)
    _record(cwd, "a-plan-CCCCCC", "a plan", bucket="plans", mtime=1_700_000_900.0)
    _record(cwd, "fan-out-DDDDDD", "a fan-out", mtime=1_700_000_950.0, extra={"fanout": {}})
    _record(cwd, "no-snap-EEEEEE", "died early", mtime=1_700_000_980.0, checkpoint=False)
    sessions = _sessions(None)
    (reply,) = _drive(_msg(1, "session/list", cwd=str(cwd)) + b"\n", sessions)
    rows = reply["result"]["sessions"]
    assert [r["sessionId"] for r in rows] == ["newer-run-BBBBBB", "older-run-AAAAAA"]
    assert rows[0]["cwd"] == str(cwd) and rows[0]["updatedAt"] == "2023-11-14T22:23:20+00:00"
    assert rows[0]["title"].startswith("second task") and "\n" not in rows[0]["title"]
    assert "nextCursor" not in reply["result"]
    (empty,) = _drive(_msg(2, "session/list") + b"\n", sessions)
    assert empty["result"] == {"sessions": []}
    opened, listed = _drive(
        _msg(3, "session/new", cwd=str(cwd)) + b"\n" + _msg(4, "session/list") + b"\n", sessions
    )
    assert "sessionId" in opened["result"]
    assert [r["sessionId"] for r in listed["result"]["sessions"]] == [
        "newer-run-BBBBBB",
        "older-run-AAAAAA",
    ]
    relative, elsewhere, paged = _drive(
        _msg(5, "session/list", cwd="relative")
        + b"\n"
        + _msg(6, "session/list", cwd=str(tmp_path / "not-a-repo"))
        + b"\n"
        + _msg(7, "session/list", cwd=str(cwd), cursor="x")
        + b"\n",
        sessions,
    )
    assert "absolute" in relative["error"]["message"]
    assert elsewhere["error"]["code"] == -32602 and "cursor" in paged["error"]["message"]


def test_resume_opens_a_recorded_run_under_its_own_id(tmp_path: pathlib.Path) -> None:
    """A resumed thread's next prompt reaches the runner with the run id set, nothing new minted."""
    cwd = _repo(tmp_path)
    _record(cwd, "quiet-run-AAAAAA", "the task", mtime=time.time())
    _record(cwd, "quiet-run-BBBBBB", "another", mtime=time.time())
    _record(cwd, "a-plan-CCCCCC", "a plan", bucket="plans", mtime=time.time())
    _record(cwd, "fan-out-DDDDDD", "a fan-out", mtime=time.time(), extra={"fanout": {}})
    seen: list[tuple[str, str, str]] = []

    def run(session: acp_session.Session, text: str) -> str:
        seen.append((session.acp_id, session.session_id, text))
        return "end_turn"

    sessions = _sessions(run)

    def resume(req_id: int, session_id: str) -> bytes:
        return _msg(req_id, "session/resume", cwd=str(cwd), sessionId=session_id, mcpServers=[])

    unknown, prefix, plan, fanout, resumed, again, prompted = _drive(
        resume(1, "gone-run-ZZZZZZ")
        + b"\n"
        + resume(2, "quiet-run")
        + b"\n"
        + resume(3, "a-plan-CCCCCC")
        + b"\n"
        + resume(4, "fan-out-DDDDDD")
        + b"\n"
        + resume(5, "quiet-run-AAAAAA")
        + b"\n"
        + resume(6, "quiet-run-AAAAAA")
        + b"\n"
        + _msg(
            7,
            "session/prompt",
            sessionId="quiet-run-AAAAAA",
            prompt=[{"type": "text", "text": "carry on"}],
        )
        + b"\n",
        sessions,
    )
    assert "no recorded run 'gone-run-ZZZZZZ'" in unknown["error"]["message"]
    assert "no recorded run 'quiet-run'" in prefix["error"]["message"]
    assert "is a plan; an editor resumes runs" in plan["error"]["message"]
    assert fanout["error"]["message"].startswith("cannot resume fan-out-DDDDDD: ")
    assert resumed["result"] == {} and again["result"] == {}
    assert prompted["result"] == {"stopReason": "end_turn"}
    assert seen == [("quiet-run-AAAAAA", "quiet-run-AAAAAA", "carry on")]
    (cwd / "sub").mkdir()
    (elsewhere,) = _drive(
        _msg(8, "session/resume", cwd=str(cwd / "sub"), sessionId="quiet-run-AAAAAA", mcpServers=[])
        + b"\n",
        sessions,
    )
    assert f"quiet-run-AAAAAA is open for {cwd}; close it" in elsewhere["error"]["message"]
    alias = tmp_path / "alias"
    alias.symlink_to(cwd)
    (same,) = _drive(
        _msg(10, "session/resume", cwd=str(alias), sessionId="quiet-run-AAAAAA", mcpServers=[])
        + b"\n",
        sessions,
    )
    assert same["result"] == {}
    (editor_servers,) = _drive(
        _msg(
            9,
            "session/resume",
            cwd=str(cwd),
            sessionId="quiet-run-BBBBBB",
            mcpServers=[{"name": "x", "command": "srv"}],
        )
        + b"\n",
        sessions,
    )
    assert "[mcp.servers]" in editor_servers["error"]["message"]


def test_resume_refuses_a_run_whose_worker_is_alive(tmp_path: pathlib.Path) -> None:
    cwd = _repo(tmp_path)
    d = _record(cwd, "busy-run-AAAAAA", "the task", mtime=time.time())
    process.write_worker_pid(d, os.getpid())
    try:
        (reply,) = _drive(
            _msg(1, "session/resume", cwd=str(cwd), sessionId="busy-run-AAAAAA") + b"\n",
            _sessions(None),
        )
    finally:
        process.clear_worker_pid(d)
    assert "is running; `agent6 attach busy-run-AAAAAA`" in reply["error"]["message"]


def test_load_replays_the_recorded_conversation_before_answering(tmp_path: pathlib.Path) -> None:
    """Loading keeps full prompts, tool outcomes and findings in journal order."""
    cwd = _repo(tmp_path)
    reviewed = review_helpers.reviewed_session(cwd)
    panel = json.loads(reviewed.logs_path.read_text())
    panel.pop("ts")
    panel["findings"] = panel["findings"][:1]
    panel["injected_finding_ids"] = ["f1"]
    task = "Inspect the parser\n\n" + "Keep every detail. " * 20
    d = _record(cwd, "quiet-run-AAAAAA", task, mtime=time.time())
    rows = [
        {"type": "session.start", "mode": "run", "user_task": task},
        {"type": "role.thinking_delta", "text": "Check the empty input."},
        {"type": "role.text_delta", "text": "I will "},
        {"type": "role.text_delta", "text": "inspect it."},
        {"type": "role.result", "role": "worker", "text": "I will inspect it."},
        {"type": "tool.call", "call_id": 1, "name": "read_file", "args": {"path": "parser.py"}},
        {
            "type": "tool.result",
            "call_id": 1,
            "name": "read_file",
            "ok": True,
            "summary": "2 lines",
            "head_tail": "def parse(text):\n    return text[0]",
            "lines_total": 2,
            "paths": ["parser.py"],
        },
        panel,
        {"type": "diff.updated", "sha": "abc1234", "patch": "+guard"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    journal = "".join(json.dumps(row) + "\n" for row in rows)
    (d / "logs.jsonl").write_text(journal, encoding="utf-8")
    sessions = _sessions(None)
    frames = _drive(
        _msg(1, "session/load", cwd=str(cwd), sessionId=d.name, mcpServers=[]) + b"\n",
        sessions,
    )
    assert frames[-1] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert all(frame["method"] == "session/update" for frame in frames[:-1])
    assert all(frame["params"]["sessionId"] == d.name for frame in frames[:-1])
    updates = [frame["params"]["update"] for frame in frames[:-1]]
    assert [update["sessionUpdate"] for update in updates] == [
        "user_message_chunk",
        "agent_thought_chunk",
        "agent_message_chunk",
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
        "agent_message_chunk",
        "agent_message_chunk",
        "agent_message_chunk",
    ]
    assert updates[0]["content"]["text"] == task.strip()
    assert updates[1]["content"]["text"] == "Check the empty input."
    assert updates[2]["content"]["text"] == "I will inspect it."
    assert updates[3]["toolCallId"] == updates[4]["toolCallId"] == f"{d.name}:1:1"
    assert updates[3]["title"] == "read_file parser.py"
    assert updates[4]["status"] == "completed"
    assert updates[4]["content"] == [
        {
            "type": "content",
            "content": {
                "type": "text",
                "text": "2 lines\ndef parse(text):\n    return text[0]\n…(2 lines)",
            },
        }
    ]
    assert updates[4]["locations"] == [{"path": str(cwd / "parser.py")}]
    assert updates[5]["content"]["text"].startswith("[agent6] Review panel: complete;")
    assert updates[6]["content"]["text"] == (
        "[agent6] Finding f1 · block · parser.py:3-4\nDefect f1\nReproduce with an empty input.\n"
    )
    assert updates[7]["content"]["text"] == "[agent6] committed abc1234 · 1 lines\n"
    assert updates[8]["content"]["text"] == "[agent6] Session passed - 1 tool · 1 commit\n"
    assert (d / "logs.jsonl").read_text(encoding="utf-8") == journal


def test_load_rejects_an_unknown_run_with_the_resume_error(tmp_path: pathlib.Path) -> None:
    """An unknown load id is invalid params, not an unsupported method or a new session."""
    cwd = _repo(tmp_path)
    (reply,) = _drive(
        _msg(1, "session/load", cwd=str(cwd), sessionId="gone-run-ZZZZZZ", mcpServers=[]) + b"\n",
        _sessions(None),
    )
    assert reply["error"] == {
        "code": -32602,
        "message": f"no recorded run 'gone-run-ZZZZZZ' in {cwd}",
    }


def test_loaded_executions_keep_distinct_tools_and_resume_the_same_run(
    tmp_path: pathlib.Path,
) -> None:
    """Replayed call ids cannot collide across executions or with the next prompt's calls."""
    cwd = _repo(tmp_path)
    d = _record(cwd, "quiet-run-AAAAAA", "first", mtime=time.time())
    call = {"type": "tool.call", "name": "run_command", "call_id": 1, "args": {"argv": ["pwd"]}}
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "first"},
        call,
        {"type": "loop.resume.start"},
        {"type": "loop.steer.injected", "text": "carry on"},
        call,
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    seen: list[tuple[str, str, int, str]] = []

    def run(session: acp_session.Session, text: str) -> str:
        seen.append((session.acp_id, session.session_id, session.execution, text))
        return "end_turn"

    sessions = _sessions(run)
    frames = _drive(
        _msg(1, "session/load", cwd=str(cwd), sessionId=d.name, mcpServers=[]) + b"\n",
        sessions,
    )
    assert frames[-1] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    updates = [frame["params"]["update"] for frame in frames[:-1]]
    assert [
        update["content"]["text"]
        for update in updates
        if "content" in update and update["sessionUpdate"] == "user_message_chunk"
    ] == ["first", "carry on"]
    tools = [update for update in updates if "toolCallId" in update]
    assert [(tool["toolCallId"], tool["status"]) for tool in tools] == [
        (f"{d.name}:1:1", "in_progress"),
        (f"{d.name}:1:1", "failed"),
        (f"{d.name}:2:1", "in_progress"),
        (f"{d.name}:2:1", "failed"),
    ]
    assert "no result (the run died)" in tools[-1]["content"][0]["content"]["text"]
    (reply,) = _drive(
        _msg(2, "session/prompt", sessionId=d.name, prompt=[{"type": "text", "text": "more"}])
        + b"\n",
        sessions,
    )
    assert reply["result"] == {"stopReason": "end_turn"}
    assert seen == [(d.name, d.name, 2, "more")]


def test_close_drops_an_idle_conversation_and_cancels_a_running_one(
    tmp_path: pathlib.Path,
) -> None:
    """After close the id is gone; a prompt in flight answers `cancelled` and the id goes then."""
    cwd = _repo(tmp_path)
    sessions = _sessions(None)
    opened = sessions.new({"cwd": str(cwd)})
    acp_id = opened["sessionId"]
    closed, gone, unknown = _drive(
        _msg(2, "session/close", sessionId=acp_id)
        + b"\n"
        + _msg(3, "session/prompt", sessionId=acp_id, prompt=[{"type": "text", "text": "x"}])
        + b"\n"
        + _msg(4, "session/close", sessionId="never-was")
        + b"\n",
        sessions,
    )
    assert closed["result"] == {} and "no session" in gone["error"]["message"]
    assert "no session 'never-was'" in unknown["error"]["message"]

    released = threading.Event()

    def run(session: acp_session.Session, _text: str) -> str:
        released.wait(timeout=5)
        return "end_turn"

    busy = _sessions(run)
    acp_id = busy.new({"cwd": str(cwd)})["sessionId"]
    out = io.BytesIO()
    connection = acp_server.ACPServer(stdin=io.BytesIO(), stdout=out, sessions=busy)
    connection._handle(  # pyright: ignore[reportPrivateUsage]
        _msg(5, "session/prompt", sessionId=acp_id, prompt=[{"type": "text", "text": "go"}])
    )
    session = busy.get({"sessionId": acp_id})
    assert session.is_running()
    connection._handle(_msg(6, "session/close", sessionId=acp_id))  # pyright: ignore[reportPrivateUsage]
    connection._handle(  # pyright: ignore[reportPrivateUsage]
        _msg(7, "session/prompt", sessionId=acp_id, prompt=[{"type": "text", "text": "more"}])
    )
    assert session.cancelled and session.closed
    released.set()
    assert session.thread is not None
    session.thread.join(timeout=5)
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    replies = {r["id"]: r for r in map(json.loads, lines) if "id" in r}
    assert replies[6]["result"] == {}
    assert replies[5]["result"] == {"stopReason": "cancelled"}
    assert "no session" in replies[7]["error"]["message"]
    with pytest.raises(acp_server.rpc.RpcError, match="no session"):
        busy.get({"sessionId": acp_id})
