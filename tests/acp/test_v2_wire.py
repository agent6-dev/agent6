# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every v2 frame agent6 sends, checked against the protocol's own v2 schema.

`schema_check.v2_errors` reads the published schema file and closes its open unions, so a
frame with a tag the schema names has to fit that tag's definition.
"""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import time
import typing
from typing import Any

import pytest

from agent6 import paths
from agent6.ui.acp import model, rpc, runner, updates, wire
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as acp_session
from agent6.viewmodel import transcript_items
from tests.acp import schema_check, test_session_lifecycle

_RECORDED = pathlib.Path(__file__).parent.parent / "unit" / "data" / "golden_session_logs.jsonl"


@pytest.fixture(autouse=True)
def _homes(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(acp_server, "VERSIONS", (1, 2))


_OPTION = {
    "configId": "effort",
    "name": "Effort",
    "category": "thought_level",
    "type": "select",
    "currentValue": "low",
    "options": [{"value": "low", "name": "low"}, {"value": "high", "name": "high"}],
}


def _events() -> list[dict[str, Any]]:
    """Return the recorded run's events, skipping the fixture's malformed lines."""
    rows = (line for line in _RECORDED.read_text(encoding="utf-8").splitlines() if line.strip())
    out: list[dict[str, Any]] = []
    for line in rows:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def _fits(update: model.Update) -> dict[str, Any]:
    """Return the update's v2 body, having checked it against the schema."""
    body = wire.v2(update)
    assert not schema_check.v2_errors("SessionUpdate", body), json.dumps(body)
    return body


def test_the_recorded_run_replays_as_valid_v2_updates(tmp_path: pathlib.Path) -> None:
    """The real journal, through the replay every `session/resume` uses."""
    sent: list[model.Update] = []
    updates.replay(
        _events(), session_id="run-x", cwd=tmp_path, objects_dir=tmp_path, notify=sent.append
    )
    kinds = {_fits(update)["sessionUpdate"] for update in sent}
    assert kinds == {
        "user_message",
        "agent_message",
        "agent_thought",
        "tool_call_update",
        "plan_update",
    }


@pytest.mark.parametrize(
    "item",
    [
        transcript_items.TranscriptItem("tool", name="run_verify", ok=False, detail="exit 1"),
        transcript_items.TranscriptItem("tool", name="grep", arg="x", ok=None),
        transcript_items.TranscriptItem("tool", name="grep", arg="x", ok=None, detail="waiting"),
        transcript_items.TranscriptItem("tool", name="read_file", arg="a.py", ok=True, tail="4"),
        transcript_items.TranscriptItem("done", ok=False, name="budget", detail="stopped"),
        transcript_items.TranscriptItem("done", ok=True, name="passed", body="all green"),
        transcript_items.TranscriptItem("commit", arg="abc1234", detail="+3 -1"),
        transcript_items.TranscriptItem("thinking", body="hmm"),
        transcript_items.TranscriptItem("operator", body="do the other thing"),
        transcript_items.TranscriptItem("task", body="the opening prompt"),
        transcript_items.TranscriptItem("text", body="the reply"),
        transcript_items.TranscriptItem("marker", body="a finding"),
    ],
)
def test_each_fold_item_renders_as_a_valid_v2_update(item: transcript_items.TranscriptItem) -> None:
    for streamed in (False, True):
        projected = updates.updates_for(
            item, cwd=pathlib.Path("/repo"), paths=("a.py",), streamed=streamed, message_id="m1"
        )
        assert projected, f"the {item.kind} item was dropped"
        for update in projected:
            _fits(update)


def test_the_title_commands_usage_plan_and_states_are_valid_v2_updates() -> None:
    info = updates.RunInfo(window=lambda _p, _m: 128_000)
    plan = updates.TaskPlan()
    sent = [update for event in _events() for update in (*info.feed(event), *plan.feed(event))]
    sent += [
        model.UsageUpdate(10, 100, None),
        model.UsageUpdate(10, 100, 0.25),
        model.ConfigOptionUpdate([_OPTION]),
        model.StateUpdate("running"),
        model.StateUpdate("requires_action"),
        model.StateUpdate("idle"),
        model.StateUpdate("idle", "end_turn"),
        model.StateUpdate("idle", "cancelled"),
        model.StateUpdate("idle", "max_turn_requests"),
        model.StateUpdate("idle", "error", rpc.RpcError(rpc.INTERNAL_ERROR, "the run stopped")),
        model.ReplayComplete(),
    ]
    kinds = {_fits(update)["sessionUpdate"] for update in sent}
    assert kinds >= {"session_info_update", "available_commands_update", "usage_update"}
    assert kinds >= {"plan_update", "state_update", "replay_complete", "config_option_update"}


def test_the_schema_check_refuses_a_named_tag_in_the_wrong_shape() -> None:
    """Unclosed, v2's open unions pass a select without its options as an unknown option type."""
    update = model.ConfigOptionUpdate([{"configId": "x", "name": "X", "type": "select"}])
    assert schema_check.v2_errors("SessionUpdate", wire.v2(update))
    misspelt = wire.v2(model.StateUpdate("idle", "end_turn")) | {"stopreason": "end_turn"}
    assert schema_check.v2_errors("SessionUpdate", misspelt)


def test_the_words_agent6_sends_are_ones_v2_names() -> None:
    """An open enum takes any string, so a misspelt status would pass the schema."""
    statuses = set(typing.get_args(model.ToolCallStatus))
    assert statuses == schema_check.named("ToolCallStatus")
    assert set(updates._TOOL_KINDS.values()) <= schema_check.named("ToolKind")
    assert set(updates._PLAN_STATUS.values()) | {"in_progress"} <= schema_check.named(
        "PlanEntryStatus"
    )
    assert set(runner._STANDING_KINDS.values()) | {"allow_once", "reject_once"} == (
        schema_check.named("PermissionOptionKind")
    )
    assert set(typing.get_args(model.State)) <= schema_check.tags("StateUpdate", "state")
    assert {"end_turn", "cancelled", "max_turn_requests", "error"} <= schema_check.tags(
        "IdleStateUpdate", "stopReason"
    )
    assert {"add", "delete", "modify"} <= schema_check.tags("DiffChange", "operation")


def test_a_usage_update_clears_a_cost_it_no_longer_has() -> None:
    """v2's cost is a patch: left out, an editor keeps showing the earlier total."""
    assert wire.v2(model.UsageUpdate(10, 100, None))["cost"] is None
    assert "cost" not in (wire.v1(model.UsageUpdate(10, 100, None)) or {})


def test_a_tool_call_is_created_by_its_first_update_and_never_returns_to_pending() -> None:
    waiting = transcript_items.TranscriptItem("tool", name="run_command", arg="ls", detail="wait")
    (created,) = updates.updates_for(waiting, wire_id="r:1:1")
    body = wire.v2(created)
    assert body["sessionUpdate"] == "tool_call_update"
    assert (body["title"], body["name"], body["kind"]) == (
        "run_command ls",
        "run_command",
        "execute",
    )
    assert body["status"] == "paused"
    (later,) = updates.updates_for(waiting, wire_id="r:1:1", announced=True)
    assert "title" not in wire.v2(later)
    stopped = transcript_items.TranscriptItem("tool", name="run_command", ok=False, detail="x")
    (cut,) = updates.updates_for(stopped, wire_id="r:1:1", announced=True, cancelled=True)
    assert wire.v2(cut)["status"] == "cancelled"
    assert (wire.v1(cut) or {})["status"] == "failed"
    assert (wire.v1(created) or {})["status"] == "pending"


def test_an_edit_is_a_diff_of_changes_with_a_patch_git_applies(tmp_path: pathlib.Path) -> None:
    """The patch is checked by Git itself: every path absolute, an add and a delete told apart."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "mod.txt").write_text("one\ntwo\nthree", encoding="utf-8")
    (repo / "gone.txt").write_text("bye\n", encoding="utf-8")
    changes = (
        model.DiffChange(str(repo / "mod.txt"), "one\ntwo\nthree", "one\n2\nthree\n"),
        model.DiffChange(str(repo / "new file.txt"), None, "hello\n"),
        model.DiffChange(str(repo / "gone.txt"), "bye\n", None),
        model.DiffChange(str(repo / "naïve\t.txt"), None, "x\n"),
    )
    body = _fits(model.ToolCallUpdate("r:1:1", "completed", text="edited", changes=changes))
    (diff,) = [entry for entry in body["content"] if entry["type"] == "diff"]
    assert [(c["operation"], c["path"]) for c in diff["changes"]] == [
        ("modify", str(repo / "mod.txt")),
        ("add", str(repo / "new file.txt")),
        ("delete", str(repo / "gone.txt")),
        ("add", str(repo / "naïve\t.txt")),
    ]
    assert diff["patch"]["format"] == "git_patch"
    applied = subprocess.run(
        ["git", "-C", str(repo), "apply", "-p0", "--unsafe-paths", "-"],
        input=diff["patch"]["text"].encode(),
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr.decode()
    assert (repo / "mod.txt").read_text(encoding="utf-8") == "one\n2\nthree\n"
    assert (repo / "new file.txt").read_text(encoding="utf-8") == "hello\n"
    assert (repo / "naïve\t.txt").read_text(encoding="utf-8") == "x\n"
    assert not (repo / "gone.txt").exists()


def _exchange(*messages: dict[str, Any], sessions: Any = None) -> list[dict[str, Any]]:
    """Feed the messages to one connection and return what came back."""
    payload = b"".join(json.dumps({"jsonrpc": "2.0", **m}).encode() + b"\n" for m in messages)
    out = io.BytesIO()
    acp_server.ACPServer(stdin=io.BytesIO(payload), stdout=out, sessions=sessions).serve()
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _initialize(version: int = 2) -> dict[str, Any]:
    info = {"name": "editor", "version": "1"}
    params: dict[str, Any] = {"protocolVersion": version, "info": info, "capabilities": {}}
    return {"id": 1, "method": "initialize", "params": params}


@pytest.mark.parametrize(("requested", "selected"), [(1, 1), (2, 2), (3, 2), (65_535, 2)])
def test_initialize_selects_the_highest_version_at_most_the_one_requested(
    requested: int, selected: int
) -> None:
    (reply,) = _exchange(_initialize(requested))
    assert reply["result"]["protocolVersion"] == selected
    if selected == 1:
        assert not schema_check.errors("InitializeResponse", reply["result"])
        assert "agentCapabilities" in reply["result"]
    else:
        assert not schema_check.v2_errors("InitializeResponse", reply["result"])
        assert reply["result"]["capabilities"] == {"session": {"prompt": {"image": {}}}}


def test_a_version_below_every_served_one_is_refused_with_the_served_list() -> None:
    (reply,) = _exchange(_initialize(0))
    assert reply["error"]["code"] == -32022
    assert reply["error"]["data"] == {"requested": 0, "supported": [1, 2]}
    assert not schema_check.v2_errors("Error", reply["error"])
    assert not schema_check.v2_errors("UnsupportedProtocolVersionErrorData", reply["error"]["data"])


def test_a_v2_connection_refuses_what_v2_forbids() -> None:
    """A null id, an `initialize` after another request, and v1's `session/load`."""
    replies = _exchange(
        _initialize(),
        {"id": None, "method": "session/list", "params": {}},
        {"id": 2, "method": "session/load", "params": {}},
        _initialize() | {"id": 3},
    )
    assert [r.get("error", {}).get("code") for r in replies] == [None, -32600, -32601, -32600]
    assert "null" in replies[1]["error"]["message"]
    assert "repeated" in replies[3]["error"]["message"]


def test_initialize_can_repeat_before_any_other_request() -> None:
    replies = _exchange(_initialize(3), _initialize(1) | {"id": 2})
    assert [r["result"]["protocolVersion"] for r in replies] == [2, 1]


def _sessions() -> acp_session.Sessions:
    return acp_session.Sessions(run=lambda _s, _t: "end_turn", state_dir_for=paths.state_dir)


def test_a_v2_session_setup_carries_the_commands_and_valid_options(tmp_path: pathlib.Path) -> None:
    cwd = test_session_lifecycle._repo(tmp_path)
    sessions = _sessions()
    sessions.build_options = lambda _session: [_OPTION]
    new = {"id": 2, "method": "session/new", "params": {"cwd": str(cwd)}}
    _, created = _exchange(_initialize(), new, sessions=sessions)
    assert not schema_check.v2_errors("NewSessionResponse", created["result"])
    names = {command["name"] for command in created["result"]["availableCommands"]}
    assert {"pin", "review", "receipt"} <= names
    assert all(command["type"] == "command" for command in created["result"]["availableCommands"])
    assert created["result"]["configOptions"][0]["configId"] == "effort"
    pick = {"sessionId": created["result"]["sessionId"], "configId": "effort", "value": "high"}
    replies = _exchange(
        _initialize(),
        {"id": 2, "method": "session/set_config_option", "params": pick},
        {"id": 3, "method": "session/set_config_option", "params": pick | {"type": "boolean"}},
        {"id": 4, "method": "session/set_config_option", "params": pick | {"type": "id"}},
        sessions=sessions,
    )
    assert [r.get("error", {}).get("code") for r in replies[1:]] == [-32602, -32602, None]
    assert not schema_check.v2_errors("SetSessionConfigOptionResponse", replies[3]["result"])


def test_a_v1_session_setup_keeps_its_own_shape(tmp_path: pathlib.Path) -> None:
    cwd = test_session_lifecycle._repo(tmp_path)
    new = {"id": 2, "method": "session/new", "params": {"cwd": str(cwd)}}
    _, created = _exchange(_initialize(1), new, sessions=_sessions())
    assert set(created["result"]) == {"sessionId", "configOptions"}


def test_an_unknown_session_is_a_missing_resource_under_v2(tmp_path: pathlib.Path) -> None:
    cwd = test_session_lifecycle._repo(tmp_path)
    unknown = {"sessionId": "no-such-AAAAAA"}
    prompt = unknown | {"prompt": [{"type": "text", "text": "hi"}]}
    replies = _exchange(
        _initialize(),
        {"id": 2, "method": "session/prompt", "params": prompt},
        {"id": 3, "method": "session/resume", "params": unknown | {"cwd": str(cwd)}},
        {"id": 4, "method": "session/close", "params": unknown},
        {"method": "session/cancel", "params": unknown},
        sessions=_sessions(),
    )
    assert [r.get("error", {}).get("code") for r in replies[1:]] == [-32002, -32002, None]
    for reply in replies[1:3]:
        assert reply["error"]["data"] == {"sessionId": "no-such-AAAAAA"}
        assert not schema_check.v2_errors("SessionNotFoundErrorData", reply["error"]["data"])
    assert replies[3] == {"jsonrpc": "2.0", "id": 4, "result": {}}
    v1 = _exchange(
        _initialize(1),
        {"id": 2, "method": "session/prompt", "params": prompt},
        sessions=_sessions(),
    )
    assert v1[1]["error"] == {"code": -32602, "message": "no session 'no-such-AAAAAA'"}


def test_a_v2_prompt_refuses_a_block_type_it_did_not_offer(tmp_path: pathlib.Path) -> None:
    cwd = test_session_lifecycle._repo(tmp_path)
    sessions = _sessions()
    opened = sessions.new({"cwd": str(cwd)})["sessionId"]
    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "audio", "data": "", "mimeType": "audio/wav"},
    ]
    prompt = {
        "id": 2,
        "method": "session/prompt",
        "params": {"sessionId": opened, "prompt": blocks},
    }
    _, refused = _exchange(_initialize(), prompt, sessions=sessions)
    assert refused["error"]["code"] == -32602
    assert "'audio'" in refused["error"]["message"]


def _recorded(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    cwd = test_session_lifecycle._repo(tmp_path)
    saved = test_session_lifecycle._record(cwd, "replay-run-AAAAAA", "Fix it", mtime=time.time())
    (saved / "logs.jsonl").write_bytes(_RECORDED.read_bytes())
    return cwd, saved.name


def test_a_v2_resume_replays_then_marks_the_end_then_reports_idle(tmp_path: pathlib.Path) -> None:
    cwd, run = _recorded(tmp_path)
    params = {"sessionId": run, "cwd": str(cwd), "replayFrom": {"type": "start"}}
    frames = _exchange(
        _initialize(), {"id": 2, "method": "session/resume", "params": params}, sessions=_sessions()
    )
    bodies = [frame["params"]["update"] for frame in frames[1:-1]]
    for frame in frames[1:-1]:
        assert not schema_check.v2_errors("UpdateSessionNotification", frame["params"]), frame
    assert [body["sessionUpdate"] for body in bodies[-2:]] == ["replay_complete", "state_update"]
    assert bodies[-1] == {"sessionUpdate": "state_update", "state": "idle"}
    assert bodies[0]["sessionUpdate"] == "user_message"
    assert bodies[0]["content"] == [{"type": "text", "text": "# Fix the bug\n\nRestore the thing."}]
    assert not schema_check.v2_errors("ResumeSessionResponse", frames[-1]["result"])
    assert frames[-1]["result"]["availableCommands"]


def test_a_v2_resume_without_a_cursor_reports_its_state_alone(tmp_path: pathlib.Path) -> None:
    cwd, run = _recorded(tmp_path)
    resume = {"id": 2, "method": "session/resume", "params": {"sessionId": run, "cwd": str(cwd)}}
    frames = _exchange(_initialize(), resume, sessions=_sessions())
    assert [frame["params"]["update"] for frame in frames[1:-1]] == [
        {"sessionUpdate": "state_update", "state": "idle"}
    ]
    cursor = resume["params"] | {"replayFrom": {"type": "message", "messageId": "m"}}
    refused = _exchange(
        _initialize(), {"id": 2, "method": "session/resume", "params": cursor}, sessions=_sessions()
    )
    assert [frame.get("error", {}).get("code") for frame in refused] == [None, -32602]


def test_a_replayed_message_keeps_its_id_and_the_models_words_share_their_calls() -> None:
    """A replay names a message as the live run did: by the row it came from."""
    rows = [
        {"type": "session.start", "event_id": "e1", "user_task": "Fix it", "mode": "run"},
        {"type": "model.call", "event_id": "e2", "actor": "worker"},
        {"type": "model.thinking_delta", "text": "hm"},
        {"type": "model.text_delta", "text": "done"},
        {"type": "model.result", "event_id": "e3", "actor": "worker", "text": "done"},
        {"type": "steer.injected", "event_id": "e4", "text": "also this", "chars": 9},
        {"type": "session.stop", "event_id": "e5", "reason": "finish_session"},
    ]
    sent: list[model.Update] = []
    updates.replay(
        rows,
        session_id="r",
        cwd=pathlib.Path("/r"),
        objects_dir=pathlib.Path("/o"),
        notify=sent.append,
    )
    ids = [(u.role, u.message_id) for u in sent if isinstance(u, model.Message)]
    assert ids[:4] == [("user", "e1"), ("thought", "e2"), ("agent", "e2"), ("user", "e4")]
    live = updates.MessageIds()
    for row in rows[:2]:
        live.feed(row)
    assert live.call == "e2", "the deltas a live run streams take the id the replay gives"


def test_a_v2_permission_request_carries_the_prompt_as_its_title() -> None:
    """The gated call is the subject by id alone; a prompt gating none has no subject."""
    request = model.PermissionRequest(
        "Allow run_command: ls",
        "run-x:1:7",
        (
            model.PermissionOption("allow", "allow_once"),
            model.PermissionOption("deny", "reject_once"),
        ),
    )
    gated = wire.permission_request(2, "s", request, request.tool_call_id)
    assert not schema_check.v2_errors("RequestPermissionRequest", gated), gated
    assert gated["title"] == "Allow run_command: ls"
    assert gated["subject"] == {"type": "tool_call", "toolCall": {"toolCallId": "run-x:1:7"}}
    bare = wire.permission_request(
        2, "s", model.PermissionRequest("Theme?", "", request.options), ""
    )
    assert not schema_check.v2_errors("RequestPermissionRequest", bare), bare
    assert "subject" not in bare


def test_an_abandoned_v2_permission_request_is_cancelled_by_id() -> None:
    """The editor dismisses a prompt nobody waits on; v1 has no such notification."""
    for version, expected in ((2, ["session/request_permission", "$/cancel_request"]), (1, None)):
        out = io.BytesIO()
        server = acp_server.ACPServer(stdin=io.BytesIO(), stdout=out, protocol_version=version)
        request = model.PermissionRequest("Go?", "", (model.PermissionOption("yes", "allow_once"),))
        assert server.request_permission("s", request, timeout_s=0.3, until=lambda: False) is None
        frames = [json.loads(line) for line in out.getvalue().splitlines()]
        if expected is None:
            assert [f["method"] for f in frames] == ["session/request_permission", "session/update"]
            continue
        assert [f["method"] for f in frames] == expected
        assert frames[1]["params"] == {"requestId": frames[0]["id"]}
        assert not schema_check.v2_errors("CancelRequestNotification", frames[1]["params"])
