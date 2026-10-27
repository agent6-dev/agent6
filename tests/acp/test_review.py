from __future__ import annotations

import io
import json
import os
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from agent6 import event_log, paths, review_records
from agent6.app import review_actions
from agent6.sessions import process
from agent6.ui.acp import runner, server, session, updates
from agent6.viewmodel import transcript
from tests import review_helpers


def _connection(
    cwd: pathlib.Path, run: Callable[[session.Session, str], str]
) -> tuple[server.ACPServer, session.Session, io.BytesIO]:
    saved = review_helpers.reviewed_session(cwd)
    current = session.Session(acp_id="editor", cwd=cwd, session_id=saved.session_id)
    sessions = session.Sessions(run=run, state_dir_for=paths.state_dir)
    sessions._by_id[current.acp_id] = current  # pyright: ignore[reportPrivateUsage]
    out = io.BytesIO()
    return server.ACPServer(io.BytesIO(), out, sessions=sessions), current, out


def _prompt(connection: server.ACPServer, current: session.Session, command: str) -> None:
    connection._handle(  # pyright: ignore[reportPrivateUsage]
        json.dumps(
            dict(
                jsonrpc="2.0",
                id=1,
                method="session/prompt",
                params=dict(sessionId=current.acp_id, prompt=[dict(type="text", text=command)]),
            )
        ).encode()
    )
    if current.thread is not None:
        current.thread.join(timeout=5)
        assert not current.thread.is_alive()


def _messages(out: io.BytesIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _notices(out: io.BytesIO) -> str:
    return "".join(
        frame.get("params", {}).get("update", {}).get("content", {}).get("text", "")
        for frame in _messages(out)
    )


def test_review_commands_are_advertised() -> None:
    commands = next(
        update["params"]["update"]["availableCommands"]
        for update in updates.RunInfo("editor").feed(dict(type="session.start", user_task="task"))
        if update["params"]["update"]["sessionUpdate"] == "available_commands_update"
    )
    assert {c["name"] for c in commands} >= {"findings", "fix", "resolve", "reopen"}
    fix = next(c for c in commands if c["name"] == "fix")
    assert "idle" in fix["description"] and "--resume" in fix["input"]["hint"]


def test_findings_resolve_and_reopen_never_reach_the_model(tmp_path: pathlib.Path) -> None:
    ran: list[str] = []

    def run(_current: session.Session, text: str) -> str:
        ran.append(text)
        return "end_turn"

    connection, current, out = _connection(tmp_path, run)
    saved = current.layout(paths.state_dir(tmp_path))
    _prompt(connection, current, "/findings")
    assert not ran
    assert "f1" in _notices(out) and "Defect f2" in _notices(out)
    assert "[agent6]" in _notices(out)
    out.seek(0)
    out.truncate()
    _prompt(connection, current, '/resolve f1 --as fixed --reason "Reproduced and fixed"')
    view = review_actions.read_review(saved.session_dir)
    assert view.findings[0].status == "resolved"
    assert view.findings[0].actions[-1].source == "acp"
    assert _notices(out).count("resolved") == 1
    assert "Reproduced and fixed" in _notices(out)
    _prompt(connection, current, '/reopen f1 --reason "Still reproduces"')
    finding = review_actions.read_review(saved.session_dir).findings[0]
    assert finding.status == "open" and finding.actions[-1].source == "acp"
    assert _notices(out).count("reopened") == 1
    assert not ran


@pytest.mark.parametrize(
    "command",
    [
        "/findings surprise",
        "/fix",
        "/resolve f1 --as fixed --reason ''",
        "/resolve f1 --as duplicate --reason same --duplicate-of missing",
        "/reopen f1",
    ],
)
def test_invalid_review_commands_are_not_model_prompts(
    tmp_path: pathlib.Path, command: str
) -> None:
    ran: list[str] = []

    def run(_current: session.Session, text: str) -> str:
        ran.append(text)
        return "end_turn"

    connection, current, out = _connection(tmp_path, run)
    _prompt(connection, current, command)
    assert not ran
    assert any(frame.get("error") for frame in _messages(out))
    assert (
        review_actions.read_review(current.layout(paths.state_dir(tmp_path)).session_dir).version
        == 1
    )


@pytest.mark.parametrize("state", ["idle", "busy", "no_snapshot", "unconfirmed"])
def test_fix_is_only_a_confirmed_idle_followup(tmp_path: pathlib.Path, state: str) -> None:
    ran: list[str] = []
    constraints: list[bool] = []

    def run(current: session.Session, text: str) -> str:
        ran.append(text)
        constraints.append(getattr(current, "require_resume", False))
        review_actions.drain(
            event_log.EventSink(current.layout(paths.state_dir(tmp_path)).logs_path),
            execution_id="followup",
            resume_steer=text,
        )
        return "end_turn"

    connection, current, out = _connection(tmp_path, run)
    saved = current.layout(paths.state_dir(tmp_path))
    if state != "no_snapshot":
        (saved.session_dir / "loop_state.json").write_text("{}")
    current.execution_live = state == "busy"
    command = "/fix f1" if state == "unconfirmed" else "/fix f1 --resume"
    _prompt(connection, current, command)
    if state == "idle":
        assert len(ran) == 1 and ran[0].startswith("Review finding f1\n")
        assert "Reproduce this report" in ran[0] and "Defect f1" in ran[0]
        assert constraints == [True]
        finding = review_actions.read_review(saved.session_dir).findings[0]
        assert finding.status == "open" and finding.fix_state == "submitted"
        assert finding.actions[0].source == "acp"
    else:
        assert not ran
        assert any(frame.get("error") for frame in _messages(out))
        assert review_actions.read_review(saved.session_dir).version == 1


def test_acp_fix_cannot_become_a_live_steer(tmp_path: pathlib.Path) -> None:
    """The action boundary also refuses an ACP Fix if another worker becomes live."""
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())
    try:
        with pytest.raises(review_actions.ReviewActionError, match="idle"):
            review_actions.act(
                saved.session_dir,
                "f1",
                review_records.FindingAction(
                    request_id="fix", expected_version=1, action="fix", resume=True
                ),
                source="acp",
                wait_s=0,
            )
    finally:
        process.clear_worker_pid(saved.session_dir)


def test_a_disappearing_resume_point_never_starts_new_work(tmp_path: pathlib.Path) -> None:
    """A Fix keeps its resume-only constraint through the prompt worker's dispatch."""
    ran: list[str] = []
    connection: server.ACPServer

    def run(current: session.Session, text: str) -> str:
        ran.append(text)
        (current.layout(paths.state_dir(tmp_path)).session_dir / "loop_state.json").unlink()
        return runner.RunBridge(connection).run(current, text)

    connection, current, out = _connection(tmp_path, run)
    (current.layout(paths.state_dir(tmp_path)).session_dir / "loop_state.json").write_text("{}")
    _prompt(connection, current, "/fix f1 --resume")
    assert current.session_id == "review-run"
    assert "no resume point" in _notices(out)
    assert "starts a new run" not in _notices(out)
    assert any(frame.get("result", {}).get("stopReason") == "refusal" for frame in _messages(out))


@pytest.mark.parametrize(
    "disposition",
    ["fixed", "false_positive", "intentional", "wont_fix", "duplicate", "pre_existing"],
)
def test_shared_transcript_emits_findings_and_each_disposition(
    tmp_path: pathlib.Path, disposition: str
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    review_actions.act(
        saved.session_dir,
        "f1",
        review_records.FindingAction(
            request_id="resolve",
            expected_version=1,
            action="resolve",
            resolution=review_records.FindingResolution.model_validate(
                dict(
                    disposition=disposition,
                    reason="Operator conclusion",
                    duplicate_of="f2" if disposition == "duplicate" else None,
                )
            ),
        ),
        source="cli",
    )
    review_actions.act(
        saved.session_dir,
        "f1",
        review_records.FindingAction(
            request_id="reopen",
            expected_version=2,
            action="reopen",
            reason="Still in scope",
        ),
        source="web",
    )
    fold = transcript.TranscriptFold()
    items = [
        item
        for line in saved.logs_path.read_text().splitlines()
        for item in fold.feed(json.loads(line))
    ]
    chunks = [
        frame["params"]["update"]["content"]["text"]
        for item in items
        for frame in updates.updates_for(item, acp_session_id="editor")
    ]
    assert chunks and all(chunk.startswith("[agent6]") for chunk in chunks)
    prose = "".join(chunks)
    assert "Defect f1" in prose and "Defect f2" in prose and "parser.py:3-4" in prose
    assert prose.count("resolved") == 1 and disposition in prose
    assert prose.count("reopened") == 1 and "Still in scope" in prose
