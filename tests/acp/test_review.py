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
from agent6.sessions import ipc, process
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


def test_review_queue_lists_the_repository_without_a_recorded_run(tmp_path: pathlib.Path) -> None:
    """`/review-queue` answers for the repository, grouped, before any run exists in the thread."""
    from tests.unit import test_review_queue as queue_tests

    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    (runs / "run-block").mkdir(parents=True)
    (runs / "run-block" / "logs.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in (queue_tests._START, *queue_tests._panel_rows(block=1), queue_tests._END)
        )
    )
    ran: list[str] = []

    def run(_current: session.Session, text: str) -> str:
        ran.append(text)
        return "end_turn"

    current = session.Session(acp_id="editor", cwd=tmp_path, session_id="")
    sessions = session.Sessions(run=run, state_dir_for=paths.state_dir)
    sessions._by_id[current.acp_id] = current  # pyright: ignore[reportPrivateUsage]
    out = io.BytesIO()
    connection = server.ACPServer(io.BytesIO(), out, sessions=sessions)
    _prompt(connection, current, "/review-queue")
    text = _notices(out)
    assert "Blocked:" in text and "run-block" in text and "1 open block finding" in text
    assert "agent6 sessions findings run-block" in text
    assert ran == []
    commands = next(
        update["params"]["update"]["availableCommands"]
        for update in updates.RunInfo("editor").feed(dict(type="session.start", user_task="task"))
        if update["params"]["update"]["sessionUpdate"] == "available_commands_update"
    )
    assert "review-queue" in {c["name"] for c in commands}


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


@pytest.mark.parametrize("kind", ["pre_existing", "changed", "unknown"])
def test_findings_show_only_pre_existing_line_origin(tmp_path: pathlib.Path, kind: str) -> None:
    """Only pre-existing lines show their base range, reason and blame in findings text."""
    connection, current, out = _connection(tmp_path, lambda _current, _text: "end_turn")
    saved = current.layout(paths.state_dir(tmp_path))
    panel = json.loads(saved.logs_path.read_text())
    origin: dict[str, Any] = dict(kind=kind, reason="The cited bytes match the reviewed base.")
    if kind == "pre_existing":
        origin.update(
            base_location=dict(path="old_parser.py", side="base", start_line=7, end_line=8),
            commits=["a" * 40, "b" * 40],
        )
    panel["findings"][0]["line_origin"] = origin
    saved.logs_path.write_text(json.dumps(panel) + "\n")

    _prompt(connection, current, "/findings")
    prose = _notices(out)
    assert "Defect f1" in prose
    if kind == "pre_existing":
        assert (
            "  Pre-existing line: old_parser.py:7-8 (base). "
            "The cited bytes match the reviewed base.\n"
            f"  Blame: {'a' * 40}, {'b' * 40}"
        ) in prose
        assert prose.count("Pre-existing line:") == 1
    else:
        assert "Pre-existing line" not in prose
        assert "Blame:" not in prose
        assert "The cited bytes match the reviewed base." not in prose


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


def test_receipt_command_never_reaches_the_worker(tmp_path: pathlib.Path) -> None:
    """Receipt inspection is advertised and stays outside the model conversation."""
    ran: list[str] = []
    connection, current, out = _connection(
        tmp_path, lambda _s, text: ran.append(text) or "end_turn"
    )
    _prompt(connection, current, "/receipt")
    assert not ran
    assert "Testing" in _notices(out) and "How to revert" in _notices(out)
    assert any(c["name"] == "receipt" for c in updates.REVIEW_COMMANDS)


def test_answer_writes_the_pending_questions_file_and_never_reaches_the_model(
    tmp_path: pathlib.Path,
) -> None:
    """`/answer` is a front-end action: it writes the answer file the run reads at its next turn."""
    ran: list[str] = []

    def run(_current: session.Session, text: str) -> str:
        ran.append(text)
        return "end_turn"

    connection, current, out = _connection(tmp_path, run)
    saved = current.layout(paths.state_dir(tmp_path))
    with saved.logs_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "session.start", "mode": "run", "user_task": "task"}) + "\n")
        fh.write(
            json.dumps(
                {
                    "type": "question.prompt",
                    "id": "question-7",
                    "questions": [{"question": "Which DB?", "options": ["pg", "sqlite"]}],
                    "wait": False,
                }
            )
            + "\n"
        )
    commands = next(
        update["params"]["update"]["availableCommands"]
        for update in updates.RunInfo("editor").feed(dict(type="session.start", user_task="task"))
        if update["params"]["update"]["sessionUpdate"] == "available_commands_update"
    )
    assert "answer" in {c["name"] for c in commands}
    _prompt(connection, current, "/answer question-7 pg")
    assert not ran
    assert "answered question-7: pg" in _notices(out)
    assert "next prompt resumes the run" in _notices(out)
    assert ipc.read_question_answers(saved.session_dir, "question-7", timeout_s=1.0) == ("pg",)
    out.seek(0)
    out.truncate()
    _prompt(connection, current, "/answer question-9 x")
    errors = [m for m in _messages(out) if "error" in m]
    assert errors and "no pending question 'question-9'" in errors[0]["error"]["message"]
    assert not ran
