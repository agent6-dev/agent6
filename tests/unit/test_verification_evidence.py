from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Literal
from unittest import mock

import pytest

from agent6 import event_log, kinds
from agent6.config import Config
from agent6.harness import _chain, _loop_resume, _loop_state, _snapshot, loop
from agent6.tools import _command_tools, _result_format, dispatch, errors, results
from agent6.vcs import git_read
from tests.unit import test_verify_finish_gate as gate_tests


def _dispatcher(
    root: pathlib.Path, journal: pathlib.Path, policy: str = "allow"
) -> dispatch.ToolDispatcher:
    config = Config.model_validate(
        {
            "harness": {"verify_command": ["true"]},
            "sandbox": {"run_commands": policy, "network": "host"},
        }
    )
    return dispatch.ToolDispatcher(root=root, config=config, events=event_log.EventSink(journal))


def _events(journal: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in journal.read_text().splitlines()]


@pytest.mark.parametrize(
    "rc,outcome", [(0, "passed"), (1, "failed"), (124, "timed_out"), (-2, "interrupted")]
)
@pytest.mark.parametrize("scoped", [False, True])
def test_attempt_records_tree_id_scope_and_output_caps(
    tmp_path: pathlib.Path, rc: int, outcome: str, scoped: bool
) -> None:
    """Harness calls carry evidence without a model tool call id."""
    repo = tmp_path / "repo"
    repo.mkdir()
    head = gate_tests._git_seed(repo)
    journal = tmp_path / "logs.jsonl"
    dispatcher = _dispatcher(repo, journal)
    raw = kinds.CommandResult(
        argv=("true",),
        returncode=rc,
        stdout="x" * 21000,
        stderr="error",
        duration_s=1.0,
        exec_failed=False,
    )
    with mock.patch("agent6.sandbox.jail.run_in_jail", return_value=raw):
        result = dispatcher.run_verify(extra_argv=("tests/test_x.py",) if scoped else ())
    start, end = _events(journal)
    assert end.get("verify_id"), end
    assert start["verify_id"] == end["verify_id"]
    assert end["tree_before"] == end["tree_after"] == git_read.tree_sha(repo, head)
    assert end["outcome"] == outcome
    assert end["scoped"] is scoped
    assert end["selected_paths"] == (["tests/test_x.py"] if scoped else [])
    assert len(end["stdout_tail"]) == 2000
    assert end["stdout_truncated"] is True and end["stderr_truncated"] is False
    assert result.to_wire()["verification"]["verify_id"] == end["verify_id"]
    assert result.to_wire()["stdout_truncated"] is True


@pytest.mark.parametrize("policy", ["deny", "ask"])
def test_denial_is_an_attempt_without_a_verdict(tmp_path: pathlib.Path, policy: str) -> None:
    """Both denial paths record missing verification rather than empty success."""
    journal = tmp_path / "logs.jsonl"
    dispatcher = _dispatcher(tmp_path, journal, policy)
    with (
        mock.patch.object(dispatcher, "_approve", return_value=False),
        pytest.raises(errors.ToolDeniedError),
    ):
        dispatcher.run_verify()
    end = _events(journal)[-1]
    assert end["outcome"] == "denied" and end["exit_code"] is None


@pytest.mark.parametrize(
    "exception,outcome",
    [(errors.ToolError("jail unavailable"), "unavailable"), (KeyboardInterrupt(), "interrupted")],
)
def test_exception_keeps_an_attempt(
    tmp_path: pathlib.Path, exception: BaseException, outcome: str
) -> None:
    """An interrupted or unavailable command leaves an evidence record."""
    journal = tmp_path / "logs.jsonl"
    dispatcher = _dispatcher(tmp_path, journal)
    with (
        mock.patch.object(_command_tools, "run_argv_in_jail", side_effect=exception),
        pytest.raises(type(exception)),
    ):
        dispatcher.run_verify()
    end = _events(journal)[-1]
    assert end["outcome"] == outcome
    assert end["stdout_truncated"] is None and end["stderr_truncated"] is None


def test_refusal_truncation_marks_the_reason_not_the_streams(tmp_path: pathlib.Path) -> None:
    """A refusal text past 2000 characters marked stderr truncated; no stream was captured."""
    journal = tmp_path / "logs.jsonl"
    dispatcher = _dispatcher(tmp_path, journal)
    with (
        mock.patch.object(
            dispatcher, "_approve_command", side_effect=errors.ToolDeniedError("x" * 2001)
        ),
        pytest.raises(errors.ToolDeniedError),
    ):
        dispatcher.run_verify()
    end = _events(journal)[-1]
    assert end["outcome"] == "denied"
    assert end["reason"] == "x" * 2000 + " (truncated)"
    assert end["stdout_truncated"] is False and end["stderr_truncated"] is False


def test_a_verifying_command_cannot_certify_its_own_edit(tmp_path: pathlib.Path) -> None:
    """Exit zero after writing the tree leaves the final tree unverified."""
    repo = tmp_path / "repo"
    repo.mkdir()
    gate_tests._git_seed(repo)
    dispatcher = _dispatcher(repo, tmp_path / "logs.jsonl")
    harness = loop.Harness(
        chain=_chain.RunChain(repo),
        config=Config.model_validate({"harness": {"verify_command": ["true"]}}),
        dispatcher=dispatcher,
        provider=mock.MagicMock(),
        logger=lambda _: None,
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)

    def edit(*args: Any, **kwargs: Any) -> results.ExecResult:
        (repo / "changed.txt").write_text("changed")
        return results.ExecResult(
            returncode=0, stdout="pass", stderr="", duration_s=1.0, exec_failed=False
        )

    with mock.patch.object(_command_tools, "run_argv_in_jail", side_effect=edit):
        result = dispatcher.run_verify()
    turn = gate_tests._turn()
    harness.gate.note_result(state, turn, result)
    notices = str(turn.tool_results)
    assert "changed the tree" in notices and "does not certify" in notices
    assert harness.gate.verification(state.verify) == "unverified"
    assert not state.verify.judged_and_untouched


def test_result_format_marks_both_stream_caps() -> None:
    """Result clipping reports completeness separately for stdout and stderr."""
    raw = kinds.CommandResult(
        argv=("true",),
        returncode=0,
        stdout="x" * 20001,
        stderr="y" * 20000,
        duration_s=0.1,
        exec_failed=False,
    )
    wire = _result_format.exec_result(raw).to_wire()
    assert wire.get("stdout_truncated") is True
    assert wire.get("stderr_truncated") is False


def test_launcher_early_capture_is_not_reported_complete() -> None:
    """An early-ended pipe capture cannot establish completeness for either stream."""
    raw = kinds.CommandResult(
        argv=("true",),
        returncode=0,
        stdout="partial",
        stderr="[agent6-jail] output capture ended early: an open pipe",
        duration_s=0.1,
        exec_failed=False,
    )
    wire = _result_format.exec_result(raw).to_wire()
    assert wire.get("stdout_truncated") is None and wire.get("stderr_truncated") is None


def test_resume_uses_content_identity_and_keeps_final_id(tmp_path: pathlib.Path) -> None:
    """A different commit with identical content preserves the observed tree and verify id."""
    head = gate_tests._git_seed(tmp_path)
    harness = gate_tests._wf(verify=True, root=tmp_path)
    tree = git_read.tree_sha(tmp_path, head)
    snap = gate_tests._snap(
        head_sha="different commit",
        last_verify_ok=True,
        verify_tree=tree,
        final_verify_id="verify-1",
    )
    state = gate_tests._resumed_state(harness, snap)
    assert state.verify.last_ok is True
    assert state.verify.final_verify_id == "verify-1"
    (tmp_path / "changed").write_text("different")
    changed = gate_tests._resumed_state(harness, snap)
    assert changed.verify.last_ok is None


def test_snapshot_and_end_carry_evidence_id(tmp_path: pathlib.Path) -> None:
    """The final verification id survives the journal and resume snapshot."""
    gate_tests._git_seed(tmp_path)
    harness = gate_tests._wf(verify=True, root=tmp_path)
    harness.resume_state_path = tmp_path / "snapshot.json"
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.final_verify_id = "verify-1"
    state.verify.verify_tree = harness.chain.tree_sha()
    state.verify.note_pass()
    _loop_resume.save_resume_snapshot(harness, state, [], next_turn=1)
    snap = _snapshot.load_session_snapshot(harness.resume_state_path)
    assert snap.final_verify_id == "verify-1"
    with mock.patch.object(loop.Harness, "emit") as emit:
        result = harness.finish(
            state, _snapshot.End("finish_session", "done", checkpoint=False), turn=1
        )
    end = next(c.kwargs for c in emit.call_args_list if c.args == ("session.end",))
    assert end["final_verify_id"] == "verify-1"
    assert dataclasses.asdict(result)["final_verify_id"] == "verify-1"


@pytest.mark.parametrize("denied", [True, False])
def test_unsuccessful_retry_replaces_an_earlier_final_pass(
    tmp_path: pathlib.Path, denied: bool
) -> None:
    """A raised attempt replaces the final id and cannot leave an earlier pass current."""
    repo = tmp_path / "repo"
    repo.mkdir()
    gate_tests._git_seed(repo)
    journal = tmp_path / "logs.jsonl"
    dispatcher = _dispatcher(repo, journal, "ask")
    harness = loop.Harness(
        chain=_chain.RunChain(repo),
        config=Config.model_validate({"harness": {"verify_command": ["true"]}}),
        dispatcher=dispatcher,
        provider=mock.MagicMock(),
        logger=lambda _: None,
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    result = results.ExecResult(
        returncode=0, stdout="pass", stderr="", duration_s=1, exec_failed=False
    )
    with (
        mock.patch.object(dispatcher, "_approve", return_value=True),
        mock.patch.object(_command_tools, "run_argv_in_jail", return_value=result),
    ):
        result = dispatcher.run_verify()
    harness.gate.note_result(state, gate_tests._turn(), result)
    error = errors.ToolDeniedError if denied else errors.ToolError
    with (
        mock.patch.object(dispatcher, "_approve", return_value=not denied),
        mock.patch.object(
            _command_tools, "run_argv_in_jail", side_effect=errors.ToolError("jail unavailable")
        ),
        pytest.raises(error),
    ):
        dispatcher.run_verify()
    harness.resume_state_path = tmp_path / "snapshot.json"
    _loop_resume.save_resume_snapshot(harness, state, [], next_turn=1)
    snap = _snapshot.load_session_snapshot(harness.resume_state_path)
    end = harness.finish(state, _snapshot.End("finish_session", "done", checkpoint=False), turn=1)
    assert end.verified == "unverified"
    latest = _events(journal)[-1]
    assert end.final_verify_id == snap.final_verify_id == latest["verify_id"]


def denied_rows(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    """Record a denied verification through the real dispatcher."""
    journal = tmp_path / "logs.jsonl"
    with pytest.raises(errors.ToolDeniedError):
        _dispatcher(tmp_path, journal, "deny").dispatch("run_verify_command", {})
    return _events(journal)


def test_denied_attempt_uses_cmd_and_keeps_reason_out_of_stderr(tmp_path: pathlib.Path) -> None:
    """A refusal has one command field and a reason, not fabricated command output."""
    rows = denied_rows(tmp_path)
    for row in rows:
        if row["type"] in ("verify.start", "verify.end"):
            assert row["cmd"] == ["true"] and "command" not in row
    end = next(row for row in rows if row["type"] == "verify.end")
    assert "run_commands" in end["reason"]
    assert end["stderr_tail"] == ""


def test_denied_attempt_is_finished_in_state_and_wire(tmp_path: pathlib.Path) -> None:
    """A null exit code on a denial denotes a finished attempt, not an in-flight gate."""
    from agent6.viewmodel import state, wire

    folded = state.fold_session(denied_rows(tmp_path))
    assert folded.last_verify is not None
    assert getattr(folded.last_verify, "outcome", None) == "denied"
    payload = wire.session_snapshot(tmp_path)
    assert payload["last_verify"]["outcome"] == "denied"
    assert "run_commands" in payload["last_verify"]["reason"]


def test_denied_attempt_transcript_names_outcome(tmp_path: pathlib.Path) -> None:
    """A denied tool's badge names denial rather than exit None."""
    from agent6.viewmodel import transcript

    items = transcript.fold_transcript(denied_rows(tmp_path))
    item = next(item for item in items if item.name == "run_verify_command")
    assert "denied" in item.detail and "None" not in item.detail


@pytest.mark.parametrize("mode", ["ask", "plan", "agent"])
def test_non_run_end_omits_empty_tree(mode: Literal["ask", "plan", "agent"]) -> None:
    """Only run-mode ends carry an observed tree field."""
    harness = gate_tests._wf(verify=False, mode=mode)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    with mock.patch.object(loop.Harness, "emit") as emit:
        harness.finish(state, _snapshot.End("finish_session", "done", checkpoint=False), turn=1)
    end = next(c.kwargs for c in emit.call_args_list if c.args == ("session.end",))
    assert "tree" not in end


def test_dispatcher_evidence_is_a_required_field() -> None:
    """An incomplete dispatcher fake cannot silently drop verification evidence."""
    harness = gate_tests._wf(verify=False)
    harness.gate.dispatcher = mock.MagicMock(spec=[])
    with pytest.raises(AttributeError, match="last_verification"):
        harness.gate.sync_evidence(_loop_state.LoopState(original_task="t", tool_calls=0).verify)
