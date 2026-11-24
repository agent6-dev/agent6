"""Observe the first request of a resumed worker and the decision in its journal."""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import budget, config, kinds, media
from agent6.app import role_providers
from agent6.harness import loop_conversation, nudges, operator_bridge
from agent6.harness.compaction import window
from agent6.providers import provider_types, request_headers
from agent6.sessions import event_log
from tests.unit import test_claude_code_provider as claude
from tests.unit import test_cold_start_rule as rules
from tests.unit import test_compaction_writers as writers
from tests.unit import test_context_compaction as drops
from tests.unit import test_harness


def _history(origin: loop_conversation.TaskSource = "operator") -> loop_conversation.Conversation:
    conv = loop_conversation.Conversation()
    conv.notice("Original goal.", origin=origin)
    conv.assistant(
        [
            {"type": "thinking", "thinking": "old reasoning", "signature": "old signature"},
            {"type": "text", "text": "Old dialogue."},
        ]
    )
    drops._reads(conv, "Old evidence." * 500, "More evidence." * 500)
    drops._add_exchange(conv, ("read_file", {"path": "hot.py"}, "Edited file body." * 500))
    drops._add_exchange(conv, ("apply_edit", {"path": "hot.py"}, "Edited."))
    drops._add_exchange(conv, ("ask_user", {}, '{"status":"answered","answers":["Exact answer."]}'))
    drops._reads(conv, "Recent evidence." * 100, "Unread evidence." * 100)
    return conv


def _resume(
    directory: pathlib.Path,
    mode: kinds.ColdStart,
    rows: list[dict[str, Any]],
    *,
    limit: float = -1,
    endpoint: bool = False,
    helper: writers.Writer | None = None,
    drop_at: int = 10**8,
    plan: pathlib.Path | None = None,
    saved: loop_conversation.Conversation | None = None,
    summarize_at: int = 10**9,
    keep_turns: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], writers.Writer]:
    path = directory / "logs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    snap = directory / "loop_state.json"
    snap.write_text(
        test_harness._resume_snapshot(
            messages=(saved or _history()).to_wire(snapshot=True),
            pins=["Binding pin."],
            original_task="Original goal.",
        ).model_dump_json()
    )
    worker = mock.MagicMock()
    worker.call.return_value = test_harness._resp("Done.")
    helper = helper or writers.Writer(writers.response("Helper handoff."))
    lines: list[str] = []
    dispatcher = mock.MagicMock(
        config=config.Config(), last_verification=None, execution_id="new", verify_withheld=False
    )
    offered = {"read_session", "list_tasks"}
    dispatcher.available_tool_names.return_value = offered

    def withheld(name: str) -> bool:
        return name not in offered

    dispatcher.tool_is_withheld.side_effect = withheld
    wf = test_harness._wf(
        provider=worker,
        mode="plan" if plan is not None else "run",
        plan_output_path=plan,
        dispatcher=dispatcher,
        events=event_log.EventSink(path),
        resume_state_path=snap,
        media_input=media.InputCapabilities(kinds.ModelRoute("p", "m")),
        logger=lines.append,
        bridge=operator_bridge.OperatorBridge(
            steer_requested=lambda: True, steer_prompt=lambda: "Continue here."
        ),
        compaction=window.CompactionSettings(
            cold_start=mode,
            idle=window.IdleCompaction("p", limit, 10**9),
            endpoint=provider_types.EndpointCompaction(writes_summary=endpoint),
            drop_at_chars=drop_at,
            summarise_at_chars=summarize_at,
            summariser="helper",
            elision_gists=False,
            helper=helper,
            helper_context_tokens=200000,
            keep_recent_turns=keep_turns,
            keep_thinking_turns=1,
        ),
        max_turns=1,
    )
    result = wf.resume(steer="Continue here.")
    if result.reason in ("plan_unreadable", "provider_error", "budget_exhausted"):
        worker.call.assert_not_called()
        sent = []
    else:
        worker.call.assert_called_once()
        sent = worker.call.call_args.kwargs["messages"]
    return sent, [json.loads(line) for line in path.read_text().splitlines()], lines, helper


@pytest.mark.parametrize("mode", ["resend", "drop", "summarize", "truncate"])
@pytest.mark.parametrize("cause", ["provider", "model", "idle"])
def test_first_cold_worker_request(
    tmp_path: pathlib.Path, mode: kinds.ColdStart, cause: str
) -> None:
    fields = {cause: "other"} if cause != "idle" else {}
    # A later worker-written summary, listing or helper call cannot warm the ordinary request.
    now = datetime.datetime.now(datetime.UTC).isoformat()
    wire, rows, lines, helper = _resume(
        tmp_path,
        mode,
        [
            rules.request(**fields),
            {"type": "compact.summarise.call", "model_call_id": "summary", "writer": "worker"},
            rules.request(ts=now, model_call_id="summary"),
            rules.request(ts=now, target="listing"),
            rules.request(ts=now, actor="helper:summariser"),
            rules.request(ts=now, type="model.result"),
        ],
        limit=5 if cause == "idle" else -1,
    )
    sent = str(wire)
    assert ("Old evidence." in sent) is (mode == "resend")
    assert ("Old dialogue." in sent) is (mode in ("resend", "drop"))
    assert ("old reasoning" in sent) is (mode == "resend")
    assert "Continue here." in sent and "Unread evidence." in sent
    assert ("Recent evidence." in sent) is (mode in ("drop", "resend"))
    assert "Exact answer." in sent
    assert len(helper.calls) == int(mode == "summarize")
    if mode == "summarize":
        assert "Helper handoff." in sent and "Old dialogue." in str(helper.calls[0]["messages"])
        assert "Old evidence." not in str(helper.calls[0]["messages"])
        assert [r["reason"] for r in rows if r["type"] == "compact.summarise.done"] == [
            "cold_start"
        ]
    if mode == "truncate":
        assert "earlier history was left out" in sent and "read_session" in sent
        assert "PROGRESS SUMMARY:" not in sent and "Binding pin." in sent and "list_tasks" in sent
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["reason"] == ("idle" if cause == "idle" else "route_changed")
    assert row["action"] == mode and (row["chars_not_sent"] > 0) is (mode != "resend")
    assert any(row["message"] in line for line in lines)
    assert mode in row["message"] and str(row["chars_not_sent"]) in row["message"]
    if cause == "idle":
        assert row["gap_s"] >= row["idle_compact_s"] == 5
    else:
        assert row["previous_route"] != row["route"] == "p/m"


@pytest.mark.parametrize("mode", ["summarize", "truncate"])
def test_a_cold_restart_states_the_task_a_parent_gave(
    tmp_path: pathlib.Path, mode: kinds.ColdStart
) -> None:
    """A child restarted cold had its task in no message."""
    sent, _, _, _ = _resume(tmp_path, mode, [rules.request(model="other")], saved=_history("model"))
    assert "TASK (the goal, verbatim):\nOriginal goal." in sent[0]["content"][0]["text"]


@pytest.mark.parametrize("mode", ["summarize", "truncate"])
def test_a_cold_restart_with_nothing_to_replace_resends(
    tmp_path: pathlib.Path, mode: kinds.ColdStart
) -> None:
    """The helper summarized an empty transcript; a truncate notice said history was left out."""
    saved = loop_conversation.Conversation()
    saved.notice("Original goal.", origin="operator")
    saved.assistant([{"type": "text", "text": "I looked and it is fine."}])
    previous = [rules.request(model="other")]
    resend, _, _, _ = _resume(tmp_path, "resend", previous, saved=saved, keep_turns=-1)
    sent, rows, _, helper = _resume(tmp_path, mode, previous, saved=saved, keep_turns=-1)
    assert sent == resend and not helper.calls
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert (row["mode"], row["action"], row["chars_not_sent"]) == (mode, "resend", 0)
    assert "resend history" in row["message"]


@pytest.mark.parametrize("forks", [1, 2])
def test_a_fork_reads_the_last_request_of_its_source(tmp_path: pathlib.Path, forks: int) -> None:
    """A fork's own journal holds no worker request, so its first request resent everything."""
    session = tmp_path / "source"
    session.mkdir()
    (session / "logs.jsonl").write_text(json.dumps(rules.request(model="other")) + "\n")
    for n in range(forks):
        parent, session = session.name, tmp_path / f"fork-{n}"
        session.mkdir()
        (session / "manifest.json").write_text(json.dumps({"parent_session_id": parent}))
    sent, rows, _, _ = _resume(session, "truncate", [])
    assert "Old evidence." not in str(sent)
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert (row["reason"], row["previous_route"], row["action"]) == (
        "route_changed",
        "p/other",
        "truncate",
    )


@pytest.mark.parametrize(
    "rows", [[], [rules.request()], [rules.request(type="model.call", model="other")]]
)
def test_no_idle_limit_or_unknown_history_does_not_rewrite(
    tmp_path: pathlib.Path, rows: list[dict[str, Any]]
) -> None:
    sent, log, _, helper = _resume(tmp_path, "truncate", rows)
    assert "Old evidence." in str(sent) and not helper.calls
    assert not any(r["type"] == "compact.cold_start" for r in log)


def test_endpoint_keeps_its_own_compaction_at_a_cold_start(tmp_path: pathlib.Path) -> None:
    sent, rows, _, helper = _resume(
        tmp_path, "truncate", [rules.request(model="other")], endpoint=True
    )
    assert "Old evidence." in str(sent) and not helper.calls
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["action"] == "endpoint" and row["chars_not_sent"] == 0


def test_a_cold_drop_strips_old_thinking_when_it_replaces_no_result(
    tmp_path: pathlib.Path,
) -> None:
    """Every old thinking block was sent with `keep_thinking_turns = 1`."""
    saved = loop_conversation.Conversation()
    saved.notice("Original goal.", origin="operator")
    for i in range(3):
        saved.assistant(
            [
                {"type": "thinking", "thinking": f"thought-{i}", "signature": "s"},
                {"type": "text", "text": f"reply {i}"},
            ]
        )
        saved.notice("continue")
    sent, rows, _, _ = _resume(tmp_path, "drop", [rules.request(model="other")], saved=saved)
    assert [f"thought-{i}" in str(sent) for i in range(3)] == [False, False, True]
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["chars_not_sent"] > 0
    assert [r["messages"] for r in rows if r["type"] == "compact.thinking_dropped"] == [2]


def test_a_cold_drop_takes_the_ordinary_checks_of_a_resend(tmp_path: pathlib.Path) -> None:
    """After a route change to a smaller window, a cold drop sent three times the summary bound."""
    prose = loop_conversation.Conversation()
    prose.notice("Original goal.", origin="operator")
    for i in range(10):
        prose.assistant([{"type": "text", "text": f"analysis {i} " + "a" * 30000}])
        prose.notice("continue")
    previous = [rules.request(model="big-window")]
    resend, _, _, _ = _resume(tmp_path, "resend", previous, saved=prose, summarize_at=100_000)
    sent, rows, _, helper = _resume(tmp_path, "drop", previous, saved=prose, summarize_at=100_000)
    assert sent == resend and len(helper.calls) == 1
    assert window.context_chars(loop_conversation.Conversation.from_wire(sent)) < 100_000
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert (row["action"], row["chars_not_sent"]) == ("drop", 0)
    assert [r["reason"] for r in rows if r["type"] == "compact.summarise.done"] == ["size"]

    # The drop bound follows too: it takes the edited read a cold drop alone leaves whole.
    previous = [rules.request(model="other")]
    sent, _, _, _ = _resume(tmp_path, "drop", previous)
    assert "Edited file body." in str(sent) and "Old evidence." not in str(sent)
    resend, _, _, _ = _resume(tmp_path, "resend", previous, drop_at=100)
    sent, _, _, _ = _resume(tmp_path, "drop", previous, drop_at=100)
    assert sent == resend and "Edited file body." not in str(sent)


@pytest.mark.parametrize("mode", ["summarize", "truncate"])
@pytest.mark.parametrize("saved_plan", [None, "Keep the local fixture.", "Superseded plan."])
def test_cold_planner_restart_counts_the_request_with_one_current_plan(
    tmp_path: pathlib.Path, mode: kinds.ColdStart, saved_plan: str | None
) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("Keep the local fixture.")
    saved = _history()
    if saved_plan is not None:
        saved.notice(f"{nudges.PLAN_ON_DISK_HEADER}\n\n{saved_plan}")
    previous = [rules.request(actor="planner", model="other")]
    resend, _, _, _ = _resume(tmp_path, "resend", previous, plan=plan, saved=saved)
    sent, rows, _, _ = _resume(tmp_path, mode, previous, plan=plan, saved=saved)
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    before, after = (
        window.context_chars(loop_conversation.Conversation.from_wire(w)) for w in (resend, sent)
    )
    assert row["chars_not_sent"] == before - after > 0
    assert str(sent).count(nudges.PLAN_ON_DISK_HEADER) == 1
    assert "Keep the local fixture." in str(sent) and "Superseded plan." not in str(sent)


@pytest.mark.parametrize("previous", [[], [rules.request(actor="planner", model="other")]])
def test_a_planner_restart_without_plan_md_keeps_the_checkpoint_plan(
    tmp_path: pathlib.Path, previous: list[dict[str, Any]]
) -> None:
    saved = _history()
    notice = f"{nudges.PLAN_ON_DISK_HEADER}\n\nKeep the orange label."
    saved.notice(notice, origin="plan")
    saved.assistant([{"type": "text", "text": "Read the fixture."}])
    sent, rows, _, helper = _resume(
        tmp_path, "resend", previous, plan=tmp_path / "plan.md", saved=saved, summarize_at=100
    )
    assert len(helper.calls) == 1 and any(r["type"] == "compact.summarise.done" for r in rows)
    snapshot = json.loads((tmp_path / "loop_state.json").read_text())
    for messages in (sent, snapshot["messages"]):
        assert [
            b["text"]
            for m in messages
            for b in m["content"]
            if b.get("text", "").startswith("[harness plan]")
        ] == [notice]


@pytest.mark.parametrize("mode", ["summarize", "resend"])
@pytest.mark.parametrize("edit", [b"Use the new fixture instead.", b"\xff", None])
def test_cold_helper_plan_edit_reaches_the_worker_or_parks_it(
    tmp_path: pathlib.Path, mode: kinds.ColdStart, edit: bytes | None
) -> None:
    """An edit during the summary must reach both the request and its checkpoint."""
    plan = tmp_path / "plan.md"
    plan.write_text("Use the old fixture.")
    previous = [rules.request(actor="planner", model="other")]
    resend, _, _, _ = _resume(tmp_path, "resend", previous, plan=plan)

    class EditingHelper(writers.Writer):
        def call(self, **kwargs: Any) -> provider_types.ProviderResponse:
            if edit is None:
                plan.rename(tmp_path / "withdrawn.md")
            else:
                plan.write_bytes(edit)
            return super().call(**kwargs)

    sent, rows, _, helper = _resume(
        tmp_path,
        mode,
        previous,
        plan=plan,
        helper=EditingHelper(writers.response()),
        summarize_at=100,
    )
    assert len(helper.calls) == 1
    cold_steps = [r for r in rows if r["type"] == "compact.cold_start"]
    if mode == "resend":
        assert [(r["action"], r["chars_not_sent"]) for r in cold_steps] == [("resend", 0)]
    if edit is None or edit == b"\xff":
        assert not sent
        assert any(r["type"] == "plan.read_failed" for r in rows)
        assert any(r["type"] == "session.stop" and r["reason"] == "plan_unreadable" for r in rows)
        return
    snapshot = json.loads((tmp_path / "loop_state.json").read_text())
    for messages in (sent, snapshot["messages"]):
        assert str(messages).count(nudges.PLAN_ON_DISK_HEADER) == 1
        assert str(messages).count(edit.decode()) == 1 and "Use the old fixture." not in str(
            messages
        )
    if mode == "summarize":
        before, after = (
            window.context_chars(loop_conversation.Conversation.from_wire(w))
            for w in (resend, sent)
        )
        (row,) = cold_steps
        assert row["chars_not_sent"] == before - after


def test_rejected_cold_summary_keeps_the_request(tmp_path: pathlib.Path) -> None:
    sent, rows, _, helper = _resume(
        tmp_path,
        "summarize",
        [rules.request(model="other")],
        helper=writers.Writer(writers.response("")),
    )
    assert len(helper.calls) == 1 and "Old evidence." in str(sent)
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["action"] == "summary_failed" and row["chars_not_sent"] == 0


@pytest.mark.parametrize(
    "failure",
    [provider_types.ProviderError("refused", status_code=403), budget.BudgetExceededError("limit")],
)
def test_a_terminal_cold_helper_refusal_reports_unchanged_history(
    tmp_path: pathlib.Path, failure: Exception
) -> None:
    sent, rows, lines, helper = _resume(
        tmp_path, "summarize", [rules.request(model="other")], helper=writers.Writer(failure)
    )
    assert not sent and len(helper.calls) == 1
    assert "Old evidence." in (tmp_path / "loop_state.json").read_text()
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert (row["reason"], row["previous_route"], row["route"]) == (
        "route_changed",
        "p/other",
        "p/m",
    )
    assert (row["mode"], row["action"], row["chars_not_sent"]) == ("summarize", "summary_failed", 0)
    assert any(row["message"] in line for line in lines)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "error_result", "buffered_text", "buffered_thinking"]
)
def test_resume_from_a_process_worker_uses_its_last_issued_call(
    tmp_path: pathlib.Path, outcome: str, stream: bool
) -> None:
    prefix = "private-prefix"
    scenario: dict[str, Any] = {"turns": [[{"text": "part of a reply"}]]}
    error = ""
    buffered = outcome.startswith("buffered_")
    if buffered:
        scenario = {
            "account_email": f"{prefix}@example.test",
            "turns": [[{outcome.removeprefix("buffered_"): prefix}]],
            "die_after_delta": True,
        }
        error = "exited 3"
    elif outcome == "error_result":
        scenario["result"] = {"subtype": "error_during_execution", "is_error": True}
        error = "error_during_execution"
    binary, _ = claude._install(tmp_path, scenario)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    transcripts = provider_types.TranscriptSink(
        tmp_path / "transcripts", objects_dir=tmp_path / "objects"
    )
    worker = role_providers.InstrumentedProvider(
        inner=claude._provider(binary, transcript_sink=transcripts),
        actor="worker",
        model="claude-haiku-4-5",
        provider_name="process",
        events=sink,
        budget=claude._budget(),
        stream_text=stream,
    )
    with (
        contextlib.closing(worker),
        (
            pytest.raises(provider_types.ProviderError, match=error)
            if error
            else contextlib.nullcontext()
        ),
    ):
        worker.call(
            system="s",
            messages=_history().to_wire(),
            tools=claude.TOOLS,
            context=request_headers.RequestContext.initiated_by(
                "user", call_context=kinds.CallContext(execution_id="previous")
            ),
        )
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert any(r["type"] in ("model.text_delta", "model.thinking_delta") for r in rows) is (
        stream and not buffered
    )
    if buffered:
        (transcript,) = (tmp_path / "transcripts").glob("*.json")
        assert prefix not in transcript.read_text() and prefix not in sink.path.read_text()
    if error:
        (attempt,) = [r for r in rows if r["type"] == "model.attempt"]
        assert attempt["process_reply"] is True and attempt["ok"] is False
    for call_id, actor, ok in (("side", "helper:summariser", True), ("not_sent", "worker", False)):
        rows.extend(
            [
                rules.request(type="model.call", actor=actor, model_call_id=call_id),
                rules.request(
                    type="model.attempt", actor=actor, model_call_id=call_id, ok=ok, request_id=None
                ),
            ]
        )
    sent, rows, _, _ = _resume(tmp_path, "truncate", rows)
    assert "Old evidence." not in str(sent) and "earlier history was left out" in str(sent)
    (cold,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert (cold["previous_route"], cold["route"]) == ("process/claude-haiku-4-5", "p/m")
