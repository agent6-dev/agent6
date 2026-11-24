"""Observe the first request of a resumed worker and the decision in its journal."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import config, kinds, media
from agent6.harness import loop_conversation, operator_bridge
from agent6.harness.compaction import window
from agent6.providers import provider_types
from agent6.sessions import event_log
from tests.unit import test_cold_start_rule as rules
from tests.unit import test_compaction_writers as writers
from tests.unit import test_context_compaction as drops
from tests.unit import test_harness


def _history() -> loop_conversation.Conversation:
    conv = loop_conversation.Conversation()
    conv.notice("Original goal.", origin="operator")
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
    helper_reply: str = "Helper handoff.",
    drop_at: int = 10**8,
) -> tuple[str, list[dict[str, Any]], list[str], writers.Writer]:
    path = directory / "logs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    snap = directory / "loop_state.json"
    snap.write_text(
        test_harness._resume_snapshot(
            messages=_history().to_wire(snapshot=True),
            pins=["Binding pin."],
            original_task="Original goal.",
        ).model_dump_json()
    )
    worker = mock.MagicMock()
    worker.call.return_value = test_harness._resp("Done.")
    helper = writers.Writer(writers.response(helper_reply))
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
            summarise_at_chars=10**9,
            elision_gists=False,
            helper=helper,
            helper_context_tokens=200000,
            keep_recent_turns=0,
            keep_thinking_turns=1,
        ),
        max_turns=1,
    )
    wf.resume(steer="Continue here.")
    worker.call.assert_called_once()
    sent = str(worker.call.call_args.kwargs["messages"])
    return sent, [json.loads(line) for line in path.read_text().splitlines()], lines, helper


@pytest.mark.parametrize("mode", ["resend", "drop", "summarize", "truncate"])
@pytest.mark.parametrize("cause", ["provider", "model", "idle"])
def test_first_cold_worker_request(
    tmp_path: pathlib.Path, mode: kinds.ColdStart, cause: str
) -> None:
    fields = {cause: "other"} if cause != "idle" else {}
    # A later worker-written summary, listing or helper call cannot warm the ordinary request.
    now = datetime.datetime.now(datetime.UTC).isoformat()
    sent, rows, lines, helper = _resume(
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


@pytest.mark.parametrize(
    "rows", [[], [rules.request()], [rules.request(type="model.call", model="other")]]
)
def test_no_idle_limit_or_unknown_history_does_not_rewrite(
    tmp_path: pathlib.Path, rows: list[dict[str, Any]]
) -> None:
    sent, log, _, helper = _resume(tmp_path, "truncate", rows)
    assert "Old evidence." in sent and not helper.calls
    assert not any(r["type"] == "compact.cold_start" for r in log)


def test_endpoint_keeps_its_own_compaction_at_a_cold_start(tmp_path: pathlib.Path) -> None:
    sent, rows, _, helper = _resume(
        tmp_path, "truncate", [rules.request(model="other")], endpoint=True
    )
    assert "Old evidence." in sent and not helper.calls
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["action"] == "endpoint" and row["chars_not_sent"] == 0


def test_a_cold_drop_does_not_run_a_second_warm_drop(tmp_path: pathlib.Path) -> None:
    sent, _, _, _ = _resume(tmp_path, "drop", [rules.request(model="other")], drop_at=100)
    assert "Edited file body." in sent and "Old evidence." not in sent


def test_rejected_cold_summary_keeps_the_request(tmp_path: pathlib.Path) -> None:
    sent, rows, _, helper = _resume(
        tmp_path, "summarize", [rules.request(model="other")], helper_reply=""
    )
    assert len(helper.calls) == 1 and "Old evidence." in sent
    (row,) = [r for r in rows if r["type"] == "compact.cold_start"]
    assert row["action"] == "summary_failed" and row["chars_not_sent"] == 0
