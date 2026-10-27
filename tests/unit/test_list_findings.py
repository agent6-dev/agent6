# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The worker lists its own review findings; acting on one stays with the operator."""

from __future__ import annotations

import functools
import pathlib

import pytest

from agent6 import event_log, paths, review_records
from agent6.app import _session, review_actions
from agent6.config import Config, RoleModel
from agent6.harness import _toolset
from agent6.sessions import layout as sessions_layout
from agent6.tools import dispatch, errors, operator_prompts, results, schema
from tests import review_helpers


def test_list_findings_is_a_run_tool_only(tmp_path: pathlib.Path) -> None:
    """A plan, an ask and a machine state neither see the tool nor get an answer from it."""
    extras = {
        mode: {cls.TOOL_NAME for cls in schema.mode_tools(mode).extras}
        for mode in ("run", "plan", "ask", "machine")
    }
    assert "list_findings" in extras["run"]
    assert all("list_findings" not in extras[mode] for mode in ("plan", "ask", "machine"))
    wired = functools.partial(results.FindingsResult, findings=())
    for mode in ("plan", "ask", "machine"):
        d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), mode=mode, findings=wired)
        with pytest.raises(errors.ToolError, match="not available in"):
            d.dispatch("list_findings", {})


def test_a_dispatcher_without_a_reader_refuses_the_tool(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), mode="run")
    with pytest.raises(errors.ToolError, match="none was wired"):
        d.dispatch("list_findings", {})


def test_the_worker_lists_open_findings_and_resolved_ones_with_all(tmp_path: pathlib.Path) -> None:
    """Open findings by default, every finding with `all`, each named as the commands take it."""
    saved = review_helpers.reviewed_session(tmp_path)
    resolve = review_records.FindingAction(
        request_id="resolve",
        expected_version=1,
        action="resolve",
        resolution=review_records.FindingResolution(
            disposition="duplicate", reason="Same defect", duplicate_of="f2"
        ),
    )
    review_actions.act(saved.session_dir, "f1", resolve, source="cli")
    d = dispatch.ToolDispatcher(
        root=tmp_path,
        config=Config(),
        mode="run",
        findings=functools.partial(review_actions.findings_result, saved.session_dir),
    )
    assert "list_findings" in {t.name for t in _toolset.tool_definitions(d, mode="run")}
    listed = d.dispatch("list_findings", {})
    assert isinstance(listed, results.FindingsResult)
    assert listed.to_wire() == {
        "findings": [
            {
                "id": "f2",
                "status": "open",
                "severity": "block",
                "category": "correctness",
                "location": "parser.py:3-4",
                "title": "Defect f2",
                "detail": "Reproduce with an empty input.",
            }
        ]
    }
    assert listed.summary() == "1 open"
    every = d.dispatch("list_findings", {"all": True})
    assert isinstance(every, results.FindingsResult)
    rows = every.to_wire()["findings"]
    assert [(r["id"], r["status"]) for r in rows] == [("f1", "resolved"), ("f2", "open")]
    assert rows[0]["resolution"] == "duplicate of f2: Same defect"
    assert every.summary() == "1 open, 1 resolved"
    assert review_actions.read_review(saved.session_dir).version == 2  # nothing recorded


def test_a_session_before_its_first_review_says_so(tmp_path: pathlib.Path) -> None:
    """An unreviewed session answers no findings and says why; a missing journal is an error."""
    saved = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "fresh-run")
    saved.ensure()
    with pytest.raises(errors.ToolError, match="no session journal"):
        review_actions.findings_result(saved.session_dir)
    sink = event_log.EventSink(saved.logs_path)
    sink.emit("session.start", user_task="t", mode="run")
    listed = review_actions.findings_result(saved.session_dir)
    assert listed.to_wire() == {"findings": [], "note": "No review panel has run."}
    assert listed.summary() == "0 open"
    sink.emit("loop.review.panel", blocked=False)
    assert review_actions.findings_result(saved.session_dir).note.startswith("Legacy review")


def test_a_pre_existing_line_is_tagged_without_lowering_severity(tmp_path: pathlib.Path) -> None:
    saved = review_helpers.reviewed_repo(tmp_path)
    listed = review_actions.findings_result(saved.session_dir)
    tagged = [f for f in listed.findings if f.pre_existing]
    assert len(tagged) == 1 and tagged[0].severity == "block"
    assert tagged[0].pre_existing.startswith("old.txt:2 (base).")


def test_the_run_lifecycle_wires_the_session_journal_to_the_tool(tmp_path: pathlib.Path) -> None:
    """A run's dispatcher reads its own journal: the session tools builder wires the reader."""
    saved = review_helpers.reviewed_session(tmp_path)
    tools = _session.build_session_tools(
        Config(),
        cwd=tmp_path,
        state_dir=paths.state_dir(tmp_path),
        layout=saved,
        isolation="none",
        mode="run",
        events=event_log.EventSink(saved.logs_path),
        prompts=operator_prompts.OperatorPrompts(session_dir=saved.session_dir),
        loop_log=lambda _line: None,
        mcp_manager=None,
        rm_role=RoleModel(provider="openrouter", model="m"),
    )
    listed = tools.dispatcher.dispatch("list_findings", {})
    assert isinstance(listed, results.FindingsResult)
    assert [f.id for f in listed.findings] == ["f1", "f2"]
    assert listed.session_id == saved.session_id  # the id the steer command the model names takes
    assert listed.to_wire()["session_id"] == saved.session_id
