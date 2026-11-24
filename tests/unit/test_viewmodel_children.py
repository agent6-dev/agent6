# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every session's children, from lineage and the dispatch records, as the surfaces read them."""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from agent6.sessions import child_dispatch, completion, process
from agent6.viewmodel import children as vm_children
from agent6.viewmodel import listing, receipt, wire


def _session(
    state: pathlib.Path,
    session_id: str,
    *,
    parent: str = "",
    number: int = 0,
    group: str = "",
    usd: float = 0.0,
    finished: bool = True,
) -> pathlib.Path:
    d = state / "sessions" / "runs" / session_id
    d.mkdir(parents=True)
    manifest: dict[str, object] = {
        "version": 2,
        "session_id": session_id,
        "mode": "run",
        "user_task": "t",
        "models": {"driver": {"provider": "p", "model": "m"}},
    }
    if parent:
        manifest["parallel"] = {
            "group": group or parent,
            "number": number,
            "parent": parent,
        }
    (d / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "t", "session_id": session_id},
        {"type": "budget.update", "usd_total": usd, "turn": 1},
    ]
    if finished:
        rows.append(
            {"type": "session.stop", "reason": "finish_session", "verification": "unverified"}
        )
    (d / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def _record(
    identity: str,
    children: list[tuple[str, child_dispatch.MergeRecord | None, float | None]],
    *,
    turn: int = 2,
) -> child_dispatch.DispatchRecord:
    return child_dispatch.DispatchRecord(
        identity=identity,
        source="model",
        owner="exec-1",
        turn=turn,
        base="a" * 40,
        spec=str(len(children)),
        children=tuple(
            child_dispatch.ChildRecord(
                session_id=sid,
                number=i,
                task="audit",
                model="reviewer" if i == 1 else None,
                merge=merge,
                usage=child_dispatch.UsageRecord(usd=usd) if usd is not None else None,
                settled=usd is not None,
            )
            for i, (sid, merge, usd) in enumerate(children, start=1)
        ),
        state="complete",
        message=f"[parallel] group {identity} complete",
        delivered=True,
    )


def _merged(sha: str = "b" * 40) -> child_dispatch.MergeRecord:
    return child_dispatch.MergeRecord(
        branch="agent6/x", status="merged", sha=sha, target_before="a" * 40, target_after=sha
    )


def test_children_come_from_the_records_in_dispatch_then_child_order(
    tmp_path: pathlib.Path,
) -> None:
    """Two groups: a finished child, one that never launched, a no-change child; each truthful."""
    state = tmp_path / "state"
    parent = _session(state, "run-abc")
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1", usd=0.5)
    _session(state, "run-abc-p2-l1", parent="run-abc", number=1, group="run-abc-p2", usd=0.25)
    never = child_dispatch.MergeRecord(
        branch="agent6/run-abc-p1-l2", status="failed", detail="never started"
    )
    child_dispatch.write_record(
        parent,
        _record("run-abc-p1", [("run-abc-p1-l1", _merged(), 0.5), ("run-abc-p1-l2", never, None)]),
        now=1.0,
    )
    no_change = child_dispatch.MergeRecord(branch="agent6/run-abc-p2-l1", status="no_changes")
    child_dispatch.write_record(
        parent, _record("run-abc-p2", [("run-abc-p2-l1", no_change, 0.25)], turn=5), now=2.0
    )

    children = vm_children.children_of(parent)
    assert [c.session_id for c in children] == ["run-abc-p1-l1", "run-abc-p1-l2", "run-abc-p2-l1"]
    first, missing, quiet = children
    assert (first.group, first.source, first.number, first.model) == (
        "run-abc-p1",
        "model",
        1,
        "reviewer",
    )
    assert (
        first.launched
        and first.status == "finished"
        and first.merged_into_parent == "run-abc"
        and first.outcome == "merged"
        and first.parent_merge_sha == "b" * 40
    )
    assert first.imported_ref == "agent6/x" and first.spend_usd == 0.5 and first.settled
    assert first.route == "p/m"
    assert not missing.launched and missing.status == "failed" and missing.outcome == "failed"
    assert missing.error == "never started" and not missing.settled and missing.spend_usd == 0.0
    assert (
        quiet.outcome == "no_changes" and quiet.parent_merge_sha == "" and quiet.spend_usd == 0.25
    )
    dispatches = vm_children.dispatches_of(parent)
    assert [d.identity for d in dispatches] == ["run-abc-p1", "run-abc-p2"]
    assert dispatches[0].state == "complete" and dispatches[1].turn == 5
    assert len(dispatches[0].children) == 2


@pytest.mark.parametrize(
    ("stop", "verify", "outcome", "detail", "word", "label"),
    [
        ("gate_stale", "failed", "retained", "", "finished", "finished · gate stale · retained"),
        ("finish_session", "failed", "retained", "", "finished", "finished · gate red · retained"),
        (
            "finish_session",
            "unverified",
            "retained",
            "",
            "finished",
            "finished · unverified · retained",
        ),
        ("finish_session", "passed", "ranked", "2 of 2", "passed", "passed · ranked 2 of 2"),
        ("finish_session", "passed", "conflict", "a.py", "passed", "passed · conflict"),
        ("answered", None, "no_changes", "", "answered", "answered · no changes"),
        ("provider_error", None, "failed", "no import", "failed", "failed · provider error"),
    ],
)
def test_parent_outcome_does_not_replace_the_childs_gate_or_stop_reason(
    tmp_path: pathlib.Path,
    stop: str,
    verify: str | None,
    outcome: str,
    detail: str,
    word: str,
    label: str,
) -> None:
    parent = _session(tmp_path, "root")
    child = _session(tmp_path, "root-p1-l1", parent="root", finished=False)
    with (child / "logs.jsonl").open("a") as log:
        log.write(
            json.dumps({"type": "session.stop", "reason": stop, "verification": verify}) + "\n"
        )
    merge = child_dispatch.MergeRecord.model_validate({"status": outcome, "detail": detail})
    child_dispatch.write_record(parent, _record("root-p1", [(child.name, merge, 0.0)]), now=1)
    (row,) = wire.manifest_header(parent)["children"]
    assert (row["status"], row["outcome"], row["status_label"]) == (word, outcome, label)
    assert row["error"] == (detail if outcome in ("conflict", "failed") else "")
    assert label in receipt.render(receipt.read_receipt(parent, tmp_path))


def test_a_missing_session_does_not_turn_a_failed_merge_into_a_launch_refusal(
    tmp_path: pathlib.Path,
) -> None:
    """The parent recorded the charge and the merge failure, not the child's exit status."""
    parent = _session(tmp_path, "root")
    merge = child_dispatch.MergeRecord(
        branch="agent6/root-p1-l1", status="failed", detail="could not lock parent ref"
    )
    child_dispatch.write_record(parent, _record("root-p1", [("root-p1-l1", merge, 0.5)]), now=1)
    (row,) = wire.manifest_header(parent)["children"]
    assert not row["launched"] and row["settled"] and row["spend_usd"] == 0.5
    assert (row["status"], row["outcome"], row["status_label"]) == (
        "unreported",
        "failed",
        "unreported · failed",
    )
    assert row["error"] == "could not lock parent ref"


def test_a_group_without_records_reads_its_children_from_lineage(tmp_path: pathlib.Path) -> None:
    """A group predating dispatch records still shows its children, with no merge to report."""
    state = tmp_path / "state"
    parent = _session(state, "fan")
    _session(state, "fan-l2", parent="fan", number=2, usd=1.0)
    _session(state, "fan-l1", parent="fan", number=1, usd=2.0)
    children = vm_children.children_of(parent)
    assert [c.session_id for c in children] == ["fan-l1", "fan-l2"]
    assert all(
        c.outcome == "" and c.launched and c.group == "fan" and c.source == "" for c in children
    )
    assert [c.spend_usd for c in children] == [2.0, 1.0]
    assert vm_children.dispatches_of(parent) == ()


def test_a_receipt_without_dispatch_records_lists_reviewers_from_lineage(
    tmp_path: pathlib.Path,
) -> None:
    parent = _session(tmp_path, "root")
    child = _session(tmp_path, "old-reviewer", parent="root", number=1, group="old-panel")
    reviewers = parent.parent.parent / "reviewers"
    reviewers.mkdir()
    manifest = child / "manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "mode": "reviewer"}))
    child.rename(reviewers / child.name)
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    assert "old-panel reviewer 1: old-reviewer" in text


def test_a_session_with_neither_records_nor_children_has_no_children(
    tmp_path: pathlib.Path,
) -> None:
    parent = _session(tmp_path / "state", "alone")
    assert vm_children.children_of(parent) == ()
    assert vm_children.dispatches_of(parent) == ()


def test_an_unreadable_record_is_reported_not_hidden(tmp_path: pathlib.Path) -> None:
    state = tmp_path / "state"
    parent = _session(state, "run-abc")
    child_dispatch.write_record(
        parent, _record("run-abc-p1", [("run-abc-p1-l1", _merged(), 0.5)]), now=1.0
    )
    child_dispatch.record_path(parent, "run-abc-p1").write_text("{broken", encoding="utf-8")
    (view,) = vm_children.dispatches_of(parent)
    assert view.state == "unreadable" and "invalid dispatch record" in view.message
    assert vm_children.children_of(parent) == ()


def test_the_header_carries_children_and_dispatches_for_an_ordinary_parent(
    tmp_path: pathlib.Path,
) -> None:
    """No group stamp is invented: the parent keeps its actions and gains its children."""
    state = tmp_path / "state"
    parent = _session(state, "run-abc")
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1", usd=0.5)
    child_dispatch.write_record(
        parent, _record("run-abc-p1", [("run-abc-p1-l1", _merged(), 0.5)]), now=1.0
    )
    header = wire.manifest_header(parent)
    assert [row["session_id"] for row in header["children"]] == ["run-abc-p1-l1"]
    (dispatch,) = header["dispatches"]
    assert dispatch["identity"] == "run-abc-p1" and dispatch["children"][0]["outcome"] == "merged"
    assert header["session_actions"]["fork"] is True and header["session_actions"]["merge"] is True
    assert "children" not in wire.manifest_header(_session(state, "alone"))


def test_the_receipt_lists_its_children_and_labels_the_inclusive_spend(
    tmp_path: pathlib.Path,
) -> None:
    state = tmp_path / "state"
    parent = _session(state, "run-abc", usd=3.0)
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1", usd=0.5)
    never = child_dispatch.MergeRecord(
        branch="agent6/run-abc-p1-l2", status="failed", detail="never started"
    )
    child_dispatch.write_record(
        parent,
        _record("run-abc-p1", [("run-abc-p1-l1", _merged(), 0.5), ("run-abc-p1-l2", never, None)]),
        now=1.0,
    )
    view = receipt.read_receipt(parent, tmp_path)
    assert [c.session_id for c in view.children] == ["run-abc-p1-l1", "run-abc-p1-l2"]
    payload = receipt.as_dict(view)
    assert (
        payload["children"][0]["spend_usd"] == 0.5 and payload["children"][1]["launched"] is False
    )
    text = receipt.render(view)
    assert "Children" in text.splitlines()
    assert (
        "run-abc-p1 attempt 1: run-abc-p1-l1 (reviewer): "
        "finished · unverified · merged into run-abc's chain at bbbbbbbbbbbb, $0.50" in text
    )
    assert "run-abc-p1 attempt 2: run-abc-p1-l2: failed: never started" in text
    assert "spend includes the children's $0.50" in text
    # A session with no children reports its counters but has no roster section.
    alone = receipt.read_receipt(_session(state, "alone"), tmp_path)
    assert alone.children == () and "Children" not in receipt.render(alone).splitlines()


def test_an_unpriced_childs_spend_reads_unknown_on_the_receipt(tmp_path: pathlib.Path) -> None:
    """A settled child whose calls were all unpriced read `$0.0000` on the receipt."""
    state = tmp_path / "state"
    parent = _session(state, "run-abc", usd=0.2)
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1")
    _session(state, "run-abc-p1-l2", parent="run-abc", number=2, group="run-abc-p1", usd=0.2)
    record = _record(
        "run-abc-p1", [("run-abc-p1-l1", _merged(), 0.0), ("run-abc-p1-l2", _merged(), 0.2)]
    )
    unpriced = record.children[0].model_copy(
        update={"usage": child_dispatch.UsageRecord(usd=0.0, usd_partial=True)}
    )
    child_dispatch.write_record(
        parent, record.model_copy(update={"children": (unpriced, record.children[1])}), now=1.0
    )
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    assert (
        "run-abc-p1 attempt 1: run-abc-p1-l1 (reviewer): "
        "finished · unverified · merged into run-abc's chain at bbbbbbbbbbbb, $?" in text
    )
    assert (
        "run-abc-p1 attempt 2: run-abc-p1-l2: finished · unverified · "
        "merged into run-abc's chain at bbbbbbbbbbbb, $0.20" in text
    )
    assert "spend includes the children's ~$0.20" in text


def test_a_running_panel_does_not_start_its_later_batches(tmp_path: pathlib.Path) -> None:
    """The whole panel is announced while only the first batch has workers."""
    parent = _session(tmp_path, "root")
    first, later = "root-panel-b1-l1", "root-panel-b2-l1"
    record = _record("root-panel", [(first, None, None), (later, None, None)])
    child_dispatch.write_record(
        parent,
        record.model_copy(
            update={"state": "running", "source": "panel", "merge_policy": "record_only"}
        ),
        now=1.0,
    )
    directory = parent.parent.parent / "reviewers" / first
    directory.mkdir(parents=True)
    process.write_worker_pid(directory, os.getpid())
    (directory / "logs.jsonl").write_text(
        '{"type":"approval.prompt","id":"approval-1","prompt":"Setup?","standing":false}\n'
    )
    rows = vm_children.children_of(parent)
    assert [(r.status, r.launched) for r in rows] == [("starting", True), ("not started", False)]


def test_a_reviewer_has_a_child_row_and_its_verdict(tmp_path: pathlib.Path) -> None:
    """A reviewed run read `(2 children)` and listed its passing reviewers as unchanged children."""
    state = tmp_path / "state"
    parent = _session(state, "run-abc", usd=1.0)
    reviewer_id = "run-abc-rpanel-l1"
    runs_dir = _session(state, reviewer_id, parent="run-abc", number=1, group="run-abc-rpanel")
    reviewer_dir = state / "sessions" / "reviewers" / reviewer_id
    reviewer_dir.parent.mkdir()
    runs_dir.rename(reviewer_dir)
    for name in ("manifest.json", "logs.jsonl"):
        text = (reviewer_dir / name).read_text(encoding="utf-8")
        (reviewer_dir / name).write_text(
            text.replace('"mode": "run"', '"mode": "reviewer"'), encoding="utf-8"
        )
    record = _record("run-abc-rpanel", [(reviewer_id, None, 0.25)])
    child_dispatch.write_record(
        parent,
        record.model_copy(update={"source": "panel", "merge_policy": "record_only"}),
        now=1.0,
    )
    summaries = [listing.summarize_session_dir(d) for d in (parent, reviewer_dir)]
    (row,) = listing.nested_rows(summaries)
    assert row.summary.session_id == "run-abc" and row.children == ()
    assert listing.children_of(state, "run-abc") == []
    completion.write_completion(
        reviewer_dir,
        completion.CompletionRecord(
            text='```json\n{"order":["b","a"],"summary":"b covers both cases"}\n```',
            stop_reason="answered",
            output_tokens=10,
            calls=1,
        ),
    )
    (row,) = wire.manifest_header(parent)["children"]
    assert tuple(row["verdict_order"]) == ("b", "a")
    assert row["verdict_summary"] == "b covers both cases"
    # The record still names the reviewer, found in its own bucket, for the panel's surfaces.
    (child,) = vm_children.children_of(parent)
    assert child.source == "panel" and child.launched and child.settled
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    roster = text.partition("\nChildren\n")[2].partition("\nHow to revert")[0]
    assert f"run-abc-rpanel reviewer 1: {reviewer_id}" in roster
    assert "spend includes the children's $0.25" in roster


def test_the_children_section_names_each_group_and_its_ranking_reviewer(
    tmp_path: pathlib.Path,
) -> None:
    """Attempt numbers restart in each dispatch; the reviewer is not another attempt 1."""
    parent = _session(tmp_path, "root")
    for group in ("root-p1", "root-p2", "root-rp2"):
        record = _record(group, [(f"{group}-l1", None, 0.25)])
        if group == "root-rp2":
            record = record.model_copy(update={"source": "panel", "merge_policy": "record_only"})
        child_dispatch.write_record(parent, record, now=1)
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    roster = text.partition("\nChildren\n")[2].partition("\nHow to revert")[0]
    assert "root-p1 attempt 1: root-p1-l1" in roster
    assert "root-p2 attempt 1: root-p2-l1" in roster
    assert "root-rp2 reviewer 1: root-rp2-l1" in roster
    assert "spend includes the children's $0.75" in roster


def test_receipt_distinguishes_legacy_child_usage_from_zero(tmp_path: pathlib.Path) -> None:
    """A settled dollar amount in an older record does not prove any token count."""
    parent = _session(tmp_path, "root")
    child_dispatch.write_record(parent, _record("g", [("child", None, 0.25)]), now=1)
    (parent / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "budget.update",
                "usd_total": 0.25,
                "settled_children": ["child"],
                "input_total": 0,
                "output_total": 0,
            }
        )
        + "\n"
    )
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    assert "Parent: in=0 out=0" in text
    assert "Children: in=unreported out=unreported" in text
    assert "TOTAL: in=unreported out=unreported" in text


def test_receipt_keeps_settlements_without_the_roster_across_resume(tmp_path: pathlib.Path) -> None:
    """The journal proves the charge even after the child records have gone."""
    parent = _session(tmp_path, "root")
    update = {
        "type": "budget.update",
        "usd_total": 0.25,
        "settled_children": ["old-child"],
        "input_total": 9,
        "output_total": 3,
        "input_settled_children": 27,
        "output_settled_children": 9,
    }
    rows = [
        {"type": "session.start"},
        update,
        update,
        {"type": "session.resume"},
        {**update, "input_settled_children": 0, "output_settled_children": 0},
    ]
    (parent / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    assert "Parent: in=18 out=6" in text
    assert "Children: in=27 out=9" in text
    assert "TOTAL: in=45 out=15" in text


def test_the_parent_header_reads_a_live_tool_and_usage_from_the_child(
    tmp_path: pathlib.Path,
) -> None:
    """Unsettled spend and a tool still running come from the child's journal."""
    state = tmp_path / "state"
    parent = _session(state, "root")
    sid = "root-p1-l1"
    child = _session(state, sid, parent="root", finished=False)
    process.write_worker_pid(child, os.getpid())
    start = time.time() - 40
    rows = [
        {"type": "session.start", "user_task": "audit", "ts": start},
        {"type": "budget.update", "input_total": 10, "output_total": 2, "usd_total": 0.2},
        {"type": "tool.call", "name": "read_file", "call_id": 1, "turn": 2},
        {"type": "tool.call", "name": "run_command", "call_id": 2, "turn": 2},
        {"type": "tool.result", "name": "read_file", "call_id": 1},
    ]
    (child / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = _record("root-p1", [(sid, None, None)]).advanced("running")
    child_dispatch.write_record(parent, rec, now=start)
    (row,) = wire.manifest_header(parent)["children"]
    assert row["status"] == "running" and row["spend_usd"] == 0.2
    assert row["turn"] == 2 and row["tokens"] == 12 and row["current_tool"] == "run_command"
    assert 40 <= row["elapsed_s"] < 60
    with (child / "logs.jsonl").open("a") as log:
        log.write(
            json.dumps({"type": "session.stop", "reason": "finish_session", "ts": start + 12})
            + "\n"
        )
    child_dispatch.write_record(parent, rec.advanced("ranking"), now=start + 12)
    (ended,) = vm_children.children_of(parent)
    assert (ended.status, ended.elapsed_s, ended.current_tool) == ("finished", 12, "")


def test_stopped_and_never_started_children_stay_in_the_header(tmp_path: pathlib.Path) -> None:
    """A stopped branch keeps its commit count and a launch refusal keeps its diagnostic."""
    state = tmp_path / "state"
    parent = _session(state, "root")
    child = _session(state, "root-p1-l1", parent="root")
    merge = child_dispatch.MergeRecord(status="stopped", branch="agent6/child", commits=2)
    child_dispatch.write_record(parent, _record("root-p1", [(child.name, merge, 0.5)]), now=1)
    refused = _record("root-p2", [("root-p2-l1", None, None)]).advanced(
        "complete", message="dispatch failed: no route"
    )
    child_dispatch.write_record(parent, refused, now=2)
    held, absent = wire.manifest_header(parent)["children"]
    assert (held["status"], held["outcome"], held["commits"]) == ("finished", "stopped", 2)
    assert held["imported_ref"] == "agent6/child"
    assert not absent["launched"] and absent["status"] == "failed" and "no route" in absent["error"]
