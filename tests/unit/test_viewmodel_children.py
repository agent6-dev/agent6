# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every session's children, from lineage and the dispatch records, as the surfaces read them."""

from __future__ import annotations

import json
import pathlib

from agent6.sessions import child_dispatch
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
    children: list[tuple[str, child_dispatch.JoinRecord | None, float | None]],
    *,
    turn: int = 2,
) -> child_dispatch.DispatchRecord:
    return child_dispatch.DispatchRecord(
        identity=identity,
        source="model",
        owner="exec-1",
        turn=turn,
        base="a" * 40,
        segments=(
            child_dispatch.SegmentRecord(
                task="audit", spec=str(len(children)), children=len(children)
            ),
        ),
        children=tuple(
            child_dispatch.ChildRecord(
                session_id=sid,
                number=i,
                task="audit",
                model="reviewer" if i == 1 else None,
                join=join,
                usage=child_dispatch.UsageRecord(usd=usd) if usd is not None else None,
                settled=usd is not None,
            )
            for i, (sid, join, usd) in enumerate(children, start=1)
        ),
        state="complete",
        message=f"[parallel] group {identity} complete",
        delivered=True,
    )


def _joined(sha: str = "b" * 40) -> child_dispatch.JoinRecord:
    return child_dispatch.JoinRecord(
        branch="agent6/x", status="joined", sha=sha, target_before="a" * 40, target_after=sha
    )


def test_children_come_from_the_records_in_dispatch_then_child_order(
    tmp_path: pathlib.Path,
) -> None:
    """Two groups: a finished child, one that never launched, a no-change child; each truthful."""
    state = tmp_path / "state"
    parent = _session(state, "run-abc")
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1", usd=0.5)
    _session(state, "run-abc-p2-l1", parent="run-abc", number=1, group="run-abc-p2", usd=0.25)
    never = child_dispatch.JoinRecord(
        branch="agent6/run-abc-p1-l2", status="failed", detail="never started"
    )
    child_dispatch.write_record(
        parent,
        _record("run-abc-p1", [("run-abc-p1-l1", _joined(), 0.5), ("run-abc-p1-l2", never, None)]),
        now=1.0,
    )
    no_change = child_dispatch.JoinRecord(branch="agent6/run-abc-p2-l1", status="no_changes")
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
        first.launched and first.status and first.outcome == "joined" and first.join_sha == "b" * 40
    )
    assert first.imported_ref == "agent6/x" and first.spend_usd == 0.5 and first.settled
    assert first.route == "p/m"
    assert not missing.launched and missing.status == "" and missing.outcome == "failed"
    assert missing.error == "never started" and not missing.settled and missing.spend_usd == 0.0
    assert quiet.outcome == "no_changes" and quiet.join_sha == "" and quiet.spend_usd == 0.25
    dispatches = vm_children.dispatches_of(parent)
    assert [d.identity for d in dispatches] == ["run-abc-p1", "run-abc-p2"]
    assert dispatches[0].state == "complete" and dispatches[1].turn == 5
    assert len(dispatches[0].children) == 2


def test_a_group_without_records_reads_its_children_from_lineage(tmp_path: pathlib.Path) -> None:
    """`run --parallel` writes no record: its children still show, with no join to report."""
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
        parent, _record("run-abc-p1", [("run-abc-p1-l1", _joined(), 0.5)]), now=1.0
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
        parent, _record("run-abc-p1", [("run-abc-p1-l1", _joined(), 0.5)]), now=1.0
    )
    header = wire.manifest_header(parent)
    assert [row["session_id"] for row in header["children"]] == ["run-abc-p1-l1"]
    (dispatch,) = header["dispatches"]
    assert dispatch["identity"] == "run-abc-p1" and dispatch["children"][0]["outcome"] == "joined"
    assert header["session_actions"]["fork"] is True and header["session_actions"]["merge"] is True
    assert "children" not in wire.manifest_header(_session(state, "alone"))


def test_the_receipt_lists_its_children_and_labels_the_inclusive_spend(
    tmp_path: pathlib.Path,
) -> None:
    state = tmp_path / "state"
    parent = _session(state, "run-abc", usd=3.0)
    _session(state, "run-abc-p1-l1", parent="run-abc", number=1, group="run-abc-p1", usd=0.5)
    never = child_dispatch.JoinRecord(
        branch="agent6/run-abc-p1-l2", status="failed", detail="never started"
    )
    child_dispatch.write_record(
        parent,
        _record("run-abc-p1", [("run-abc-p1-l1", _joined(), 0.5), ("run-abc-p1-l2", never, None)]),
        now=1.0,
    )
    view = receipt.read_receipt(parent, tmp_path)
    assert [c.session_id for c in view.children] == ["run-abc-p1-l1", "run-abc-p1-l2"]
    payload = receipt.as_dict(view)
    assert (
        payload["children"][0]["spend_usd"] == 0.5 and payload["children"][1]["launched"] is False
    )
    text = receipt.render(view)
    assert "Children" in text
    assert "child 1 run-abc-p1-l1 (reviewer): joined at bbbbbbbbbbbb, $0.50" in text
    assert "child 2 run-abc-p1-l2: failed: never started" in text
    assert "spend includes the children's $0.50" in text
    # A session with no children renders no section and the review fields are untouched.
    alone = receipt.read_receipt(_session(state, "alone"), tmp_path)
    assert alone.children == () and "Children" not in receipt.render(alone)


def test_a_panel_reviewer_is_evidence_of_its_panel_and_no_child(tmp_path: pathlib.Path) -> None:
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
        record.model_copy(update={"source": "panel", "join_policy": "record_only"}),
        now=1.0,
    )
    summaries = [listing.summarize_session_dir(d) for d in (parent, reviewer_dir)]
    (row,) = listing.nested_rows(summaries)
    assert row.summary.session_id == "run-abc" and row.children == ()
    assert listing.children_of(state, "run-abc") == []
    assert "children" not in wire.manifest_header(parent)
    # The record still names the reviewer, found in its own bucket, for the panel's surfaces.
    (child,) = vm_children.children_of(parent)
    assert child.source == "panel" and child.launched and child.settled
    text = receipt.render(receipt.read_receipt(parent, tmp_path))
    assert "Children" not in text and "child 1" not in text
