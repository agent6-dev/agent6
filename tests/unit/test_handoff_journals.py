"""The handoff measures use the six queued nodes and observed child activity."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any

import pytest


def _event(t: int, kind: str, **fields: Any) -> dict[str, Any]:
    return dict(
        ts=datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat(), type=kind, **fields
    )


def _journal() -> list[dict[str, Any]]:
    from bench.handoff import job

    titles = job.TITLES
    queued = {f"{i:04}": title for i, title in enumerate(titles, 2)}
    graph = {k: dict(title=v, status="passed") for k, v in queued.items()}
    graph["0007"]["status"] = "pending"
    # A copy with a passed result is not its original listed task.
    graph["0008"] = dict(title=titles[-1], status="passed")
    activity = "parallel.child_activity"
    usage = dict(input_tokens=6, cache_read_tokens=3, cache_write_tokens=1, output_tokens=2)
    return [
        _event(0, "session.start"),
        *[_event(1, "task.queued", task_id=k, title=v) for k, v in queued.items()],
        _event(2, "model.call", model_call_id="call"),
        _event(2, "model.attempt", model_call_id="call", attempt=1, **usage),
        _event(2, "model.attempt", model_call_id="call", attempt=2, **usage),
        _event(2, "model.result", model_call_id="call", **usage),
        _event(2, "tool.call", name="parallel", args_preview={"task": titles[0]}),
        _event(2, "nudge.task_stuck", task_id="0002"),
        _event(3, activity, child="a", activity="started", detail=titles[0]),
        _event(4, activity, child="b", activity="started", detail=titles[1]),
        _event(5, "tool.call", name="parallel", args_preview={"task_id": "0004"}),
        _event(5, "tool.result", name="parallel", ok=True, summary="nothing dispatched"),
        _event(8, activity, child="a", activity="finished"),
        _event(9, activity, child="c", activity="started", detail=titles[0]),
        _event(11, activity, child="c", activity="stopped"),
        _event(12, activity, child="b", activity="failed"),
        _event(13, "graph.update", nodes=graph),
        _event(16, "session.stop", turns=4, reason="max_turns"),
    ]


def _measure(root: pathlib.Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the parent's journal beside one journal per child, as the session directory does."""
    from bench.handoff import journals

    path = root / "handoff" / "logs.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    attempt = json.dumps(_event(0, "model.attempt", provider="scripted", model="worker"))
    for child in "abc":
        (root / child).mkdir()
        (root / child / path.name).write_text(attempt + "\n")
    return journals.measure(path, "text", "scripted/worker")


def test_under_two_excludes_full_intervals_and_counts_each_waiting_task(
    tmp_path: pathlib.Path,
) -> None:
    row = _measure(tmp_path, _journal())
    assert row["under_two_s"] == 9  # [1,4), [8,9), [11,16)
    assert row["wait_by_task_s"] == dict(
        zip((f"{i:04}" for i in range(2, 8)), [2, 3, 6, 6, 6, 9], strict=True)
    )
    assert row["started_twice"] == 1 and row["never_started"] == 4
    assert row["all_accepted"] is False
    assert (row["turns"], row["stuck_notices"], row["off_arm_calls"]) == (4, 1, 1)
    assert (row["input_tokens"], row["output_tokens"]) == (12, 4)
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (6, 2)
    assert row["stop_reason"] == "max_turns" and row["issues"] == []


@pytest.mark.parametrize(
    "missing", ["ts", "detail", "status", "input_tokens", "cache_write_tokens"]
)
def test_missing_fields_leave_only_their_measures_unknown(
    tmp_path: pathlib.Path, missing: str
) -> None:
    events = _journal()
    if missing in ("ts", "detail"):
        del next(e for e in events if e.get("activity") == "started")[missing]
    elif missing == "status":
        del next(e for e in events if e["type"] == "graph.update")["nodes"]["0007"][missing]
    else:
        del next(e for e in events if e["type"] == "model.attempt")[missing]
    row = _measure(tmp_path, events)
    assert row["turns"] == 4 and row["output_tokens"] == 4
    if missing in ("ts", "detail"):
        assert row["under_two_s"] is None and row["wait_by_task_s"] is None
        assert row["never_started"] == (None if missing == "detail" else 4)
    elif missing == "status":
        assert row["all_accepted"] is None
    elif missing == "input_tokens":
        assert row["input_tokens"] is None
    else:
        assert row["cache_write_tokens"] is None and row["input_tokens"] == 12
        return
    assert row["issues"]


def test_a_child_on_another_route_or_without_a_journal_is_an_issue(tmp_path: pathlib.Path) -> None:
    """A child's route is read from its own journal; an unreadable one costs no parent measure."""
    from bench.handoff import journals

    _measure(tmp_path, _journal())
    other = _event(0, "model.attempt", provider="scripted", model="alternate")
    (tmp_path / "a" / "logs.jsonl").write_text(json.dumps(other) + "\n")
    (tmp_path / "b" / "logs.jsonl").unlink()
    row = journals.measure(tmp_path / "handoff" / "logs.jsonl", "text", "scripted/worker")
    used, unknown = row["issues"]
    assert used == "child a used scripted/alternate"
    assert unknown.startswith("child b route unknown: ")
    assert row["under_two_s"] == 9 and row["all_accepted"] is False


@pytest.mark.parametrize("missing", ["session.stop", "task.queued", "end", "logs.jsonl"])
def test_incomplete_runs_do_not_suggest_an_empty_queue(
    tmp_path: pathlib.Path, missing: str
) -> None:
    from bench.handoff import journals

    if missing == "logs.jsonl":
        row = journals.measure(tmp_path / missing, "text", "scripted/worker")
    else:
        events = [e for e in _journal() if e["type"] != missing]
        if missing == "end":
            events = [e for e in events if e.get("activity") != "finished"]
        row = _measure(tmp_path, events)
    assert row["under_two_s"] is None and row["issues"]
    if missing != "end":
        assert row["all_accepted"] is None and row["never_started"] is None
