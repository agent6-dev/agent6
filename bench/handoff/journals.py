"""Measure a single handoff from its parent journal."""

from __future__ import annotations

import collections
import datetime
import json
import pathlib
from typing import Any

from bench.handoff import job

from agent6.graph import order

type Event = dict[str, Any]


class MissingEvidence(Exception):
    """A journal cannot establish a measurement."""


def _timestamp(event: Event) -> datetime.datetime:
    try:
        stamp = datetime.datetime.fromisoformat(event["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingEvidence(f"missing or invalid {event['type']} timestamp") from exc
    if stamp.tzinfo is None:
        raise MissingEvidence(f"no timezone on {event['type']} timestamp")
    return stamp


def _wait(events: list[Event], ids: dict[str, str]) -> tuple[dict[str, float], float]:
    """Integrate per-task and wall time only while work waits below the ceiling."""
    elapsed = dict.fromkeys(ids.values(), 0.0)
    wall = 0.0
    waiting: set[str] = set()
    running: set[str] = set()
    ticks = sorted(((_timestamp(e), i, e) for i, e in enumerate(events)))
    previous = ticks[0][0]
    for now, _, e in ticks:
        seconds = (now - previous).total_seconds()
        previous = now
        if waiting and len(running) < job.CHILDREN:
            wall += seconds
            for task_id in waiting:
                elapsed[task_id] += seconds
        if e["type"] == "task.queued":
            waiting.add(e["task_id"])
        elif e["type"] == "graph.update":
            for task_id in waiting.copy():
                status = e["nodes"].get(task_id, {}).get("status")
                if status is None:
                    raise MissingEvidence("missing queued task status during wait")
                if status not in order.OPEN_STATUSES:
                    waiting.remove(task_id)
        elif e["type"] == "parallel.child_activity":
            child, activity = e["child"], e["activity"]
            if activity == "started":
                if child in running:
                    raise MissingEvidence("child started again without an end")
                running.add(child)
                waiting.discard(ids[e["detail"]])
            elif activity != "steered":
                # A failed launch may report failure without ever starting.
                if activity != "failed" and child not in running:
                    raise MissingEvidence("child ended without a start")
                running.discard(child)
    if running or ticks[-1][2]["type"] != "session.stop":
        raise MissingEvidence("incomplete child or parent time interval")
    return {k: round(s, 6) for k, s in elapsed.items()}, round(wall, 6)


def _queue(rows: list[Event], result: Event, arm: str) -> None:
    queued = [e for e in rows if e["type"] == "task.queued"]
    ids = {e["title"]: e["task_id"] for e in queued}
    if tuple(e["title"] for e in queued) != job.TITLES or len(set(ids.values())) != len(job.TITLES):
        raise MissingEvidence("expected the six operator tasks in order")
    # tool.call keeps a refused call's arguments: a tuple compares a non-string without hashing it.
    result["off_arm_calls"] = sum(
        not (
            args.get("task_id") in ids.values() and not args.get("task")
            if arm == "id"
            else args.get("task") in job.TITLES and not args.get("task_id")
        )
        for e in rows
        if e["type"] == "tool.call" and e["name"] == "parallel"
        for args in [e["args_preview"]]
    )
    last = next((e["nodes"] for e in reversed(rows) if e["type"] == "graph.update"), {})
    statuses = {i: last.get(i, {}).get("status") for i in ids.values()}
    if None in statuses.values():
        result["issues"].append("no final status for every original task")
    else:
        result["all_accepted"] = all(v == "passed" for v in statuses.values())
    counts = dict.fromkeys(ids.values(), 0)
    for e in rows:
        if e["type"] == "parallel.child_activity" and e["activity"] == "started":
            if (title := e.get("detail")) not in ids:
                raise MissingEvidence("child start does not name one exact queued task")
            counts[ids[title]] += 1
    result.update(
        starts_by_task=counts,
        started_twice=sum(n >= 2 for n in counts.values()),
        never_started=sum(n == 0 for n in counts.values()),
    )
    events = [e for e in rows if e["type"] in _TIMELINE]
    result["wait_by_task_s"], result["under_two_s"] = _wait(events, ids)


_TIMELINE = {"graph.update", "parallel.child_activity", "session.stop", "task.queued"}
_KINDS = _TIMELINE | {
    "model.attempt",
    "model.call",
    "model.result",
    "nudge.task_stuck",
    "session.start",
    "tool.call",
}


def _usage(rows: list[Event], result: Event) -> None:
    attempts = [e for e in rows if e["type"] == "model.attempt"]
    calls = {e["model_call_id"] for e in rows if e["type"] == "model.call"}
    ended = {e["model_call_id"] for e in rows if e["type"] == "model.result"}
    if not attempts or calls != ended or calls != {a["model_call_id"] for a in attempts}:
        result["issues"].append("missing parent model call or usage")
        return
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        counts = [v for a in attempts if type(v := a.get(key)) is int and v >= 0]
        if len(counts) == len(attempts):
            result[key] = sum(counts)
        elif key in ("input_tokens", "output_tokens"):
            result["issues"].append(f"unreported parent {key}")


def measure(path: pathlib.Path, arm: str) -> dict[str, Any]:
    """Return full-run counts, with nulls and reasons for missing evidence."""
    result: dict[str, Any] = dict.fromkeys(
        (
            "stop_reason",
            "turns",
            "stuck_notices",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "all_accepted",
            "off_arm_calls",
            "starts_by_task",
            "started_twice",
            "never_started",
            "wait_by_task_s",
            "under_two_s",
        )
    )
    result["issues"] = []
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [e for line in stream if (e := json.loads(line))["type"] in _KINDS]
        kinds = collections.Counter(e["type"] for e in rows)
        if kinds["session.start"] != 1 or kinds["session.stop"] != 1:
            raise MissingEvidence("expected one parent start and stop")
        stop = next(e for e in rows if e["type"] == "session.stop")
        result.update(
            stop_reason=stop["reason"], turns=stop["turns"], stuck_notices=kinds["nudge.task_stuck"]
        )
        _usage(rows, result)
        _queue(rows, result, arm)
    except (MissingEvidence, OSError, ValueError, KeyError, TypeError) as exc:
        result["issues"].append(str(exc))
    return result
