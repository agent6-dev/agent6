"""Count journal evidence from one resumed compaction trial."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Sequence
from typing import Any

from bench import compaction_arms

type Row = dict[str, Any]


def _read_paths(rows: Sequence[Row]) -> Iterable[tuple[int, int | None, str | None]]:
    """Yield completed read paths in dispatch order, with their call and turn numbers."""
    results = {
        (row.get("execution_id"), row.get("call_id")): row
        for row in rows
        if row["type"] == "tool.result" and row.get("name") == "read_file"
    }
    calls = (row for row in rows if row["type"] == "tool.call")
    for n, row in enumerate(calls, 1):
        if row.get("name") != "read_file":
            continue
        key = row.get("execution_id"), row.get("call_id")
        ok = results.get(key, {}).get("ok") if None not in key else None
        if ok is False:
            continue
        raw = row.get("args_preview", {}).get("path")
        path = (
            posixpath.normpath(raw)
            if isinstance(raw, str) and raw and not re.search(r"… \(\d+ chars\)\Z", raw)
            else ""
        )
        relative = path and not path.startswith(("/", "../")) and path != ".."
        yield n, row.get("turn"), path if relative and ok is True else None


def _rereads(
    prefix: Sequence[Row], rows: Sequence[Row], start: int
) -> tuple[int | None, int | None]:
    """Count repeated read dispatches inside each window, including within-window repeats."""
    seen = {path for _, _, path in _read_paths(prefix)}
    six = ten = 0
    six_known = ten_known = True
    for n, turn, path in _read_paths(rows):
        uncertain = path is None or (None in seen and path not in seen)
        repeat = path is not None and path in seen
        if n <= 6:
            six += repeat
            six_known &= not uncertain
        if turn is None or start <= turn < start + 10:
            ten += repeat
            ten_known &= not uncertain and turn is not None
        seen.add(path)
    return six if six_known else None, ten if ten_known else None


def _input(row: Row, *, uncached: bool = False) -> int | None:
    fresh = row.get("input_tokens")
    read, write = row.get("cache_read_tokens"), row.get("cache_write_tokens")
    if fresh is None or (uncached and read is None and write is None):
        return None
    return fresh + (write or 0) + (0 if uncached else read or 0)


def _usage(rows: Sequence[Row], requests: Sequence[Row], worker: Row) -> Row:
    attempts = [r for r in rows if r["type"] == "model.attempt"]
    by_id = {r["request_id"]: r for r in attempts if r.get("request_id")}
    first, first_worker = (by_id.get(r["request_id"], {}) for r in (requests[0], worker))
    # A request without a settlement or a retry without usage has an unknown bill.
    complete = all(r["request_id"] in by_id for r in requests)
    uncached = [_input(r, uncached=True) for r in attempts]
    totals = [(_input(r), r.get("output_tokens")) for r in attempts]
    return {
        "first_request_input_tokens": _input(first),
        "first_request_cache_read_tokens": first.get("cache_read_tokens"),
        "first_worker_input_tokens": _input(first_worker),
        "first_worker_cache_read_tokens": first_worker.get("cache_read_tokens"),
        "uncached_input_tokens": sum(v for v in uncached if v is not None)
        if complete and None not in uncached
        else None,
        "total_tokens": sum(i + o for i, o in totals if i is not None and o is not None)
        if complete and all(i is not None and o is not None for i, o in totals)
        else None,
    }


def measure(prefix: Sequence[Row], rows: Sequence[Row], arm: str, recall: str | None) -> Row:
    """Measure one completed execution after checking its first ordinary request boundary.

    Args:
        prefix: The frozen journal before the restart turn.
        rows: Only rows appended by the trial.
        arm: The requested keep or cold arm.
        recall: The expected finish line, or None without a labelled steer.

    Returns:
        Observed outcomes, with None for unavailable evidence.

    Raises:
        ValueError: The run, treatment or stopping boundary cannot be identified.
    """
    resumes = [r for r in rows if r["type"] == "session.resume"]
    if len(resumes) != 1 or not (execution := resumes[0].get("execution_id")):
        raise ValueError("expected one resumed execution")
    stops = [r for r in rows if r["type"] == "session.stop" and r.get("execution_id") == execution]
    if len(stops) != 1 or rows.index(stops[0]) < rows.index(resumes[0]):
        raise ValueError("no unique stop for the resumed execution")
    stop = stops[0]
    start = resumes[0]["turn"]
    rows = [
        r
        for r in rows[rows.index(resumes[0]) : rows.index(stop) + 1]
        if r.get("execution_id") in (None, execution)
    ]
    calls = {r["model_call_id"] for r in rows if r["type"] == "compact.summarise.call"}
    requests: list[Row] = []
    ordinary: list[Row] = []
    turns: set[int] = set()
    turn: int | None = None
    for row in rows:
        if row["type"] == "budget.turn":
            turn = row["turn"]
        if row["type"] == "provider.request" and row.get("target") == "inference":
            requests.append(row)
            if row.get("actor") == "worker" and row["model_call_id"] not in calls:
                ordinary.append(row)
                if turn is None:
                    raise ValueError("ordinary request has no worker turn")
                turns.add(turn)
    if not ordinary:
        raise ValueError("no ordinary worker request")
    boundary = rows.index(ordinary[0])
    before = rows[:boundary]
    if arm in compaction_arms.COLD:
        mode = compaction_arms.COLD[arm].cold
        if not any(
            r["type"] == "compact.cold_start" and r.get("mode") == mode and r.get("action") == mode
            for r in before
        ):
            raise ValueError(f"cold start did not apply {mode} before the first worker request")
    elif not any(
        r["type"] == "compact.summarise.done"
        and r.get("writer") == "worker"
        and r.get("reason") == "requested"
        for r in before
    ):
        raise ValueError("no accepted requested worker restart before the first worker request")
    six, ten = _rereads(prefix, rows[boundary:], start)
    verification = stop.get("verification") if stop.get("scoped") is False else None
    finished = stop.get("reason") == "finish_session"
    answer = stop.get("summary")
    return _usage(rows, requests, ordinary[0]) | {
        "turns": len(turns),
        "rereads_6_calls": six,
        "rereads_10_turns": ten,
        "second_restart": any(r["type"] == "compact.summarise.done" for r in rows[boundary:]),
        "stop_reason": stop.get("reason"),
        "verification": stop.get("verification"),
        "passed": finished and verification == "passed"
        if verification in ("passed", "failed")
        else None,
        "exact_recall": recall in answer.splitlines()
        if finished and recall and isinstance(answer, str)
        else None,
    }
