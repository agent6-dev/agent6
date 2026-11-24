from __future__ import annotations

from typing import Any

import pytest
from bench import compaction_counts as counts

from agent6.tools import result_format


def _read(call: int, path: str, turn: int, *, ok: bool = True) -> list[dict[str, Any]]:
    key = {"call_id": call, "name": "read_file", "turn": turn, "execution_id": "e"}
    return [
        dict(type="tool.call", **key, args_preview=result_format.truncate_args({"path": path})),
        dict(type="tool.result", **key, ok=ok),
    ]


def _request(role: str, num: int, fresh: int, cached: int | None) -> list[dict[str, Any]]:
    key = dict(model_call_id=f"m{num}", request_id=f"r{num}", actor=role, execution_id="e")
    return [
        dict(type="provider.request", target="inference", **key),
        dict(
            type="model.attempt",
            attempt=1,
            input_tokens=fresh,
            cache_read_tokens=cached,
            output_tokens=5,
            ok=True,
            **key,
        ),
        dict(type="model.result", input_tokens=123456, **key),
    ]


def test_counts_only_resumed_attempts_and_completed_reads() -> None:
    prefix = [
        *_read(1, "old", 2),
        *_read(2, "missing", 3, ok=False),
        *_request("worker", 7, 1000, 0),
    ]
    rows = [
        dict(type="session.resume", turn=12, execution_id="e"),
        dict(type="budget.turn", turn=12),
        dict(type="compact.summarize.call", model_call_id="m1"),
        *_request("helper", 1, 20, 3),
        dict(type="compact.summarize.done", writer="helper", reason="cold_start"),
        dict(type="compact.cold_start", mode="summarize", action="summarize"),
        *_request("worker", 2, 40, 2),
        *_read(1, "./old", 12),
        *_read(2, "missing", 12),
        *_read(3, "old", 12, ok=False),
        *_read(4, "new", 12),
        *_read(5, "new", 12),
        *_read(6, "fresh", 12),
        *_read(7, "old", 12),
        dict(type="budget.turn", turn=22),
        dict(type="compact.summarize.done", reason="size"),
        *_request("worker", 3, 4, 1),
        *_read(8, "old", 22),
        dict(
            type="session.stop",
            execution_id="e",
            reason="finish_session",
            verification="passed",
            scoped=False,
            summary="Done.\nAmber",
            turns=22,
        ),
    ]
    row = counts.measure(prefix, rows, "summarize", "Amber")
    assert row["first_request_input_tokens"] == 23
    assert row["first_worker_input_tokens"] == 42
    assert row["uncached_input_tokens"] == 64 and row["total_tokens"] == 85
    assert row["rereads_6_calls"] == 2 and row["rereads_10_turns"] == 3
    assert row["turns"] == 2 and row["second_restart"]
    assert row["passed"] and row["exact_recall"]
    rows[-1].update(verification="not_applicable", summary="Got Amber!")
    assert counts.measure(prefix, rows, "summarize", "Amber")["passed"] is None
    assert counts.measure(prefix, rows, "summarize", "Amber")["exact_recall"] is False


def test_unreported_cache_split_and_failed_treatment_are_not_zero() -> None:
    rows = [
        dict(type="session.resume", turn=7, execution_id="e"),
        dict(type="budget.turn", turn=7),
        dict(type="compact.cold_start", action="drop", mode="drop"),
        *_request("worker", 1, 10, None),
        dict(type="session.stop", reason="max_turns", execution_id="e", summary="Amber"),
    ]
    row = counts.measure([], rows, "drop", "Amber")
    assert row["exact_recall"] is None and row["uncached_input_tokens"] is None
    with pytest.raises(ValueError, match="cold"):
        counts.measure([], rows, "truncate", None)
    with pytest.raises(ValueError, match="restart"):
        counts.measure([], rows, "baseline", None)
    with pytest.raises(ValueError, match="stop"):
        counts.measure([], rows[:-1], "drop", None)


def test_clipped_read_names_make_only_the_affected_window_unknown() -> None:
    """Two different files can have the same journal preview, even its length suffix."""
    common = "p" * 200
    prefix = [*_read(1, common, 2), *_read(2, common + "a", 3)]
    reads = [r for n in range(1, 7) for r in _read(n, common, 7)]
    clipped = _read(7, common + "b", 16)
    assert prefix[-2]["args_preview"] == clipped[0]["args_preview"]
    rows = [
        dict(type="session.resume", turn=7, execution_id="e"),
        dict(type="budget.turn", turn=7),
        dict(type="compact.cold_start", action="resend", mode="resend"),
        *_request("worker", 1, 10, 0),
        *reads,
        *clipped,
        dict(type="session.stop", reason="finish_session", execution_id="e"),
    ]
    row = counts.measure(prefix, rows, "resend", None)
    assert row["rereads_6_calls"] == 6 and row["rereads_10_turns"] is None
    for r in clipped:
        r["turn"] = 17
    assert counts.measure(prefix, rows, "resend", None)["rereads_10_turns"] == 6


def test_clipped_seed_read_cannot_prove_a_new_path_is_fresh() -> None:
    """The missing part of an earlier path can name the file read on resume."""
    prefix = _read(1, "./" * 101 + "file", 3)
    rows = [
        dict(type="session.resume", turn=7, execution_id="e"),
        dict(type="budget.turn", turn=7),
        dict(type="compact.cold_start", action="drop", mode="drop"),
        *_request("worker", 1, 10, 0),
        *_read(1, "file", 7),
        dict(type="session.stop", reason="finish_session", execution_id="e"),
    ]
    row = counts.measure(prefix, rows, "drop", None)
    assert row["rereads_6_calls"] is None and row["rereads_10_turns"] is None
