"""A resume reads the route and time of the same ordinary request attempt."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import kinds
from agent6.harness.compaction import cold_start


def request(**fields: Any) -> dict[str, Any]:
    return {
        "ts": "2026-02-02T10:00:00+00:00",
        "type": "provider.request",
        "actor": "worker",
        "provider": "p",
        "model": "m",
        "target": "inference",
        "execution_id": "previous",
        "model_call_id": "ordinary",
        "request_id": "attempt",
        **fields,
    }


@pytest.mark.parametrize(
    "limit,now,cold", [(-1, 1000, False), (10, 109.999, False), (10, 110, True)]
)
def test_last_attempt_sets_the_idle_boundary(
    tmp_path: pathlib.Path, limit: float, now: float, cold: bool
) -> None:
    log = tmp_path / "logs.jsonl"
    rows = [
        request(ts="1970-01-01T00:00:00Z", model="other"),
        request(ts="1970-01-01T01:01:40+01:00"),
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trigger = cold_start.detect(
        log, actor="worker", route=kinds.ModelRoute("p", "m"), limit=limit, now=now
    )
    assert (trigger is not None) is cold
    if trigger:
        assert (trigger.reason, trigger.gap_s) == ("idle", 10)


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"ts": "unknown"}, None),
        ({"ts": "unknown", "model": "new"}, "route_changed"),
        ({"provider": "", "ts": "1970-01-01T00:00:00Z"}, "idle"),
    ],
)
def test_unknown_facts_do_not_supply_a_trigger(
    tmp_path: pathlib.Path, fields: dict[str, Any], reason: str | None
) -> None:
    log = tmp_path / "logs.jsonl"
    log.write_text(json.dumps(request(**fields)) + "\n")
    trigger = cold_start.detect(
        log, actor="worker", route=kinds.ModelRoute("p", "m"), limit=10, now=20
    )
    assert (trigger.reason if trigger else None) == reason


def test_a_process_reply_confirms_the_opening_time_not_the_reply_time(
    tmp_path: pathlib.Path,
) -> None:
    log = tmp_path / "logs.jsonl"
    rows = [
        request(type="model.call", ts="1970-01-01T00:00:10Z"),
        request(type="model.attempt", ts="1970-01-01T00:00:30Z", ok=True, request_id=None),
        {"type": "compact.summarise.call", "model_call_id": "summary"},
        request(type="model.call", model_call_id="summary", model="other"),
        request(type="model.attempt", model_call_id="summary", ok=True, request_id=None),
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trigger = cold_start.detect(
        log, actor="worker", route=kinds.ModelRoute("p", "m"), limit=20, now=30
    )
    assert trigger is not None and (trigger.reason, trigger.gap_s) == ("idle", 20)
