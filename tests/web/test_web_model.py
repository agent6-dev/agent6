# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the pure web payload builders (no HTTP)."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.web import model


def _run(cwd: Path, run_id: str, events: list[dict[str, object]]) -> Path:
    d = model.runs_root(cwd) / run_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_run_summary_captures_cost_and_status(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "the task"},
            {"type": "budget.update", "usd_total": 0.0123},
            {"type": "run.end", "all_passed": True},
        ],
    )
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["mode"] == "run"
    assert s["task"] == "the task"
    assert s["status"] == "passed"
    assert s["usd"] == 0.0123


def test_run_summary_survives_torn_utf8_tail(tmp_path: Path) -> None:
    # A live writer can leave the log's last line torn mid multibyte UTF-8
    # sequence; the hub summary must fold the complete lines, not raise.
    d = model.runs_root(tmp_path) / "torn"
    d.mkdir(parents=True)
    full = json.dumps({"type": "role.text_delta", "text": "café"}, ensure_ascii=False).encode()
    cut = full.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é
    head = json.dumps({"type": "run.start", "mode": "run", "user_task": "torn tail"}).encode()
    (d / "logs.jsonl").write_bytes(head + b"\n" + full[:cut])
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["task"] == "torn tail"


def test_transcript_payload_empty_when_no_transcripts(tmp_path: Path) -> None:
    d = _run(tmp_path, "r2", [{"type": "run.start", "user_task": "x"}])
    payload = model.transcript_payload(d)
    assert payload == {"run_id": "r2", "turns": []}


def test_reasoning_snapshot_empty_without_state_log(tmp_path: Path) -> None:
    # A machine dir with no states/ subtree has no agent reasoning to fold.
    md = model.machines_root(tmp_path) / "m1"
    md.mkdir(parents=True)
    assert model.machine_reasoning_snapshot(md) == {}


def test_run_dir_for_rejects_traversal(tmp_path: Path) -> None:
    _run(tmp_path, "good-run", [{"type": "run.start"}])
    assert model.run_dir_for(tmp_path, "good-run") is not None
    for bad in ("..", ".", "", "../good-run", "a/b", "..\\x"):
        assert model.run_dir_for(tmp_path, bad) is None


def test_machine_dir_for_rejects_traversal(tmp_path: Path) -> None:
    (model.machines_root(tmp_path) / "m1").mkdir(parents=True)
    assert model.machine_dir_for(tmp_path, "m1") is not None
    for bad in ("..", "../m1", "a/b", ""):
        assert model.machine_dir_for(tmp_path, bad) is None


def test_hub_payload_shape(tmp_path: Path) -> None:
    _run(tmp_path, "r3", [{"type": "run.start", "mode": "plan"}])
    hub = model.hub_payload(tmp_path)
    assert [r["id"] for r in hub["runs"]] == ["r3"]
    assert hub["machines"] == []
