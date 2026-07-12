# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` renders readable windowed hits, not raw JSON lines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.ui.cli.history_cmds import (
    _parse_rg_matches,  # pyright: ignore[reportPrivateUsage]
    _render_history_hits,  # pyright: ignore[reportPrivateUsage]
    _run_id_from_path,  # pyright: ignore[reportPrivateUsage]
    _window,  # pyright: ignore[reportPrivateUsage]
)


def _rg_match(path: str, line: str, col: int) -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
                "submatches": [{"start": col, "end": col + 1}],
            },
        }
    )


def test_window_clips_a_huge_line_around_the_match() -> None:
    line = "x" * 500 + "NEEDLE" + "y" * 500
    out = _window(line, 500)
    assert "NEEDLE" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) < 200  # not the whole 1000+ char line


def test_window_collapses_json_escaped_newlines() -> None:
    assert _window("the\\nfirst message here", 0) == "the first message here"


def test_run_id_from_path_finds_the_run_dir_child() -> None:
    assert _run_id_from_path(Path("/s/runs/deep-poppy-AB/logs.jsonl")) == "deep-poppy-AB"
    assert _run_id_from_path(Path("/s/asks/quiet-fox-CD/transcripts/0003.json")) == "quiet-fox-CD"


def test_parse_extracts_event_type_and_time_for_logs_jsonl() -> None:
    event = {"ts": "2026-07-12T09:15:30.1Z", "type": "tool.call", "name": "grep"}
    out = _parse_rg_matches(_rg_match("/s/runs/r1/logs.jsonl", json.dumps(event), 40))
    assert len(out) == 1
    assert out[0].run_id == "r1"
    assert out[0].kind == "tool.call"
    assert out[0].when == "09:15:30"


def test_transcripts_share_one_label(capsys: pytest.CaptureFixture[str]) -> None:
    # The same snippet across cumulative transcript snapshots collapses to one
    # (xN) line, labelled "transcript", not per-file.
    lines = "\n".join(
        _rg_match(f"/s/runs/r1/transcripts/000{i}.json", '  "text": "hello NEEDLE"', 12)
        for i in (3, 5, 7)
    )
    hits = _parse_rg_matches(lines)
    assert all(h.kind == "transcript" for h in hits)
    _render_history_hits(hits, Path("/s/runs"))
    out = capsys.readouterr().out
    assert "(x3)" in out  # three identical snapshot hits collapsed
    assert out.count("hello NEEDLE") == 1
