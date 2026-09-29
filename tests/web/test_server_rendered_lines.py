# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web client shows lines the server renders; it keeps no copy of a Python rendering decision.

client.js carried a `fmtUsd` twin of `format_usd`, a task-glyph map, a `format_compare` mirror and a
`format_transition` mirror, each with a "keep in sync" comment and no pin. The cost twin had
drifted: Python's `%.4f` rounds half to even on the binary value and JS `toFixed` rounds half away,
so 0.15625 printed `$0.1562` on the CLI and `$0.1563` on the web.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import layout
from agent6.ui.cli import main
from agent6.ui.web import page
from agent6.viewmodel import format, listing, state, wire

ROUTER = """
machine = "router"
version = 1
initial = "route"

[budget]
max_transitions = 10

[states.route]
kind = "branch"
when = [{ else = true, goto = "done" }]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _run(tmp_path: pathlib.Path, name: str, events: list[dict[str, object]]) -> pathlib.Path:
    d = layout.bucket_dir(paths.state_dir(tmp_path), "runs") / name
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_the_page_carries_no_cost_formatter_and_no_glyph_map() -> None:
    assert "fmtUsd" not in page.PAGE_HTML and "toFixed(4)" not in page.PAGE_HTML
    assert "★" not in page.PAGE_HTML and "▸" not in page.PAGE_HTML


def test_the_budget_line_is_rendered_once() -> None:
    assert (
        format.budget_usd_text(0.42, partial=False, usd_cap=0.0, usd_prior_executions=0.0)
        == "$0.42"
    )
    assert format.budget_usd_text(0.42, partial=True, usd_cap=-1, usd_prior_executions=0.0) == (
        "~$0.42 (unlimited)"
    )
    assert format.budget_usd_text(0.42, partial=False, usd_cap=1.0, usd_prior_executions=0.0) == (
        "$0.42 / $1.00"
    )
    assert format.budget_usd_text(0.42, partial=False, usd_cap=1.0, usd_prior_executions=0.1) == (
        "$0.42 · execution $0.32 / $1.00"
    )


def test_a_partial_total_marks_the_execution_figure_too() -> None:
    """A resumed execution's dollar figure carries the `~` its cumulative total carries."""
    assert format.budget_usd_text(0.42, partial=True, usd_cap=1.0, usd_prior_executions=0.1) == (
        "~$0.42 · execution ~$0.32 / $1.00"
    )


def test_the_hub_row_and_the_run_view_carry_rendered_cells(tmp_path: pathlib.Path) -> None:
    spent = _run(
        tmp_path,
        "spent",
        [
            {"type": "session.start", "mode": "run", "user_task": "x"},
            {"type": "budget.update", "usd_total": 0.15625, "usd_cap": 1.0},
        ],
    )
    clean = _run(tmp_path, "clean", [{"type": "session.start", "mode": "run", "user_task": "y"}])
    assert listing.summary_row(listing.summarize_session_dir(spent))["cost"] == "$0.16"
    assert listing.summary_row(listing.summarize_session_dir(clean))["cost"] == ""
    assert (
        listing.summary_row(listing.summarize_session_dir(clean), winner=True)["id_cell"]
        == "clean ★"
    )
    assert listing.summary_row(listing.summarize_session_dir(clean))["id_cell"] == "clean"
    assert wire.session_snapshot(spent)["budget"]["usd_text"] == "$0.16 / $1.00"

    (spent / "manifest.json").write_text(
        json.dumps({"compare": {"rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}}),
        encoding="utf-8",
    )
    assert wire.session_snapshot(spent)["compare"]["line"] == "rank 1/2 · winner · judge"


def test_task_rows_carry_their_glyph() -> None:
    views = state.task_tree_views(
        {"a": {"title": "t", "status": "passed", "children": ["b"]}, "b": {"title": "u"}}, "b"
    )
    assert [(v.glyph, v.is_cursor) for v in views] == [("✓", False), ("▸", True)]


def test_machine_transitions_and_spend_arrive_rendered(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    f = tmp_path / "router.asm.toml"
    f.write_text(ROUTER, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()
    md = layout.machines_root(paths.state_dir(tmp_path)) / "router"
    snap = wire.machine_snapshot(md)
    (first, *_rest) = snap["transitions"]
    assert (
        first["line"] == f"[{first['seq']}] {first['state']} --{first['label']}--> {first['goto']}"
    )
    assert first["state"] == "route" and first["goto"] == "done"
    assert snap["spend"]["text"] == "$0.0000"
    marks = {st["name"]: st["mark"] for st in snap["states"]}
    assert marks["done"] == "▸" and marks["route"] == "·"
