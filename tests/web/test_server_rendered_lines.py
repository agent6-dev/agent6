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
from agent6.viewmodel import format, listing, wire
from agent6.viewmodel.session import state

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


def _usd_text(usd: float, *, partial: bool, execution_usd: float, usd_cap: float | None) -> str:
    cell = format.format_usage_cell(usd, partial=partial)
    return format.budget_usd_text(
        usd, execution_usd=execution_usd, execution_partial=partial, usd_cap=usd_cap, cell=cell
    )


def test_the_budget_line_is_rendered_once() -> None:
    assert _usd_text(0.42, partial=False, execution_usd=0.42, usd_cap=None) == "$0.42"
    assert _usd_text(0.42, partial=True, execution_usd=0.42, usd_cap=-1) == "~$0.42 (unlimited)"
    assert _usd_text(0.42, partial=False, execution_usd=0.42, usd_cap=1.0) == "$0.42 / $1.00"
    # A plan-metered run's figure is the listings' cell, the cap text unchanged.
    assert format.budget_usd_text(
        0.0, execution_usd=0.0, execution_partial=False, usd_cap=1.0, cell="2.5pt"
    ) == ("2.5pt / $1.00")
    # Earlier executions spent: this execution's figure follows the session's.
    assert _usd_text(0.42, partial=False, execution_usd=0.32, usd_cap=1.0) == (
        "$0.42 · execution $0.32 / $1.00"
    )
    # A refused ledger (cap 0) and an unknown cap (None) both show the bare figure.
    assert _usd_text(0.42, partial=False, execution_usd=0.42, usd_cap=0.0) == "$0.42"


@pytest.mark.parametrize(
    ("spent", "partial", "usd_text"),
    [
        (0.0, False, "~$0.10 · execution $0 / $1.00"),
        (0.05, False, "~$0.15 · execution $0.05 / $1.00"),
        (0.32, True, "~$0.42 · execution ~$0.32 / $1.00"),
    ],
)
def test_the_execution_figure_carries_its_own_lower_bound_mark(
    tmp_path: pathlib.Path, spent: float, partial: bool, usd_text: str
) -> None:
    """A priced execution after an unpriced one read `execution $?` for its `$0`, `~` for its spend.

    The session's figure stays a lower bound; the execution's says how this execution was priced.
    """
    resumed = _run(
        tmp_path,
        "resumed",
        [
            {"type": "session.start", "mode": "run", "user_task": "x"},
            {"type": "budget.update", "usd_total": 0.1, "usd_partial": True, "usd_cap": 1.0},
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
            {"type": "session.resume", "mode": "run"},
            {"type": "budget.update", "usd_total": spent, "usd_partial": partial, "usd_cap": 1.0},
        ],
    )
    assert wire.session_snapshot(resumed)["budget"]["usd_text"] == usd_text


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
    assert listing.summary_row(listing.summarize_session_dir(spent))["usage"] == "$0.16"
    assert listing.summary_row(listing.summarize_session_dir(clean))["usage"] == "$0"
    assert (
        listing.summary_row(listing.summarize_session_dir(clean), winner=True)["id_cell"]
        == "clean ★"
    )
    assert listing.summary_row(listing.summarize_session_dir(clean))["id_cell"] == "clean"
    assert wire.session_snapshot(spent)["budget"]["usd_text"] == "$0.16 / $1.00"

    (spent / "manifest.json").write_text(
        json.dumps({"placement": {"rank": 1, "of": 2, "winner": True, "group": "fan"}}),
        encoding="utf-8",
    )
    owner = _run(tmp_path, "fan", [])
    (owner / "manifest.json").write_text(
        json.dumps(
            {
                "comparison": {
                    "group": "fan",
                    "ranked_by": "judge",
                }
            }
        ),
        encoding="utf-8",
    )
    assert wire.session_snapshot(spent)["placement"]["line"] == "rank 1/2 · winner · judge"


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
    assert snap["spend"]["text"] == "$0"
    marks = {st["name"]: st["mark"] for st in snap["states"]}
    assert marks["done"] == "▸" and marks["route"] == "·"
