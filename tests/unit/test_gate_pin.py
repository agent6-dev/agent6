# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The manifest's gate pin: who writes it, and what keeps it true.

Every viewer, the baseline check and the next leg read the gate from here, so a
pin that goes stale is a surface that lies about what judged the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.manifest import pin_gate, write_run_manifest
from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.events import EventSink
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import read_manifest


def _layout(tmp_path: Path) -> RunLayout:
    layout = RunLayout(state_dir=tmp_path, run_id="brave-elk-BBBBBB")
    layout.ensure()
    write_run_manifest(
        layout,
        run_id=layout.run_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    return layout


def _sink(tmp_path: Path) -> EventSink:
    return EventSink(tmp_path / "logs.jsonl")


def _quiet() -> tuple[Reporter, list[str]]:
    said: list[str] = []
    return Reporter(out=said.append, err=said.append), said


def test_a_gate_adopted_mid_leg_re_pins(tmp_path: Path) -> None:
    """The stamp and the re-stamp were separate wiring, present only on a fresh
    run: a RESUMED leg that adopted a gate left a manifest reading gateless
    while a gate was live."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.run_dir, (), "", events=events, reporter=reporter)
    assert read_manifest(layout.run_dir).workflow.verify_command == ()

    events.emit("loop.verify_inferred", command=["pytest", "-q"], source="agents_md", adopted_at=3)

    pinned = read_manifest(layout.run_dir).workflow
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_a_preflight_inference_is_not_an_adoption(tmp_path: Path) -> None:
    """The same event fires at run start with no `adopted_at`; re-pinning on it
    would relabel a configured gate."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.run_dir, ("make", "check"), "configured", events=events, reporter=reporter)
    events.emit("loop.verify_inferred", command=["pytest"], source="repo_signals")
    pinned = read_manifest(layout.run_dir).workflow
    assert pinned.verify_command == ("make", "check")
    assert pinned.verify_origin == "configured"


def test_a_pin_that_cannot_be_written_is_reported(tmp_path: Path) -> None:
    """EventSink swallows a listener's exceptions so a UI consumer cannot break
    the run -- which silently ate the re-pin's failure too."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, said = _quiet()
    pin_gate(layout.run_dir, (), "", events=events, reporter=reporter)
    layout.manifest_path.unlink()
    events.emit("loop.verify_inferred", command=["pytest"], source="agents_md", adopted_at=1)
    assert any("could not record this run's verify gate" in line for line in said)


def test_a_fork_inherits_the_gate_its_source_was_judged_by(tmp_path: Path) -> None:
    """Derived from the current config instead, a source whose gate was inferred
    or adopted forked to a run every surface called gateless."""
    dst = RunLayout(state_dir=tmp_path, run_id="quiet-fox-AAAAAA")
    dst.ensure()
    write_run_manifest(
        dst,
        run_id=dst.run_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # no verify_command configured, as the source had none
        gate=(("pytest", "-q"), "adopted"),
    )
    pinned = read_manifest(dst.run_dir).workflow
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_the_end_of_run_block_never_runs_a_second_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline check ran INSIDE print_run_end, which the run calls while
    still holding the repo and worker locks: a full second gate then kept the
    checkout for up to verify_timeout_s after the run visibly ended, and
    `agent6 run` refused, naming a run the operator had watched finish."""
    import json

    from agent6.app import finalize
    from agent6.budget import BudgetTracker
    from agent6.workflows._run_state import RunResult

    rd = tmp_path / "runs" / "r1"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "run.end", "reason": "finish_run", "all_passed": False}) + "\n",
        encoding="utf-8",
    )

    def _no_second_gate(*_a: object, **_k: object) -> None:
        pytest.fail("a second gate ran inside the lock scope")

    monkeypatch.setattr(finalize, "gate_on_base", _no_second_gate)
    finalize.print_run_end(
        RunResult(
            completed=True,
            reason="finish_run",
            summary="s",
            iterations=1,
            tool_calls=1,
            verified="failed",
        ),
        layout=RunLayout(state_dir=tmp_path, run_id="r1"),
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
    )
