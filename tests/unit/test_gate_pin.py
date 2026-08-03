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


def test_the_baseline_gates_where_a_fork_actually_started(tmp_path: Path) -> None:
    """A fork inherits its parent's base_sha, but it STARTS at the sha it was
    cut from. Gating the parent's base told the operator "the gate passes on
    the base commit, so this run broke it" about breakage the fork inherited."""
    import json

    from agent6.app import finalize
    from agent6.app.baseline import Baseline
    from agent6.workflows._run_state import RunResult

    rd = tmp_path / "runs" / "fork1"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "run.end", "reason": "finish_run", "all_passed": False}) + "\n",
        encoding="utf-8",
    )
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "run_id": "fork1",
                "mode": "run",
                "base_sha": "a" * 40,
                "forked_from_sha": "b" * 40,
                "workflow": {"verify_command": ["true"], "verify_origin": "configured"},
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def _capture(_cwd: object, base: str, **_k: object) -> Baseline:
        seen.append(base)
        return Baseline(ran=True, returncode=0, detail="")

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    monkey.setattr(finalize, "gate_on_base", _capture)
    try:
        finalize.print_baseline(
            RunResult(
                completed=True,
                reason="finish_run",
                summary="s",
                iterations=1,
                tool_calls=1,
                verified="failed",
            ),
            layout=RunLayout(state_dir=tmp_path, run_id="fork1"),
            cfg=Config(),
            isolation="none",
            reporter=Reporter(out=lambda _m: None, err=lambda _m: None),
        )
    finally:
        monkey.undo()
    assert seen == ["b" * 40], "the fork was gated against its PARENT's base"


def test_a_run_records_the_isolation_it_actually_ran_under(tmp_path: Path) -> None:
    """`auto` degrades. A manifest stamping the knob told every surface "auto",
    which says nothing about whether the run was confined."""
    layout = RunLayout(state_dir=tmp_path, run_id="quiet-fox-AAAAAA")
    layout.ensure()
    write_run_manifest(
        layout,
        run_id=layout.run_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # sandbox.isolation defaults to "auto"
        isolation="hardened",
    )
    assert read_manifest(layout.run_dir).policy.isolation == "hardened"


def test_an_empty_gate_never_carries_an_origin(tmp_path: Path) -> None:
    """`configured` beside `()` is self-contradictory on disk, and the next
    leg reads that origin back."""
    layout = _layout(tmp_path)
    reporter, _said = _quiet()
    pin_gate(layout.run_dir, (), "", events=_sink(tmp_path), reporter=reporter)
    pinned = read_manifest(layout.run_dir).workflow
    assert (pinned.verify_command, pinned.verify_origin) == ((), "")
