# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The manifest's gate pin: who writes it, and what keeps it true.

Every viewer, the baseline check and the next execution read the gate from here.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import event_log
from agent6.app import reporter as app_reporter
from agent6.app import stamps
from agent6.config import Config
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest


def _layout(tmp_path: pathlib.Path) -> sessions_layout.SessionLayout:
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="brave-elk-BBBBBB")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),
    )
    return layout


def _sink(tmp_path: pathlib.Path) -> event_log.EventSink:
    return event_log.EventSink(tmp_path / "logs.jsonl")


def _quiet() -> tuple[app_reporter.Reporter, list[str]]:
    said: list[str] = []
    return app_reporter.Reporter(out=said.append, err=said.append), said


def test_a_gate_adopted_mid_execution_re_pins(tmp_path: pathlib.Path) -> None:
    """A resumed execution that adopts a gate re-stamps the manifest, as a fresh run does."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    stamps.pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    assert sessions_manifest.read_manifest(layout.session_dir).harness.verify_command == ()

    events.emit("loop.verify_inferred", command=["pytest", "-q"], source="agents_md", adopted_at=3)

    pinned = sessions_manifest.read_manifest(layout.session_dir).harness
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_an_un_adopted_gate_re_pins_gateless(tmp_path: pathlib.Path) -> None:
    """The un-adopt rides the same event with an empty command; the manifest reads gateless."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    stamps.pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    events.emit("loop.verify_inferred", command=["pytest", "-q"], source="agents_md", adopted_at=3)
    events.emit("loop.verify_inferred", command=[], source="unadopted", adopted_at=5)
    pinned = sessions_manifest.read_manifest(layout.session_dir).harness
    assert pinned.verify_command == () and pinned.verify_origin == "unadopted"


def test_a_preflight_inference_is_not_an_adoption(tmp_path: pathlib.Path) -> None:
    """The event at run start carries no `adopted_at`, so it does not relabel a configured gate."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, _said = _quiet()
    stamps.pin_gate(
        layout.session_dir, ("make", "check"), "configured", events=events, reporter=reporter
    )
    events.emit("loop.verify_inferred", command=["pytest"], source="repo_signals")
    pinned = sessions_manifest.read_manifest(layout.session_dir).harness
    assert pinned.verify_command == ("make", "check")
    assert pinned.verify_origin == "configured"


def test_a_pin_that_cannot_be_written_is_reported(tmp_path: pathlib.Path) -> None:
    """The re-pin's failure is reported although EventSink swallows listener exceptions."""
    layout = _layout(tmp_path)
    events = _sink(tmp_path)
    reporter, said = _quiet()
    stamps.pin_gate(layout.session_dir, (), "", events=events, reporter=reporter)
    layout.manifest_path.unlink()
    events.emit("loop.verify_inferred", command=["pytest"], source="agents_md", adopted_at=1)
    assert any("could not record this run's verify gate" in line for line in said)


def test_a_fork_inherits_the_gate_its_source_was_judged_by(tmp_path: pathlib.Path) -> None:
    """A fork inherits the source's pinned gate, not the current config's."""
    dst = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    dst.ensure()
    stamps.write_session_manifest(
        dst,
        session_id=dst.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # no verify_command configured, as the source had none
        gate=(("pytest", "-q"), "adopted"),
    )
    pinned = sessions_manifest.read_manifest(dst.session_dir).harness
    assert pinned.verify_command == ("pytest", "-q")
    assert pinned.verify_origin == "adopted"


def test_nothing_runs_a_second_gate_at_the_end_of_a_run(tmp_path: pathlib.Path) -> None:
    """The gate is pinned at the run's start and read by every later surface; no teardown gate."""
    from agent6.app import finalize as finalize_mod
    from agent6.app import resume, run

    for module in (finalize_mod, run, resume):
        src = pathlib.Path(module.__file__ or "").read_text(encoding="utf-8")
        assert "gate_on_base" not in src, f"{module.__name__} still runs a second gate"


def test_a_red_gate_nobody_checked_says_so_and_names_the_check(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end block answers whether verify passed from the run's last verify, or says so."""
    import json

    from agent6 import budget
    from agent6.app import finalize
    from agent6.harness import _snapshot

    rd = tmp_path / "sessions" / "runs" / "r1"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": False}) + "\n",
        encoding="utf-8",
    )
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "r1",
                "mode": "run",
                "base_sha": "a" * 40,
                "harness": {"verify_command": ["uv", "run", "pytest"]},
            }
        ),
        encoding="utf-8",
    )
    finalize.print_session_end(
        _snapshot.SessionResult(
            completed=True,
            reason="finish_session",
            summary="s",
            iterations=1,
            tool_calls=1,
            verified="failed",
        ),
        layout=sessions_layout.SessionLayout(state_dir=tmp_path, session_id="r1"),
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=app_reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "nothing checked it before this run started" in out
    assert "uv run pytest" in out


def test_a_run_records_the_isolation_it_actually_ran_under(tmp_path: pathlib.Path) -> None:
    """The manifest stamps the resolved level: `auto` says nothing about the run's confinement."""
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="t",
        base_sha="0" * 40,
        base_branch="main",
        run_branch=None,
        cfg=Config(),  # sandbox.isolation defaults to "auto"
        isolation="hardened",
    )
    assert sessions_manifest.read_manifest(layout.session_dir).policy.isolation == "hardened"


def test_an_empty_gate_never_carries_an_origin(tmp_path: pathlib.Path) -> None:
    """`configured` beside `()` is self-contradictory on disk; the next execution reads it."""
    layout = _layout(tmp_path)
    reporter, _said = _quiet()
    stamps.pin_gate(layout.session_dir, (), "", events=_sink(tmp_path), reporter=reporter)
    pinned = sessions_manifest.read_manifest(layout.session_dir).harness
    assert (pinned.verify_command, pinned.verify_origin) == ((), "")
