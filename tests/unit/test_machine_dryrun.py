# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Phase 5: `machine test` dry-run — per-state synthesis + per-branch routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.machine import dry_run, load_machine
from agent6.machine._semantics import validate_finish_payload
from agent6.machine.dryrun import synthesize_record

# tool -> branch -> (agent | tool) -> terminal, with a typed capture + an enum.
DEMO = """
machine = "demo"
version = 1
initial = "scan"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
approved = { type = "bool", value = false }

[vars.code]
items = { type = "list[str]", default = [] }

[vars.agent]
verdict = { type = "review", default = { label = "low", score = 0 } }

[schemas.scan_result]
items = "list[str]"

[schemas.review]
label = { type = "str", enum = ["low", "high"] }
score = "int"

[states.scan]
kind = "tool"
command = ["scan"]
output_schema = "scan_result"
capture = { set = { items = "{{ result.items }}" } }
timeout_secs = 5
on = { ok = "check", nonzero = "stop_fail", timeout = "stop_fail" }

[states.check]
kind = "branch"
when = [
  { if = "approved", goto = "judge" },
  { else = true, goto = "stop_ok" },
]

[states.judge]
kind = "agent"
model = "claude-x"
prompt = "review {{ items | json }}"
output_schema = "review"
capture = { finish_json = "verdict" }
timeout_secs = 30
on = { ok = "stop_ok", failed = "stop_fail", budget_exhausted = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: Path, text: str = DEMO) -> Path:
    f = tmp_path / "m.asm.toml"
    f.write_text(text, encoding="utf-8")
    return f


# --- schema synthesis -------------------------------------------------------


def test_synthesize_record_is_schema_valid(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    payload = synthesize_record(spec, "review")
    # enum field -> first member; scalar -> zero value.
    assert payload == {"label": "low", "score": 0}
    # And it passes the same strict check the live agent path uses.
    assert validate_finish_payload(spec, "review", payload) == []


def test_synthesize_handles_lists(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    assert synthesize_record(spec, "scan_result") == {"items": []}


# --- per-state dry-run ------------------------------------------------------


def test_dry_run_states_route_and_capture(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec)
    by_name = {s.name: s for s in report.states}
    # branch state is reported separately, not in the per-state pass.
    assert "check" not in by_name
    assert by_name["scan"].ok and by_name["scan"].goto == "check"
    assert "captures items" in by_name["scan"].detail
    assert by_name["judge"].ok and by_name["judge"].goto == "stop_ok"
    assert "captures verdict" in by_name["judge"].detail
    assert by_name["stop_ok"].kind == "terminal" and by_name["stop_ok"].ok
    assert report.ok


# --- per-branch routing -----------------------------------------------------


def test_branch_routes_to_else_by_default(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec)  # approved defaults to false
    check = next(b for b in report.branches if b.name == "check")
    assert check.goto == "stop_ok"
    assert check.predicate == "else"
    assert check.ok


def test_branch_fixture_steers_routing(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path))
    report = dry_run(spec, {"approved": True})
    check = next(b for b in report.branches if b.name == "check")
    assert check.clause_index == 0
    assert check.goto == "judge"
    assert check.predicate == "approved"


def test_branch_on_empty_record_default_synthesizes_fields(tmp_path: Path) -> None:
    # The realistic shape: an agent verdict var with the required `default = {}`
    # routed by a branch reading `verdict.field`. The dry-run must synthesize
    # the schema-zero record so the predicate evaluates instead of erroring on
    # a missing field (which made every such machine fail `machine test`).
    text = DEMO.replace(
        'verdict = { type = "review", default = { label = "low", score = 0 } }',
        'verdict = { type = "review", default = {} }',
    ).replace(
        '{ if = "approved", goto = "judge" }',
        '{ if = "verdict.score > 0", goto = "judge" }',
    )
    spec = load_machine(_write(tmp_path, text))
    report = dry_run(spec)
    check = next(b for b in report.branches if b.name == "check")
    assert check.ok, check.detail
    assert check.goto == "stop_ok"  # zero score -> else
    # A fixture still wins over the synthesized record.
    report2 = dry_run(spec, {"verdict": {"label": "high", "score": 5}})
    check2 = next(b for b in report2.branches if b.name == "check")
    assert check2.goto == "judge"


# --- CLI surface ------------------------------------------------------------


def test_cli_machine_test_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agent6.cli import main

    f = _write(tmp_path)
    assert main(["machine", "test", str(f)]) == 0
    out = capsys.readouterr().out
    assert "per-state dry-run" in out
    assert "per-branch routing" in out
    assert "dry-run passed" in out


def test_cli_machine_test_with_blackboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.cli import main

    f = _write(tmp_path)
    bb = tmp_path / "bb.toml"
    bb.write_text("approved = true\n", encoding="utf-8")
    assert main(["machine", "test", str(f), "--blackboard", str(bb)]) == 0
    out = capsys.readouterr().out
    assert "judge" in out  # branch now routes to the agent state


def test_cli_machine_test_runs_check_first(tmp_path: Path) -> None:
    from agent6.cli import main

    # An invalid machine (goto target missing) must fail like `machine check`.
    bad = DEMO.replace('goto = "judge"', 'goto = "nope"')
    f = _write(tmp_path, bad)
    assert main(["machine", "test", str(f)]) == 1


def test_cli_machine_test_missing_fixture(tmp_path: Path) -> None:
    from agent6.cli import main

    f = _write(tmp_path)
    assert main(["machine", "test", str(f), "--blackboard", str(tmp_path / "nope.toml")]) == 2


def test_cli_machine_test_bad_fixture_toml(tmp_path: Path) -> None:
    from agent6.cli import main

    f = _write(tmp_path)
    bb = tmp_path / "bb.toml"
    bb.write_text("not = valid = toml", encoding="utf-8")
    assert main(["machine", "test", str(f), "--blackboard", str(bb)]) == 2
