# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Exercise the real-world harness with a local task and no provider calls."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

import pytest

from agent6 import config as agent6_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "bench" / "realworld" / "run_realworld.sh"
_SUMMARIZE = _ROOT / "bench" / "realworld" / "summarize.py"
_STUB = """#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

with Path(os.environ['STUB_LOG']).open('a') as log:
    log.write(json.dumps({
        'argv': sys.argv[1:],
        'jail': os.environ.get('AGENT6_JAIL_BIN', ''),
        'away': os.environ.get('AGENT6_DETACHED_AWAY', ''),
    }) + '\\n')
assert sys.argv[1:3] == ['--config', 'agent6.toml']
args = sys.argv[3:]
if args == ['config', 'show']:
    sys.exit(0)
if args[0] == 'run':
    if os.environ.get('STUB_EXIT') == '2':
        sys.exit(2)
    print('[agent6] session id: stub-1', file=sys.stderr, flush=True)
    sign = os.environ.get('STUB_COST_SIGN', '=')
    partial = '+' if sign == '~' else ''
    print(f'TOTAL: in=10 out=5 cost{sign}$0.4200{partial} of $1.00', flush=True)
    if os.environ.get('STUB_PASS') == '1':
        Path('value.txt').write_text('restored\\n')
    time.sleep(float(os.environ.get('STUB_SLEEP', '0')))
    sys.exit(int(os.environ.get('STUB_EXIT', '4')))
if args == ['sessions', 'show', 'stub-1', '--json']:
    if os.environ.get('STUB_SESSION_FAIL') == '1':
        sys.exit(1)
    print(json.dumps({
        'session_id': 'stub-1',
        'reason': os.environ.get('STUB_REASON', 'finish_session'),
        'model': 'stub-provider/stub-model',
        'plan_consumed': json.loads(os.environ.get('STUB_PLAN', '1.5')),
        'plan_cap': 2,
        'input_tokens': 10,
        'output_tokens': 5,
        'cost_usd': 0.42,
    }))
    sys.exit(0)
raise SystemExit(f'unexpected argv: {args!r}')
"""


@pytest.fixture
def bench_repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Build a provider-free checkout at the harness's AGENT6_REPO boundary."""
    for key in os.environ:
        if key.startswith(("AGENT6_", "STUB_", "PER_TASK_")) or key == "BENCH_ROOT":
            monkeypatch.delenv(key)
    repo = tmp_path / "repo"
    tasks = repo / "bench" / "realworld" / "tasks"
    tasks.mkdir(parents=True)
    task = {
        "name": "fixture",
        "description": "Local file restoration",
        "files": {"value.txt": "restored\n"},
        "install": [],
        "break": [{"path": "value.txt", "find": "restored", "replace": "broken"}],
        "task_md": "Restore value.txt without changing tests.",
        "verify_command": [
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('value.txt').read_text() == 'restored\\n'",
        ],
    }
    (tasks / "fixture.json").write_text(json.dumps(task), encoding="utf-8")
    binary = repo / ".venv" / "bin" / "agent6"
    binary.parent.mkdir(parents=True)
    binary.write_text(_STUB, encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("AGENT6_REPO", str(repo))
    monkeypatch.setenv("BENCH_ROOT", str(repo / "results"))
    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    monkeypatch.setenv("STUB_LOG", str(repo / "calls.jsonl"))
    # An operator's own launcher, which the harness passes through to the run.
    monkeypatch.setenv("AGENT6_JAIL_BIN", sys.executable)
    return repo


@dataclasses.dataclass(frozen=True, slots=True)
class BenchResult:
    """Hold the harness's recorded row and rendered output."""

    row: dict[str, Any]
    stdout: str
    markdown: str


def _run_bench(repo: pathlib.Path) -> BenchResult:
    process = subprocess.run(
        ["bash", str(_SCRIPT)], cwd=repo, capture_output=True, text=True, timeout=90, check=False
    )
    assert process.returncode == 0, process.stdout + process.stderr
    root = repo / "results"
    row = json.loads((root / "_logs" / "fixture_index" / "result.json").read_text())
    markdown = (root / "results_index.md").read_text()
    return BenchResult(row, process.stdout, markdown)


@pytest.mark.parametrize("sign", ["=", "~"])
def test_cost_summary_accepts_exact_and_estimated_spend(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, sign: str
) -> None:
    """A priced TOTAL with cost=$ must not silently become zero spend."""
    monkeypatch.setenv("STUB_COST_SIGN", sign)
    result = _run_bench(bench_repo)
    assert result.row["cost_usd"] == 0.42
    assert "$0.4200" in result.markdown


@pytest.mark.parametrize("plan", ["1.5", "null", '"unreadable"'])
def test_plan_points_come_from_the_session_record(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, plan: str
) -> None:
    """Subscription usage survives aggregation instead of disappearing behind dollar cost."""
    monkeypatch.setenv("STUB_PLAN", plan)
    result = _run_bench(bench_repo)
    want = 1.5 if plan == "1.5" else 0
    assert result.row["plan_points"] == want
    assert f"plan={want}pt" in result.stdout
    assert "plan pts" in result.markdown
    assert result.row["session_id"] == "stub-1"
    assert result.row["model"] == "stub-provider/stub-model"
    assert "Model: stub-provider/stub-model." in result.markdown


@pytest.mark.parametrize(
    ("exit_code", "reason", "unscored"),
    [
        (4, "finish_session", None),
        (1, "max_turns", None),
        (1, "provider_error", "provider error"),
        (1, "verify_command_unexecutable", "verify command cannot run in the jail"),
        (1, "", "agent6 exit 1, end reason unread"),
        (2, "", "agent6 refused to start (exit 2)"),
        (130, "steer_abort", "agent6 exit 130"),
        (137, "", "timeout after 3600s"),
    ],
)
def test_end_reason_separates_unscored_runs_from_agent_failures(
    bench_repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    reason: str,
    unscored: str | None,
) -> None:
    """An infrastructure failure is not evidence that the agent failed the task."""
    monkeypatch.setenv("STUB_EXIT", str(exit_code))
    monkeypatch.setenv("STUB_REASON", reason)
    result = _run_bench(bench_repo)
    assert result.row["agent_exit"] == exit_code
    assert result.row["verify_pass"] is False
    if unscored:
        assert "Total: 0/0 verify pass, 1 unscored" in result.markdown
        assert f"UNSCORED: {unscored}" in result.markdown
    else:
        assert "Total: 0/1 verify pass, 0 unscored" in result.markdown
        assert "| FAIL |" in result.markdown
    assert result.row["unscored"] == unscored
    assert result.row["end_reason"] == reason
    if exit_code == 2:
        calls = [
            json.loads(line)["argv"]
            for line in (bench_repo / "calls.jsonl").read_text().splitlines()
        ]
        assert all("sessions" not in argv for argv in calls)


@pytest.mark.parametrize("provider_failed", [False, True])
def test_final_verify_does_not_score_a_provider_failure_even_on_a_passing_tree(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, provider_failed: bool
) -> None:
    """The observed verify stays green while a provider failure remains unscored."""
    monkeypatch.setenv("STUB_PASS", "1")
    monkeypatch.setenv("STUB_EXIT", "1" if provider_failed else "0")
    monkeypatch.setenv("STUB_REASON", "provider_error" if provider_failed else "finish_session")
    result = _run_bench(bench_repo)
    assert result.row["verify_pass"] is True
    want = "0/0 verify pass, 1 unscored" if provider_failed else "1/1 verify pass, 0 unscored"
    assert want in result.markdown


def test_unreadable_session_cannot_turn_exit_one_into_a_scored_failure(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed sessions show cannot supply an agent-side reason for exit 1."""
    monkeypatch.setenv("STUB_EXIT", "1")
    monkeypatch.setenv("STUB_SESSION_FAIL", "1")
    result = _run_bench(bench_repo)
    assert result.row["unscored"] == "agent6 exit 1, end reason unread"
    assert result.row["plan_points"] == 0
    assert "0/0 verify pass, 1 unscored" in result.markdown


def test_setup_failure_is_not_in_the_pass_rate(bench_repo: pathlib.Path) -> None:
    """A missing break.find prevents setup, not an attempted solution."""
    task_file = bench_repo / "bench" / "realworld" / "tasks" / "fixture.json"
    task = json.loads(task_file.read_text())
    task["break"][0]["find"] = "absent"
    task_file.write_text(json.dumps(task))
    result = _run_bench(bench_repo)
    assert "0/0 verify pass, 1 unscored" in result.markdown
    assert result.row["unscored"] == "setup failed (exit 1)"
    assert result.row["plan_points"] == 0
    assert not (bench_repo / "calls.jsonl").exists()


def test_run_timeout_is_unscored(bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unattended worker cannot leave the harness waiting without a wall bound."""
    monkeypatch.setenv("STUB_SLEEP", "2")
    monkeypatch.setenv("AGENT6_REALWORLD_TASK_TIMEOUT_S", "1")
    result = _run_bench(bench_repo)
    assert result.row["agent_exit"] == 124
    assert result.row["unscored"] == "timeout after 1s"
    assert "0/0 verify pass, 1 unscored" in result.markdown


@pytest.mark.parametrize("override", [False, True])
def test_worker_and_caps_use_operator_config_and_explicit_overrides(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """A generated provider table must not hijack the operator's worker or hide the real cap."""
    if override:
        monkeypatch.setenv("AGENT6_REALWORLD_MODEL", "chatgpt/test-model")
        monkeypatch.setenv("PER_TASK_BUDGET_USD", "2.50")
        monkeypatch.setenv("PER_TASK_MAX_PLAN_POINTS", "2")
        monkeypatch.setenv("AGENT6_DETACHED_AWAY", "wait")
    result = _run_bench(bench_repo)
    config = tomllib.loads((bench_repo / "results" / "fixture_index" / "agent6.toml").read_text())
    assert config["budget"]["max_usd"] == (2.5 if override else 1.0)
    assert "providers" not in config
    assert "models" not in config
    assert config["budget"].get("max_plan_points") == (2 if override else None)
    loaded = agent6_config.Config.model_validate(config)
    assert loaded.budget.max_usd == (2.5 if override else 1.0)
    assert loaded.budget.max_plan_points == (2.0 if override else -1.0)
    calls = [json.loads(line) for line in (bench_repo / "calls.jsonl").read_text().splitlines()]
    run = next(call for call in calls if call["argv"][2] == "run")
    assert run["argv"][3:] == (
        ["--model", "chatgpt/test-model", "Restore value.txt without changing tests."]
        if override
        else ["Restore value.txt without changing tests."]
    )
    assert run["away"] == ("wait" if override else "deny")
    assert run["jail"] == sys.executable
    assert f"${'2.50' if override else '1.00'} metered" in result.markdown
    assert f"{'2' if override else 'operator config'} plan points" in result.markdown


def test_bundled_jail_lookup_needs_no_cargo_build(
    bench_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout without a cargo target must reach agent6's own launcher lookup."""
    monkeypatch.delenv("AGENT6_JAIL_BIN")
    _run_bench(bench_repo)
    calls = [json.loads(line) for line in (bench_repo / "calls.jsonl").read_text().splitlines()]
    assert all(call["jail"] == "" for call in calls)


def test_summarizer_excludes_unscored_rows_even_when_verify_passed(tmp_path: pathlib.Path) -> None:
    """Aggregation must not count setup or provider failures as agent attempts."""
    logs = tmp_path / "_logs"
    for name, passed, unscored in [
        ("pass", True, None),
        ("fail", False, None),
        ("provider", True, "provider error"),
        ("setup", False, "setup failed (exit 1)"),
    ]:
        task_dir = logs / f"{name}_index"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(
            json.dumps({"task": name, "verify_pass": passed, "unscored": unscored, "cost_usd": 0.1})
        )
    process = subprocess.run(
        [sys.executable, str(_SUMMARIZE), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "| provider | UNSCORED: provider error |" in process.stdout
    assert "| setup | UNSCORED: setup failed (exit 1) |" in process.stdout
    assert "Total: 1/2 pass, 2 unscored, $0.4000" in process.stdout
