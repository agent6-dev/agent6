from __future__ import annotations

import collections
import io
import json
import pathlib
import subprocess
import sys
import tomllib

import pytest
from bench import compaction_arms, compaction_slot, compaction_small, compaction_worker
from bench import compaction_bench as bench

from agent6.config import model as config_model
from agent6.ui.cli import exec_cmds


def test_live_worker_requires_full_in_its_own_config() -> None:
    arm = compaction_arms.KEEP["baseline"]
    config = compaction_slot.overlay(
        arm, bench.SOL, cold=False, small=True, gate=compaction_small.GATE
    )
    assert tomllib.loads(config).get("sandbox", {}).get("isolation", "auto") == "full"


def test_live_scorer_pins_full_at_command_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The final scorer launcher must refuse rather than fall back after a live worker."""
    policy = config_model.Config()
    assert policy.sandbox.isolation == "auto"
    monkeypatch.setattr(
        sys, "stdin", io.TextIOWrapper(io.BytesIO(policy.model_dump_json().encode()))
    )

    def inspect(cfg: config_model.Config, cwd: pathlib.Path, argv: tuple[str, ...]) -> int:
        assert cfg.sandbox.isolation == "full"
        return 0

    monkeypatch.setattr(exec_cmds, "exec_new", inspect)
    assert compaction_worker.grade("dummy") == 0


def test_scripted_opening_needs_no_command_grant(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three scripted reads start under the default attended command policy."""
    for key in ("CONFIG", "STATE", "CACHE", "DATA"):
        monkeypatch.setenv(f"XDG_{key}_HOME", str(tmp_path / key.lower()))
    seed = compaction_small.prepare(tmp_path / "six-modules")
    assert seed.turn == 4 and seed.recall == compaction_small.LABEL


def test_small_pilot_grades_edits_and_counts_the_new_execution(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single passing module does not pass the task, even with a passing gate."""
    for key in ("CONFIG", "STATE", "CACHE", "DATA"):
        monkeypatch.setenv(f"XDG_{key}_HOME", str(tmp_path / key.lower()))
    profile = tmp_path / "config/agent6/config.toml"
    profile.parent.mkdir(parents=True)
    profile.write_text('[sandbox]\nrun_commands="allow"\n')
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.compaction_bench",
            "small",
            "--pilot",
            "pointers",
            "--scripted",
            "--out",
            tmp_path / "out",
        ],
        capture_output=True,
        text=True,
        timeout=160,
        check=False,
    )
    result = tmp_path / "out/runs.jsonl"
    assert proc.returncode == 0, (proc.stderr, result.read_text() if result.exists() else "")
    (row,) = map(json.loads, result.read_text().splitlines())
    assert row["verification"] == "passed" and row["groups"] == 1 and row["passed"] is False
    assert row["turn"] == 4 and row["exact_recall"] and row["turns"] >= 2
    assert row["total_tokens"] > row["first_worker_input_tokens"]
    assert pathlib.Path(row["journal"]).exists()
    table = subprocess.check_output(
        [sys.executable, "-m", "bench.compaction_bench", "table", result], text=True
    )
    assert "0/1" in table and "1/1" in table and "errors" in table


@pytest.mark.parametrize("family", ["keep", "cold"])
def test_run_set_replication_and_overlap(family: str, tmp_path: pathlib.Path) -> None:
    """Run all replay cells, even when all ten small trials pass with overlapping ranges."""
    arms = list(bench.ARMS[family])
    seen: list[bench.Row] = []
    opening = compaction_slot.Opening(tmp_path, "a" * 40, 7, "digest", "label")

    def attempt(row: bench.Row) -> None:
        row.update(
            passed=True,
            turns=6,
            rereads_6_calls=2,
            second_restart=False,
            uncached_input_tokens=27,
            exact_recall=True,
        )
        seen.append(row)

    case = bench.Case(opening, attempt)
    cases = {"small": case, "replay": case} if family == "keep" else {"replay": case}
    bench.run_set(family, cases, tmp_path, stop_on_overlap=family == "keep")
    assert collections.Counter(r["arm"] for r in seen if r["task"] == "replay") == dict.fromkeys(
        arms, 5
    )
    assert collections.Counter(r["arm"] for r in seen if r["task"] == "small") == (
        dict.fromkeys(arms[:2], 5) if family == "keep" else {}
    )
    assert {r["model"] for r in seen} == {"chatgpt/gpt-6.2-sol"}
    seen.clear()
    final = tmp_path / "final"
    final.mkdir()
    bench.run_set(family, {"replay": case}, final, finalists=arms[-2:])
    assert collections.Counter(r["arm"] for r in seen) == dict.fromkeys(arms[-2:], 3)
    assert {r["model"] for r in seen} == {"chatgpt/gpt-6.2-astra"}
    with pytest.raises(ValueError, match="distinct"):
        bench.run_set(family, cases, final, finalists=[arms[0], arms[0]])
    assert not bench.overlap([seen[0]] * 5, [seen[0] | {"uncached_input_tokens": None}] * 5)
