# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared run-listing helpers (run_mtime, task_snippet, summarize_run_dir)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent6.runs.manifest import CompareStamp
from agent6.viewmodel import (
    is_run_husk,
    is_winner,
    run_compare,
    run_mtime,
    summarize_run_dir,
    task_snippet,
)
from agent6.viewmodel.format import format_compare


def test_run_mtime_prefers_log_over_dir(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (1000.0, 1000.0))
    os.utime(d, (5000.0, 5000.0))  # dir bumped later (a viewer wrote frontend.pid)
    assert run_mtime(d) == 1000.0  # keyed off the log, not the dir


def test_run_mtime_falls_back_to_dir(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    os.utime(d, (2000.0, 2000.0))
    assert run_mtime(d) == 2000.0  # no log yet -> dir mtime


def test_task_snippet_skips_seeded_file_block() -> None:
    task = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="a.py">\ndef f(): pass\nSHOULD NOT SHOW\n</file>\n\n'
        "why is the broker slow?\n\n## Answer\n"
    )
    assert task_snippet(task) == "why is the broker slow?"


def test_task_snippet_plain_task() -> None:
    assert task_snippet("add a --json flag\nmore detail") == "add a --json flag"


def test_task_snippet_falls_back_to_stripped_text() -> None:
    assert task_snippet("   ") == ""


def _stamp(run_dir: Path, compare: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"compare": compare}), encoding="utf-8")


def test_run_compare_and_is_winner_read_the_manifest_block(tmp_path: Path) -> None:
    # The fixture writes the legacy `group` key; the model ignores it (old-shape
    # compat), so a fan-out lane recorded before the dedup still reads its stamp.
    win = tmp_path / "win"
    _stamp(win, {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"})
    assert is_winner(win) is True
    assert isinstance(run_compare(win), CompareStamp)
    loser = tmp_path / "loser"
    _stamp(loser, {"group": "fan", "rank": 2, "of": 2, "winner": False, "ranked_by": "judge"})
    assert is_winner(loser) is False
    # A run outside any fan-out (no manifest / no compare block) reads as None.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run_compare(plain) is None and is_winner(plain) is False


def test_format_compare_headline_and_rationale() -> None:
    won = format_compare(
        CompareStamp(rank=1, of=3, winner=True, ranked_by="judge", rationale="cleanest diff")
    )
    assert won == ("rank 1/3 · winner · judge", "cleanest diff")
    # A loser, mechanical, no rationale.
    lost = format_compare(
        CompareStamp(rank=2, of=3, winner=False, ranked_by="mechanical", rationale="")
    )
    assert lost == ("rank 2/3 · mechanical", "")
    # No stamp -> None.
    assert format_compare(None) is None


# --- summarize_run_dir / status_word (shared by TUI hub, web hub, runs list) --


def _write_run(base: Path, sub: str, run_id: str, events: list[dict[str, object]]) -> Path:
    import json

    rd = base / sub / run_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return rd


def test_summary_reads_mode_task_and_passed(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "fix [the] bug"},
            {"type": "tool.call", "name": "read_file"},
            {"type": "budget.update", "usd_total": 0.12},
            {"type": "run.end", "all_passed": True, "reason": "finish_run"},
        ],
    )
    s = summarize_run_dir(rd)
    assert (s.mode, s.task, s.status, s.reason) == ("run", "fix [the] bug", "passed", "")
    assert s.cost_usd == 0.12


def test_summary_failure_carries_its_reason(tmp_path: Path) -> None:
    """The core truth fix: a provider_error death reads 'failed · provider_error',
    never a neutral 'done' the operator scrolls past."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "all_passed": False, "reason": "provider_error"},
        ],
    )
    s = summarize_run_dir(rd)
    assert (s.status, s.reason) == ("failed", "provider_error")


def test_summary_stop_is_not_a_failure(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "all_passed": False, "reason": "steer_abort"},
        ],
    )
    assert summarize_run_dir(rd).status == "stopped"


def test_summary_interrupt_reads_as_stopped(tmp_path: Path) -> None:
    # A Ctrl-C interrupt is the operator's own act, like steer_abort -- not a
    # failure the listing should flag red.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "all_passed": False, "reason": "interrupted"},
        ],
    )
    assert summarize_run_dir(rd).status == "stopped"


def test_summary_resume_unfinishes(tmp_path: Path) -> None:
    """A detached resume appends past the first run.end; the run is running
    again, not whatever it last ended as."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "all_passed": False, "reason": "steer_abort"},
            {"type": "loop.resume.start", "iteration": 2},
        ],
    )
    assert summarize_run_dir(rd, stale_after_s=10_000_000).status == "running"


def test_summary_running_and_stale(tmp_path: Path) -> None:
    rd = _write_run(tmp_path, "runs", "r2", [{"type": "run.start", "mode": "plan"}])
    assert summarize_run_dir(rd, stale_after_s=10_000_000).status == "running"
    assert summarize_run_dir(rd, stale_after_s=0.0).status == "stale"


def test_summary_dead_worker_reads_stale_at_once(tmp_path: Path) -> None:
    # A killed run (worker.pid points at a dead process, no run.end) must not
    # read "running" for the whole silence window; the pid probe settles it now.
    rd = _write_run(tmp_path, "runs", "r3", [{"type": "run.start", "mode": "run"}])
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # beyond pid_max: never alive
    assert summarize_run_dir(rd, stale_after_s=10_000_000).status == "stale"


def test_summary_live_worker_stays_running_past_the_silence_window(tmp_path: Path) -> None:
    # The converse: a live worker blocked in a long provider call emits no
    # events, but it is not stale -- the pid probe wins over log silence.
    import os

    rd = _write_run(tmp_path, "runs", "r4", [{"type": "run.start", "mode": "run"}])
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert summarize_run_dir(rd, stale_after_s=0.0).status == "running"


def test_summary_ask_task_comes_from_transcript(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path,
        "asks",
        "a1",
        [
            {"type": "run.start", "mode": "ask", "user_task": '<file path="a.py">\nx'},
            {"type": "run.end", "all_passed": True},
        ],
    )
    (rd / "transcript.md").write_text(
        "# agent6 ask\n\n## Question\n\nwhat is the default port?\n", encoding="utf-8"
    )
    s = summarize_run_dir(rd)
    assert task_snippet(s.task) == "what is the default port?"


def test_summary_no_logs(tmp_path: Path) -> None:
    rd = tmp_path / "runs" / "empty"
    rd.mkdir(parents=True)
    s = summarize_run_dir(rd)
    assert (s.status, s.task) == ("?", "(no logs)")


def test_summary_plan_reads_planned_not_passed(tmp_path: Path) -> None:
    # A plan pass ends via finish_planning (its only clean exit) with
    # all_passed=True; it gates nothing, so it must read "planned", not "passed".
    rd = _write_run(
        tmp_path,
        "runs",
        "p1",
        [
            {"type": "run.start", "mode": "plan", "user_task": "plan the refactor"},
            {"type": "run.end", "all_passed": True, "reason": "finish_planning"},
        ],
    )
    s = summarize_run_dir(rd)
    assert (s.mode, s.status, s.reason) == ("plan", "planned", "")
    # A real run still reads "passed" (finish_run + all_passed) -- unchanged.
    rd2 = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "all_passed": True, "reason": "finish_run"},
        ],
    )
    assert summarize_run_dir(rd2).status == "passed"


def test_summary_manifest_only_fork_shows_mode_and_task(tmp_path: Path) -> None:
    # A `fork --no-run` fork has a manifest (mode + task) but no logs yet; the
    # listing must show them, not a blank "? ? (no logs)".
    rd = tmp_path / "runs" / "child"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "plan", "user_task": "carry this forward"}), encoding="utf-8"
    )
    s = summarize_run_dir(rd)
    assert (s.mode, s.task, s.status) == ("plan", "carry this forward", "?")


def test_summary_launching_run_reads_starting(tmp_path: Path) -> None:
    # A run with no verify_command spends ~80s inferring one BEFORE run.start.
    # During it the log has a role.call (the inference LLM call) but no run.start,
    # and the worker is alive -- it must read "starting" (its real mode+task from
    # the manifest), not a blank "? / (no task) / running" that looks missing.
    import os

    rd = _write_run(tmp_path, "runs", "boot", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "refactor the loop"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker
    s = summarize_run_dir(rd, stale_after_s=0.0)
    assert (s.mode, s.task, s.status) == ("run", "refactor the loop", "starting")


def test_summary_pre_start_dead_worker_is_neutral_not_stale(tmp_path: Path) -> None:
    # The converse: no run.start and no LIVE worker (a `fork --no-run`, or a run
    # that died in preflight) must NOT read a false "stale" -- it never claimed to
    # be running. It stays the neutral "?".
    rd = _write_run(tmp_path, "runs", "dead", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "t"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # never alive
    assert summarize_run_dir(rd, stale_after_s=0.0).status == "?"


def test_summary_cost_sums_across_resume_legs(tmp_path: Path) -> None:
    # Each resume leg starts a fresh budget (usd_total resets to 0). The listing
    # total must be the cumulative spend across legs, not just the latest leg's.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "budget.update", "usd_total": 0.02},  # leg 1 ends at $0.02
            {"type": "run.end", "all_passed": False, "reason": "budget_exhausted"},
            {"type": "loop.resume.start", "iteration": 3},
            {"type": "budget.update", "usd_total": 0.003},
            {"type": "budget.update", "usd_total": 0.007},  # leg 2 ends at $0.007
            {"type": "run.end", "all_passed": True, "reason": "finish_run"},
        ],
    )
    s = summarize_run_dir(rd)
    assert abs(s.cost_usd - 0.027) < 1e-9  # 0.02 (leg 1) + 0.007 (leg 2), not 0.007


def test_is_run_husk(tmp_path: Path) -> None:
    # Neither manifest nor logs: never started, a husk.
    husk = tmp_path / "husk"
    husk.mkdir()
    assert is_run_husk(husk)
    # Either file makes it a real run.
    with_logs = tmp_path / "with-logs"
    with_logs.mkdir()
    (with_logs / "logs.jsonl").write_text("", encoding="utf-8")
    assert not is_run_husk(with_logs)
    with_manifest = tmp_path / "with-manifest"
    with_manifest.mkdir()
    (with_manifest / "manifest.json").write_text("{}", encoding="utf-8")
    assert not is_run_husk(with_manifest)
    # A dir with neither file but a LIVE worker.pid is a launching run in its
    # pre-manifest preflight window, not a husk -- keep it listed (as "starting").
    import os

    launching = tmp_path / "launching"
    launching.mkdir()
    (launching / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert not is_run_husk(launching)
    # ... but a dead worker.pid with no files is still a husk.
    dead = tmp_path / "dead-husk"
    dead.mkdir()
    (dead / "worker.pid").write_text("999999999", encoding="utf-8")
    assert is_run_husk(dead)


def test_summary_survives_a_valid_json_non_object_line(tmp_path: Path) -> None:
    # A valid-JSON line that isn't an object (a torn or adversarial writer) must
    # not crash the listing fold -- one bad line otherwise took down the whole
    # hub / `runs list` / TUI home. It's skipped like an unparseable line.
    rd = tmp_path / "runs" / "weird"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "run.start", "user_task": "do a thing"})
        + "\n"
        + "[1, 2, 3]\n"  # valid JSON, not a dict
        + '"a bare string"\n'
        + json.dumps({"type": "run.end", "all_passed": True, "reason": "finish_run"})
        + "\n",
        encoding="utf-8",
    )
    s = summarize_run_dir(rd)  # must not raise
    assert s.task == "do a thing"
    assert s.status == "passed"
