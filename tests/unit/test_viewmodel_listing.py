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
