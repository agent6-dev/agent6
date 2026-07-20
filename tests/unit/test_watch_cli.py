# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for the unified `agent6 attach <target>` (run + machine, --json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.ui.cli import main

# A branch -> terminal machine: no model/jail, reaches a journaled end at once.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _make_run(tmp_path: Path, run_id: str, events: list[dict[str, object]]) -> None:
    runs = resolved_state_dir(tmp_path) / "runs" / run_id
    runs.mkdir(parents=True)
    body = "".join(json.dumps(e) + "\n" for e in events)
    (runs / "logs.jsonl").write_text(body, encoding="utf-8")


def test_watch_run_json_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A target that resolves to a run id (here by exact match) yields the folded
    # RunState as JSON -- the same wire form a web client reads.
    monkeypatch.chdir(tmp_path)
    _make_run(
        tmp_path,
        "willing-glen-001",
        [
            {"type": "run.start", "user_task": "demo"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
        ],
    )
    assert main(["attach", "willing-glen-001", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["user_task"] == "demo"
    assert out["tool_calls"][0]["name"] == "grep"


def test_watch_machine_json_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A target that is not a run but names a machine instance routes to the
    # machine fold.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    assert main(["attach", "tiny", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["machine"] == "tiny"
    assert out["current"] == "done"
    assert out["ended"]["status"] == "ok"


def test_watch_unknown_target_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["attach", "nope"]) == 2
    assert "no run or machine matches" in capsys.readouterr().err


def test_watch_ambiguous_prefix_surfaces_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An ambiguous run prefix must report the ambiguity, not fall through to a
    # machine lookup and print "no run or machine matches".
    monkeypatch.chdir(tmp_path)
    _make_run(tmp_path, "willing-glen-001", [{"type": "run.start"}])
    _make_run(tmp_path, "willing-glen-002", [{"type": "run.start"}])
    assert main(["attach", "willing-glen"]) == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "no run or machine matches" not in err


def test_attach_to_a_crashed_run_ends_readonly_with_a_truthful_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crashed worker never writes run.end: attach used to replay, re-ask the
    dead worker's pending approval, then follow forever behind a "working"
    spinner while `runs show` called the same run stopped. With a stale
    worker.pid it must render read-only, never prompt, and end with the
    truthful crashed line."""
    import threading

    from agent6.ui.cli import plan_watch as pw

    _make_run(
        tmp_path,
        "dead-run",
        [
            {"type": "run.start", "user_task": "t"},
            {"type": "tool.call", "name": "run_command", "args": {}},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "run?"},
        ],
    )
    monkeypatch.chdir(tmp_path)
    run_dir = resolved_state_dir(tmp_path) / "runs" / "dead-run"
    (run_dir / "worker.pid").write_text("999999", encoding="utf-8")

    def _no_prompt(*a: object, **k: object) -> None:
        raise AssertionError("a dead worker's prompt must not be re-asked")

    monkeypatch.setattr(pw._CliFrontEnd, "handle", _no_prompt)  # pyright: ignore[reportPrivateUsage]
    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(main(["attach", "dead-run"])), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "attach failed to terminate on a crashed run"
    assert result == [0]
    err = capsys.readouterr().err
    assert "crashed or killed" in err
