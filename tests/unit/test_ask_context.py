# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ask --run/--continue` digest + `--file` seed helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.cli._ask import (
    build_ask_run_digest as _build_ask_run_digest,
)
from agent6.cli._ask import (
    seed_files as _seed_files,
)
from agent6.config_layer import resolved_state_dir


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _make_run(tmp_path: Path) -> str:
    # A repo with a base commit + a run branch that changed a file, plus a
    # synthetic runs/<id>/ manifest + logs.jsonl under the out-of-tree state dir.
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-qb", "agent6/run")
    (tmp_path / "m.py").write_text("x = 2  # changed by the run\n", encoding="utf-8")
    _git(tmp_path, "commit", "-aqm", "run change")
    rid = "sunny-otter-AAA111"
    run_dir = resolved_state_dir(tmp_path) / "runs" / rid
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"user_task": "make x equal 2", "base_sha": base_sha, "run_branch": "agent6/run"}
        ),
        encoding="utf-8",
    )
    (run_dir / "logs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "run.start", "user_task": "make x equal 2"}),
                json.dumps({"type": "tool.call", "name": "apply_edit", "args": "m.py"}),
                json.dumps({"type": "verify.end", "exit_code": 0}),
                json.dumps({"type": "run.end", "reason": "finish_run", "iterations": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rid


def test_ask_run_digest_includes_task_diff_and_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_run_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "make x equal 2" in digest  # the run's task
    assert "changed by the run" in digest  # the diff
    assert "reason=finish_run" in digest  # the outcome
    assert rid in digest  # identifies the prior run
    # Run state is out of the workspace; the digest says so rather than pointing
    # the jailed worker at unreachable paths.
    assert "outside the workspace" in digest


def test_ask_run_digest_continue_picks_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_run_digest(tmp_path, "", latest=True)
    assert digest is not None and "make x equal 2" in digest


def test_ask_run_digest_unknown_run_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (resolved_state_dir(tmp_path) / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _build_ask_run_digest(tmp_path, "nope", latest=False) is None


def test_seed_files_wraps_and_skips_missing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    out = _seed_files(tmp_path, ["a.py", "missing.py"])
    assert '<file path="a.py">' in out
    assert "print('a')" in out
    assert "missing" not in out  # missing file skipped, not crashed


def test_ask_question_snippet_skips_digest_tags() -> None:
    from agent6.cli._ask import ask_question_snippet as _ask_question_snippet

    t = (
        "# agent6 ask\n\n## Question\n\n"
        '<prior-run id="x">stuff</prior-run>\n\nwhy is the broker slow?\n\n'
        "## Answer\n\nbecause\n"
    )
    assert _ask_question_snippet(t) == "why is the broker slow?"
    # plain question (no tags)
    assert _ask_question_snippet("## Question\n\nwhat does fib do?\n\n## Answer\n") == (
        "what does fib do?"
    )
    with_file = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="a.py">\nprint("a")\n</file>\n\n'
        "what does this file do?\n\n## Answer\n\nprints a\n"
    )
    assert _ask_question_snippet(with_file) == "what does this file do?"
    with_answer_heading_in_file = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="notes.md">\n## Answer\nbody\n</file>\n\n'
        "what does this note say?\n\n## Answer\n\nbody\n"
    )
    assert _ask_question_snippet(with_answer_heading_in_file) == "what does this note say?"


def test_ask_repl_multi_turn_carries_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:

    from agent6.cli._ask import run_ask_repl as _run_ask_repl
    from agent6.graph.storage import RunLayout
    from agent6.workflows.loop import RunResult

    class _FakeWf:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, q: str) -> RunResult:
            self.calls.append(q)
            return RunResult(
                completed=True,
                reason="silent_finish",
                summary=f"answer-{len(self.calls)}",
                iterations=1,
                tool_calls=0,
            )

    class _FakeBudget:
        def is_exhausted(self) -> bool:
            return False

        def format_summary(self) -> str:
            return "cost: $0.00"

    layout = RunLayout(state_dir=resolved_state_dir(tmp_path), run_id="x", subdir="asks")
    layout.run_dir.mkdir(parents=True)
    wf = _FakeWf()
    inputs = iter(["a follow-up", "/quit"])

    def _fake_input(*_a: object) -> str:
        return next(inputs)

    monkeypatch.setattr("builtins.input", _fake_input)

    result = _run_ask_repl(wf, _FakeBudget(), layout, first_question="first question")  # type: ignore[arg-type]

    assert wf.calls[0] == "first question"  # turn 1 verbatim
    # turn 2 carried the prior Q&A as context
    assert "a follow-up" in wf.calls[1]
    assert "answer-1" in wf.calls[1]
    out = capsys.readouterr().out
    assert "answer-1" in out and "answer-2" in out
    assert result.summary == "answer-2"
    # cumulative transcript written
    assert "## Q2" in (layout.run_dir / "transcript.md").read_text(encoding="utf-8")
