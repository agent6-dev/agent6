# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions review`: a finished session's journal folds into a digest
the reviewer role reads, and the review prints and is saved, with nothing
written to the repo or the memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.harness.run_review import RunReviewError, run_digest, run_review
from agent6.kinds import RoleName
from agent6.memory import add, record_use
from agent6.paths import state_dir
from agent6.providers import ProviderError, ProviderResponse, ToolDefinition
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main
from agent6.ui.cli import sessions_review as review_mod

_EVENTS: list[dict[str, Any]] = [
    {
        "type": "session.start",
        "session_id": "run-AAAA11",
        "mode": "run",
        "user_task": "fix the parser",
    },
    {"type": "role.result", "role": "worker", "text": "Reading the parser."},
    {"type": "tool.call", "name": "read_file", "args": {"path": "src/p.py"}, "call_id": 1},
    {"type": "tool.result", "name": "read_file", "ok": True, "summary": "40 bytes", "call_id": 1},
    {"type": "tool.call", "name": "run_command", "args": {"cmd": "pytest"}, "call_id": 2},
    {
        "type": "tool.result",
        "name": "run_command",
        "ok": False,
        "summary": "exit 1: 2 failed",
        "call_id": 2,
    },
    {"type": "loop.steer.injected", "chars": 20, "text": "do not touch the lexer"},
    {"type": "loop.decision.recorded", "question": "keep the old API?", "answer": "yes"},
    {"type": "loop.no_progress.nudge", "iteration": 4},
    {
        "type": "verify.end",
        "cmd": ["pytest"],
        "exit_code": 1,
        "duration_s": 2.5,
        "stdout_tail": "",
        "stderr_tail": "E  assert 1 == 2",
    },
    {
        "type": "verify.end",
        "cmd": ["pytest"],
        "exit_code": 0,
        "duration_s": 2.0,
        "stdout_tail": "3 passed",
        "stderr_tail": "",
    },
    {"type": "role.result", "role": "reviewer", "text": "a side seat's answer"},
    {"type": "role.result", "role": "worker", "text": "Done: the lexer is untouched."},
    {"type": "budget.update", "usd_total": 0.25},
    {"type": "session.end", "reason": "finish_session", "iterations": 6, "all_passed": True},
]


def _write_session(
    repo: Path, session_id: str = "run-AAAA11", events: list[dict[str, Any]] | None = None
) -> SessionLayout:
    layout = SessionLayout(state_dir=state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {"version": 2, "session_id": session_id, "mode": "run", "user_task": "fix the parser"}
        )
        + "\n",
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        "\n".join(json.dumps(e) for e in (events if events is not None else _EVENTS)) + "\n",
        encoding="utf-8",
    )
    return layout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_digest_folds_what_the_reviewer_needs(repo: Path) -> None:
    layout = _write_session(repo)
    record_use(layout.state_dir, session="run-AAAA11", wrote=("lexer-rule",), read={"old": 1})
    record_use(layout.state_dir, session="run-other", wrote=("other",), read={})
    add(layout.state_dir, "money-rounding", "Money rounds half-up on the cent.")

    d = run_digest(layout)

    assert (d.session_id, d.mode, d.task) == ("run-AAAA11", "run", "fix the parser")
    assert (d.end_reason, d.verify, d.iterations) == ("finish_session", "passed", 6)
    assert (d.tool_calls, d.tool_errors, d.cost_usd) == (2, 1, 0.25)
    assert d.steers == ("do not touch the lexer",)
    assert d.decisions == (("keep the old API?", "yes"),)
    assert [(v.exit_code, v.tail) for v in d.verify_runs] == [
        (1, "E  assert 1 == 2"),
        (0, "3 passed"),
    ]
    assert d.errors_by_tool == (("run_command", 1),)
    assert d.first_errors == ("[run_command] exit 1: 2 failed",)
    assert d.notices == (("no_progress.nudge", 1),)
    assert d.memory_wrote == ("lexer-rule",)
    assert d.memory_index == "- money-rounding: Money rounds half-up on the cent."
    assert "assistant: Done: the lexer is untouched." in d.conversation
    assert "a side seat's answer" not in d.conversation
    text = d.render()
    assert text.startswith(
        "session run-AAAA11 (run): fix the parser\n"
        "ended: finish_session; verify passed; 6 iterations, 2 tool calls (1 failed), $0.25"
    )
    for heading in (
        "operator steers (1):",
        "rulings recorded (1):",
        "verify runs (2):",
        "tool errors (1 of 2 calls):",
        "harness notices: no_progress.nudge x1",
        "memory facts this session wrote: lexer-rule",
        "memory index (every run on this repo is shown it):\n- money-rounding:",
        "conversation (tail):",
    ):
        assert heading in text


def test_every_writer_of_a_fact_is_credited_not_only_the_first_and_last(repo: Path) -> None:
    """A creates a fact, B edits it, C edits it: reviewing B credited B with
    no memory write, because the record kept only the first and the last
    writer."""
    layout = _write_session(repo)
    record_use(layout.state_dir, session="run-A", wrote=("fact",), read={})
    record_use(layout.state_dir, session="run-AAAA11", wrote=("fact",), read={})
    record_use(layout.state_dir, session="run-C", wrote=("fact",), read={})
    assert run_digest(layout).memory_wrote == ("fact",)


def test_a_red_gate_and_an_empty_journal_read_truthfully(repo: Path) -> None:
    red = [
        e
        for e in _EVENTS
        if e["type"] != "session.end" and not (e["type"] == "verify.end" and e["exit_code"] == 0)
    ]
    red.append(
        {"type": "session.end", "reason": "no_progress", "iterations": 9, "all_passed": False}
    )
    layout = _write_session(repo, events=red)
    d = run_digest(layout)
    assert (d.end_reason, d.verify) == ("no_progress", "failed")

    empty = _write_session(repo, session_id="run-BBBB22", events=[])
    d = run_digest(empty)
    # Nothing observed the tree: "unverified", never a word that claims a gate
    # was absent or green.
    assert (d.end_reason, d.verify, d.tool_calls, d.steers) == ("", "unverified", 0, ())
    assert "conversation (tail):" in d.render()


def test_a_plan_or_an_ask_has_no_gate_to_pass(repo: Path) -> None:
    """A plan's and an ask's `session.end` carries `all_passed: true` (nothing
    gated them), and the digest read that as "verify passed": a plan that ran
    no verify was reviewed as green."""
    plan = [
        {"type": "session.start", "session_id": "plan-1", "mode": "plan", "user_task": "plan it"},
        {"type": "session.end", "reason": "finish_planning", "iterations": 3, "all_passed": True},
    ]
    layout = _write_session(repo, session_id="plan-1", events=plan)
    layout.manifest_path.write_text(
        json.dumps({"version": 2, "session_id": "plan-1", "mode": "plan", "user_task": "plan it"})
        + "\n",
        encoding="utf-8",
    )
    d = run_digest(layout)
    assert (d.mode, d.verify) == ("plan", "not gated")
    assert "verify not gated" in d.render()


def test_the_digest_clips_a_runaway_index_like_the_prompt_does(repo: Path) -> None:
    """The index reaches the reviewer under the same cap the prompt applies,
    with the same marker, so one runaway index cannot flood a review call."""
    from agent6.memory import INDEX_INJECT_CAP

    layout = _write_session(repo)
    for i in range(120):
        add(layout.state_dir, f"fact-{i:03d}", "x" * 60)
    d = run_digest(layout)
    assert len(d.memory_index) <= INDEX_INJECT_CAP
    assert d.memory_index.endswith("... (index clipped; read MEMORY.md for the rest)")


def test_the_gate_word_agrees_with_the_listing_scan(repo: Path) -> None:
    """Three shapes the digest got wrong against `sessions show`: a resumed
    run whose red verify was in execution 1 and whose execution 2 ran none reads
    "unverified" (the scan resets at the resume); a gated run whose gate never
    ran (all_passed false, no verify.end) reads "unverified", not "not
    gated"; a killed run (a green verify, no session.end) reads "unverified"."""
    from agent6.viewmodel.listing import scan_session_log

    resumed = [
        {"type": "session.start", "session_id": "r", "mode": "run", "user_task": "t"},
        {
            "type": "verify.end",
            "cmd": ["pytest"],
            "exit_code": 1,
            "duration_s": 1.0,
            "stdout_tail": "",
            "stderr_tail": "E boom",
        },
        {"type": "session.end", "reason": "budget_exhausted", "iterations": 4, "all_passed": False},
        {"type": "loop.resume.start", "session_id": "r", "mode": "run"},
        {"type": "session.end", "reason": "finish_session", "iterations": 6, "all_passed": False},
    ]
    layout = _write_session(repo, session_id="run-resumed", events=resumed)
    d = run_digest(layout)
    assert d.verify == "unverified"
    assert scan_session_log(layout.logs_path).verify_verdict() is None

    never_ran = [
        {"type": "session.start", "session_id": "n", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": False},
    ]
    layout = _write_session(repo, session_id="run-never", events=never_ran)
    assert run_digest(layout).verify == "unverified"

    killed = [
        {"type": "session.start", "session_id": "k", "mode": "run", "user_task": "t"},
        {
            "type": "verify.end",
            "cmd": ["pytest"],
            "exit_code": 0,
            "duration_s": 1.0,
            "stdout_tail": "ok",
            "stderr_tail": "",
        },
    ]
    layout = _write_session(repo, session_id="run-killed", events=killed)
    assert run_digest(layout).verify == "unverified"

    gateless = [
        {"type": "session.start", "session_id": "g", "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": None},
    ]
    layout = _write_session(repo, session_id="run-gateless", events=gateless)
    assert run_digest(layout).verify == "not gated"


def test_the_caps_are_named_not_silent(repo: Path) -> None:
    many = [{"type": "loop.steer.injected", "chars": 1, "text": f"steer {i}"} for i in range(25)]
    layout = _write_session(repo, events=many)
    d = run_digest(layout)
    assert len(d.steers) == 20
    assert d.steers_total == 25
    assert "operator steers (20, 5 more not shown):" in d.render()


@dataclass
class _FakeProvider:
    response_text: str = "## Outcome\nfinished green"
    raise_error: bool = False
    last_system: str = ""
    last_user: str = ""

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> ProviderResponse:
        if self.raise_error:
            raise ProviderError("boom")
        self.last_system = system
        self.last_user = str(messages[0]["content"])
        return ProviderResponse(
            text=self.response_text,
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_run_review_hands_the_record_and_agents_md_to_the_reviewer() -> None:
    provider = _FakeProvider()
    out = run_review(provider, digest="session x: task", agents_md="# rules")  # type: ignore[arg-type]
    assert out.startswith("## Outcome")
    assert provider.last_user == "AGENTS.md:\n# rules\n\nRUN RECORD:\nsession x: task"
    assert "Candidate memory facts" in provider.last_system
    assert "Memory entries the record contradicts" in provider.last_system
    # Not a fault: a finish the memory backstop deferred once, a task's own
    # starting red verify. Not a fact: the code's state at the time, the
    # index's own entries.
    for line in (
        "not a fault, unless",
        "the task, not a fault",
        "Not the code's state at the time",
        "or the memory index already state",
    ):
        assert line in provider.last_system
    with pytest.raises(RunReviewError, match="provider call failed"):
        run_review(_FakeProvider(raise_error=True), digest="x")  # type: ignore[arg-type]
    with pytest.raises(RunReviewError, match="empty"):
        run_review(_FakeProvider(response_text="  "), digest="x")  # type: ignore[arg-type]


def _reviewer_config() -> Config:
    return Config.model_validate(
        {
            "providers": {"local": {"api_format": "openai", "base_url": "https://example.test/v1"}},
            "models": {"reviewer": {"provider": "local", "model": "reviewer"}},
        }
    )


def test_the_verb_prints_and_saves_the_review(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_session(repo)
    provider = _FakeProvider(
        response_text="## Outcome\nfinished green\n\n## Candidate memory facts\n- lexer-rule: ..."
    )
    cfg = _reviewer_config()

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    monkeypatch.setattr(review_mod, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_mod, "build_role_provider", MagicMock(return_value=provider))

    rc = main(["sessions", "review", "run-AAAA11"])

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "## Outcome\nfinished green\n\n## Candidate memory facts\n- lexer-rule: ...\n"
    assert "reviewing run: run-AAAA11" in out.err
    assert provider.last_user.startswith("RUN RECORD:\nsession run-AAAA11 (run): fix the parser")
    saved = sorted((state_dir(repo) / "reviews").glob("*-review.md"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8").startswith("# review: run run-AAAA11\n\n## Outcome")
    assert f"review saved: {saved[0]}" in out.err
    # Read-only: the memory store gained nothing.
    assert not (state_dir(repo) / "memory").exists()


def test_the_verb_reviews_any_session_and_picks_the_newest_across_buckets(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verb resolved through the git verbs' resolver, which refused a
    model-git run, a fan-out and a session with no manifest, and with no id
    picked the newest RUN while a plan that just ended was newer. The review
    reads a journal: any session by id, the newest session without one."""
    import time as _time

    provider = _FakeProvider(response_text="## Outcome\nok")
    cfg = _reviewer_config()

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    monkeypatch.setattr(review_mod, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_mod, "build_role_provider", MagicMock(return_value=provider))

    modelgit = _write_session(repo, session_id="run-MODEL1")
    modelgit.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "run-MODEL1",
                "mode": "run",
                "user_task": "t",
                "git_control": "model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["sessions", "review", "run-MODEL1"]) == 0
    assert "reviewing run: run-MODEL1" in capsys.readouterr().err

    bare = _write_session(repo, session_id="run-BARE01")
    bare.manifest_path.unlink()
    assert main(["sessions", "review", "run-BARE01"]) == 0
    capsys.readouterr()

    _time.sleep(0.05)
    plan = SessionLayout(state_dir=state_dir(repo), session_id="plan-NEWEST", subdir="plans")
    plan.ensure()
    plan.manifest_path.write_text(
        json.dumps({"version": 2, "session_id": "plan-NEWEST", "mode": "plan", "user_task": "p"})
        + "\n",
        encoding="utf-8",
    )
    plan.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "plan", "user_task": "p"}) + "\n",
        encoding="utf-8",
    )
    assert main(["sessions", "review"]) == 0
    err = capsys.readouterr().err
    assert "reviewing the newest session: plan-NEWEST" in err


def test_the_model_flag_reaches_the_reviewer(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_session(repo)
    cfg = _reviewer_config()
    seen: list[str] = []

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    def build(cfg_used: Config, role: RoleName, **_k: object) -> _FakeProvider:
        route = cfg_used.models.resolve(role)
        seen.append(route.model if route is not None else "")
        return _FakeProvider()

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    monkeypatch.setattr(review_mod, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_mod, "build_role_provider", build)
    assert main(["sessions", "review", "run-AAAA11", "--model", "local/other"]) == 0
    capsys.readouterr()
    assert seen == ["other"]


def test_the_verb_refuses_an_unknown_session_and_a_provider_it_cannot_build(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_session(repo)
    assert main(["sessions", "review", "nope"]) == 2
    assert "nope" in capsys.readouterr().err

    cfg = _reviewer_config()

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    monkeypatch.setattr(review_mod, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(
        review_mod, "build_role_provider", MagicMock(side_effect=ProviderError("no key"))
    )
    assert main(["sessions", "review", "run-AAAA11", "--model", "local/other"]) == 2
    assert "provider init failed: no key" in capsys.readouterr().err


def test_a_failed_reviewer_call_is_reported(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_session(repo)
    cfg = _reviewer_config()

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    monkeypatch.setattr(review_mod, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(
        review_mod, "build_role_provider", MagicMock(return_value=_FakeProvider(raise_error=True))
    )
    assert main(["sessions", "review"]) == 2
    err = capsys.readouterr().err
    assert "reviewing the newest session: run-AAAA11" in err
    assert "REVIEW FAILED: provider call failed: boom" in err


def test_a_live_session_is_refused_before_any_call(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run's record is not complete; the review waits for its end."""
    import os

    from agent6.sessions.ipc import write_worker_pid

    layout = _write_session(repo, events=_EVENTS[:1])
    write_worker_pid(layout.session_dir, os.getpid())
    monkeypatch.setattr(
        "agent6.ui.cli.review_cmds.load_effective",
        MagicMock(side_effect=AssertionError("config loaded")),
    )
    assert main(["sessions", "review", "run-AAAA11"]) == 2
    err = capsys.readouterr().err
    assert "run-AAAA11 is live" in err and "agent6 stop run-AAAA11" in err
