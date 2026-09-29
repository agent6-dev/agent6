# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ask --from/--from-latest` digest + `--file` seed helpers."""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from agent6 import paths
from agent6.ui.cli import _ask


def _build_ask_session_digest(cwd: pathlib.Path, session_id: str, *, latest: bool) -> str | None:
    """The seed text for tests concerned with digest contents."""
    seed = _ask.build_session_seed(cwd, session_id, latest=latest)
    return seed.text if seed is not None else None


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _make_run(tmp_path: pathlib.Path) -> str:
    # A base commit, a run branch that changed a file, and a synthetic manifest and log.
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
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / rid
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {"user_task": "make x equal 2", "base_sha": base_sha, "run_branch": "agent6/run"}
        ),
        encoding="utf-8",
    )
    (session_dir / "logs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session.start", "user_task": "make x equal 2"}),
                json.dumps({"type": "tool.call", "name": "apply_edit", "args": "m.py"}),
                json.dumps({"type": "verify.end", "exit_code": 0}),
                json.dumps({"type": "session.end", "reason": "finish_session", "iterations": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rid


def test_ask_run_digest_includes_task_diff_and_outcome(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "make x equal 2" in digest  # the run's task
    assert "changed by the run" in digest  # the diff
    assert "reason=finish_session" in digest  # the outcome
    assert rid in digest  # identifies the prior run
    # Run state is out of the workspace; the digest says so instead of naming unreachable paths.
    assert "outside the workspace" in digest


def test_ask_run_digest_continue_picks_a_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, "", latest=True)
    assert digest is not None and "make x equal 2" in digest


def test_ask_run_digest_unknown_run_returns_none(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (paths.state_dir(tmp_path) / "sessions" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _build_ask_session_digest(tmp_path, "nope", latest=False) is None


def test_ask_from_latest_no_sessions_names_the_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error names the flag the operator typed, `--from-latest`."""
    (paths.state_dir(tmp_path) / "sessions" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert _build_ask_session_digest(tmp_path, "", latest=True) is None
    assert "--from-latest" in capsys.readouterr().err


def test_seed_files_wraps_and_skips_missing(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    out = _ask.seed_files(tmp_path, ["a.py", "missing.py"])
    assert '<file path="a.py">' in out
    assert "print('a')" in out
    assert "missing" not in out  # missing file skipped, not crashed


def test_ask_transcript_snippet_skips_digest_tags() -> None:
    """The ask snippet is the question, past the headers and a seeded block."""
    from agent6.viewmodel import task_snippet

    t = (
        "# agent6 ask\n\n## Question\n\n"
        '<prior-run id="x">stuff</prior-run>\n\nwhy is the broker slow?\n\n'
        "## Answer\n\nbecause\n"
    )
    assert task_snippet(t) == "why is the broker slow?"
    # plain question (no tags)
    assert task_snippet("## Question\n\nwhat does fib do?\n\n## Answer\n") == "what does fib do?"
    with_file = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="a.py">\nprint("a")\n</file>\n\n'
        "what does this file do?\n\n## Answer\n\nprints a\n"
    )
    assert task_snippet(with_file) == "what does this file do?"
    with_answer_heading_in_file = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="notes.md">\n## Answer\nbody\n</file>\n\n'
        "what does this note say?\n\n## Answer\n\nbody\n"
    )
    assert task_snippet(with_answer_heading_in_file) == "what does this note say?"


def test_ask_repl_multi_turn_carries_context(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.harness import _snapshot
    from agent6.sessions import layout as sessions_layout

    class _FakeWf:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, q: str) -> _snapshot.SessionResult:
            self.calls.append(q)
            return _snapshot.SessionResult(
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

    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="x", subdir="asks"
    )
    layout.session_dir.mkdir(parents=True)
    wf = _FakeWf()
    inputs = iter(["a follow-up", "/quit"])

    def _fake_input(*_a: object) -> str:
        return next(inputs)

    monkeypatch.setattr("builtins.input", _fake_input)

    result = _ask.run_ask_repl(wf, _FakeBudget(), layout, first_question="first question")  # type: ignore[arg-type]

    assert wf.calls[0] == "first question"  # turn 1 verbatim
    # turn 2 carried the prior Q&A as context
    assert "a follow-up" in wf.calls[1]
    assert "answer-1" in wf.calls[1]
    out = capsys.readouterr().out
    assert "answer-1" in out and "answer-2" in out
    assert result.summary == "answer-2"
    # cumulative transcript written
    assert "## Q2" in (layout.session_dir / "transcript.md").read_text(encoding="utf-8")


def test_ask_transcript_snippet_reads_interactive_transcripts(tmp_path: pathlib.Path) -> None:
    """The ask snippet skips the REPL headers `## Q1` and `## A1` too."""
    from agent6.sessions import layout as sessions_layout
    from agent6.viewmodel import task_snippet

    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="ask-x")
    layout.ensure()
    _ask.save_ask_repl_transcript(
        layout, [("why is the broker slow?", "because"), ("more?", "sure")]
    )
    text = (layout.session_dir / "transcript.md").read_text(encoding="utf-8")
    assert task_snippet(text) == "why is the broker slow?"


# `agent6 ask` runs in any directory; the context loader and prompt degrade instead of raising.


def test_load_repo_summary_outside_git(tmp_path: pathlib.Path) -> None:
    from agent6.harness import _context

    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    summary = _context.load_repo_summary(tmp_path)
    assert summary.is_git is False
    assert summary.branch == "" and summary.head_sha == ""
    assert summary.recent_log == "" and summary.repo_map == ""
    assert summary.file_count == 0
    assert "notes.txt" in summary.top_level and "sub/" in summary.top_level


def test_system_prompt_names_non_git_directory(tmp_path: pathlib.Path) -> None:
    from agent6 import kinds
    from agent6.config import load_config
    from agent6.harness import _prompt_blocks  # pyright: ignore[reportPrivateUsage]

    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(
        "\n".join(
            (
                "[agent6]",
                "config_version = 1",
                "[providers.anthropic]",
                'api_format = "anthropic"',
                'api_key_env = "ANTHROPIC_API_KEY"',
                "[models.worker]",
                'provider = "anthropic"',
                'model = "x"',
            )
        ),
        encoding="utf-8",
    )
    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="",
        head_sha="",
        file_count=0,
        top_level=("notes.txt",),
        agents_md="",
        recent_log="",
        is_git=False,
    )
    prompt = _prompt_blocks.build_system_prompt(
        config=load_config(cfg_path), repo=repo, mode="ask", skills=None
    )
    assert "not a git repository" in prompt
    assert "branch=" not in prompt  # no fake repo header


def test_prompt_revision_context_names_non_git_directory(tmp_path: pathlib.Path) -> None:
    """Outside git the reviser context names the situation instead of a fake empty repo header."""
    from agent6 import kinds
    from agent6.harness import _prompt_revision

    repo = kinds.RepoSummary(
        root=tmp_path,
        branch="",
        head_sha="",
        file_count=0,
        top_level=("notes.txt",),
        agents_md="",
        recent_log="",
        is_git=False,
    )
    ctx = _prompt_revision.format_prompt_revision_context(repo)
    assert "not a git repository" in ctx
    assert "branch=" not in ctx  # no fake repo header
    git_repo = kinds.RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="a" * 40,
        file_count=3,
        top_level=("x.py",),
        agents_md="",
        recent_log="",
    )
    assert "branch=main" in _prompt_revision.format_prompt_revision_context(git_repo)


def test_ask_repl_prompt_uses_default_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle ask> prompt runs under the default SIGINT handler, so one Ctrl-C exits.

    No step is in flight there; the run's escalating handler would print a pause banner and arm a
    phantom pause menu.
    """
    import signal
    from typing import Any, cast

    from agent6.harness import loop

    fired: list[object] = []

    def steer_style_handler(_signum: int, _frame: object) -> None:
        fired.append(True)  # swallows, like the escalation's stage-1 arm

    prev = signal.signal(signal.SIGINT, steer_style_handler)
    try:
        seen: list[object] = []

        def fake_input(prompt: str = "") -> str:
            seen.append(signal.getsignal(signal.SIGINT))
            raise KeyboardInterrupt  # the operator leaves the REPL

        monkeypatch.setattr("builtins.input", fake_input)
        result = _ask.run_ask_repl(
            cast("loop.Harness", object()),
            cast("Any", object()),
            cast("Any", object()),
            first_question="",
        )
        assert result.reason == "ask_repl_empty"
        assert seen == [signal.default_int_handler]
        # and the surrounding run handler is back after the prompt
        assert signal.getsignal(signal.SIGINT) is steer_style_handler
    finally:
        signal.signal(signal.SIGINT, prev)


def test_ask_run_digest_survives_non_utf8_diff(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run digest survives a non-UTF-8 diff: bytes are decoded lossily."""
    rid = _make_run(tmp_path)
    (tmp_path / "latin.txt").write_bytes(b"caf\xe9 r\xe9sum\xe9\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "latin-1 bytes")
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "latin.txt" in digest
    assert "changed by the run" in digest


def test_ask_run_digest_pruned_branch_falls_back_to_merge_stamp(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a squash-merged run branch is pruned, the digest diffs the merge stamp's commit."""
    rid = _make_run(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / rid
    m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    _git(tmp_path, "checkout", "-q", m["base_sha"])
    _git(tmp_path, "merge", "--squash", "agent6/run")
    _git(tmp_path, "commit", "-qm", "squash-merge run")
    merge_sha = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "branch", "-D", "agent6/run")
    m["merged"] = {"into": "master", "sha": merge_sha, "ts": "2026-07-24T00:00:00Z"}
    (session_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "changed by the run" in digest
    assert "run branch pruned" in digest


def test_ask_run_digest_fast_forward_merge_keeps_earlier_commits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast-forwarded run's digest diffs base..merged, keeping every commit.

    The stamp's `tip` equals its sha there, and `sha^..sha` would keep only the last commit.
    """
    rid = _make_run(tmp_path)  # leaves one commit on agent6/run
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / rid
    m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "second.py").write_text("y = 3  # second run commit\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "second run change")
    tip = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "-b", "main-line", m["base_sha"])
    _git(tmp_path, "merge", "-q", "--ff-only", "agent6/run")
    assert _git(tmp_path, "rev-parse", "HEAD") == tip  # a true fast-forward
    _git(tmp_path, "branch", "-D", "agent6/run")
    m["merged"] = {"into": "main-line", "sha": tip, "tip": tip, "ts": "2026-07-26T00:00:00Z"}
    (session_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "second run commit" in digest
    assert "changed by the run" in digest  # the FIRST commit is not dropped
    assert "fast-forward" in digest  # the label says what the range is


def test_ask_run_digest_does_not_call_a_present_branch_pruned(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed diff with the branch present is not reported as "run branch pruned".

    A base_sha that no longer resolves fails the diff with the branch still there.
    """
    rid = _make_run(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / rid
    m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    _git(tmp_path, "checkout", "-q", m["base_sha"])
    _git(tmp_path, "merge", "--squash", "agent6/run")
    _git(tmp_path, "commit", "-qm", "squash-merge run")
    m["merged"] = {"into": "master", "sha": _git(tmp_path, "rev-parse", "HEAD"), "ts": "2026-07"}
    m["base_sha"] = "deadbeef" * 5  # unreachable base, branch untouched
    (session_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "changed by the run" in digest  # still falls back to the merge commit
    assert "run branch pruned" not in digest  # ...without lying about why


def test_ask_run_digest_reports_unavailable_diff(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diff the repo cannot produce is reported as unavailable, never as an empty diff."""
    rid = _make_run(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / rid
    m = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    _git(tmp_path, "checkout", "-q", m["base_sha"])
    _git(tmp_path, "branch", "-D", "agent6/run")
    monkeypatch.chdir(tmp_path)
    digest = _build_ask_session_digest(tmp_path, rid, latest=False)
    assert digest is not None
    assert "diff unavailable" in digest


def _session(
    tmp_path: pathlib.Path, bucket: str, sid: str, mode: str, *, run_branch: str | None
) -> None:
    d = paths.state_dir(tmp_path) / "sessions" / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": sid,
                "mode": mode,
                "user_task": f"the {mode} task",
                "base_sha": "0" * 40,
                "run_branch": run_branch,
            }
        ),
        encoding="utf-8",
    )
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True}) + "\n",
        encoding="utf-8",
    )


def test_a_session_that_wrote_no_code_shows_no_diff(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan and an ask cut no branch, so their digest shows no diff."""
    _make_run(tmp_path)  # a repo with real, unrelated commits on HEAD
    _session(tmp_path, "runs", "plan-only-BBB222", "plan", run_branch=None)
    monkeypatch.chdir(tmp_path)

    digest = _build_ask_session_digest(tmp_path, "plan-only-BBB222", latest=False)

    assert digest is not None
    assert "the plan task" in digest
    assert "wrote no code" in digest
    assert "changed by the run" not in digest, "an unrelated diff was attributed to the plan"


def test_from_latest_skips_a_machine_draft(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-latest skips a machine draft.

    A draft is an authoring log, not a session with a task and an outcome.
    """
    rid = _make_run(tmp_path)
    # A real draft, newer than the run: a husk with no manifest is skipped by every listing anyway.
    _session(tmp_path, "machines", "draft-CCC333", "machine", run_branch=None)
    monkeypatch.chdir(tmp_path)

    digest = _build_ask_session_digest(tmp_path, "", latest=True)

    assert digest is not None and rid in digest
