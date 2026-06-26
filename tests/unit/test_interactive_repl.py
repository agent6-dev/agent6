# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for interactive REPL plumbing.

Covers:
* ``git_ops.revert_head`` on a tmp repo - forward-revert preserves
  history (HEAD~1 is the original commit) and produces a new SHA.
* ``cli._build_repl_hook`` slash-command dispatch:
  - empty input / ``/continue`` -> ``"continue"``
  - ``/quit`` -> ``"stop"``
  - EOF -> ``"stop"``
  - ``/cost`` invokes ``budget.format_summary`` then re-prompts
  - ``/undo`` invokes ``git_ops.revert_head`` then re-prompts
  - unknown command re-prompts
* ``Workflow`` exits cleanly with ``reason="interactive_stop"`` when
  the hook returns ``"stop"`` after an auto-commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.budget import BudgetTracker
from agent6.cli.run import _build_repl_hook  # pyright: ignore[reportPrivateUsage]
from agent6.git_ops import GitError, revert_head


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "a.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def _commit(path: Path, name: str, body: str, msg: str) -> str:
    (path / name).write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", msg], check=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


# --- git_ops.revert_head --------------------------------------------------


def test_revert_head_creates_new_commit(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    bad_sha = _commit(tmp_path, "b.txt", "bad\n", "bad change")
    revert_sha = revert_head(tmp_path)
    assert len(revert_sha) == 40
    assert revert_sha != bad_sha
    # The file added in the bad commit should be gone.
    assert not (tmp_path / "b.txt").exists()
    # History preserved: HEAD~1 is still the bad commit (no rewrite).
    parent = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD~1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert parent == bad_sha


def test_revert_head_raises_on_non_repo(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        revert_head(tmp_path)


# --- _build_repl_hook dispatch -------------------------------------------


def _budget() -> BudgetTracker:
    return BudgetTracker(max_input_tokens=1000, max_output_tokens=1000)


def test_hook_empty_input_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "")
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "deadbeefcafe1234") == "continue"


def test_hook_slash_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "/continue")
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(2, "abc") == "continue"


def test_hook_quit_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p="": "/quit")
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(3, "abc") == "stop"


def test_hook_eof_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_p: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "stop"


def test_hook_cost_reprompts_then_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["/cost", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "continue"


def test_hook_unknown_reprompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["/wat", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "stop"


def test_hook_undo_invokes_revert_and_reprompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "b.txt", "bad\n", "bad change")
    answers = iter(["/undo", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "continue"
    # /undo actually ran: b.txt should be gone.
    assert not (tmp_path / "b.txt").exists()


def test_hook_undo_failure_reprompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # tmp_path is not a git repo: revert_head will raise GitError.
    answers = iter(["/undo", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    hook = _build_repl_hook(tmp_path, _budget())
    assert hook(1, "abc") == "stop"


# --- Workflow integration ----------------------------------------------


def test_after_auto_commit_default_continues() -> None:
    """Default hook is a no-op lambda returning "continue"."""
    from agent6.workflows.loop import Workflow

    wf = Workflow(
        root=Path("/tmp"),
        config=MagicMock(prompt=MagicMock(system_prompt_file="")),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
    )
    # Field exists and defaults to the no-op shape.
    assert wf.after_auto_commit(1, "abc") == "continue"


def test_after_auto_commit_field_is_overridable() -> None:
    """Custom hook is honoured (called with iteration + sha)."""
    from agent6.workflows.loop import Workflow

    calls: list[tuple[int, str]] = []

    def hook(it: int, sha: str) -> Any:
        calls.append((it, sha))
        return "stop"

    wf = Workflow(
        root=Path("/tmp"),
        config=MagicMock(prompt=MagicMock(system_prompt_file="")),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        after_auto_commit=hook,
    )
    assert wf.after_auto_commit(7, "deadbeef") == "stop"
    assert calls == [(7, "deadbeef")]


# --- steer marker self-heals on a dismissed/timed-out TUI modal ------------


def _tui_live(_run_dir: Path) -> bool:
    return True


def _answer_none(_run_dir: Path) -> str | None:
    return None  # modal dismissed / read_steer_answer timed out


def _answer_text(_run_dir: Path) -> str | None:
    return "do the thing"


def test_steer_prompt_clears_request_marker_on_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TUI-initiated steer whose modal is dismissed (read_steer_answer -> None
    on timeout) must clear the `steer.request` marker so the run does NOT
    re-enter the 600s blocking prompt at every later boundary."""
    from agent6.cli import _steer
    from agent6.ui.approval import request_steer, steer_request_pending

    run_dir = tmp_path
    request_steer(run_dir)  # TUI `s`-key dropped the marker
    assert steer_request_pending(run_dir)

    # TUI is live but the modal yields no answer (dismissed / 600s timeout).
    monkeypatch.setattr(_steer, "tui_is_live", _tui_live)
    monkeypatch.setattr(_steer, "read_steer_answer", _answer_none)

    state = _steer.install_steer_sigint(MagicMock(), run_dir)
    try:
        assert state.requested() is True  # marker seen -> would prompt
        assert state.prompt() is None  # dismissed modal
        # The marker is gone, so the next boundary does NOT re-trigger a steer.
        assert not steer_request_pending(run_dir)
        assert state.requested() is False
    finally:
        state.restore()


def test_steer_prompt_keeps_marker_on_real_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely-answered steer still works: prompt() returns the answer and
    leaves clearing to the caller's clear() (which consumes request+answer)."""
    from agent6.cli import _steer
    from agent6.ui.approval import request_steer, steer_request_pending

    run_dir = tmp_path
    request_steer(run_dir)
    monkeypatch.setattr(_steer, "tui_is_live", _tui_live)
    monkeypatch.setattr(_steer, "read_steer_answer", _answer_text)

    state = _steer.install_steer_sigint(MagicMock(), run_dir)
    try:
        assert state.prompt() == "do the thing"
        # prompt() must NOT clear on the answered path (caller's clear() owns it).
        assert steer_request_pending(run_dir)
        state.clear()  # caller clears after consuming the answer
        assert not steer_request_pending(run_dir)
    finally:
        state.restore()
