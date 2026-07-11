# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for three loop/resume bugs:

#3  end-of-iteration resume snapshot (don't replay already-executed tools)
#12 completion-relevant scalars survive a resume (metric / verify-settled)
#10 final checkpoint commits a dirty worktree on a gated-run success exit
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from agent6.workflows._metric import MetricSample as _MetricSample
from agent6.workflows._run_state import load_resume_snapshot
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
)


def _silent(_: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    return Workflow(**defaults)


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


# --- #12: completion-relevant scalars round-trip + restore -----------------


def test_snapshot_persists_completion_scalars(tmp_path: Path) -> None:
    """verify_ever_passed / gateless_ever_committed / metric summary are written
    and load back, instead of resetting to their fresh-run defaults."""
    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            require_verify_to_finish=False,
            verify_command=(),
            metric=SimpleNamespace(goal="maximize"),
        )
    )
    wf = _wf(resume_state_path=snap, config=config)
    state = _LoopState(original_task="t", tool_calls=2)
    state.verify_ever_passed = True
    state.gateless_ever_committed = True
    state.metric_history.append(_MetricSample(label="x", score=27.0, returncode=0, at_ceiling=True))
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=2, next_iteration=4, root_task_id=None, state=state
    )
    loaded = load_resume_snapshot(snap)
    assert loaded.verify_ever_passed is True
    assert loaded.gateless_ever_committed is True
    assert loaded.metric_best_score == 27.0
    assert loaded.metric_at_ceiling is True


def test_old_snapshot_without_scalars_loads_with_safe_defaults(tmp_path: Path) -> None:
    """A pre-field snapshot (no completion scalars) still loads, with defaults."""
    snap = tmp_path / "loop_state.json"
    snap.write_text(
        json.dumps(
            {
                "version": 1,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_resume_snapshot(snap)
    assert loaded.verify_ever_passed is False
    assert loaded.gateless_ever_committed is False
    assert loaded.metric_best_score is None
    assert loaded.metric_at_ceiling is False


def test_resume_seeds_state_from_snapshot_scalars() -> None:
    """_drive_loop restores verify_ever_passed and a synthetic at-ceiling metric
    sample so the metric/verify-settled stop logic doesn't regress on resume.

    Drives a single iteration that immediately finishes; the assertion is that
    the loop saw the restored at-ceiling history (no early-finish rejection).
    """
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            require_verify_to_finish=False,
            verify_command=(),
            metric=SimpleNamespace(goal="maximize"),
        )
    )
    provider = MagicMock()
    provider.call.return_value = SimpleNamespace(
        text="",
        tool_uses=({"id": "t1", "name": "finish_run", "input": {"summary": "done"}},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        raw={
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "finish_run",
                    "input": {"summary": "done"},
                }
            ]
        },
    )
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = {"ok": True}
    wf = _wf(provider=provider, dispatcher=dispatcher, config=config, mode="run")

    captured: dict[str, Any] = {}
    orig = wf._metric_at_ceiling  # pyright: ignore[reportPrivateUsage]

    def _spy(history: list[Any]) -> bool:
        captured["at_ceiling"] = orig(history)
        return captured["at_ceiling"]

    wf._metric_at_ceiling = _spy  # type: ignore[method-assign]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=[],
        tool_calls=0,
        start_iteration=3,
        root_task_id=None,
        original_task="go",
        metric_best_score=27.0,
        metric_at_ceiling=True,
    )
    assert result.completed is True
    assert result.reason == "finish_run"
    # The early-finish guard consulted the restored at-ceiling history.
    assert captured.get("at_ceiling") is True


# --- #3: end-of-iteration snapshot (no replay of executed tools) -----------


def test_snapshot_written_after_tool_dispatch_advances_iteration(tmp_path: Path) -> None:
    """After a full iteration (assistant turn + tool dispatch + tool_results),
    the snapshot must advance to next_iteration and include the executed turn,
    so a crash before the next pre-call snapshot resumes AFTER the tools."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    snap = repo / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            require_verify_to_finish=False, verify_command=(), metric=SimpleNamespace(goal=None)
        )
    )
    provider = MagicMock()
    # Iter 1: a run_command tool_use (non-idempotent side effect).
    # Iter 2: finish_run.
    provider.call.side_effect = [
        SimpleNamespace(
            text="",
            tool_uses=({"id": "a1", "name": "run_command", "input": {"command": "echo hi"}},),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "a1",
                        "name": "run_command",
                        "input": {"command": "echo hi"},
                    }
                ]
            },
        ),
        SimpleNamespace(
            text="",
            tool_uses=({"id": "f1", "name": "finish_run", "input": {"summary": "done"}},),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            raw={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "f1",
                        "name": "finish_run",
                        "input": {"summary": "x"},
                    }
                ]
            },
        ),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = {"returncode": 0, "stdout": "hi", "stderr": ""}
    dispatcher.set_run_root_node_id = MagicMock()

    events: list[dict[str, Any]] = []
    wf = _wf(
        root=repo,
        provider=provider,
        dispatcher=dispatcher,
        config=config,
        mode="run",
        resume_state_path=snap,
    )
    orig_save = wf._save_resume_snapshot  # pyright: ignore[reportPrivateUsage]
    orig_call = wf._call_with_retry  # pyright: ignore[reportPrivateUsage]
    orig_compact = wf._maybe_compact  # pyright: ignore[reportPrivateUsage]

    def _spy_save(**kw: Any) -> None:
        orig_save(**kw)
        events.append(
            {
                "kind": "save",
                "next_iteration": kw["next_iteration"],
                "messages": json.loads(json.dumps(kw["messages"])),
            }
        )

    def _spy_call(*a: Any, **kw: Any) -> Any:
        events.append({"kind": "provider_call"})
        return orig_call(*a, **kw)

    def _spy_compact(msgs: Any) -> bool:
        events.append({"kind": "compact"})
        return orig_compact(msgs)

    wf._save_resume_snapshot = _spy_save  # type: ignore[method-assign]
    wf._call_with_retry = _spy_call  # type: ignore[method-assign]
    wf._maybe_compact = _spy_compact  # type: ignore[method-assign]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="go",
    )

    # The KEY guarantee: a snapshot advancing to next_iteration=2 (with the
    # executed iter-1 turn) must be written at the END of iter 1 -- i.e. AFTER
    # the first provider call but BEFORE iter 2's compaction/pre-call snapshot.
    # That closes the crash window between tool dispatch and iter-2's pre-call
    # save. On the old code the FIRST save after provider-call-1 was iter-2's
    # OWN pre-call save, which happens AFTER iter-2's compaction.
    kinds = [ev["kind"] for ev in events]
    first_call = kinds.index("provider_call")
    second_compact = next(i for i, k in enumerate(kinds) if k == "compact" and i > first_call)
    end_of_iter_saves = [
        ev
        for i, ev in enumerate(events)
        if first_call < i < second_compact and ev["kind"] == "save"
    ]
    assert end_of_iter_saves, (
        "expected an end-of-iteration snapshot between the 1st provider call"
        " and iter-2's compaction (the post-tool-dispatch crash window)"
    )
    post = [s for s in end_of_iter_saves if s["next_iteration"] == 2]
    assert post, "end-of-iteration snapshot must advance next_iteration to 2"
    msgs = post[0]["messages"]
    assert any(m.get("role") == "assistant" for m in msgs), "assistant turn must be snapshotted"
    has_tool_result = any(
        m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        for m in msgs
    )
    assert has_tool_result, "executed tool_result must be in the advanced snapshot"


# --- #10: final checkpoint commits a dirty worktree on a gated run ---------


def test_final_checkpoint_commits_dirty_worktree_on_gated_run(tmp_path: Path) -> None:
    """A run_command-authored edit left uncommitted on a gated run is captured
    by _final_checkpoint so it isn't lost from git history at exit."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            require_verify_to_finish=False,
            verify_command=("pytest", "-q"),
            metric=SimpleNamespace(goal=None),
        )
    )
    wf = _wf(root=repo, config=config, mode="run")
    # Worker edited a file via run_command; never re-verified, never committed.
    (repo / "edit.txt").write_text("a real edit\n")
    head_before = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf._final_checkpoint(5)  # pyright: ignore[reportPrivateUsage]

    head_after = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head_after != head_before, "dirty worktree must be committed on exit"
    status = sp.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert status == "", "worktree must be clean after the final checkpoint"
    subject = sp.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert "checkpoint" in subject


def test_final_checkpoint_noop_when_clean_or_not_run_mode(tmp_path: Path) -> None:
    """No commit when the tree is clean, and never in non-run mode."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    config = SimpleNamespace(
        workflow=SimpleNamespace(
            require_verify_to_finish=False,
            verify_command=("pytest",),
            metric=SimpleNamespace(goal=None),
        )
    )
    head = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    wf_clean = _wf(root=repo, config=config, mode="run")
    wf_clean._final_checkpoint(1)  # pyright: ignore[reportPrivateUsage]
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )

    # Dirty tree, but plan mode -> still no commit.
    (repo / "edit.txt").write_text("plan-mode edit\n")
    wf_plan = _wf(root=repo, config=config, mode="plan")
    wf_plan._final_checkpoint(1)  # pyright: ignore[reportPrivateUsage]
    assert (
        sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        == head
    )
