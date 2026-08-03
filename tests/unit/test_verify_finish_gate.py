# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The verify finish gate: finish_session can never report 'passed' over a red or
stale verify (honest default), and require_verify_to_finish turns that into an
opt-in hard gate. Both ground on _tree_is_verify_green."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
)


def _wf(
    *,
    verify: bool,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
    root: Path = Path("/tmp"),
) -> Workflow:
    data: dict[str, Any] = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Workflow(
        root=root,
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        mode=mode,
    )


def _green(wf: Workflow, **state_kw: Any) -> bool | None:
    state = _LoopState(original_task="t", tool_calls=0, **state_kw)
    return wf._tree_is_verify_green(state)  # pyright: ignore[reportPrivateUsage]


def test_no_verify_command_is_not_gated() -> None:
    # Nothing to gate on -> None -> finish is always an honest pass.
    assert _green(_wf(verify=False), last_verify_ok=None) is None
    assert _green(_wf(verify=False), last_verify_ok=False) is None


def test_green_only_when_last_verify_passed_and_tree_unedited() -> None:
    wf = _wf(verify=True)
    assert _green(wf, last_verify_ok=True, edited_since_verify=False) is True
    # Never verified, or last verify failed -> not green.
    assert _green(wf, last_verify_ok=None) is False
    assert _green(wf, last_verify_ok=False) is False
    # A green verify that has since been edited over is stale -> not green.
    assert _green(wf, last_verify_ok=True, edited_since_verify=True) is False


def test_require_verify_to_finish_defaults_off() -> None:
    assert Config().workflow.require_verify_to_finish is False


def _verified(wf: Workflow, **state_kw: Any) -> str:
    state = _LoopState(original_task="t", tool_calls=0, **state_kw)
    return wf._verification(state)  # pyright: ignore[reportPrivateUsage]


def test_verification_carries_the_same_verdict_the_event_does() -> None:
    """SessionResult.verified is the app layer's copy of session.end.all_passed's
    grounding, so exit code, auto-merge, and the notify hook read the verify
    truth instead of `completed` (true for any deliberate finish)."""
    assert _verified(_wf(verify=True), last_verify_ok=True, edited_since_verify=False) == "passed"
    assert _verified(_wf(verify=True), last_verify_ok=False) == "failed"
    # Green but edited since: stale, not verified.
    assert _verified(_wf(verify=True), last_verify_ok=True, edited_since_verify=True) == "failed"
    # Gateless: nothing ever gated this run, so there is no verdict to claim.
    assert _verified(_wf(verify=False), last_verify_ok=None) == "not_applicable"


def test_plan_and_ask_are_never_gated_on_verify() -> None:
    """plan/ask end clean whatever the tree looks like -- finish_planning and
    the ask answer both emit session.end all_passed=True -- so they have no verify
    verdict to report. Reporting one made `agent6 plan` exit 4 (preflight
    INFERS a verify command for plan, and plan never runs it, so the tree read
    as red) while its own journal and every listing said passed."""
    for mode in ("plan", "ask"):
        assert _verified(_wf(verify=True, mode=mode), last_verify_ok=None) == "not_applicable"
        assert _verified(_wf(verify=True, mode=mode), last_verify_ok=False) == "not_applicable"


def test_a_command_that_dirties_the_tree_invalidates_the_verify_pass(tmp_path: Path) -> None:
    """A green verify must not survive a run_command that changed the tree:
    edited_since_verify was set only by apply_edit/apply_patch, so a model
    could verify green, then mutate through run_command (or an MCP tool) and
    still finish reporting verified="passed" -- defeating exit 4,
    require_verify_to_finish, and the auto-merge gate together. Grounded on
    git, so a read-only command keeps the pass it had."""
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)

    dirty = _wf(verify=True, root=tmp_path)._left_the_tree_dirty  # pyright: ignore[reportPrivateUsage]

    assert dirty("run_command") is False  # clean tree: a read-only probe costs nothing
    assert dirty("grep") is False  # never asked of in-process read tools
    (tmp_path / "a.txt").write_text("mutated\n", encoding="utf-8")
    assert dirty("run_command") is True
    assert dirty("mcp__srv__write") is True
    # verify/metric are the operator's own gates; their caches must not
    # invalidate the pass they just produced.
    assert dirty("run_verify_command") is False
    assert dirty("run_metric_command") is False
