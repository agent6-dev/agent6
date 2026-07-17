# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration coverage for Workflow critic modes other than
``before_finish`` (which is exercised in test_critic.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent6.tools.results import ExecResult, RawResult
from tests.unit.test_critic import (
    _finish_tool_use,  # pyright: ignore[reportPrivateUsage]
    _resp,  # pyright: ignore[reportPrivateUsage]
    _resp_with_tool_use,  # pyright: ignore[reportPrivateUsage]
    _wf,  # pyright: ignore[reportPrivateUsage]
)


def _verify_pass_tool_use(tu_id: str) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tu_id,
        "name": "run_verify_command",
        "input": {},
    }


# --- periodic critic ----------------------------------------------------


def test_periodic_critic_fires_every_n_iterations() -> None:
    """critic_mode=periodic with critic_period=2 must call the critic
    on iters 2 and 4 only, never on iters 1 or 3."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),
        _resp_with_tool_use("t4", _verify_pass_tool_use("v4")),
        _resp_with_tool_use("done", _finish_tool_use("f", "summary")),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("looks fine\n\nVERDICT: SATISFIED")
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = ExecResult(
        returncode=0, stdout="", stderr="", duration_s=0.0, exec_failed=False
    )
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="periodic",
        critic_period=2,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    # Each verify pass commits real progress (the normal success path), so the
    # verify-settled detector stays dormant and all 5 iterations run.
    with patch("agent6.workflows.loop.commit_all", return_value="sha"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )
    assert result.iterations == 5
    assert result.reason == "finish_run"
    # iters 2 and 4 trigger periodic critic. iter 5 is finish_run which
    # under critic_mode=periodic is NOT gated -> total 2 critic calls.
    assert critic.call.call_count == 2


def test_periodic_critic_injects_text_into_next_user_msg() -> None:
    """When the periodic critic returns text, it must appear in the
    user message appended after the firing iteration."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("* CONSIDER X\n\nVERDICT: SATISFIED")
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = ExecResult(
        returncode=0, stdout="", stderr="", duration_s=0.0, exec_failed=False
    )
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="periodic",
        critic_period=1,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        messages=messages,
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
    )
    iter1_user_msg = messages[2]
    text_blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[critic]" in b["text"] for b in text_blocks)
    assert any("CONSIDER X" in b["text"] for b in text_blocks)


# --- on_verify_fail critic ---------------------------------------------


def test_on_verify_fail_critic_fires_only_on_nonzero_exit() -> None:
    """critic_mode=on_verify_fail: critic called only on iters where
    run_verify_command returned exit != 0. Passing verify must NOT fire."""
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),  # passes
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),  # FAILS
        _resp_with_tool_use("t3", _verify_pass_tool_use("v3")),  # passes
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("hmm\n\nVERDICT: NEEDS_WORK")
    dispatcher = MagicMock()
    dispatcher.dispatch.side_effect = [
        ExecResult(returncode=0, stdout="", stderr="", duration_s=0.0, exec_failed=False),
        ExecResult(
            returncode=1, stdout="", stderr="test x failed", duration_s=0.0, exec_failed=False
        ),
        ExecResult(returncode=0, stdout="", stderr="", duration_s=0.0, exec_failed=False),
        RawResult({"ok": True}),
    ]
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="on_verify_fail",
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        messages=messages,
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
    )
    assert result.iterations == 4
    assert result.reason == "finish_run"
    assert critic.call.call_count == 1
    iter2_user_msg = messages[4]
    text_blocks = [b for b in iter2_user_msg["content"] if b.get("type") == "text"]
    assert any("[critic]" in b["text"] for b in text_blocks)


def test_on_verify_fail_critic_skipped_when_no_verify_call() -> None:
    """A pure-edit iteration with no run_verify_command in tool_uses
    must not fire on_verify_fail critic (no failure signal)."""
    edit_tool = {
        "type": "tool_use",
        "id": "e1",
        "name": "list_dir",
        "input": {"path": "."},
    }
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("editing", edit_tool),
        _resp_with_tool_use("done", _finish_tool_use("f", "ok")),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("VERDICT: SATISFIED")
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"entries": []})
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="on_verify_fail",
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        messages=messages,
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
    )
    assert critic.call.call_count == 0
