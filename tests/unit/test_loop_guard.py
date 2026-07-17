# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""degenerate-loop guard in Workflow._drive_loop.

When the worker calls the same (tool_name, args) back-to-back >=3 times,
the workflow appends a one-shot "loop-guard" text block to the next user
turn telling the worker the result has not changed and to pivot. Behaviour
observed live with Kimi K2.6 on the perf takehome: 15 consecutive
`read_file(path="problem.py")` calls returning the same 19826 bytes,
followed by went_quiet.
"""

from __future__ import annotations

import subprocess as _sp
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.providers import ProviderResponse
from agent6.tools.results import RawResult
from agent6.workflows.loop import Workflow


def _silent(_msg: str) -> None:
    return None


def _resp_with_tool(name: str, args: dict[str, Any], tu_id: str = "tu1") -> ProviderResponse:
    """Provider response with a single tool_use."""
    block = {"type": "tool_use", "id": tu_id, "name": name, "input": args}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


def _resp_text(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    _sp.run(["git", "add", "x.txt"], cwd=repo, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _build_wf(repo: Path, provider: MagicMock, dispatcher: MagicMock) -> Workflow:
    return Workflow(
        root=repo,
        config=MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=10,
    )


def _loop_guard_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """Extract the text of every [loop-guard] block injected into user turns."""
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if text.startswith("[loop-guard]"):
                out.append(text)
    return out


def test_loop_guard_fires_on_three_identical_calls(tmp_path: Path) -> None:
    """3 back-to-back identical tool calls -> one loop-guard notice appended."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # Turns 1-3: same read_file call. Turn 4: silent finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t3"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read the file")

    assert result.completed is True
    assert provider.call.call_count == 4

    # Reconstruct messages from the provider call history (last call's
    # messages arg holds the full conversation).
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    assert len(notices) == 1, f"expected exactly one notice, got {len(notices)}: {notices}"
    assert "read_file" in notices[0]
    assert "3 times" in notices[0]


def test_loop_guard_does_not_fire_when_args_change(tmp_path: Path) -> None:
    """Different args every turn -> no loop-guard notice."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "c.txt"}, tu_id="t3"),
        _resp_with_tool("read_file", {"path": "d.txt"}, tu_id="t4"),
        _resp_text("ok"),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "x"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read several files")

    assert result.completed is True
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    assert _loop_guard_blocks(final_messages) == []


def test_loop_guard_does_not_re_fire_back_to_back(tmp_path: Path) -> None:
    """Once notice is emitted at iter N, do not emit again at iter N+1 even if streak continues."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # 5 identical calls then finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(5)
    ] + [_resp_text("ok")]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.run("loop")

    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    # The guard fires once when streak hits 3. The
    # `repeat_warning_emitted_at < iteration - 1` gate suppresses
    # re-emission at iter 4 (consecutive) but allows re-emission at
    # iter 5 (one-iteration gap) if the streak persists. So we expect
    # 1 or 2 notices, but NOT one per iteration.
    assert 1 <= len(notices) <= 2, f"expected 1-2 notices, got {len(notices)}"


def test_loop_guard_kills_run_when_streak_passes_threshold(tmp_path: Path) -> None:
    """Notice is advisory; when the streak reaches
    `loop_guard_kill_threshold` the run terminates."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    # 12 identical calls. Threshold=5 -> kill at iter 5.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ] + [_resp_text("never reached")]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = Workflow(
        root=repo,
        config=MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=20,
        loop_guard_kill_threshold=5,
    )
    result = wf.run("loop forever")

    assert result.completed is False
    assert result.reason == "loop_guard_killed"
    assert provider.call.call_count == 5
    assert "read_file" in result.summary
    assert "5x" in result.summary or "5 " in result.summary


def test_loop_guard_kill_disabled_when_threshold_zero(tmp_path: Path) -> None:
    """`loop_guard_kill_threshold = 0` restores notice-only behaviour."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(6)
    ] + [_resp_text("done")]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"content": "hi\n"})

    wf = Workflow(
        root=repo,
        config=MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        provider_retry_count=0,
        provider_retry_delay_s=0.0,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    result = wf.run("loop")

    assert result.completed is True
    assert provider.call.call_count == 7
