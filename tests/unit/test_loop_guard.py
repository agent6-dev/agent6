# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The degenerate-loop guard in Harness._drive_loop.

The same (tool_name, args) called back-to-back three times draws a one-shot loop-guard block telling
the worker the result has not changed.
"""

from __future__ import annotations

import dataclasses
import itertools
import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

import pytest

from agent6.harness import guards, loop, loop_resume, provider_call, run_chain
from agent6.providers import ProviderResponse
from agent6.sessions import journal
from agent6.tools import results


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


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    subprocess.run(["git", "add", "x.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _build_wf(
    repo: pathlib.Path, provider: mock.MagicMock, dispatcher: mock.MagicMock
) -> loop.Harness:
    return loop.Harness(
        chain=run_chain.RunChain(
            repo,
            ref="refs/agent6/guard",
            fallback_parent=subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
            ).stdout.strip(),
        ),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file=""),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_at_start=True,
                verify_retries=2,
            ),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=10,
    )


def _loop_guard_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """Extract the text of every [loop-guard] block injected into user messages."""
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


def test_loop_guard_fires_on_three_identical_calls(tmp_path: pathlib.Path) -> None:
    """3 back-to-back identical tool calls -> one loop-guard notice appended."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    # Turns 1-3: same read_file call. Turn 4: silent finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t3"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read the file")

    assert result.completed is True
    assert provider.call.call_count == 4

    # The last call's messages arg holds the full conversation.
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    assert len(notices) == 1, f"expected exactly one notice, got {len(notices)}: {notices}"
    assert "read_file" in notices[0]
    assert "3 times" in notices[0]


def test_polling_a_growing_result_is_not_a_spiral(tmp_path: pathlib.Path) -> None:
    """Polling a background job with `read_background` is not a repeated call; other repeats are.

    `run_command`'s own description tells the model to poll, with args that never change until the
    job ends.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    provider.call.side_effect = [
        *(_resp_with_tool("read_background", {"id": "bg-1"}, tu_id=f"t{i}") for i in range(1, 10)),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    tail = iter(f"line {i}\n" * i for i in range(1, 20))

    def _growing_tail(*_args: object, **_kwargs: object) -> results.RawResult:
        return results.RawResult({"output": next(tail)})

    dispatcher.dispatch.side_effect = _growing_tail

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("watch the build")

    assert result.completed is True, result.reason
    assert result.reason != "loop_guard_killed"
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    assert _loop_guard_blocks(final_messages) == []

    # The same shape on a non-poll tool is a spiral: a command re-run with identical arguments.
    spiraller = mock.MagicMock()
    spiraller.call.side_effect = itertools.chain(
        (_resp_with_tool("run_command", {"argv": ["pytest"]}, tu_id=f"s{i}") for i in range(1, 10)),
        itertools.repeat(_resp_text("ok")),
    )
    spins = mock.MagicMock(operator_wait_s=0.0, execution_id="execution", last_verification=None)
    runs = iter(f"1 failed in {i}.0s\n" for i in range(1, 20))

    def _timestamped(*_args: object, **_kwargs: object) -> results.RawResult:
        return results.RawResult({"stdout": next(runs)})

    spins.dispatch.side_effect = _timestamped
    wf2 = _build_wf(repo, spiraller, spins)
    wf2.run("make the tests pass")
    last2 = spiraller.call.call_args_list[-1]
    msgs2: list[dict[str, Any]] = last2.kwargs.get("messages") or last2.args[1]
    assert _loop_guard_blocks(msgs2), "a repeated command with a changing duration is a spiral"


def test_loop_guard_does_not_fire_when_args_change(tmp_path: pathlib.Path) -> None:
    """Different args every turn -> no loop-guard notice."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_with_tool("read_file", {"path": "c.txt"}, tu_id="t3"),
        _resp_with_tool("read_file", {"path": "d.txt"}, tu_id="t4"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})

    wf = _build_wf(repo, provider, dispatcher)
    result = wf.run("read several files")

    assert result.completed is True
    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    assert _loop_guard_blocks(final_messages) == []


def test_loop_guard_does_not_re_fire_back_to_back(tmp_path: pathlib.Path) -> None:
    """Once notice is emitted at turn N, do not emit again at turn N+1 even if streak continues."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    # 5 identical calls then finish.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(5)
    ] + [_resp_text("ok")]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.run("loop")

    last_args = provider.call.call_args_list[-1]
    final_messages: list[dict[str, Any]] = last_args.kwargs.get("messages") or last_args.args[1]
    notices = _loop_guard_blocks(final_messages)
    # Fires at streak 3, suppressed at turn 4, re-emitted at turn 5: two notices, not one per turn.
    assert 1 <= len(notices) <= 2, f"expected 1-2 notices, got {len(notices)}"


def test_loop_guard_kills_run_when_streak_passes_threshold(tmp_path: pathlib.Path) -> None:
    """The notice is advisory; at `loop_guard_kill_threshold` the streak ends the run."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    # 12 identical calls. Threshold=5 -> kill at turn 5.
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ] + [_resp_text("never reached")]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = loop.Harness(
        chain=run_chain.RunChain(repo),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file=""),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_at_start=True,
                verify_retries=2,
            ),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=5)
    result = wf.run("loop forever")

    assert result.completed is False
    assert result.reason == "loop_guard_killed"
    assert provider.call.call_count == 5
    assert "read_file" in result.summary
    assert "5x" in result.summary or "5 " in result.summary


def test_loop_guard_kill_disabled_when_threshold_zero(tmp_path: pathlib.Path) -> None:
    """`loop_guard_kill_threshold = 0` restores notice-only behaviour."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(6)
    ] + [_resp_text("done")]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = loop.Harness(
        chain=run_chain.RunChain(repo),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file=""),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_at_start=True,
                verify_retries=2,
            ),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    result = wf.run("loop")

    assert result.completed is True
    assert provider.call.call_count == 7


def _knobs(cfg: Any, **knobs: Any) -> Any:
    """The mocked config with `[harness]` guard knobs set."""
    for key, value in knobs.items():
        setattr(cfg.harness, key, value)
    return cfg


def _gated_wf(
    repo: pathlib.Path, provider: mock.MagicMock, dispatcher: mock.MagicMock, **kw: Any
) -> loop.Harness:
    """Return a harness with a gate it never runs itself (`verify_when = "never"`).

    An editing turn is checkpointed on the chain whatever the gate policy, so a run_command-authored
    edit reaches git at its turn, and an abnormal end checkpoints whatever is left.
    """
    return loop.Harness(
        chain=run_chain.RunChain(
            repo,
            ref="refs/agent6/guard",
            fallback_parent=subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
            ).stdout.strip(),
        ),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file=""),
            git=types.SimpleNamespace(
                control="agent6",
                commit_per_turn=True,
                commit=types.SimpleNamespace(
                    checkpoint=types.SimpleNamespace(message="agent6"),
                    name="",
                    email="",
                    trailer="",
                ),
            ),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                settled_stop_after=12,
                stagnation_notice_after_s=300.0,
                verify_command=("true",),
                verify_when="never",
                verify_at_start=True,
                verify_retries=2,
            ),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=_silent,
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        **kw,
    )


def _dirtying_dispatcher(repo: pathlib.Path) -> mock.MagicMock:
    """A dispatcher whose tool leaves an uncommitted edit, as run_command does."""
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )

    def dispatch(*_args: Any, **_kwargs: Any) -> results.RawResult:
        (repo / "edit.txt").write_text("run_command wrote this\n")
        return results.RawResult({"content": "hi\n"})

    dispatcher.dispatch.side_effect = dispatch
    return dispatcher


def _chain_holds(repo: pathlib.Path, path: str) -> bool:
    """Return whether the chain tip's tree holds the file: the edit reached git."""
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"refs/agent6/guard:{path}"],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def test_loop_guard_kill_checkpoints_the_dirty_worktree(tmp_path: pathlib.Path) -> None:
    """A harness-initiated stop does not drop run_command-authored edits.

    Every surface (diff, merge, resume, score) reads git history, not the worktree.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ] + [_resp_text("never reached")]

    wf = _gated_wf(
        repo,
        provider,
        _dirtying_dispatcher(repo),
        max_turns=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=5)
    result = wf.run("loop forever")

    assert result.reason == "loop_guard_killed"
    assert _chain_holds(repo, "edit.txt")


def test_max_turns_stop_checkpoints_the_dirty_worktree(tmp_path: pathlib.Path) -> None:
    """The contract the loop-guard kill has to match."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id=f"t{i}") for i in range(12)
    ]

    wf = _gated_wf(
        repo,
        provider,
        _dirtying_dispatcher(repo),
        max_turns=3,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    result = wf.run("keep going")

    assert result.reason == "max_turns"
    assert _chain_holds(repo, "edit.txt")


def test_budget_exhausted_checkpoints_the_dirty_worktree(tmp_path: pathlib.Path) -> None:
    """Every harness-initiated end checkpoints, a BudgetExceededError on the next call included."""
    from agent6 import budget

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        budget.BudgetExceededError("input cap reached"),
    ]
    wf = _gated_wf(repo, provider, _dirtying_dispatcher(repo), max_turns=5)
    result = wf.run("do the thing")
    assert result.reason == "budget_exhausted"
    assert _chain_holds(repo, "edit.txt")


def test_provider_error_checkpoints_the_dirty_worktree(tmp_path: pathlib.Path) -> None:
    """A fatal provider error is a harness-initiated end too: the run's edits land in git."""
    from agent6.providers import ProviderError

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        ProviderError("HTTP 401", status_code=401),
    ]
    wf = _gated_wf(repo, provider, _dirtying_dispatcher(repo), max_turns=5)
    result = wf.run("do the thing")
    assert result.reason == "provider_error"
    assert _chain_holds(repo, "edit.txt")


def test_went_quiet_checkpoints_the_dirty_worktree(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A model that starves into empty turns after real edits keeps them in git history."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        _resp_text(""),  # no text, no tool_use -> went_quiet
    ]
    wf = _gated_wf(
        repo,
        provider,
        _dirtying_dispatcher(repo),
        max_turns=5,
    )
    wf.config = _knobs(wf.config, went_quiet_max_nudges=0)
    result = wf.run("do the thing")
    assert result.reason == "went_quiet"
    assert _chain_holds(repo, "edit.txt")


def test_unexecutable_verify_abort_checkpoints_the_dirty_worktree(tmp_path: pathlib.Path) -> None:
    """A verify that cannot execute in the jail still ends with the run's edits checkpointed."""
    from agent6.tools import tool_errors

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        _resp_with_tool("run_verify_command", {}, tu_id="tu2"),
        _resp_text("never reached"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )

    def dispatch(name: str, *_a: Any, **_k: Any) -> results.RawResult:
        if name == "run_verify_command":
            raise tool_errors.OperatorCommandUnexecutableError(
                "verify binary missing from the jail PATH"
            )
        (repo / "edit.txt").write_text("run_command wrote this\n")
        return results.RawResult({"content": "hi\n"})

    dispatcher.dispatch.side_effect = dispatch
    wf = _gated_wf(repo, provider, dispatcher, max_turns=5)
    result = wf.run("do the thing")
    assert result.reason == "verify_command_unexecutable"
    assert _chain_holds(repo, "edit.txt")


def _stagnation_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """Every [stagnation] block injected into user messages."""
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.startswith("[stagnation]"):
                    out.append(text)
    return out


def _final_messages(provider: mock.MagicMock) -> list[dict[str, Any]]:
    last_args = provider.call.call_args_list[-1]
    return last_args.kwargs.get("messages") or last_args.args[1]


def test_stagnation_notice_fires_once_without_attempts(tmp_path: pathlib.Path) -> None:
    """Wall clock past the threshold with no edit and no verify injects the notice once.

    Recall spirals make few calls with long reasoning between them, so the identical-signature guard
    never sees them.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        # An attemptless prose end gets one silent-no-work nudge round first.
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.config = _knobs(wf.config, stagnation_notice_after_s=1e-9)
    result = wf.run("investigate")
    assert result.completed is True
    notices = _stagnation_blocks(_final_messages(provider))
    assert len(notices) == 1, notices
    # No verify command here, so the notice names no gate the prompt says the run lacks.
    assert "nothing edited yet" in notices[0]
    assert "verify" not in notices[0]

    # A gate the policy withholds is not a gate either: `run_commands = "deny"` takes it away.
    denied = mock.MagicMock()
    denied.call.side_effect = itertools.chain(
        [_resp_with_tool("read_file", {"path": "x.txt"}, tu_id="d1")],
        itertools.repeat(_resp_text("ok")),
    )
    no_commands = mock.MagicMock(
        operator_wait_s=0.0, execution_id="execution", last_verification=None
    )
    no_commands.dispatch.return_value = results.RawResult({"content": "x"})
    no_commands.command_policy.return_value = "deny"
    wf3 = _build_wf(repo, denied, no_commands)
    wf3.config.harness.verify_command = ("true",)
    wf3.config = _knobs(wf3.config, stagnation_notice_after_s=1e-9)
    wf3.run("investigate")
    assert "nothing edited yet" in _stagnation_blocks(_final_messages(denied))[0]

    # With a gate the run can actually reach, the same notice names it.
    gated = mock.MagicMock()
    gated.call.side_effect = itertools.chain(
        [_resp_with_tool("read_file", {"path": "x.txt"}, tu_id="g1")],
        itertools.repeat(_resp_text("ok")),
    )
    wf2 = _build_wf(repo, gated, dispatcher)
    wf2.config.harness.verify_command = ("true",)
    wf2.config = _knobs(wf2.config, stagnation_notice_after_s=1e-9)
    wf2.run("investigate")
    assert "no edit and no verify" in _stagnation_blocks(_final_messages(gated))[0]


def test_stagnation_ignores_time_blocked_on_the_operator(tmp_path: pathlib.Path) -> None:
    """The stagnation clock is the model's own time; an hour at an approval is not research."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )

    def dispatch(*_args: object, **_kwargs: object) -> results.RawResult:
        dispatcher.operator_wait_s = 3600.0  # the approval the model's own call waited on
        return results.RawResult({"content": "x"})

    dispatcher.dispatch.side_effect = dispatch
    wf = _build_wf(repo, provider, dispatcher)
    wf.config = _knobs(wf.config, stagnation_notice_after_s=1e-9)
    result = wf.run("investigate")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_stagnation_counts_from_the_window_not_from_the_start_gates_approval(
    tmp_path: pathlib.Path,
) -> None:
    """The start gate's approval wait, counted before the window opened, was subtracted again."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "a.txt"}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "b.txt"}, tu_id="t2"),
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        execution_id="execution", last_verification=None, operator_wait_s=3600.0
    )  # waited on before the first turn
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.config = _knobs(wf.config, stagnation_notice_after_s=1e-9)
    assert wf.run("investigate").completed is True
    assert len(_stagnation_blocks(_final_messages(provider))) == 1


def test_stagnation_ignores_the_start_gates_time(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start gate that outlasted the knob fired the notice at the model's first turn."""
    clock = [0.0]
    monkeypatch.setattr(guards, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = itertools.chain(
        [_resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1")],
        itertools.repeat(_resp_text("ok")),
    )
    gates: list[int] = []

    def run_verify(**_kw: object) -> results.ExecResult:
        if not gates:
            clock[0] += 1.0  # past the knob, then timed out: no baseline either
        gates.append(124 if not gates else 0)
        return results.ExecResult(
            returncode=gates[-1],
            stdout="",
            stderr="",
            duration_s=1.0,
            exec_failed=False,
            timed_out=len(gates) == 1,
        )

    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    dispatcher.command_policy.return_value = "allow"
    dispatcher.run_verify.side_effect = run_verify
    wf = _build_wf(repo, provider, dispatcher)
    wf.config.harness.verify_command = ("true",)
    wf.config = _knobs(wf.config, verify_when="finish", stagnation_notice_after_s=0.9)
    wf.run("investigate")
    assert gates[0] == 124
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_stagnation_ignores_the_finish_gates_time(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finish gate that outlasted the knob fired the notice when its red result came back."""
    clock = [0.0]
    monkeypatch.setattr(guards, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = itertools.repeat(
        _resp_with_tool("finish_session", {"summary": "done"})
    )
    gates: list[int] = []

    def run_verify(**_kw: object) -> results.ExecResult:
        clock[0] += 12.3  # the gate alone outlasts the knob
        gates.append(1)
        return results.ExecResult(
            returncode=1,
            stdout="F.........\n1 failed, 9 passed in 12.3s\n",
            stderr="",
            duration_s=12.3,
            exec_failed=False,
        )

    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    dispatcher.command_policy.return_value = "allow"
    dispatcher.run_verify.side_effect = run_verify
    wf = _build_wf(repo, provider, dispatcher)
    # A base the gate can compare the clean tree with: its red is the base's, no verify of the
    # run's own, so only the window's accounting keeps the notice quiet.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    wf.chain = dataclasses.replace(wf.chain, base_sha=head)
    wf.config.harness.verify_command = ("true",)
    wf.config = _knobs(
        wf.config,
        verify_when="finish",
        verify_at_start=False,
        verify_retries=1,
        stagnation_notice_after_s=0.9,
    )
    logged: list[str] = []
    wf.logger = logged.append
    result = wf.run("check the gate")
    assert gates and result.reason == "gate_red_at_base"
    assert not [line for line in logged if "stagnation:" in line], logged


def test_stagnation_notice_suppressed_by_an_edit(tmp_path: pathlib.Path) -> None:
    """An edit attempt before the threshold means no notice: the guard targets attemptless runs."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("apply_edit", {"path": "x.txt", "edits": []}, tu_id="t1"),
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t2"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.config = _knobs(wf.config, stagnation_notice_after_s=1e-9)
    result = wf.run("fix it")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_stagnation_notice_zero_disables(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}, tu_id="t1"),
        _resp_text("ok"),
        _resp_text("ok"),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "x"})
    wf = _build_wf(repo, provider, dispatcher)
    wf.config = _knobs(wf.config, stagnation_notice_after_s=0.0)
    result = wf.run("look around")
    assert result.completed is True
    assert _stagnation_blocks(_final_messages(provider)) == []


def test_unlimited_turns_is_minus_one(tmp_path: pathlib.Path) -> None:
    """[harness].max_turns = -1 runs unbounded, not range(start, 0)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        *(_resp_with_tool("read_file", {"path": f"x{i}.txt"}, tu_id=f"t{i}") for i in range(12)),
        _resp_text("ok"),
    ]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.max_turns = -1
    result = wf.run("read the files")

    assert result.completed is True
    assert provider.call.call_count == 13


def test_resume_execution_rearms_the_turn_allowance(tmp_path: pathlib.Path) -> None:
    """A resumed execution gets a fresh max_turns window relative to its own start."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [_resp_text("done")]
    dispatcher = mock.MagicMock(
        last_verification=None,
        execution_id="execution",
        operator_wait_s=0.0,
        verify_withheld=False,
    )
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    wf = _build_wf(repo, provider, dispatcher)
    wf.max_turns = 5
    wf.resume_state_path = tmp_path / "loop_state.json"
    from agent6.harness import loop_state

    (
        loop_state.LoopState(original_task="t", tool_calls=0).system,
        loop_state.LoopState(original_task="t", tool_calls=0).tool_calls,
        loop_state.LoopState(original_task="t", tool_calls=0).root_task_id,
    ) = "s", 0, None
    loop_resume.save_resume_snapshot(
        wf, loop_state.LoopState(original_task="t", tool_calls=0), [], next_turn=6
    )
    result = wf.resume()

    assert result.completed is True
    assert provider.call.call_count == 1


def test_the_notice_fires_at_three_and_re_arms_after_a_quiet_turn() -> None:
    """The third identical call draws the notice, the next turn is quiet, the one after hears it.

    A shorter streak draws nothing.
    """
    from agent6.harness import guards, loop_conversation, loop_state
    from tests.unit.turn_context import turn_context

    state = loop_state.LoopState(original_task="t", tool_calls=0)
    ctx = turn_context()

    def turn(turn: int) -> loop_state.TurnState:
        return loop_state.TurnState(
            number=turn,
            resp=mock.MagicMock(),
            assistant=loop_conversation.AssistantMessage((), ()),
        )

    for _ in range(2):
        state.spiral.note_call('read_file:{"path": "x.txt"}')
    assert guards.loop_guard_notice(turn(2), state, ctx) is None
    state.spiral.note_call('read_file:{"path": "x.txt"}')
    nudge = guards.loop_guard_notice(turn(3), state, ctx)
    assert nudge is not None and "`read_file`" in nudge.text and "3 times" in nudge.text
    assert nudge.event == journal.LoopLoopGuardTriggered(turn=3, tool="read_file", streak=3)
    state.spiral.note_call('read_file:{"path": "x.txt"}')
    assert guards.loop_guard_notice(turn(4), state, ctx) is None
    state.spiral.note_call('read_file:{"path": "x.txt"}')
    again = guards.loop_guard_notice(turn(5), state, ctx)
    assert again is not None and "5 times" in again.text


def test_the_kill_is_a_hard_stop_at_the_threshold_and_off_at_zero() -> None:
    """At the threshold the stop names the tool and the streak everywhere; below it, nothing.

    In the summary, the event fields and the log line; the knob at 0 disables it.
    """
    from agent6.harness import guards, loop_conversation, loop_state
    from tests.unit.turn_context import turn_context

    state = loop_state.LoopState(original_task="t", tool_calls=0)
    turn = loop_state.TurnState(
        number=5, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    ctx = turn_context(loop_guard_kill_threshold=5)
    for _ in range(4):
        state.spiral.note_call('read_file:{"path": "x.txt"}')
    assert guards.loop_guard_kill(turn, state, ctx) is None
    state.spiral.note_call('read_file:{"path": "x.txt"}')
    stop = guards.loop_guard_kill(turn, state, ctx)
    assert stop is not None and stop.soft == "" and stop.declared == ""
    end = stop.end()
    assert end.reason == "loop_guard_killed" and end.completed is False
    assert end.summary == (
        "loop-guard killed run: `read_file` called 5x in a row with identical"
        " arguments (threshold 5)"
    )
    assert end.fields == {"tool": "read_file", "streak": 5}
    assert stop.log == (
        "LOOP: loop_guard_killed at turn 5 - read_file called 5x in a row (threshold=5)"
    )
    off = turn_context(loop_guard_kill_threshold=0)
    assert guards.loop_guard_kill(turn, state, off) is None


def test_a_checkpoint_names_the_subtask_the_turn_worked_on(tmp_path: pathlib.Path) -> None:
    """An editing turn's chain commit carries the surfaced subtask in its `Agent6-Task` trailer."""
    from agent6 import budget
    from agent6.graph import curator as graph_curator
    from agent6.graph import models
    from agent6.sessions import layout as sessions_layout
    from agent6.vcs import git_read

    repo = tmp_path / "repo"
    _init_repo(repo)
    curator = graph_curator.GraphCurator(
        sessions_layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    )

    def draft(title: str) -> models.TaskNodeDraft:
        return models.TaskNodeDraft(title=title, depends_on=(), created_by="planner")

    root = curator.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=draft("the task")))
    sub = curator.add_subtask(
        models.AddSubtaskIntent(parent_id=root.id, draft=draft("write edit.txt"))
    )
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_with_tool("read_file", {"path": "x.txt"}),
        budget.BudgetExceededError("input cap reached"),
    ]
    wf = _gated_wf(repo, provider, _dirtying_dispatcher(repo), max_turns=5, curator=curator)
    wf.run("the task")

    body = subprocess.run(
        ["git", "log", "-1", "--format=%B", "refs/agent6/guard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"{git_read.TASK_TRAILER}: {sub.id}" in body, body
