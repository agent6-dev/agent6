# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for critic-in-loop helpers and Workflow wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.providers import ProviderError, ProviderResponse
from agent6.tools.results import RawResult
from agent6.workflows import loop as loopmod
from agent6.workflows._conversation import Conversation
from agent6.workflows.loop import Workflow


def _silent(_msg: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        # Gateless by default (verify_command=()) so a bare MagicMock's truthy
        # attr doesn't make the verify finish-gate think a red verify is pending.
        "config": MagicMock(
            prompt=MagicMock(system_prompt_file=""),
            workflow=MagicMock(verify_command=(), require_verify_to_finish=False),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        "provider_retry_delay_s": 0.01,
    }
    defaults.update(kw)
    return Workflow(**defaults)


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


# --- _parse_critic_verdict ----------------------------------------------


def test_parse_critic_verdict_satisfied() -> None:
    text = "* did the thing\n* fine\n\nVERDICT: SATISFIED\n"
    assert loopmod.parse_critic_verdict(text) is True  # pyright: ignore[reportPrivateUsage]


def test_parse_critic_verdict_needs_work() -> None:
    text = "* broken\n\nVERDICT: NEEDS_WORK"
    assert loopmod.parse_critic_verdict(text) is False  # pyright: ignore[reportPrivateUsage]


def test_parse_critic_verdict_missing_defaults_to_needs_work() -> None:
    text = "lol I forgot to include a verdict"
    assert loopmod.parse_critic_verdict(text) is False  # pyright: ignore[reportPrivateUsage]


def test_parse_critic_verdict_case_insensitive() -> None:
    text = "x\n\nverdict: satisfied"
    assert loopmod.parse_critic_verdict(text) is True  # pyright: ignore[reportPrivateUsage]


# --- _format_messages_tail_for_critic -----------------------------------


def test_format_messages_tail_renders_roles() -> None:
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "doing it"},
                {"type": "tool_use", "name": "run_command", "input": {}, "id": "x"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": "output"}],
        },
    ]
    out = loopmod.format_messages_tail_for_critic(msgs)  # pyright: ignore[reportPrivateUsage]
    assert "internal reasoning" not in out  # thinking blocks are stripped
    assert "hello" in out
    assert "doing it" in out
    assert "run_command" in out
    assert "output" in out


# --- _run_critic --------------------------------------------------------


def test_run_critic_returns_none_when_no_provider() -> None:
    wf = _wf()
    assert (
        wf._run_critic(  # pyright: ignore[reportPrivateUsage]
            task="t", messages=[], trigger="periodic", iteration=1
        )
        is None
    )


def test_run_critic_returns_critique_on_success() -> None:
    critic = MagicMock()
    critic.call.return_value = _resp("* fine\n\nVERDICT: SATISFIED")
    wf = _wf(critic_provider=critic, critic_mode="periodic")
    out = wf._run_critic(  # pyright: ignore[reportPrivateUsage]
        task="t", messages=[], trigger="periodic", iteration=1
    )
    assert out is not None
    assert out.satisfied is True
    assert "VERDICT: SATISFIED" in out.text


def test_run_critic_returns_none_on_provider_error() -> None:
    critic = MagicMock()
    critic.call.side_effect = ProviderError("upstream 500")
    wf = _wf(critic_provider=critic, critic_mode="on_verify_fail")
    out = wf._run_critic(  # pyright: ignore[reportPrivateUsage]
        task="t", messages=[], trigger="verify_failed", iteration=2
    )
    assert out is None


# --- _drive_loop integration: before_finish revocation -----------------


def _finish_tool_use(tu_id: str = "tu1", summary: str = "done") -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tu_id,
        "name": "finish_run",
        "input": {"summary": summary},
    }


def _resp_with_tool_use(text: str, tool_use: dict[str, Any]) -> ProviderResponse:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks.append({"type": "tool_use", **tool_use})
    return ProviderResponse(
        text=text,
        tool_uses=(tool_use,),
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": blocks},
    )


def test_before_finish_critic_revokes_finish_and_injects_critique() -> None:
    """When critic returns NEEDS_WORK on a finish_run, the loop must NOT
    return finish_run on this iteration; instead it appends a `[critic]`
    text block to the user message and continues to the next iter."""
    # Worker turn 1: calls finish_run (rejected by critic).
    # Worker turn 2: emits plain text, no tool_use -> silent_finish exit.
    # silent_finish now also goes through critic, so the
    # iter-2 critic must return SATISFIED for the loop to exit cleanly.
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("attempting to finish", _finish_tool_use("tu1", "wrap up")),
        # iters 2-3: early prose on an untouched tree is bounced by the
        # (free, pre-critic) no-work gate; the critic sees the iter-4 prose.
        _resp("ok, looks good"),
        _resp("ok, looks good"),
        _resp("ok, looks good"),
    ]
    critic = MagicMock()
    critic.call.side_effect = [
        _resp("* still TODOs left\n\nVERDICT: NEEDS_WORK"),
        _resp("* now fine\n\nVERDICT: SATISFIED"),
    ]

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})

    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}],
        }
    ]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=(conversation := Conversation.from_wire(messages)),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    # First finish revoked by the critic; the early prose turns bounce off
    # the no-work gate (no critic spend); the iter-4 prose exits.
    assert result.iterations == 4
    assert result.reason == "silent_finish"
    assert critic.call.call_count == 2
    # The user message appended after iter 1 must carry the critic block.
    # Each iteration appends assistant then user, so 's user is at index 2.
    iter1_user_msg = conversation.to_wire()[2]
    assert iter1_user_msg["role"] == "user"
    content_blocks = iter1_user_msg["content"]
    critic_blocks = [b for b in content_blocks if b.get("type") == "text"]
    assert any("[critic]" in b["text"] for b in critic_blocks)
    assert any("NEEDS_WORK" in b["text"] for b in critic_blocks)


def test_before_finish_critic_satisfied_accepts_finish() -> None:
    """When critic says SATISFIED on a finish_run, the loop returns
    immediately with reason=finish_run on that same iteration."""
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use(
        "wrapping up", _finish_tool_use("tu1", "all done")
    )
    critic = MagicMock()
    critic.call.return_value = _resp("* clean\n\nVERDICT: SATISFIED")

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})

    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=Conversation.from_wire(messages),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert result.iterations == 1
    assert result.reason == "finish_run"
    assert result.summary == "all done"
    assert critic.call.call_count == 1


def test_before_finish_rejection_cap_lets_finish_through() -> None:
    """After `max_consecutive_critic_rejections` back-to-back NEEDS_WORK
    verdicts, the next finish_run must be accepted even if the critic
    still disagrees. Prevents stubborn-worker budget burn."""
    # Worker calls finish_run every iteration. With cap=2, iters 1 and 2
    # get rejected; iter 3's finish goes through with the rejection-cap
    # message attached.
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("try 1", _finish_tool_use("tu1", "v1")),
        _resp_with_tool_use("try 2", _finish_tool_use("tu2", "v2")),
        _resp_with_tool_use("try 3", _finish_tool_use("tu3", "v3")),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("* nope\n\nVERDICT: NEEDS_WORK")

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})

    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
        max_consecutive_critic_rejections=2,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=(conversation := Conversation.from_wire(messages)),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert result.iterations == 3
    assert result.reason == "finish_run"
    assert critic.call.call_count == 3
    # The iter-3 user message must carry the cap-reached critic warning.
    # iter N user msg is at index 2N (initial user + 2 entries per iter).
    iter3_user_msg = conversation.to_wire()[6]
    critic_text = next(b["text"] for b in iter3_user_msg["content"] if b.get("type") == "text")
    assert "rejection cap" in critic_text.lower() or "cap was" in critic_text.lower()


def test_before_finish_satisfied_resets_rejection_counter() -> None:
    """A SATISFIED verdict must reset the consecutive-rejection counter
    so a later transient NEEDS_WORK doesn't get instantly cap-accepted."""
    # NEEDS_WORK -> rejected.
    # SATISFIED -> accepted. Counter should reset.
    worker = MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("try 1", _finish_tool_use("tu1", "v1")),
        _resp_with_tool_use("try 2", _finish_tool_use("tu2", "v2")),
    ]
    critic = MagicMock()
    critic.call.side_effect = [
        _resp("VERDICT: NEEDS_WORK"),
        _resp("VERDICT: SATISFIED"),
    ]
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({"ok": True})

    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
        max_consecutive_critic_rejections=2,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=Conversation.from_wire(messages),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.iterations == 2
    assert result.reason == "finish_run"
    assert result.summary == "v2"


def test_critic_mode_off_never_calls_critic() -> None:
    """With critic_mode='off' but a critic_provider set, the critic is
    never invoked. Guards against accidental wiring leaks."""
    worker = MagicMock()
    worker.call.return_value = _resp_with_tool_use("done", _finish_tool_use("tu1", "summary"))
    critic = MagicMock()
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = RawResult({})
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="off",
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=Conversation.from_wire(messages),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert critic.call.call_count == 0


# --- : silent_finish goes through before_finish critic ---------


def test_silent_finish_critic_revokes_and_continues() -> None:
    """When the agent silent-finishes (text but no tool_use)
    under critic_mode=before_finish, a NEEDS_WORK critique must keep
    the loop running and inject the critique into the next user msg."""
    # iter 1: silent_finish, critic NEEDS_WORK -> loop continues.
    # iter 2: silent_finish, critic SATISFIED -> loop exits.
    worker = MagicMock()
    worker.call.side_effect = [
        # iters 1-2 bounce off the pre-critic no-work gate
        _resp("still nothing done"),
        _resp("still nothing done"),
        _resp("I think I'm done already"),
        _resp("ok now actually done"),
    ]
    critic = MagicMock()
    critic.call.side_effect = [
        _resp("* nothing was done\n\nVERDICT: NEEDS_WORK"),
        _resp("* fine\n\nVERDICT: SATISFIED"),
    ]
    dispatcher = MagicMock()
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nfix it\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=(conversation := Conversation.from_wire(messages)),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.iterations == 4
    assert result.reason == "silent_finish"
    assert result.summary == "ok now actually done"
    assert critic.call.call_count == 2
    # critique injected as a user msg after the iter-3 assistant turn.
    iter1_user_msg = conversation.to_wire()[6]
    assert iter1_user_msg["role"] == "user"
    text_blocks = [b for b in iter1_user_msg["content"] if b.get("type") == "text"]
    assert any("[critic]" in b["text"] for b in text_blocks)
    assert any("silent finish" in b["text"].lower() for b in text_blocks)


def test_silent_finish_critic_cap_lets_finish_through() -> None:
    """Stubborn silent-finish + always-NEEDS_WORK
    critic must hit the rejection cap and accept the finish."""
    worker = MagicMock()
    worker.call.side_effect = [
        # iters 1-2 bounce off the pre-critic no-work gate
        _resp("done v0a"),
        _resp("done v0b"),
        _resp("done v1"),
        _resp("done v2"),
        _resp("done v3"),
    ]
    critic = MagicMock()
    critic.call.return_value = _resp("VERDICT: NEEDS_WORK")
    dispatcher = MagicMock()
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        critic_provider=critic,
        critic_mode="before_finish",
        max_consecutive_critic_rejections=2,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
    ]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=Conversation.from_wire(messages),
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.iterations == 5
    assert result.reason == "silent_finish"
    assert result.summary == "done v3"
    assert critic.call.call_count == 3


def test_silent_finish_critic_off_bypasses() -> None:
    """critic_mode != before_finish must NOT gate silent_finish.
    A silent finish under periodic/on_verify_fail/off exits immediately."""
    worker = MagicMock()
    worker.call.return_value = _resp("done")
    critic = MagicMock()
    dispatcher = MagicMock()
    for mode in ("off", "on_verify_fail", "periodic"):
        worker.call.reset_mock(return_value=False, side_effect=False)
        worker.call.return_value = _resp("done")
        critic.call.reset_mock(return_value=False, side_effect=False)
        critic.call.return_value = _resp("VERDICT: NEEDS_WORK")
        wf = _wf(
            provider=worker,
            dispatcher=dispatcher,
            critic_provider=critic,
            critic_mode=mode,  # pyright: ignore[reportArgumentType]
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}
        ]
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="S",
            conversation=Conversation.from_wire(messages),
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
        assert result.reason == "silent_finish"
        # the (critic-independent) early no-work gate bounces twice first
        assert result.iterations == 3
        # periodic/on_verify_fail don't fire on silent_finish either -
        # critic was wired but had no trigger condition matched.
        assert critic.call.call_count == 0, f"mode={mode} called critic unexpectedly"
