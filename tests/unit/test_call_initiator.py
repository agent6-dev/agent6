# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The worker's call names its initiator: the operator after the task, a steer or an answer."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds
from agent6.harness import loop_conversation, loop_snapshot, provider_call
from agent6.providers import ProviderError, ProviderResponse, request_headers
from tests.unit import test_harness


def _noop(*_args: object, **_kwargs: object) -> None:
    """Accept any call and do nothing."""


def _ok() -> ProviderResponse:
    return ProviderResponse(
        text="ok",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={},
    )


def _caller(provider: Any, *, retry_count: int = 0) -> provider_call.ProviderCaller:
    return provider_call.ProviderCaller(
        provider=provider,
        retry_count=retry_count,
        retry_delay_s=0.0,
        retry_max_delay_s=0.0,
        temperature=None,
        should_abort=lambda: False,
        should_interrupt=lambda: False,
        log=lambda _m: None,
        emit=_noop,
    )


def test_the_initiator_follows_the_latest_notice_origin() -> None:
    conv = loop_conversation.Conversation()
    assert conv.initiator == "agent"
    conv.notice("TASK: do it", origin="operator")
    assert conv.initiator == "user"
    conv.notice("[harness verify] the gate's baseline is green")  # a preamble keeps the turn
    assert conv.initiator == "user"
    conv.assistant([{"type": "text", "text": "working"}])
    assert conv.initiator == "agent"
    conv.notice("[no-progress] nudge")
    assert conv.initiator == "agent"
    conv.notice("OPERATOR STEERING: faster", origin="operator")
    assert conv.initiator == "user"


def test_checkpoint_preserves_notice_authors_but_the_provider_gets_plain_text(
    tmp_path: pathlib.Path,
) -> None:
    """A resumed steer keeps its author even beside an identical harness notice."""
    snap = tmp_path / "loop_state.json"
    harness = test_harness._wf(root=tmp_path, resume_state_path=snap)
    state = test_harness._state()
    conv = loop_conversation.Conversation()
    for origin in ("operator", "harness"):
        conv.notice("same words", origin=origin)
    wire = harness._turn_pre_call(
        conversation=conv,
        state=state,
        ctx=harness._turn_context(state, turn=1, execution_start=1),
        tools=[],
    )
    assert isinstance(wire, list) and wire == conv.to_wire()
    assert all("origin" not in b for m in wire for b in m["content"])
    for path in (snap, tmp_path / "checkpoints" / "0001.json"):
        saved = loop_snapshot.load_session_snapshot(path)
        resumed = loop_conversation.Conversation.from_wire(saved.messages)
        assert resumed.messages == conv.messages
        assert resumed.initiator == "user"
        assert resumed.to_wire() == wire


def test_the_caller_passes_the_initiator_as_a_partial_context() -> None:
    seen: list[Any] = []

    class _Provider:
        def call(self, **kw: Any) -> ProviderResponse:
            seen.append(kw["context"])
            return _ok()

    caller = _caller(_Provider())
    caller.call("s", [], [], 10, initiator="user")
    caller.call("s", [], [], 10)
    assert [c.initiator for c in seen] == ["user", "agent"]
    assert all(
        isinstance(c, request_headers.RequestContext) and c.conversation_id == "" for c in seen
    )


def test_a_retried_call_keeps_its_logical_call_id() -> None:
    """A transient error's retry carries the call id of the attempt it repeats."""
    seen: list[str] = []

    class _Flaky:
        def call(self, **kw: Any) -> ProviderResponse:
            # As the instrumented provider does: mint an id only when the caller sent none.
            call_context = kw["context"].call_context or kinds.CallContext()
            seen.append(call_context.model_call_id)
            if len(seen) == 1:
                raise ProviderError("busy", status_code=503)
            return _ok()

    _caller(_Flaky(), retry_count=1).call("s", [], [], 10)
    assert len(seen) == 2
    assert len(set(seen)) == 1, seen


def test_a_given_call_context_is_shared_by_every_attempt() -> None:
    """The caller's context, with its execution, reaches each attempt unchanged."""
    seen: list[kinds.CallContext] = []

    class _Flaky:
        def call(self, **kw: Any) -> ProviderResponse:
            seen.append(kw["context"].call_context)
            if len(seen) == 1:
                raise ProviderError("busy", status_code=503)
            return _ok()

    given = kinds.CallContext(model_call_id="given", execution_id="exec-1")
    _caller(_Flaky(), retry_count=1).call("s", [], [], 10, call_context=given)
    assert seen == [given, given]


def test_streamed_deltas_carry_the_call_identity(tmp_path: pathlib.Path) -> None:
    """Each delta carries its call's actor, logical ID and execution."""
    import json

    from agent6 import budget as agent6_budget
    from agent6.app import role_providers
    from agent6.sessions import event_log

    class _Streaming:
        def call(self, **kw: Any) -> ProviderResponse:
            kw["text_delta_callback"]("hi")
            kw["thinking_delta_callback"]("hmm")
            return _ok()

    logs = tmp_path / "logs.jsonl"
    wrapper = role_providers.InstrumentedProvider(
        inner=_Streaming(),
        actor="reviewer:security",
        model="m",
        provider_name="p",
        events=event_log.EventSink(logs),
        budget=agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        stream_text=True,
    )
    given = kinds.CallContext(model_call_id="given", execution_id="exec-1")
    wrapper.call(
        system="s",
        messages=[],
        context=request_headers.RequestContext.initiated_by("agent", call_context=given),
    )
    rows = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    by_type = {row["type"]: row for row in rows}
    for name in ("model.call", "model.text_delta", "model.thinking_delta", "model.result"):
        assert by_type[name]["actor"] == "reviewer:security", name
        assert by_type[name]["model_call_id"] == "given", name
        assert by_type[name]["execution_id"] == "exec-1", name


def _wrapped(inner: Any, tmp_path: pathlib.Path) -> Any:
    from agent6 import budget as agent6_budget
    from agent6.app import role_providers
    from agent6.sessions import event_log

    return role_providers.InstrumentedProvider(
        inner=inner,
        actor="worker",
        model="m",
        provider_name="p",
        events=event_log.EventSink(tmp_path / "logs.jsonl"),
        budget=agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
    )


def _rows(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    import json

    return [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_a_retried_call_is_one_lifecycle_with_an_attempt_row_each(tmp_path: pathlib.Path) -> None:
    """A transient error leaves model.call once, a row per attempt and model.result once."""
    calls: list[int] = []

    class _Flaky:
        def call(self, **kw: Any) -> ProviderResponse:
            calls.append(kw["context"].attempt)
            if len(calls) == 1:
                raise ProviderError("busy", status_code=503)
            return _ok()

    _caller(_wrapped(_Flaky(), tmp_path), retry_count=1).call("s", [], [], 10)
    rows = [r for r in _rows(tmp_path) if r["type"].startswith("model.")]
    assert [r["type"] for r in rows] == [
        "model.call",
        "model.attempt",
        "model.attempt",
        "model.result",
    ]
    assert len({r["model_call_id"] for r in rows}) == 1
    assert {r["actor"] for r in rows} == {"worker"}
    first, second = rows[1], rows[2]
    assert (first["attempt"], first["ok"], first["error"]) == (1, False, "busy")
    assert (second["attempt"], second["ok"]) == (2, True)
    assert rows[3]["ok"] is True and calls == [1, 2]


def test_a_permanent_error_closes_the_call_on_its_first_attempt(tmp_path: pathlib.Path) -> None:
    """A status no retry recovers ends the lifecycle at once: one attempt, one failed result."""

    class _Refusing:
        def call(self, **kw: Any) -> ProviderResponse:
            raise ProviderError("bad request", status_code=400)

    with pytest.raises(ProviderError):
        _caller(_wrapped(_Refusing(), tmp_path), retry_count=3).call("s", [], [], 10)
    types = [r["type"] for r in _rows(tmp_path) if r["type"].startswith("model.")]
    assert types == ["model.call", "model.attempt", "model.result"]


def test_an_empty_tool_call_response_is_an_attempt_of_the_same_call(
    tmp_path: pathlib.Path,
) -> None:
    """The blind retry of an empty tool-call response stays inside the one lifecycle."""
    responses = iter(
        [
            ProviderResponse(
                text="",
                tool_uses=(),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            _ok(),
        ]
    )

    class _Empty:
        def call(self, **kw: Any) -> ProviderResponse:
            return next(responses)

    _caller(_wrapped(_Empty(), tmp_path), retry_count=1).call("s", [], [], 10)
    rows = [r for r in _rows(tmp_path) if r["type"].startswith("model.")]
    assert [r["type"] for r in rows] == [
        "model.call",
        "model.attempt",
        "model.attempt",
        "model.result",
    ]
    assert rows[1]["ok"] is True and rows[3]["text"] == "ok"


def test_a_model_assignment_is_the_task_and_not_an_operator_quote() -> None:
    from tests.unit import test_restart_retention

    wf = test_harness._wf(task_source="model", curator=None)
    with mock.patch.object(wf, "_drive_loop") as drive:
        wf.run("model assignment")
    conv = drive.call_args.kwargs["conversation"]
    conv.assistant([{"type": "text", "text": "done"}])
    test_restart_retention._restart(
        conv, task="model assignment", keep_recent_chars=100, keep_recent_turns=0
    )
    notice = test_restart_retention._notice(conv)
    assert "TASK (the goal, verbatim):\nmodel assignment" in notice
    assert "Operator message:" not in notice
