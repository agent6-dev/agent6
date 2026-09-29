# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the Harness loop.

Provider retry, operator steering, the tool-error ladder, finish gates and the other drive-loop
mechanics, driven with scripted providers and dispatchers. Termination reasons are exercised
end-to-end in the integration suite.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from agent6.config import Config
from agent6.harness._advice import Stop, with_open_tasks
from agent6.harness._chain import RunChain
from agent6.harness._compaction import CompactionSettings
from agent6.harness._conversation import AssistantTurn, Conversation, Notice
from agent6.harness._finish_gates import (
    SILENT_END_GATES,
    FinishCall,
    FinishGates,
    task_finish_nudge,
)
from agent6.harness._guards import SettledGuard, settled_end, verify_settled
from agent6.harness._metric import MetricGuard, metric_plateau
from agent6.harness._operator import OperatorBridge
from agent6.harness._provider_call import (
    CallSettings,
    ProviderCaller,
    is_empty_tool_call_response,
    reasoning_starvation,
)
from agent6.harness._quiet_turns import QuietGuard
from agent6.harness._reviewer import Reviewer, ReviewSettings
from agent6.harness._snapshot import SNAPSHOT_VERSION, End
from agent6.harness._verify_verdict import VerifyVerdict
from agent6.harness.loop import Harness, LoopState, TurnState
from agent6.providers import ProviderError, ProviderResponse
from agent6.tools.mcp_client import MCPToolDescriptor
from agent6.tools.results import ExecResult, MetricResult, RawResult, ToolResult
from tests.unit.turn_context import turn_context

# The `[git]` surface the loop reads, empty as a real Config carries it unset.
_GIT_STUB = SimpleNamespace(
    control="agent6",
    commit_per_step=True,
    commit=SimpleNamespace(
        checkpoint=SimpleNamespace(message="agent6"), name="", email="", trailer=""
    ),
)


class _StubDispatcher:
    """The dispatcher surface the loop reads besides `dispatch`.

    The loop rebuilds its tool list every turn, so a stub must answer more than `dispatch`; the
    defaults filter no tools and withhold nothing.
    """

    dag_available = True

    def available_tool_names(self) -> tuple[str, ...]:
        return ()

    def mcp_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        """No MCP servers, so nothing to add to the per-turn tool list."""
        return ()

    def metric_configured(self) -> bool:
        return True

    def skills_available(self) -> bool:
        return False

    def command_policy(self) -> str:
        return "ask"

    def tool_is_withheld(self, name: str) -> bool:
        """The "ask" policy withholds nothing: the operator is asked at call time."""
        return False

    def settle_background(self) -> None:
        """The turn boundary observes background commands; these tests start none."""


def _silent(_msg: str) -> None:
    return None


def _knobs(cfg: Any, **knobs: Any) -> Any:
    """The mocked config with `[harness]` guard knobs set."""
    for key, value in knobs.items():
        setattr(cfg.harness, key, value)
    return cfg


def _wf(
    root: Path | None = None,
    *,
    ref: str | None = "refs/agent6/test",
    fallback_parent: str | None = None,
    branch: str | None = None,
    per_step: bool = True,
    base_sha: str = "",
    **kw: Any,
) -> Harness:
    """Construct a Harness with mocks for everything not under test; caller kwargs win."""
    if root is not None and fallback_parent is None:
        # Mirror run.py's wiring: the chain's first parent is HEAD at start.
        fallback_parent = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    # A live chain by default, so the auto-commit paths run; tests patch `_chain.chain_commit`.
    defaults: dict[str, Any] = {
        "chain": RunChain(
            root or Path("/tmp"),
            ref=ref,
            branch=branch,
            fallback_parent=fallback_parent,
            per_step=per_step,
            base_sha=base_sha,
        ),
        "config": MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            harness=MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_retries=2,
            ),
        ),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        "call": CallSettings(retry_delay_s=0.01),  # keep tests fast
    }
    defaults.update(kw)
    return Harness(**defaults)


def _state(**kw: Any) -> Any:
    """Minimal LoopState for _save_resume_snapshot call sites."""
    from agent6.harness.loop import LoopState

    defaults: dict[str, Any] = {"original_task": "t", "tool_calls": 0}
    defaults.update(kw)
    return LoopState(**defaults)


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _tn(node_id: str, **fields: Any) -> Any:
    """A TaskNode fixture from a partial dict of node fields.

    model_construct keeps short readable ids ("a", "b"), whose sort order first_ready_subtask relies
    on.
    """
    from agent6.graph.models import TaskNode

    base: dict[str, Any] = {
        "id": node_id,
        "parent_id": None,
        "title": "t",
        "rationale": "",
        "acceptance": "",
        "relevant_paths": (),
        "depends_on": (),
        "children": (),
        "status": "pending",
        "created_at": _T0,
        "updated_at": _T0,
        "created_by": "planner",
        "commit_sha": "",
        "notes": "",
    }
    base.update(fields)
    return TaskNode.model_construct(**base)


def _typed(nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert the tests' dict-of-dicts node literals into the curator's typed nodes."""
    return {nid: _tn(nid, **d) for nid, d in nodes.items()}


def test_ask_silent_finish_ends_as_answered_not_silent_finish() -> None:
    # In ask mode a prose answer ends as "answered", not "silent_finish".
    wf = _wf(mode="ask")
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "The answer is 42.", Conversation(), _state(), _turn(iteration=2), _ctx(wf, _state())
    )
    assert result is not None
    assert result.reason == "answered"
    assert result.completed is True
    assert result.summary == "The answer is 42."  # ask keeps the whole answer


def test_run_silent_finish_stays_silent_finish() -> None:
    # In run mode (edited + verified) a no-tool prose turn is still an implicit silent_finish.
    wf = _wf(mode="run")
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        _turn(iteration=5),
        _ctx(wf, _state()),
    )
    assert result is not None
    assert result.reason == "silent_finish"


class _EventCapture:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, /, **fields: Any) -> None:
        self.events.append({"type": event_type, **fields})


def _cfg_with_verify() -> Any:
    return MagicMock(
        prompt=MagicMock(system_prompt_file=""),
        harness=MagicMock(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=("true",),
            verify_when="never",
            verify_retries=2,
            metric=SimpleNamespace(goal=None),
        ),
    )


def test_run_silent_finish_over_red_verify_is_not_passed() -> None:
    """A silent finish over a red or stale verify ends all_passed=False, like finish_session."""
    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev)
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "I tried, but the tests still fail.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True, last_ok=False)),
        _turn(iteration=5),
        _ctx(wf, _state()),
    )
    assert result is not None and result.reason == "silent_finish"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_run_silent_finish_over_green_verify_stays_passed() -> None:
    """The mirror: a clean green tree still ends passed (no false negative)."""
    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev)
    wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done, all green.",
        Conversation(),
        _state(
            ever_edited=True,
            verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=False),
        ),
        _turn(iteration=5),
        _ctx(wf, _state()),
    )
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is True


def test_run_silent_finish_gateless_is_ungated_not_passed() -> None:
    """A gateless run's silent finish ends with all_passed=None, which words as "finished"."""
    from agent6.viewmodel.listing import status_word

    ev = _EventCapture()
    wf = _wf(mode="run", events=ev)  # _wf's default config has no verify_command
    result = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "READY",
        Conversation(),
        _state(ever_edited=True),
        _turn(iteration=5),
        _ctx(wf, _state()),
    )
    assert result is not None and result.reason == "silent_finish"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is None
    assert status_word(finished=True, all_passed=None, end_reason="silent_finish") == (
        "finished",
        "",
    )


def _turn(**kw: Any) -> Any:
    """A bare TurnState for direct turn-phase method tests."""
    from agent6.harness.loop import TurnState

    defaults: dict[str, Any] = {
        "iteration": 1,
        "resp": _resp(""),
        "assistant": AssistantTurn((), ()),
    }
    defaults.update(kw)
    return TurnState(**defaults)


def _ctx(wf: Harness, state: Any, iteration: int = 1) -> Any:
    return wf._turn_context(state, iteration=iteration, execution_start=1)  # pyright: ignore[reportPrivateUsage]


def _plateau_stop() -> Stop:
    """The metric plateau's stop as the advisor decides it: a grounded end."""
    end = End("metric_plateau", "score plateaued at 10", completed=True, verdict="grounded")
    return Stop(lambda: end, soft="metric_plateau", declared="metric_plateau")


def _settle(wf: Harness, state: Any, turn: Any) -> Any:
    """The settled advisor's answer, applied through the loop; a stop runs the end gates at once."""
    ctx = _ctx(wf, state, turn.iteration)
    return wf._take(state, turn, ctx, verify_settled(turn, state, ctx))  # pyright: ignore[reportPrivateUsage]


def test_finish_planning_salvages_a_title_only_plan(tmp_path: Path) -> None:
    # Weak models put the plan in `summary`; the fold must produce a plan.md with real content.
    plan_path = tmp_path / "plan.md"
    wf = _wf(mode="plan", plan_output_path=plan_path)
    wf._capture_finish(  # pyright: ignore[reportPrivateUsage]
        _turn(),
        "finish_planning",
        {
            "summary": "1. Add the --count flag. 2. Update the parser help. 3. Add a test.",
            "plan_markdown": "# Plan: Add --count flag",
        },
    )
    text = plan_path.read_text(encoding="utf-8")
    assert "# Plan: Add --count flag" in text  # the title is kept
    assert "Add the --count flag" in text  # the summary was folded in as the body


def test_finish_planning_keeps_a_real_plan_markdown(tmp_path: Path) -> None:
    # A proper plan_markdown is written verbatim; the summary is NOT appended.
    plan_path = tmp_path / "plan.md"
    wf = _wf(mode="plan", plan_output_path=plan_path)
    wf._capture_finish(  # pyright: ignore[reportPrivateUsage]
        _turn(),
        "finish_planning",
        {"summary": "short blurb", "plan_markdown": "# Plan: X\n\n1. real step\n2. another"},
    )
    text = plan_path.read_text(encoding="utf-8")
    assert "real step" in text and "short blurb" not in text


def _resp(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def _tool_resp(
    name: str,
    tool_input: dict[str, Any] | None = None,
    *,
    tool_id: str = "tool-1",
) -> ProviderResponse:
    payload = tool_input or {}
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": payload}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tool_id, "name": name, "input": payload},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


# --- ProviderCaller -------------------------------------------------------


def _never() -> bool:
    return False


def _no_log(_msg: str) -> None:
    pass


def _no_emit(_event: str, **_fields: Any) -> None:
    pass


def _caller(provider: Any, *, retry_count: int = 0, retry_delay_s: float = 0.01) -> ProviderCaller:
    return ProviderCaller(
        provider=provider,
        retry_count=retry_count,
        retry_delay_s=retry_delay_s,
        retry_max_delay_s=0.01,
        temperature=0.0,
        should_abort=_never,
        should_interrupt=_never,
        log=_no_log,
        emit=_no_emit,
    )


def _call(caller: ProviderCaller) -> ProviderResponse:
    return caller.call(system="s", messages=[], tools=[], max_tokens=16384)


def test_caller_first_try_returns() -> None:
    """No ProviderError: one call, returned as is."""
    provider = MagicMock()
    provider.call.return_value = _resp("first")
    assert _call(_caller(provider)).text == "first"
    assert provider.call.call_count == 1


def test_caller_succeeds_on_retry() -> None:
    """ProviderError on the first call, success on the retry: the retry is returned."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("transient 529"), _resp("retried")]
    assert _call(_caller(provider, retry_count=1)).text == "retried"
    assert provider.call.call_count == 2


def test_caller_reraises_after_retries_exhausted() -> None:
    """Two ProviderErrors with retry_count=1: the last error bubbles."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("flake 1"), ProviderError("flake 2")]
    with pytest.raises(ProviderError, match="flake 2"):
        _call(_caller(provider, retry_count=1))
    assert provider.call.call_count == 2


def test_caller_never_retries_an_abort() -> None:
    """ProviderAborted (operator stop) bubbles immediately, never retried."""
    from agent6.providers import ProviderAborted

    provider = MagicMock()
    provider.call.side_effect = [ProviderAborted("stopped"), _resp("late")]
    with pytest.raises(ProviderAborted):
        _call(_caller(provider, retry_count=3))
    assert provider.call.call_count == 1
    assert provider.call.call_args.kwargs["should_abort"] is _never


def test_caller_never_retries_a_steer_interrupt() -> None:
    """ProviderInterrupted (a steer mid-stream) bubbles immediately; a retry would re-hit it."""
    from agent6.providers import ProviderInterrupted

    provider = MagicMock()
    provider.call.side_effect = [ProviderInterrupted("steer"), _resp("late")]
    with pytest.raises(ProviderInterrupted):
        _call(_caller(provider, retry_count=3))
    assert provider.call.call_count == 1
    assert provider.call.call_args.kwargs["should_interrupt"] is _never


def test_caller_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 carrying retry_after_s waits at least that long, not the shorter computed backoff."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.harness._provider_call.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429 rate limited", status_code=429, retry_after_s=50.0),
        _resp("ok"),
    ]
    assert _call(_caller(provider, retry_count=1)).text == "ok"
    assert slept and slept[0] >= 50.0


def test_caller_clamps_retry_after_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile or buggy Retry-After cannot hang the run: clamped to the ceiling."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.harness._provider_call.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429", status_code=429, retry_after_s=9999.0),
        _resp("ok"),
    ]
    _call(_caller(provider, retry_count=1))
    assert slept and slept[0] <= 120.0


def _empty_tool_call_resp() -> ProviderResponse:
    """A self-contradictory response: stop_reason=tool_calls with no tool_use and no text."""
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="tool_calls",
        input_tokens=1,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def test_caller_retries_empty_tool_call_response() -> None:
    """An empty finish=tool_calls response is retried and the recovered response returned."""
    provider = MagicMock()
    provider.call.side_effect = [_empty_tool_call_resp(), _tool_resp("read_file", {"path": "x"})]
    out = _call(_caller(provider, retry_count=4, retry_delay_s=0.001))
    assert out.tool_uses
    assert provider.call.call_count == 2


def test_caller_returns_last_empty_after_exhausting() -> None:
    """When every attempt is empty the last empty response is returned; went_quiet takes over."""
    provider = MagicMock()
    provider.call.return_value = _empty_tool_call_resp()
    out = _call(_caller(provider, retry_count=2, retry_delay_s=0.001))
    assert out.stop_reason == "tool_calls" and not out.tool_uses
    assert provider.call.call_count == 3


def test_reasoning_starvation_counts_only_a_cap_cut_turn() -> None:
    """The count is the thinking a cap-cut, billed turn spent; any other turn counts 0."""
    starved = ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="length",
        input_tokens=1,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "thinking", "thinking": "x" * 40}]},
    )
    assert reasoning_starvation(starved) == 40
    assert reasoning_starvation(replace(starved, stop_reason="end_turn")) == 0
    assert reasoning_starvation(replace(starved, output_tokens=0)) == 0
    assert reasoning_starvation(replace(starved, raw={"content": []})) == 0


def test_is_empty_tool_call_response_discriminates() -> None:
    assert is_empty_tool_call_response(_empty_tool_call_resp())
    assert not is_empty_tool_call_response(_resp("hi"))  # has text -> a silent finish
    assert not is_empty_tool_call_response(_tool_resp("read_file"))  # has a tool_use
    # length-truncated reasoning starvation is handled separately, not retried here.
    starved = ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="length",
        input_tokens=1,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert not is_empty_tool_call_response(starved)


def test_call_with_retry_default_rides_out_multiple_flaps() -> None:
    """The default retry budget survives more than one consecutive transient disconnect."""
    provider = MagicMock()
    disconnect = ProviderError("Server disconnected without sending a response")
    provider.call.side_effect = [disconnect, disconnect, disconnect, _resp("recovered")]
    wf = _wf(provider=provider)  # uses the default provider_retry_count
    out = wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert out.text == "recovered"
    assert provider.call.call_count == 4


def test_call_with_retry_zero_retries_no_retry() -> None:
    """provider_retry_count=0 -> single attempt, no retry on error."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("nope")]
    wf = _wf(provider=provider, call=CallSettings(retry_count=0, retry_delay_s=0.01))
    with pytest.raises(ProviderError, match="nope"):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1


def test_call_with_retry_does_not_swallow_non_provider_errors() -> None:
    """RuntimeError (etc.) must propagate without retry."""
    provider = MagicMock()
    provider.call.side_effect = [RuntimeError("not a provider error")]
    wf = _wf(provider=provider, call=CallSettings(retry_count=3, retry_delay_s=0.01))
    with pytest.raises(RuntimeError, match="not a provider error"):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1


def test_call_with_retry_skips_retry_on_permanent_status() -> None:
    """A permanent client error (402) re-raises on the first failure without consuming a retry."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 402: Insufficient credits", status_code=402),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=3, retry_delay_s=0.01))
    with pytest.raises(ProviderError, match="402"):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1


def test_call_with_retry_never_retries_a_fatal_error() -> None:
    """A fatal ProviderError re-raises at once; without the flag a status-less error retries."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("claude not signed in", fatal=True),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=3, retry_delay_s=0.01))
    with pytest.raises(ProviderError, match="not signed in"):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1


@pytest.mark.parametrize(
    "status",
    [307, 400, 401, 402, 403, 404, 405, 413, 415, 422, 426, 431, 451],
)
def test_call_with_retry_skips_retry_on_all_permanent_statuses(status: int) -> None:
    """Every status in _NON_RETRYABLE_HTTP_STATUSES re-raises on the first failure."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError(f"provider error {status}", status_code=status),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=3, retry_delay_s=0.01))
    with pytest.raises(ProviderError, match=str(status)):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1


@pytest.mark.parametrize("status", [408, 409, 425, 429])
def test_call_with_retry_keeps_anthropic_transient_client_statuses(status: int) -> None:
    """A timeout, a conflict, too-early and a rate limit are the 4xx a blind retry can outlive."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError(f"provider error {status}", status_code=status),
        _resp("recovered"),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=1, retry_delay_s=0.01))
    assert wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384).text == "recovered"
    assert provider.call.call_count == 2


def test_call_with_retry_still_retries_transient_5xx() -> None:
    """A 503 carries a status but is not in the permanent set, so the retry path applies."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 503: upstream", status_code=503),
        _resp("recovered"),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=1, retry_delay_s=0.01))
    out = wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert out.text == "recovered"
    assert provider.call.call_count == 2


# --- exponential backoff with jitter -------------------------------------


def test_call_with_retry_exponential_backoff() -> None:
    """Attempt N sleeps provider_retry_delay_s * 2 ** (attempt - 1), scaled by the jitter factor."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("flake 1"),
        ProviderError("flake 2"),
        ProviderError("flake 3"),
        _resp("success"),
    ]
    wf = _wf(
        provider=provider,
        call=CallSettings(retry_count=3, retry_delay_s=2.0, retry_max_delay_s=30.0),
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=0.75),
    ):
        out = wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert out.text == "success"
    assert provider.call.call_count == 4
    assert sleep_calls[0] == pytest.approx(1.5)  # 2.0 * 2**0 * 0.75
    assert sleep_calls[1] == pytest.approx(3.0)  # 2.0 * 2**1 * 0.75
    assert sleep_calls[2] == pytest.approx(6.0)  # 2.0 * 2**2 * 0.75


def test_call_with_retry_backoff_capped_at_max_delay() -> None:
    """Exponential backoff is capped at provider_retry_max_delay_s."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("flake 1"),
        ProviderError("flake 2"),
        ProviderError("flake 3"),
        ProviderError("flake 4"),
        _resp("success"),
    ]
    wf = _wf(
        provider=provider,
        call=CallSettings(retry_count=4, retry_delay_s=2.0, retry_max_delay_s=5.0),
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=1.0),
    ):
        out = wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert out.text == "success"
    assert provider.call.call_count == 5
    assert sleep_calls[0] == pytest.approx(2.0)  # min(2.0 * 2**0, 5.0)
    assert sleep_calls[1] == pytest.approx(4.0)  # min(2.0 * 2**1, 5.0)
    assert sleep_calls[2] == pytest.approx(5.0)  # min(2.0 * 2**2, 5.0) capped
    assert sleep_calls[3] == pytest.approx(5.0)  # min(2.0 * 2**3, 5.0) capped


def test_call_with_retry_backoff_skips_sleep_on_permanent_status() -> None:
    """A permanent status re-raises at once with no sleep, whatever provider_retry_count allows."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("Insufficient credits", status_code=402),
    ]
    wf = _wf(provider=provider, call=CallSettings(retry_count=3, retry_delay_s=10.0))
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        pytest.raises(ProviderError, match="Insufficient credits"),
    ):
        wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_count == 1
    assert sleep_calls == []


# --- temperature wiring (Amp 2) -----------------------------------


def test_call_with_retry_pins_default_temperature_to_zero() -> None:
    """Default Harness.temperature is 0.0 and every provider.call receives it.

    A None temperature lets a router pick the model's often high default, with observable
    degeneration.
    """
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider)
    wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_args.kwargs["temperature"] == 0.0


def test_call_with_retry_honours_overridden_temperature() -> None:
    """Operators who set `[models.worker].temperature = 0.7` get it threaded through verbatim."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, call=CallSettings(temperature=0.7, retry_delay_s=0.01))
    wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_args.kwargs["temperature"] == 0.7


def test_call_with_retry_passes_through_none_temperature() -> None:
    """An explicit `temperature = None` lets the provider pick."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, call=CallSettings(temperature=None, retry_delay_s=0.01))
    wf.caller.call(system="s", messages=[], tools=[], max_tokens=16384)
    assert provider.call.call_args.kwargs["temperature"] is None


# --- automatic metric feedback ------------------------------------------


def test_the_metric_is_sampled_once_per_state_of_the_tree(tmp_path: Path) -> None:
    """The metric is not sampled on dirt alone: an uncommitting run stays dirty throughout.

    A benchmark re-run every turn, read-only ones included, is not free.
    """
    import subprocess as sp

    from agent6.tools.results import MetricResult

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        cwd=repo,
        check=True,
    )

    calls: list[str] = []

    class _Dispatcher(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            calls.append(name)
            return MetricResult(
                returncode=0,
                stdout="CYCLES: 42\n",
                stderr="",
                duration_s=0.1,
                exec_failed=False,
                score=42.0,
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(root=repo, config=config, dispatcher=_Dispatcher(), per_step=False)
    state = _state()

    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # the run's edit
    wf._sample_metric(state, _turn(iteration=1), sha="")  # pyright: ignore[reportPrivateUsage]
    wf._sample_metric(state, _turn(iteration=2), sha="")  # pyright: ignore[reportPrivateUsage]

    assert calls == ["run_metric_command"], "a read-only turn re-ran the benchmark"

    (repo / "a.py").write_text("x = 3\n", encoding="utf-8")  # a new state of the tree
    wf._sample_metric(state, _turn(iteration=3), sha="")  # pyright: ignore[reportPrivateUsage]

    assert calls == ["run_metric_command", "run_metric_command"]


@pytest.mark.parametrize("commit_per_step", [True, False])
def test_drive_loop_auto_runs_metric_after_verify_pass(
    tmp_path: Path, commit_per_step: bool
) -> None:
    """After a green verify the harness runs the metric itself and injects a history block."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []
            self.saw_metric_feedback = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            messages = kwargs["messages"]
            self.calls.append(messages)
            if len(self.calls) == 1:
                return _tool_resp("run_verify_command")
            rendered = str(messages[-1])
            self.saw_metric_feedback = self.saw_metric_feedback or (
                "[harness metric]" in rendered
                and "score=42" in rendered
                and "first parsed metric sample" in rendered
            )
            if len(self.calls) == 2:
                # A read-only turn after the verify: the tree is unchanged, so no second sample.
                return _tool_resp("read_file", {"path": "x"}, tool_id="tool-read")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="CYCLES: 42\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=42.0,
                )
            if name == "read_file":
                return RawResult({"content": "x = 1\n"})
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=4,
        # `[git].commit_per_step` governs the commit, not the measurement the prompt promises.
        per_step=commit_per_step,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "finish_session"
    assert provider.saw_metric_feedback is True
    # Exactly one reading: sampling on dirt alone re-ran the benchmark every turn.
    assert dispatcher.calls == [
        "run_verify_command",
        "run_metric_command",
        "read_file",
        "finish_session",
    ]


def test_drive_loop_tracks_iterations_reached(tmp_path: Path) -> None:
    """The loop records the absolute iteration it is driving on the Harness.

    The app-level KeyboardInterrupt fallbacks then emit a session.end with a truthful count; a
    resumed start_iteration proves it is not a zero-based counter.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.n = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.n += 1
            if self.n == 1:
                return _tool_resp("run_verify_command")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    assert wf.iterations_reached == 0  # untouched before the loop runs
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]

    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=7,  # a resumed run picks up mid-way
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    # verify ran at iter 7, finish_session at iter 8 -> the loop reached iteration 8.
    assert wf.iterations_reached == 8


@pytest.mark.parametrize(
    ("ending", "expected_reason"),
    [
        ("budget", "budget_exhausted"),
        ("provider", "provider_error"),
        ("quiet", "went_quiet"),
        ("iterations", "max_iterations"),
    ],
)
def test_abnormal_end_keeps_an_observed_red_verdict(
    tmp_path: Path, ending: str, expected_reason: str
) -> None:
    """A terminal fault after an observed red does not revert verification to not_applicable."""
    from agent6.budget import BudgetExceededError

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")
            if ending == "budget":
                raise BudgetExceededError("token cap")
            if ending == "provider":
                raise ProviderError("provider unavailable", fatal=True)
            if ending == "quiet":
                return _resp("")
            raise AssertionError("max_iterations should stop before another call")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return ExecResult(
                returncode=1, stdout="red", stderr="", duration_s=5.0, exec_failed=False
            )

    wf = _wf(
        root=tmp_path,
        config=_cfg_with_verify(),
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=1 if ending == "iterations" else 3,
    )
    wf.config = _knobs(wf.config, went_quiet_max_nudges=0)
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
        ),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert result.reason == expected_reason
    assert result.verified == "failed"


def test_provider_error_summary_is_concise_not_the_raw_body(tmp_path: Path) -> None:
    """A permanent provider error's raw body lands in one diagnostic log line, not the end block.

    The body can carry a noisy account user_id; the summary keeps the failure and HTTP status.
    """
    raw_body = 'OpenRouter API error 400: {"error":"bad model","user_id":"user_SECRET"}'

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            raise ProviderError(raw_body, status_code=400)

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    logs: list[str] = []
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        logger=logs.append,
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "provider_error"
    assert "provider error" in result.summary and "HTTP 400" in result.summary
    assert "user_SECRET" not in result.summary  # the raw blob is NOT re-echoed here
    assert any("user_SECRET" in line for line in logs)  # kept once, in the log line


def test_fatal_provider_error_ends_the_run_with_its_text(tmp_path: Path) -> None:
    """A fatal ProviderError (agent6's own remedy text, no status) ends the run after one call.

    The summary carries that text: with no status there is no hint.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            raise ProviderError("Claude Code is not signed in; run `claude auth login`", fatal=True)

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=MagicMock(),
        max_iterations=3,
        call=CallSettings(retry_count=3, retry_delay_s=0.01),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "provider_error"
    assert provider.calls == 1
    assert "claude auth login" in result.summary
    assert "HTTP" not in result.summary and "agent6 connect" not in result.summary


def test_exhausted_provider_retries_keep_the_attempt_count_and_reason(tmp_path: Path) -> None:
    """The terminal retry is counted and its statusless reason reaches the run summary."""
    reason = "Server disconnected without sending a response"
    provider = MagicMock()
    provider.call.side_effect = ProviderError(reason)
    logs: list[str] = []
    wf = _wf(
        root=tmp_path,
        provider=provider,
        dispatcher=MagicMock(),
        logger=logs.append,
        call=CallSettings(retry_count=2, retry_delay_s=0),
    )

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(
            [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]
        ),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert provider.call.call_count == 3
    assert any("3 attempts" in line for line in logs)
    assert "3 attempts" in result.summary
    assert reason in result.summary


def test_a_call_the_providers_front_end_refused_is_an_error_result_not_a_dispatch(
    tmp_path: Path,
) -> None:
    """A call the claude_code CLI rejected is not run; the refusal is the call's error result."""
    from dataclasses import replace as _replace

    from agent6.harness._conversation import ToolResultItem

    provider = MagicMock()
    refused = _replace(
        _tool_resp("read_file", {"path": "a"}, tool_id="t1"),
        refused={"t1": "InputValidationError: bad input"},
    )
    provider.call.side_effect = [refused, _resp("done")]
    dispatcher = MagicMock()
    logs: list[str] = []
    wf = _wf(
        root=tmp_path,
        provider=provider,
        dispatcher=dispatcher,
        logger=logs.append,
        max_iterations=2,
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]
    )

    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=conversation,
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    dispatcher.dispatch.assert_not_called()
    results = [
        it
        for turn in conversation.turns
        for it in getattr(turn, "items", ())
        if isinstance(it, ToolResultItem)
    ]
    assert results and "InputValidationError: bad input" in results[0].content
    assert any("tool_error" in line and "InputValidationError" in line for line in logs)


class _OneShotSteer:
    """A file-bridge steer stand-in that fires once, returning *text*."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._fired = False

    def requested(self) -> bool:
        return not self._fired

    def prompt(self) -> str:
        return self.text

    def clear(self) -> None:
        self._fired = True


def _resume_snapshot(**kw: Any) -> Any:
    from agent6.harness._snapshot import SessionSnapshot

    defaults: dict[str, Any] = {
        "system": "system",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "TASK:\nmean"}]}],
        "tool_calls": 4,
        "next_iteration": 5,
        "root_task_id": None,
        "original_task": "add a mean() function",
        "verify_command": (),
    }
    defaults.update(kw)
    return SessionSnapshot(**defaults)


def test_resume_seeded_steer_drives_a_finished_run(tmp_path: Path) -> None:
    """`resume --steer` on a finished run injects the follow-up before the first provider call.

    Otherwise the resumed conversation silent-finishes on iteration 1, before the steer poll.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            rendered = str(kwargs["messages"])
            self.calls.append(rendered)
            if "median" not in rendered:
                # The buggy path: the follow-up never reached the model, which re-confirmed.
                return _resp("The mean() function is already done.")
            # The follow-up is present: act on it, then finish.
            if len(self.calls) == 1:
                return _tool_resp("run_command", {"command": "add median"}, tool_id="m1")
            return _tool_resp("finish_session", {"summary": "added median()"}, tool_id="m2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            return RawResult({"content": "ok"})

    steer = _OneShotSteer("add a median() function too")
    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
        bridge=OperatorBridge(
            steer_requested=steer.requested, steer_prompt=steer.prompt, steer_clear=steer.clear
        ),
    )
    snapshot = _resume_snapshot()

    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system=snapshot.system,
            conversation=Conversation.from_wire(snapshot.messages),
            tool_calls=snapshot.tool_calls,
            start_iteration=snapshot.next_iteration,
            root_task_id=snapshot.root_task_id,
            original_task=snapshot.original_task,
            resume_from=snapshot,
        )

    # The seeded steer entered the conversation BEFORE the first provider call.
    assert "median" in provider.calls[0]
    # It drove the run to a real finish, not a dropped-steer silent finish.
    assert result.reason == "finish_session"
    assert result.completed is True
    assert len(provider.calls) >= 2  # at least one more iteration than the silent finish


def test_resume_without_steer_does_not_poll_up_front(tmp_path: Path) -> None:
    """A resume with no `--steer` puts no phantom OPERATOR STEERING block on the wire."""
    captured: list[str] = []

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            captured.append(str(kwargs["messages"]))
            return _resp("done")

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    # No steer callables: steer_requested() is False, so the up-front check is a no-op.
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        max_iterations=5,
    )
    snapshot = _resume_snapshot(
        messages=[{"role": "user", "content": [{"type": "text", "text": "TASK:\nengaged"}]}],
        verify_ever_passed=True,
    )

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system=snapshot.system,
        conversation=Conversation.from_wire(snapshot.messages),
        tool_calls=snapshot.tool_calls,
        start_iteration=snapshot.next_iteration,
        root_task_id=snapshot.root_task_id,
        original_task=snapshot.original_task,
        resume_from=snapshot,
    )

    assert result.reason == "silent_finish"  # unchanged behaviour without a seeded steer
    # The named property: nothing was injected ahead of the first resumed call.
    assert captured and "OPERATOR STEERING" not in captured[0]


def test_drive_loop_auto_metric_unexecutable_aborts_gracefully(tmp_path: Path) -> None:
    """An unexecutable metric command aborts the run the same way on both paths.

    OperatorCommandUnexecutableError is a sibling of ToolError, not a subclass, so the auto path's
    `except ToolError` did not catch it.
    """
    from agent6.tools.dispatch import OperatorCommandUnexecutableError

    class ProviderStub:
        # Always pass verify and never call run_metric_command, so the auto path triggers it.
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _tool_resp("run_verify_command")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                raise OperatorCommandUnexecutableError("metric command '/x/uv' not in jail")
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path, config=config, provider=provider, dispatcher=dispatcher, max_iterations=5
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is False
    assert result.reason == "verify_command_unexecutable"
    # The auto path triggered it: verify ran, then the auto metric raised.
    assert dispatcher.calls == ["run_verify_command", "run_metric_command"]


def test_a_denied_auto_metric_is_withheld_for_the_rest_of_the_run(tmp_path: Path) -> None:
    """The operator's no to the automatic metric holds for the run, like a denied verify."""
    from agent6.tools.dispatch import ToolDeniedError

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _tool_resp("run_verify_command")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                raise ToolDeniedError("denied by the operator")
            raise AssertionError(f"unexpected tool: {name}")

    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    ev = _EventCapture()
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=dispatcher,
        max_iterations=3,
        events=ev,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    trees = iter(["t1", "t2", "t3"])
    with (
        patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"),
        patch.object(RunChain, "tree_sha", side_effect=lambda: next(trees)),
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.reason == "max_iterations"
    assert dispatcher.calls.count("run_verify_command") == 3
    assert dispatcher.calls.count("run_metric_command") == 1
    failed = [e for e in ev.events if e["type"] == "loop.metric.auto_failed"]
    assert len(failed) == 1 and "withheld for the rest of the run" in str(failed[0]["error"])


def test_drive_loop_no_verified_commit_when_edit_follows_verify_in_turn(tmp_path: Path) -> None:
    """A turn that edits after a green verify does not auto-commit as verified.

    The edit changed the tree the verify validated; edit-then-verify still commits.
    """

    def _multi(*names: str) -> ProviderResponse:
        tus = tuple({"id": f"t{i}", "name": n, "input": {}} for i, n in enumerate(names))
        return ProviderResponse(
            text="",
            tool_uses=tus,
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [{"type": "tool_use", **tu} for tu in tus]},
        )

    class ProviderStub:
        def __init__(self) -> None:
            self.n = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.n += 1
            if self.n == 1:
                # verify (green) THEN edit, in that order, in ONE turn.
                return _multi("run_verify_command", "apply_edit")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "apply_edit":
                return RawResult({"ok": True})
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nx"}]}]

    commits: list[str] = []

    def _fake_commit(root: Any, subject: str) -> str:
        del root
        commits.append(subject)
        return f"sha{len(commits)}"

    with patch("agent6.harness._chain.chain_commit", side_effect=_fake_commit):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The verify->edit turn produced no 'verify passed' commit (old code did).
    assert commits == []
    assert result.reason == "finish_session"


def test_worker_max_tokens_starvation_backoff() -> None:
    """A metric run backs off the lifted ceiling after two consecutive quiet turns.

    A one-off quiet keeps the full recovery room; non-metric runs are unaffected.
    """
    metric_cfg = SimpleNamespace(
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        )
    )
    wf = _wf(config=metric_cfg)
    wmt = wf._worker_max_tokens  # pyright: ignore[reportPrivateUsage]
    full = max(wf.call.per_call_max_tokens, wf.call.metric_task_max_tokens)
    assert wmt(_state(quiet=QuietGuard(went_quiet_nudges_used=0))) == full
    assert (
        wmt(_state(quiet=QuietGuard(went_quiet_nudges_used=1))) == full
    )  # one-off quiet: full room
    assert (
        wmt(_state(quiet=QuietGuard(went_quiet_nudges_used=2))) == wf.call.per_call_max_tokens
    )  # spiral: back off
    assert wmt(_state(quiet=QuietGuard(went_quiet_nudges_used=3))) == wf.call.per_call_max_tokens

    # Non-metric run: always per_call, regardless of the quiet streak.
    plain = _wf(
        config=SimpleNamespace(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            harness=SimpleNamespace(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_when="never",
                verify_retries=2,
                verify_command=("true",),
                metric=None,
                verify_timeout_s=60.0,
                verify_infer=True,
            ),
        )
    )
    pwmt = plain._worker_max_tokens  # pyright: ignore[reportPrivateUsage]
    assert (
        pwmt(_state(quiet=QuietGuard(went_quiet_nudges_used=0))) == plain.call.per_call_max_tokens
    )
    assert (
        pwmt(_state(quiet=QuietGuard(went_quiet_nudges_used=2))) == plain.call.per_call_max_tokens
    )


def test_drive_loop_starvation_backoff_breaks_the_spiral(tmp_path: Path) -> None:
    """A model quiet at the full ceiling but active under a tighter cap recovers via the backoff.

    The stub's behaviour is keyed on the cap it receives, so the backoff changes the run's outcome,
    not just a number.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.caps_seen: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            cap = kwargs["max_tokens"]
            self.caps_seen.append(cap)
            if cap >= 65536:
                # Full ceiling: a reasoning binge that emits nothing actionable.
                return ProviderResponse(
                    text="",
                    tool_uses=(),
                    stop_reason="end_turn",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    raw={"content": []},
                )
            # Tightened cap: the model is forced to act.
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
        call=CallSettings(
            per_call_max_tokens=16384, metric_task_max_tokens=65536, retry_delay_s=0.01
        ),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Recovered (finished) rather than dying on went_quiet.
    assert result.reason == "finish_session"
    # Two quiet turns at the lifted ceiling, then the backoff to per_call.
    assert provider.caps_seen[:3] == [65536, 65536, 16384]


def test_drive_loop_finishes_on_metric_plateau(tmp_path: Path) -> None:
    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Improves to 50, then ties: three plateaus draw a pivot nudge, the fourth stops.
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.harness._chain.chain_commit",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5", "sha6", "sha7", "sha8"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "metric_plateau"
    assert "performance per dollar" in result.summary
    # 8 verify+metric pairs: samples 5-7 each draw a pivot nudge, sample 8 stops.
    assert dispatcher.calls == ["run_verify_command", "run_metric_command"] * 8


def test_drive_loop_plateau_nudges_before_stopping(tmp_path: Path) -> None:
    """The first plateau injects a pivot nudge and keeps the run going."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_plateau_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness plateau]" in rendered:
                self.saw_plateau_nudge = True
                return _tool_resp("finish_session", {"summary": "pivoted"}, tool_id="fin")
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.harness._chain.chain_commit",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The plateau at the 5th sample injected a pivot nudge; the worker finished on its own terms.
    assert provider.saw_plateau_nudge is True
    assert result.reason == "finish_session"


def test_drive_loop_plateau_final_nudge_fires_in_final_budget_slice(tmp_path: Path) -> None:
    """Ties while budget is high do not exhaust the plateau patience.

    The final-slice nudge still fires once the budget crosses the threshold.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_final_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "[harness plateau]" in str(kwargs["messages"][-1]):
                self.saw_final_nudge = True
            # Vary the call each turn so the repeat guard (10 identical calls) stays out of the way.
            return _tool_resp(
                "run_verify_command", {"n": self.calls}, tool_id=f"verify-{self.calls}"
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.metric_count = 0
            # Ties 5-8 land in runway and consume no patience; 9-11 draw the final nudge, 12 stops.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 8)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                self.metric_count += 1
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=20,
    )
    # The budget fraction follows the measurement count: samples 5-8 see 80% left, 9+ see 10%.
    wf._budget_fraction_remaining = lambda: 0.8 if dispatcher.metric_count <= 8 else 0.1  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.harness._chain.chain_commit",
        side_effect=[f"sha{i}" for i in range(20)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # The FINAL nudge must have fired in the final slice before the run stopped.
    assert provider.saw_final_nudge is True
    assert result.reason == "metric_plateau"
    # Runway ties did not consume patience, so the run went well past sample 9.
    assert dispatcher.metric_count >= 12


def test_drive_loop_plan_finish_nudge_fires_once_at_iter_cap(tmp_path: Path) -> None:
    """A planner that never calls finish_planning gets one finish nudge at the plan turn cap.

    Pins the off-by-one (iteration - start + 1 >= cap) and the one-shot latch.
    """
    from agent6.harness._nudges import (
        PLAN_BUDGET_NUDGE,
        PLAN_NUDGE_AFTER_ITERS,
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            # never finish on our own -> the loop must force the issue
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            assert name == "read_file"
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="plan",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=PLAN_NUDGE_AFTER_ITERS + 3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nplan a feature"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Injected exactly once, on the turn-cap iteration; the latch keeps it to one.
    assert provider.nudged_on == [PLAN_NUDGE_AFTER_ITERS]


def test_drive_loop_plan_finish_nudge_fires_on_low_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finish nudge also fires early when the token budget runs low."""
    from agent6.harness import loop as loopmod
    from agent6.harness._nudges import PLAN_BUDGET_NUDGE

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Harness, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="plan",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=5,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nplan"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Budget already below the threshold -> nudge on the very first turn, once.
    assert provider.nudged_on == [1]


def test_drive_loop_run_budget_nudge_forces_verify_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-metric `run` gets a one-shot wrap-up nudge when budget runs low."""
    from agent6.harness import loop as loopmod
    from agent6.harness._nudges import RUN_BUDGET_NUDGE

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Harness, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if RUN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("list_dir", {"path": "."}, tool_id=f"l-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"content": "..."})

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=4,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # fires once, on the first turn at/below the threshold, and only once.
    assert provider.nudged_on == [1]


def test_drive_loop_verify_settled_nudges_then_stops(tmp_path: Path) -> None:
    """A worker spinning after a green verify is nudged once, then stopped as verify_settled."""
    from agent6.harness._nudges import VERIFY_SETTLED_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # -> verify passes
            # then spin on read-only commands forever (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.saw_nudge is True
    assert result.reason == "verify_settled"
    assert result.completed is True


def test_drive_loop_settle_after_unreverified_edits_is_not_passed(tmp_path: Path) -> None:
    """A green verify followed by unverified edits settles as 'settled', stale-green summary.

    The settle end grounds on the same tree probe as finish_session.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # green
            if self.calls == 2:  # then an edit nothing re-verifies
                return _tool_resp(
                    "apply_edit",
                    {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 2\n"}]},
                    tool_id="e1",
                )
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=30,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "settled"
    assert "never re-verified" in result.summary
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_settle_after_a_failed_reverify_reports_the_red_gate() -> None:
    """A settled tree whose latest verify failed reports that red, not "no reverify"."""
    wf = _wf(mode="run", config=_cfg_with_verify())
    state = _state(verify=VerifyVerdict(ever_passed=True, last_ok=False))

    with patch.object(RunChain, "dirty", return_value=False):
        result = wf._finish(state, settled_end(state, _ctx(wf, state)), iteration=8)  # pyright: ignore[reportPrivateUsage]

    assert result is not None and result.reason == "settled"
    assert result.verified == "failed"
    assert "verify gate is still red" in result.summary


def test_drive_loop_verify_settled_does_not_fire_before_first_verify(tmp_path: Path) -> None:
    """The settled detector stays dormant until verify has passed once."""
    from agent6.harness._nudges import VERIFY_SETTLED_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="fin")
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input["summary"]})
            return RawResult({"content": "..."})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path, config=config, mode="run", provider=provider, dispatcher=DispatcherStub()
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # never verified -> never nudged/stopped by the settled detector
    assert provider.saw_nudge is False
    assert result.reason == "finish_session"


def test_drive_loop_verify_settled_neutral_on_reverify(tmp_path: Path) -> None:
    """Re-running verify on a green tree is active work; it does not count toward the settle."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")  # always re-verify

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason != "verify_settled"


def test_drive_loop_verify_settled_dormant_on_metric_runs(tmp_path: Path) -> None:
    """On a metric run, post-verify measure and read iterations do not trip the settled stop.

    Completion is owned by the metric early-finish and plateau logic.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # verify passes
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=None,
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    # goal set -> this is a metric run (still mode=="run")
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    # would have been killed at idle 6 without the metric gate
    assert result.reason != "verify_settled"


def test_metric_plateau_nudge_states_the_remaining_budget() -> None:
    """The plateau notice is one fact with the run's remaining budget."""
    from agent6.harness._metric import metric_plateau_nudge as _metric_plateau_nudge

    assert "the remaining budget is unknown" in _metric_plateau_nudge(None)
    assert "80% of the budget remains" in _metric_plateau_nudge(0.80)
    assert _metric_plateau_nudge(0.20).startswith("[harness plateau]")


def test_drive_loop_plateau_keeps_nudging_while_budget_high(tmp_path: Path) -> None:
    """A metric plateau does not end the run while most of the budget is unspent."""
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.plateau_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness plateau]" in rendered:
                self.plateau_nudges_seen += 1
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Plateaus at the 5th sample and stays flat thereafter.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 20)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    # Huge ceilings keep fraction_remaining ~1.0, so the plateau never becomes terminal.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    max_iters = 12
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        max_iterations=max_iters,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.harness._chain.chain_commit",
        side_effect=[f"sha{i}" for i in range(1, max_iters + 2)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # Ran out the iteration cap instead of stopping, nudging past the patience of 3.
    assert result.reason == "max_iterations"
    assert provider.plateau_nudges_seen > 3


def test_drive_loop_rejects_early_finish_while_budget_high(tmp_path: Path) -> None:
    """An early finish_session on a metric run is rejected a few times before it is honoured."""
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            # Vary the summary so the loop-guard repeat detector stays quiet.
            return _tool_resp(
                "finish_session",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    # Huge ceilings keep fraction_remaining ~1.0, well above the final slice.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    # Rejected for the fixed patience of 3, then honoured on the 4th call.
    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 3
    assert provider.calls == 4


def test_drive_loop_honors_finish_without_budget_signal(tmp_path: Path) -> None:
    """With no budget tracker an early finish_session is honoured at once; no deadlock."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"finish-{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=None,
        max_iterations=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )

    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 1


def test_tool_calls_after_finish_session_are_not_executed(tmp_path: Path) -> None:
    """A tool call after finish_session in the same message gets an error, never a dispatch."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            uses = (
                {"id": "f1", "name": "finish_session", "input": {"summary": "done"}},
                {"id": "c1", "name": "run_command", "input": {"command": "rm -rf build"}},
            )
            return ProviderResponse(
                text="",
                tool_uses=uses,
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={"content": [{"type": "tool_use", **u} for u in uses]},
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            self.calls.append(name)
            return RawResult({"ok": True})

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        budget=None,
        max_iterations=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "finish_session"
    assert dispatcher.calls == ["finish_session"]
    assert result.tool_calls == 1


def test_metric_at_fraction_ceiling_detects_maxed_score() -> None:
    from agent6.harness._metric import (
        metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
    )

    # Maxed-out fraction: numerator == score == denominator.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 27.0, pattern=r"SCORE: (\d+)") is True
    assert _metric_at_fraction_ceiling("passed 5 / 5 checks", 5.0, pattern=r"passed (\d+)") is True
    # Partial score is not the ceiling.
    assert _metric_at_fraction_ceiling("SCORE: 26/27\n", 26.0, pattern=r"SCORE: (\d+)") is False
    # Score that does not match the numerator is ignored.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 26.0, pattern=r"SCORE: (\d+)") is False
    # Unbounded metric (raw count, no denominator) never trips the ceiling.
    assert _metric_at_fraction_ceiling("CYCLES: 1487\n", 1487.0, pattern=r"CYCLES: (\d+)") is False


def test_metric_at_fraction_ceiling_scans_only_the_score_line() -> None:
    from agent6.harness._metric import (
        metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
    )

    # A tqdm 100/100 in stderr equals the score; the real score line has no denominator.
    text = "SCORE: 100\n100%|##########| 100/100 [00:03<00:00, 33.1it/s]\n"
    assert _metric_at_fraction_ceiling(text, 100.0, pattern=r"SCORE: (\d+)") is False
    # A genuine maxed fraction ON the score-pattern line still trips it.
    assert _metric_at_fraction_ceiling("junk 3/3\nSCORE: 27/27\n", 27.0, pattern=r"SCORE: (\d+)")
    # No score-pattern match at all -> conservative False.
    assert _metric_at_fraction_ceiling("100/100\n", 100.0, pattern=r"SCORE: (\d+)") is False


def test_drive_loop_honors_finish_at_metric_ceiling(tmp_path: Path) -> None:
    """A finish_session at a maximize metric's provable ceiling (SCORE: N/N) is honoured at once."""
    from agent6.budget import BudgetTracker

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            # First turn: pass verify (the auto-metric reports the ceiling). Then try to finish.
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")
            return _tool_resp(
                "finish_session",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del raw_input
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                return MetricResult(
                    returncode=0,
                    stdout="SCORE: 27/27\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=27.0,
                )
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="maximize", pattern=r"SCORE:\s*([\d.]+)"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    # Huge ceilings keep fraction_remaining ~1.0: without the ceiling guard the finish is rejected.
    budget = BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.harness._chain.chain_commit",
        side_effect=[f"sha{i}" for i in range(1, 22)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    # Honoured on the very first finish_session, with no budget nudges.
    assert result.reason == "finish_session"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 2


# --- tier-aware metric targets --------------------------------------------


def test_extract_metric_targets_ignores_arrow_output() -> None:
    """Grader progress arrows ('epoch 2 -> 27.0') are not thresholds."""
    from agent6.harness._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = "epoch 1 -> 12.0\nepoch 2 -> 27.0\nbest => 30\nSCORE: 27\n"
    assert _extract_metric_targets(text, goal="maximize") == ()
    # Real assert-style thresholds still extract.
    assert _extract_metric_targets("assert score > 25", goal="maximize") == (25.0,)


def test_extract_metric_targets_minimize_picks_upper_bounds() -> None:
    from agent6.harness._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = (
        "assert cycles() < 18532\n"
        "assert cycles() < 1_487\n"
        "assert cycles() < 1579\n"
        "some unrelated > 99 noise\n"
    )
    targets = _extract_metric_targets(text, goal="minimize")
    # Only `<`/`<=` bounds, de-duplicated, order preserved.
    assert targets == (18532.0, 1487.0, 1579.0)


def test_extract_metric_targets_maximize_picks_lower_bounds() -> None:
    from agent6.harness._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = "assert score > 0.80\nassert score >= 0.95\nassert other < 5\n"
    targets = _extract_metric_targets(text, goal="maximize")
    assert targets == (0.80, 0.95)


def test_next_metric_target_minimize_returns_nearest_unmet() -> None:
    from agent6.harness._metric import next_metric_target as _next_metric_target

    targets = (147734.0, 18532.0, 1579.0, 1487.0)
    # At 8256 the nearest unmet threshold is the largest one still below the score.
    assert _next_metric_target(targets, 8256.0, "minimize") == 1579.0
    # Once under everything, no target remains.
    assert _next_metric_target(targets, 1000.0, "minimize") is None


def test_next_metric_target_maximize_returns_nearest_unmet() -> None:
    from agent6.harness._metric import next_metric_target as _next_metric_target

    targets = (0.50, 0.80, 0.95)
    assert _next_metric_target(targets, 0.83, "maximize") == 0.95
    assert _next_metric_target(targets, 0.99, "maximize") is None


def test_next_metric_target_equality_is_unmet() -> None:
    # Thresholds come from strict comparisons, so a score exactly on one has not met it.
    from agent6.harness._metric import next_metric_target as _next_metric_target

    assert _next_metric_target((1487.0,), 1487.0, "minimize") == 1487.0
    assert _next_metric_target((0.95,), 0.95, "maximize") == 0.95
    # Strictly beyond the threshold in the improving direction -> met.
    assert _next_metric_target((1487.0,), 1486.0, "minimize") is None
    assert _next_metric_target((0.95,), 0.96, "maximize") is None


def test_format_metric_feedback_shows_next_target() -> None:
    from agent6.harness._metric import (
        MetricSample as _MetricSample,
    )
    from agent6.harness._metric import (
        format_metric_feedback as _format_metric_feedback,
    )

    history = [
        _MetricSample(label="a", score=20000.0, returncode=0),
        _MetricSample(
            label="b",
            score=8256.0,
            returncode=0,
            targets=(18532.0, 1579.0, 1487.0),
        ),
    ]
    text = _format_metric_feedback(history, goal="minimize")
    assert "next target: below 1579" in text
    assert "current 8256" in text


def test_worker_max_tokens_lifts_cap_on_metric_runs() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        config=config,
        mode="run",
        call=CallSettings(
            per_call_max_tokens=16384, metric_task_max_tokens=32768, retry_delay_s=0.01
        ),
    )
    assert wf._worker_max_tokens(_state()) == 32768  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_without_metric() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        config=config,
        mode="run",
        call=CallSettings(
            per_call_max_tokens=16384, metric_task_max_tokens=32768, retry_delay_s=0.01
        ),
    )
    assert wf._worker_max_tokens(_state()) == 16384  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_in_plan_mode() -> None:
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        config=config,
        mode="plan",
        call=CallSettings(
            per_call_max_tokens=16384, metric_task_max_tokens=32768, retry_delay_s=0.01
        ),
    )
    assert wf._worker_max_tokens(_state()) == 16384  # pyright: ignore[reportPrivateUsage]


# --- tier-2 summarise-and-restart compaction ------------------------------


def _long_history(n_pairs: int) -> list[dict[str, Any]]:
    """Return a task message followed by bulky tool_use and tool_result pairs."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize the kernel"}]}
    ]
    for i in range(n_pairs):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"i": i}}
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "X" * 5000}],
            }
        )
    return msgs


def _restart_via_wire(wf: Harness, messages: list[dict[str, Any]], *, state: Any = None) -> None:
    conversation = Conversation.from_wire(messages)
    wf.compactor.summarise_and_restart(conversation, state if state is not None else _state())
    messages[:] = conversation.to_wire()


def _compact_via_wire(wf: Harness, messages: list[dict[str, Any]], *, state: Any = None) -> bool:
    conversation = Conversation.from_wire(messages)
    out = wf.compactor.compact(conversation, state if state is not None else _state())
    messages[:] = conversation.to_wire()
    return out


def _read_history(*reads: tuple[str, str]) -> list[dict[str, Any]]:
    """An original task message plus one read_file exchange per (path, content)."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}
    ]
    for i, (path, content) in enumerate(reads):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{i}",
                        "name": "read_file",
                        "input": {"path": path},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}],
            }
        )
    return msgs


def test_tier1_compact_event_names_what_was_elided() -> None:
    """loop.compact.dropped carries the elided call identities, not just a count."""
    ev = _EventCapture()
    wf = _wf(
        events=ev,
        compaction=CompactionSettings(
            drop_at_chars=1500, summarise_at_chars=10**9, elision_gists=False
        ),
    )
    msgs = _read_history(("a.py", "X" * 1000), ("b.py", "X" * 1000), ("c.py", "X" * 1000))
    _compact_via_wire(wf, msgs)
    dropped = [e for e in ev.events if e["type"] == "loop.compact.dropped"]
    assert dropped and dropped[-1]["calls"] == ["read_file a.py"]


def test_tier1_gist_event_carries_paths() -> None:
    import json

    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("docs/spec.md: spec facts distilled")
    wf = _wf(
        events=ev,
        compaction=CompactionSettings(
            summariser=summariser, drop_at_chars=1800, summarise_at_chars=10**9, elision_gists=True
        ),
    )
    doc = json.dumps({"content": "authoritative spec. " * 300})
    msgs = _read_history(("docs/spec.md", doc), ("b.py", "x" * 500), ("c.py", "y" * 500))
    _compact_via_wire(wf, msgs)
    gists = [e for e in ev.events if e["type"] == "loop.compact.gists"]
    assert gists
    assert gists[-1]["paths"] == ["docs/spec.md"]
    assert gists[-1]["demoted_paths"] == []


def test_summarise_done_event_carries_summary_text() -> None:
    """The full summary rides the event so surfaces can show what the model works from."""
    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("done: tried A; best=42 at sha9")
    wf = _wf(events=ev, compaction=CompactionSettings(summariser=summariser))
    _restart_via_wire(wf, _long_history(6))
    done = [e for e in ev.events if e["type"] == "loop.compact.summarise.done"]
    assert done and done[-1]["summary"] == "done: tried A; best=42 at sha9"


def test_forced_compact_threads_focus_to_summariser() -> None:
    """`/compact <focus>` reaches the summariser prompt and the loop.compact.requested event."""
    ev = _EventCapture()
    summariser = MagicMock()
    summariser.call.return_value = _resp("s")
    cleared: list[bool] = []
    wf = _wf(
        events=ev,
        compaction=CompactionSettings(
            summariser=summariser, drop_at_chars=10**9, summarise_at_chars=10**9
        ),
        bridge=OperatorBridge(
            compact_requested=lambda: "weigh the auth decisions",
            compact_clear=lambda: cleared.append(True),
        ),
    )
    assert _compact_via_wire(wf, _long_history(3)) is True
    assert cleared == [True]
    sent = str(summariser.call.call_args)
    assert "Operator focus for this summary" in sent
    assert "weigh the auth decisions" in sent
    req = [e for e in ev.events if e["type"] == "loop.compact.requested"]
    assert req and req[-1]["focus"] == "weigh the auth decisions"


def test_forced_compact_below_the_floor_says_it_was_refused() -> None:
    """A `/compact` refused by the history floor emits its own event, so surfaces can say so."""
    ev = _EventCapture()
    summariser = MagicMock()
    cleared: list[bool] = []
    wf = _wf(
        events=ev,
        compaction=CompactionSettings(
            summariser=summariser, drop_at_chars=10**9, summarise_at_chars=10**9
        ),
        bridge=OperatorBridge(
            compact_requested=lambda: "keep the auth work",
            compact_clear=lambda: cleared.append(True),
        ),
    )
    assert _compact_via_wire(wf, _long_history(1)) is False
    assert cleared == [True]
    summariser.call.assert_not_called()
    refused = [e for e in ev.events if e["type"] == "loop.compact.refused"]
    assert refused, "the consumed request must say why nothing happened"
    assert "history" in str(refused[-1].get("reason", ""))


def test_forced_compact_plain_keeps_prompt_unfocused() -> None:
    """A plain `/compact` forces tier-2 with the automatic tier-2 prompt, byte-identical."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("s")
    wf = _wf(
        compaction=CompactionSettings(
            summariser=summariser, drop_at_chars=10**9, summarise_at_chars=10**9
        ),
        bridge=OperatorBridge(compact_requested=lambda: ""),
    )
    assert _compact_via_wire(wf, _long_history(3)) is True
    assert "Operator focus" not in str(summariser.call.call_args)


def test_summarise_and_restart_reinjects_pins_verbatim() -> None:
    """Pins are re-shown verbatim in the restart message; the summariser does not restate them."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary text")
    wf = _wf(compaction=CompactionSettings(summariser=summariser))
    st = _state(pins=["never touch schema files", "goal:\nship X"])
    messages = _long_history(6)
    _restart_via_wire(wf, messages, state=st)
    text = messages[1]["content"][0]["text"]
    assert "PINNED operator instructions (verbatim):" in text
    assert "1. never touch schema files" in text
    assert "2. goal:\nship X" in text
    assert text.index("PINNED operator") < text.index("PROGRESS SUMMARY:")
    assert "does not restate" in str(summariser.call.call_args)


def test_summarise_and_restart_replaces_history() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("done: tried A (kept), B (reverted); best=42 at sha9")
    wf = _wf(compaction=CompactionSettings(summariser=summariser))
    messages = _long_history(6)
    original = messages[0]

    _restart_via_wire(wf, messages)

    # Collapsed to (original task, restart-with-summary).
    assert len(messages) == 2
    assert messages[0] == original
    text = messages[1]["content"][0]["text"]
    assert "[harness context restart]" in text
    assert "best=42 at sha9" in text
    # The summariser saw the worker provider's content, not the worker itself.
    summariser.call.assert_called_once()


def test_summarise_and_restart_applies_dag_checkoff() -> None:
    """Tier-2 compaction applies the summariser's task bookkeeping to the curator.

    Completed tasks pass, discovered ones queue, the block is stripped from the restart, and
    hallucinated ids are ignored.
    """

    class _FakeClient:
        def __init__(self) -> None:
            self._nodes = {
                "01ROOT": {"parent_id": None, "status": "in_progress", "title": "review repo"},
                "01DONE": {"parent_id": "01ROOT", "status": "pending", "title": "audit providers"},
                "01OPEN": {"parent_id": "01ROOT", "status": "pending", "title": "audit sandbox"},
            }
            self.passed: list[str] = []
            self.added: list[tuple[str | None, str]] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def cursor(self) -> str | None:
            return None

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

        def add_subtask(self, intent: Any) -> Any:
            self.added.append((intent.parent_id, intent.draft.title))
            return MagicMock()

    fake = _FakeClient()
    summariser = MagicMock()
    summariser.call.return_value = _resp(
        "Progress: finished the providers audit.\n\n"
        '```checkoff\n{"completed_ids": ["01DONE", "01HALLUCINATED"], '
        '"new_tasks": ["fix the budget rounding bug"]}\n```'
    )
    logged: list[str] = []
    wf = _wf(
        compaction=CompactionSettings(summariser=summariser), curator=fake, logger=logged.append
    )
    messages = _long_history(6)
    _restart_via_wire(wf, messages, state=_state(root_task_id="01ROOT"))

    assert fake.passed == ["01DONE"]  # valid completed id passed; hallucinated id ignored
    assert fake.added == [("01ROOT", "fix the budget rounding bug")]  # queued under the root
    # The log reports what landed: the cap and a refused status both shrink the request.
    assert any("check-off -- passed 1, queued 1" in line for line in logged), logged
    restart_text = messages[1]["content"][0]["text"]
    assert "providers audit" in restart_text
    assert "checkoff" not in restart_text  # bookkeeping block stripped from the restart


class _FakeGraph:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self._nodes = nodes

    def nodes(self) -> dict[str, Any]:
        return _typed(self._nodes)

    def cursor(self) -> str | None:
        return None

    def update_status(self, intent: Any) -> None:
        self._nodes[intent.id]["status"] = intent.new_status


def test_task_finish_gate_nudges_open_subtasks_then_caps() -> None:
    """The gate refuses while a subtask is open, naming the blockers, then yields after patience."""
    from agent6.harness._nudges import TASK_FINISH_PATIENCE

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "sub2": {"parent_id": "root", "status": "passed", "title": "audit sandbox"},  # done
    }
    wf = _wf(curator=_FakeGraph(nodes))
    st = _state()
    for i in range(1, TASK_FINISH_PATIENCE + 1):
        nudge = task_finish_nudge(wf._open_subtasks(), st.gates)  # pyright: ignore[reportPrivateUsage]
        assert nudge is not None and "sub1: audit providers" in nudge
        assert "audit sandbox" not in nudge  # passed subtask not listed
        assert st.gates.task_nudges_used == i
    assert task_finish_nudge(wf._open_subtasks(), st.gates) is None  # pyright: ignore[reportPrivateUsage]
    assert "1 open task(s): audit providers" in with_open_tasks("done", wf._open_subtasks())  # pyright: ignore[reportPrivateUsage]


def test_a_plans_tasks_neither_gate_nor_decorate_its_finish() -> None:
    """A plan's task DAG is its deliverable, open by design: no gate, no decorated receipt."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "plan"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "step one"},
    }
    wf = _wf(curator=_FakeGraph(nodes))
    wf.mode = "plan"
    assert task_finish_nudge(wf._open_subtasks(), _state().gates) is None  # pyright: ignore[reportPrivateUsage]
    assert with_open_tasks("planned", wf._open_subtasks()) == "planned"  # pyright: ignore[reportPrivateUsage]
    turn = _turn()
    state = _state()
    assert (
        wf._end_gates(  # pyright: ignore[reportPrivateUsage]
            state, turn, _ctx(wf, state), ending="silent_finish", gates=SILENT_END_GATES
        )
        is None
    )
    assert turn.end_returned is False and turn.tool_results == []


def test_a_settled_end_over_open_subtasks_after_the_cap_keeps_its_verdict() -> None:
    """With the gate's cap spent, the settled stop goes through, naming the open subtasks."""
    from agent6.harness._nudges import TASK_FINISH_PATIENCE, VERIFY_SETTLED_STOP_AFTER

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "audit providers"},
    }
    wf = _wf(curator=_FakeGraph(nodes))
    state = _state(
        verify=VerifyVerdict(ever_passed=True, last_ok=True),
        settled=SettledGuard(tree="tree", idle=VERIFY_SETTLED_STOP_AFTER - 1),
        gates=FinishGates(task_nudges_used=TASK_FINISH_PATIENCE),
    )
    turn = _turn()
    with patch.object(RunChain, "tree_sha", return_value="tree"):
        assert _settle(wf, state, turn) is None
        assert turn.stops and turn.end_returned is False
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]

    assert result is not None and result.reason == "verify_settled"
    assert result.summary.startswith("verify passed and the worker stopped making changes")
    assert "1 open task(s): audit providers" in result.summary


def test_a_settled_end_from_the_scoped_gate_reads_scoped() -> None:
    """A verify_settled end carries `scoped` like the grounded ends do."""
    from agent6.harness._nudges import VERIFY_SETTLED_STOP_AFTER

    ev = _EventCapture()
    wf = _wf(events=ev)
    state = _state(
        verify=VerifyVerdict(ever_passed=True, last_ok=True, scoped=True),
        settled=SettledGuard(tree="tree", idle=VERIFY_SETTLED_STOP_AFTER - 1),
    )
    turn = _turn()
    with patch.object(RunChain, "tree_sha", return_value="tree"):
        assert _settle(wf, state, turn) is None
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.reason == "verify_settled"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends[-1]["all_passed"] is True and ends[-1]["scoped"] is True


def test_task_finish_gate_allows_finish_without_open_subtasks() -> None:
    """Only subtasks gate: the always-pending root never blocks a finish; no curator, no gate."""
    root_only = _FakeGraph({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    assert task_finish_nudge(_wf(curator=root_only)._open_subtasks(), _state().gates) is None  # pyright: ignore[reportPrivateUsage]
    assert task_finish_nudge(_wf(curator=None)._open_subtasks(), _state().gates) is None  # pyright: ignore[reportPrivateUsage]


def test_verify_settled_end_is_refused_while_a_subtask_is_open() -> None:
    """The automatic settled ending passes the same task gate as finish_session."""
    from agent6.harness._nudges import VERIFY_SETTLED_STOP_AFTER

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "audit providers"},
    }
    wf = _wf(curator=_FakeGraph(nodes))
    state = _state(
        verify=VerifyVerdict(ever_passed=True, last_ok=True),
        settled=SettledGuard(tree="tree", idle=VERIFY_SETTLED_STOP_AFTER - 1),
    )
    turn = _turn()
    with patch.object(RunChain, "tree_sha", return_value="tree"):
        result = _settle(wf, state, turn)

    assert result is None
    assert turn.stops == []
    assert turn.end_returned is True
    assert any(
        isinstance(item, Notice) and "sub1: audit providers" in item.text
        for item in turn.tool_results
    )


def test_metric_plateau_end_is_refused_while_a_subtask_is_open() -> None:
    """A metric ceiling cannot end passed while a task is open; the refusal reaches the model."""
    from agent6.harness._metric import MetricSample

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "optimize"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "measure variant"},
    }
    wf = _wf(curator=_FakeGraph(nodes))
    state = _state(
        verify=VerifyVerdict(ever_passed=True, last_ok=True),
        metric=MetricGuard(
            history=[MetricSample(label="ceiling", score=10, returncode=0, at_ceiling=True)]
        ),
    )
    turn = _turn(metric_plateau_finish="score reached its ceiling")

    ctx = _ctx(wf, state)
    result = wf._take(state, turn, ctx, metric_plateau(turn, state, ctx))  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert turn.stops == []
    assert turn.end_returned is True
    assert any(
        isinstance(item, Notice) and "sub1: measure variant" in item.text
        for item in turn.tool_results
    )


# --- surface-current-task -------------------------------------------------


def test_current_task_id_prefers_open_cursor() -> None:
    """The cursor wins when it still points at an open subtask, even if an earlier one is open."""
    from agent6.harness.loop import current_task_id  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "in_progress", "title": "b"},
    }
    assert current_task_id(_typed(nodes), "b") == "b"  # cursor respected
    assert current_task_id(_typed(nodes), None) == "a"  # no cursor -> first open subtask
    # Stale cursor (points at a closed task) -> recompute the frontier.
    nodes["b"]["status"] = "passed"
    assert current_task_id(_typed(nodes), "b") == "a"
    # Cursor on the auto-root is not a focus target -> first open subtask.
    assert current_task_id(_typed(nodes), "root") == "a"


def test_first_ready_subtask_respects_deps_and_order() -> None:
    """The frontier skips a subtask whose dependency is open; roots and done tasks never show."""
    from agent6.harness._dag_focus import first_ready_subtask as _first_ready_subtask

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "passed", "title": "a"},  # done
        "b": {"parent_id": "root", "status": "pending", "title": "b", "depends_on": ["c"]},
        "c": {"parent_id": "root", "status": "pending", "title": "c"},
    }
    # b is blocked on c (pending) -> c is the first ready subtask.
    assert _first_ready_subtask(_typed(nodes)) == "c"
    # Once c is done, b unblocks.
    nodes["c"]["status"] = "obsolete"
    assert _first_ready_subtask(_typed(nodes)) == "b"
    # Everything done -> nothing ready (the finish-gate, not this, ends the run).
    nodes["b"]["status"] = "passed"
    assert _first_ready_subtask(_typed(nodes)) is None


def test_first_ready_subtask_prefers_leaf_over_decomposed_parent() -> None:
    """A subtask with open children is a container: the frontier surfaces its first ready leaf."""
    from agent6.harness._dag_focus import (
        current_task_id as _current_task_id,
    )
    from agent6.harness._dag_focus import (
        first_ready_subtask as _first_ready_subtask,
    )

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a", "b"]},
        "a": {"parent_id": "root", "status": "in_progress", "title": "a", "children": ["a1", "a2"]},
        "a1": {"parent_id": "a", "status": "pending", "title": "a1"},
        "a2": {"parent_id": "a", "status": "pending", "title": "a2"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    assert _first_ready_subtask(_typed(nodes)) == "a1"  # the parent 'a' is skipped as a container
    assert _current_task_id(_typed(nodes), "a") == "a1"  # stale cursor on the parent falls through
    # Once the children are done, the parent becomes a focusable leaf again.
    nodes["a1"]["status"] = "passed"
    nodes["a2"]["status"] = "passed"
    assert _first_ready_subtask(_typed(nodes)) == "a"


def test_first_ready_subtask_surfaces_a_parent_over_a_failed_child() -> None:
    """A failed child is not open, so the frontier surfaces the parent again."""
    from agent6.harness._dag_focus import first_ready_subtask as _first_ready_subtask

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a"]},
        "a": {"parent_id": "root", "status": "in_progress", "title": "a", "children": ["a1"]},
        "a1": {"parent_id": "a", "status": "failed", "title": "a1"},
    }
    assert _first_ready_subtask(_typed(nodes)) == "a"


def test_current_task_banner_carries_title_acceptance_paths() -> None:
    from agent6.harness.loop import current_task_banner  # pyright: ignore[reportPrivateUsage]

    banner = current_task_banner(
        "01TASK",
        _tn("01TASK", title="audit providers", acceptance="no bugs left", relevant_paths=("a.py",)),
    )
    assert "Current task (01TASK): audit providers" in banner
    assert "Acceptance: no bugs left" in banner
    assert "Relevant paths: a.py" in banner
    assert "ONE task to completion" in banner
    # Absent acceptance/paths are simply omitted, not rendered empty.
    bare = current_task_banner("01X", _tn("01X", title="t"))
    assert "Acceptance:" not in bare and "Relevant paths:" not in bare
    # Decompose invites a finer plan for a large childless task; off by default.
    assert "child subtasks" not in bare
    rec = current_task_banner("01X", _tn("01X", title="t"), decompose=True)
    assert "child subtasks under it (parent_id=01X)" in rec
    has_kids = current_task_banner("01X", _tn("01X", title="t", children=("01Y",)), decompose=True)
    assert "child subtasks" not in has_kids


def test_graph_update_snapshot_payload_is_wire_stable(tmp_path: Path) -> None:
    """The `graph.update` event's node projection is a frozen wire surface.

    Each node is exactly {title, status, parent_id, children, created_by, standing} plus a top-level
    cursor; a run dir written before a field existed lacks it and every reader defaults it. Driven
    with a real curator and Harness, so the emitted bytes are pinned.
    """
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import (
        AddSubtaskIntent,
        SetCursorIntent,
        TaskNodeDraft,
        UpdateStatusIntent,
    )
    from agent6.sessions.layout import SessionLayout

    cur = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    child = cur.add_subtask(
        AddSubtaskIntent(parent_id=root.id, draft=TaskNodeDraft(title="child", created_by="worker"))
    )
    cur.update_status(UpdateStatusIntent(id=child.id, new_status="in_progress"))
    cur.set_cursor(SetCursorIntent(id=child.id))

    captured: list[tuple[str, dict[str, Any]]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            captured.append((event_type, fields))

    wf = _wf(curator=cur, events=_Events())
    wf._emit_graph_snapshot()  # pyright: ignore[reportPrivateUsage]

    assert len(captured) == 1
    etype, fields = captured[0]
    assert etype == "graph.update"
    assert fields == {
        "nodes": {
            root.id: {
                "title": "root",
                "status": "pending",
                "parent_id": None,
                "children": [child.id],
                "created_by": "planner",
                "standing": False,
            },
            child.id: {
                "title": "child",
                "status": "in_progress",
                "parent_id": root.id,
                "children": [],
                "created_by": "worker",
                "standing": False,
            },
        },
        "cursor": child.id,
    }
    # children serialize as a JSON list, the on-disk contract old run dirs hold.
    assert isinstance(fields["nodes"][root.id]["children"], list)


def test_decompose_prompt_describes_nested_phases() -> None:
    from agent6.prompts.loop import DAG_RULES_DECOMPOSE, dag_rules_block

    assert dag_rules_block(True) == DAG_RULES_DECOMPOSE
    # Phases with child subtasks (parent_id), and the re-plan-when-large rule.
    assert "phases" in DAG_RULES_DECOMPOSE.lower()
    assert "parent_id" in DAG_RULES_DECOMPOSE
    assert "large" in DAG_RULES_DECOMPOSE.lower()


class _FakeCurator:
    """In-memory GraphCurator stand-in: nodes / cursor / set_cursor / update_status."""

    def __init__(self, nodes: dict[str, dict[str, Any]], cursor: str | None = None) -> None:
        self._nodes = nodes
        self._cursor = cursor
        self.cursor_sets: list[str | None] = []
        self.status_sets: list[tuple[str, str]] = []

    def nodes(self) -> dict[str, Any]:
        return _typed(self._nodes)

    def cursor(self) -> str | None:
        return self._cursor

    def set_cursor(self, intent: Any) -> None:
        self._cursor = intent.id
        self.cursor_sets.append(intent.id)

    def update_status(self, intent: Any) -> None:
        self.status_sets.append((intent.id, intent.new_status))
        self._nodes[intent.id]["status"] = intent.new_status


def _surface(wf: Harness, st: Any, messages: list[dict[str, Any]]) -> None:
    """Wire-in/wire-out driver so the tests keep asserting on message dicts."""
    conversation = Conversation.from_wire(messages)
    wf._maybe_surface_current_task(conversation, st)  # pyright: ignore[reportPrivateUsage]
    messages[:] = conversation.to_wire()


def test_surface_current_task_surfaces_advances_then_quiets() -> None:
    """The first call surfaces the focus banner and marks the task in_progress; a repeat is mute."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "b": {"parent_id": "root", "status": "pending", "title": "audit sandbox"},
    }
    cur = _FakeCurator(nodes)
    wf = _wf(curator=cur)
    st = _state()
    messages: list[dict[str, Any]] = []

    _surface(wf, st, messages)
    assert len(messages) == 1
    assert "audit providers" in messages[0]["content"][0]["text"]
    assert cur.cursor_sets == ["a"]  # cursor advanced onto the focus task
    assert cur.status_sets == [("a", "in_progress")]  # reflected as being worked
    assert st.focus.surfaced_task_id == "a"

    # Same focus -> no new banner, no redundant cursor/status writes.
    _surface(wf, st, messages)
    assert len(messages) == 1
    assert cur.cursor_sets == ["a"]
    assert cur.status_sets == [("a", "in_progress")]  # no second write for the same task

    # Worker finishes task a -> next turn focus advances to b.
    nodes["a"]["status"] = "passed"
    _surface(wf, st, messages)
    assert len(messages) == 2
    assert "audit sandbox" in messages[1]["content"][0]["text"]
    assert cur.cursor_sets == ["a", "b"]
    assert cur.status_sets == [("a", "in_progress"), ("b", "in_progress")]
    assert st.focus.surfaced_task_id == "b"


def test_surface_current_task_skips_status_write_when_already_in_progress() -> None:
    """A task already in_progress is surfaced without a redundant update_status write."""
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "in_progress", "title": "audit providers"},
        },
        cursor="a",
    )
    wf = _wf(curator=cur)
    messages: list[dict[str, Any]] = []
    _surface(wf, _state(), messages)
    assert len(messages) == 1  # banner still surfaced
    assert cur.status_sets == []  # already in_progress -> no redundant status write
    assert cur.cursor_sets == []  # cursor already on it -> no redundant set_cursor


def test_surface_current_task_resurfaces_after_compaction_reset() -> None:
    """A tier-2 restart resets surfaced_task_id, so the next call re-injects the focus banner."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    _surface(wf, st, messages)
    assert len(messages) == 1
    st.focus.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert len(messages) == 2  # re-surfaced after the restart wiped the banner


def test_surface_current_task_noop_cases() -> None:
    """The stuck-task nudge is a no-op without open subtasks, a curator, or run mode."""
    root_only = _FakeCurator({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    msgs: list[dict[str, Any]] = []
    _surface(_wf(curator=root_only), _state(), msgs)
    assert msgs == [] and root_only.cursor_sets == []

    _surface(_wf(curator=None), _state(), msgs)
    assert msgs == []

    open_sub = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "pending", "title": "t"},
            "a": {"parent_id": "root", "status": "pending", "title": "a"},
        }
    )
    _surface(_wf(curator=open_sub, mode="plan"), _state(), msgs)
    assert msgs == [] and open_sub.cursor_sets == []  # plan mode does not surface


def _stuck_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if "without concluding it" in m["content"][0]["text"])


def test_surface_current_task_stuck_nudge_fires_periodically_then_caps() -> None:
    """The split/pass/skip nudge re-fires on the same stuck task and caps at _STUCK_NUDGE_MAX."""
    from agent6.harness._dag_focus import STUCK_NUDGE_MAX, STUCK_ON_TASK_AFTER

    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    wf = _wf(curator=cur)
    st = _state()
    messages: list[dict[str, Any]] = []
    # One nudge after the first period, but not before it.
    for _ in range(STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0  # turns_on_task is _STUCK_ON_TASK_AFTER-1 here
    _surface(wf, st, messages)
    assert _stuck_count(messages) == 1  # crossed the first period
    # Keep grinding well past the cap; it re-fires periodically then stops.
    for _ in range((STUCK_NUDGE_MAX + 2) * STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == STUCK_NUDGE_MAX
    assert st.focus.stuck_nudges_fired == STUCK_NUDGE_MAX


def test_surface_current_task_stuck_nudge_resets_on_progress() -> None:
    """Forward motion (a task marked passed, focus advances) resets the grind counter."""
    from agent6.harness._dag_focus import STUCK_ON_TASK_AFTER

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(STUCK_ON_TASK_AFTER - 1):  # grind almost to the threshold on a
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    nodes["a"]["status"] = "passed"  # progress -> focus advances to b
    for _ in range(3):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    assert st.focus.last_focus_id == "b" and st.focus.turns_on_task < STUCK_ON_TASK_AFTER


def test_surface_current_task_stuck_counter_survives_compaction() -> None:
    """A tier-2 restart resets the banner but not the grind counter: compaction is not progress."""
    wf = _wf(
        curator=_FakeCurator(
            {
                "root": {"parent_id": None, "status": "in_progress", "title": "r"},
                "a": {"parent_id": "root", "status": "pending", "title": "a"},
            }
        )
    )
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(5):
        _surface(wf, st, messages)
    assert st.focus.turns_on_task == 4
    st.focus.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert st.focus.turns_on_task == 5  # kept climbing across the restart
    assert st.focus.last_focus_id == "a"


def test_surface_decompose_resets_grind_counter() -> None:
    """Decomposing the focus task moves focus to the first new leaf and resets the grind counter."""
    nodes: dict[str, dict[str, Any]] = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a"]},
        "a": {"parent_id": "root", "status": "pending", "title": "a", "children": []},
    }
    wf = _wf(curator=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(5):
        _surface(wf, st, messages)
    assert st.focus.last_focus_id == "a" and st.focus.turns_on_task == 4
    # Worker splits 'a' into a child -> 'a' becomes a container, focus moves to a1.
    nodes["a"]["status"] = "in_progress"
    nodes["a"]["children"] = ["a1"]
    nodes["a1"] = {"parent_id": "a", "status": "pending", "title": "a1"}
    _surface(wf, st, messages)
    assert st.focus.last_focus_id == "a1"  # focus advanced to the new leaf
    assert st.focus.turns_on_task == 0  # grind counter reset by the decompose


def test_maybe_compact_returns_restart_signal() -> None:
    """Compact returns True only when a tier-2 restart replaced the history."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    wf = _wf(compaction=CompactionSettings(summariser=summariser, summarise_at_chars=500_000))
    # Below the tier-2 threshold -> no restart, returns False.
    short = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _compact_via_wire(wf, short) is False
    # Over the threshold -> restart, returns True.
    big = _big_text_history("TASK: x", blocks=8, block_chars=100_000)
    assert _compact_via_wire(wf, big) is True
    # History was replaced: [task, restart+summary, verbatim recent tail].
    assert len(big) == 3


def test_tier2_summariser_is_told_the_task() -> None:
    """The summary's goal line comes from the task, never from a transcript that starts mid-work."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    wf = _wf(compaction=CompactionSettings(summariser=summariser, summarise_at_chars=500_000))
    big = _big_text_history("TASK: x", blocks=8, block_chars=100_000)
    st = _state()
    st.original_task = "make the tests pass"
    assert _compact_via_wire(wf, big, state=st) is True
    user_msg = summariser.call.call_args.kwargs["messages"][0]["content"]
    assert "TASK (the goal, verbatim):\nmake the tests pass" in user_msg


def test_a_restart_does_not_re_fire_on_the_next_iteration() -> None:
    """The growth floor counts the request prefix, as the trigger does.

    Measured on the conversation alone, the floor sat below the total the moment a restart landed
    near the threshold, so tier 2 re-fired with no new turns.
    """
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    # The prefix alone clears the threshold, so the floor is all that spaces the summariser calls.
    wf = _wf(compaction=CompactionSettings(summariser=summariser, summarise_at_chars=60_000))
    msgs = _big_text_history("TASK: x", blocks=8, block_chars=20_000)
    st = _state()
    prefix = 100_000

    conversation = Conversation.from_wire(msgs)
    assert wf.compactor.compact(conversation, st, prefix_chars=prefix) is True
    assert summariser.call.call_count == 1

    # Same conversation, one iteration later, nothing added.
    assert wf.compactor.compact(conversation, st, prefix_chars=prefix) is False
    assert summariser.call.call_count == 1


def test_compact_request_forces_a_tier2_restart() -> None:
    """An operator compact.request forces one tier-2 restart at the next boundary, then clears."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    pending = {"req": True}
    wf = _wf(
        compaction=CompactionSettings(summariser=summariser, summarise_at_chars=500_000),
        bridge=OperatorBridge(
            compact_requested=lambda: "" if pending["req"] else None,
            compact_clear=lambda: pending.__setitem__("req", False),
        ),
    )
    small = _big_text_history("TASK: x", blocks=2, block_chars=100)  # nowhere near tier 2
    assert _compact_via_wire(wf, small) is True
    assert pending["req"] is False  # the marker was consumed
    assert len(small) == 2  # history replaced by (task + summary)
    # No re-trigger without a fresh request (and still below the threshold).
    assert _compact_via_wire(wf, small) is False


def test_stop_request_ends_the_run_at_the_step_boundary(tmp_path: Path) -> None:
    """A stop.request ends the run at the iteration boundary in the resumable steer_abort shape."""
    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            tid = f"t{self.calls}"
            return ProviderResponse(
                text="working",
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": "working"},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            del name, raw_input
            return RawResult({"ok": True})

    provider = ProviderStub()
    pending = {"stop": True}
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        events=EventSink(tmp_path / "logs.jsonl"),
        bridge=OperatorBridge(
            stop_requested=lambda: pending["stop"],
            stop_clear=lambda: pending.__setitem__("stop", False),
        ),
        max_iterations=30,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: x"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "steer_abort"  # the resumable stopped shape
    assert "stopped the run after step 1" in result.summary
    assert provider.calls == 1  # the step completed; no second turn started
    assert pending["stop"] is False  # the marker was consumed


def test_drive_loop_resurfaces_current_task_after_compaction(tmp_path: Path) -> None:
    """A tier-2 restart mid-run wipes the focus banner, so the next nudge pass re-surfaces it."""
    import json

    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"f{self.calls}")
            big = "y" * 3000  # accumulates each turn so tier-2 fires mid-run
            tid = f"t{self.calls}"
            return ProviderResponse(
                text=big,
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": big},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class SummariserStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _resp("SUMMARY of progress so far")  # no checkoff block

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})
            return RawResult({"ok": True})

    events = EventSink(tmp_path / "logs.jsonl")
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "review"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        compaction=CompactionSettings(
            summariser=SummariserStub(), drop_at_chars=256_000, summarise_at_chars=5_000
        ),
        events=events,
        curator=cur,  # low so tier-2 fires mid-run
        budget=None,
        max_iterations=30,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: review"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id="root",
            original_task="t",
        )
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # tier-2 restart happened
    assert "loop.task.surfaced" in types
    # The focus banner re-surfaces AFTER the restart wiped it.
    restart_at = types.index("loop.compact.summarise.done")
    assert "loop.task.surfaced" in types[restart_at + 1 :]


def test_summarise_and_restart_falls_back_to_worker_provider() -> None:
    worker = MagicMock()
    worker.call.return_value = _resp("summary text")
    wf = _wf(provider=worker, compaction=CompactionSettings(summariser=None))
    messages = _long_history(4)

    _restart_via_wire(wf, messages)

    assert len(messages) == 2
    worker.call.assert_called_once()


def test_summarise_and_restart_keeps_history_on_empty_summary() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("   ")
    wf = _wf(compaction=CompactionSettings(summariser=summariser))
    messages = _long_history(5)
    before = list(messages)

    _restart_via_wire(wf, messages)

    # Empty summary -> message list untouched (fail-safe).
    assert messages == before


def test_summarise_and_restart_rejects_checkoff_without_a_summary() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp(
        '```checkoff\n{"completed_ids": ["01DONE"], "new_tasks": []}\n```'
    )
    curator = MagicMock()
    curator.nodes.return_value = _typed(
        {
            "01ROOT": {"parent_id": None, "status": "in_progress", "title": "review repo"},
            "01DONE": {
                "parent_id": "01ROOT",
                "status": "pending",
                "title": "audit providers",
            },
        }
    )
    wf = _wf(compaction=CompactionSettings(summariser=summariser), curator=curator)
    conversation = Conversation.from_wire(_long_history(5))
    before = conversation.to_wire()

    restarted = wf.compactor.summarise_and_restart(conversation, _state())

    assert restarted is False
    assert conversation.to_wire() == before
    curator.update_status.assert_not_called()


def test_summarise_and_restart_keeps_history_on_provider_error() -> None:
    summariser = MagicMock()
    summariser.call.side_effect = ProviderError("boom")
    wf = _wf(compaction=CompactionSettings(summariser=summariser))
    messages = _long_history(5)
    before = list(messages)

    _restart_via_wire(wf, messages)

    assert messages == before


# --- _maybe_handle_steer --------------------------------------------------


def _steer_via_wire(
    wf: Harness, messages: list[dict[str, Any]], *, iteration: int, state: Any
) -> str | None:
    conversation = Conversation.from_wire(messages)
    try:
        return wf.steering.handle(conversation, iteration, state)
    finally:
        messages[:] = conversation.to_wire()


def test_steer_noop_when_not_requested() -> None:
    """steer_requested() returns False -> _maybe_handle_steer is a no-op."""
    wf = _wf()  # default steer_requested = lambda: False
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=1, state=_state())
    assert result is None
    assert messages == []


def test_steer_injects_instruction() -> None:
    """Requested + non-empty prompt text -> instruction appended to messages."""
    cleared: list[bool] = []
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=lambda: "focus on perf_takehome.py first",
        ),
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=3, state=_state())
    assert result is None
    assert cleared == [True], "steer_clear must be called even on success"
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "text"
    assert "OPERATOR STEERING" in block["text"]
    assert "focus on perf_takehome.py first" in block["text"]


def test_steer_empty_text_continues_without_inject() -> None:
    """Operator answered blank/whitespace -> continue with no message."""
    cleared: list[bool] = []
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=lambda: "   ",
        ),
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=2, state=_state())
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_none_text_continues_without_inject() -> None:
    """Operator EOF'd (None) -> continue with no message."""
    cleared: list[bool] = []
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=lambda: None,
        ),
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=2, state=_state())
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_pin_records_and_injects_marked_notice() -> None:
    ev = _EventCapture()
    st = _state()
    wf = _wf(
        events=ev,
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: None,
            steer_prompt=lambda: "/pin never touch the schema files",
        ),
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=3, state=st) is None
    assert st.pins == ["never touch the schema files"]
    block = messages[0]["content"][0]["text"]
    assert "pinned:" in block and "survives context compaction" in block
    assert "never touch the schema files" in block
    added = [e for e in ev.events if e["type"] == "loop.pin.added"]
    assert added and added[-1]["text"] == "never touch the schema files"
    assert added[-1]["count"] == 1


def test_steer_pin_over_cap_delivers_as_ordinary_steer() -> None:
    """A pin past the total cap still reaches the model now as a plain steer.

    Only the survives-compaction durability is refused, loudly.
    """
    from agent6.harness._operator import PINS_MAX_CHARS

    ev = _EventCapture()
    st = _state(pins=["x" * (PINS_MAX_CHARS - 10)])
    wf = _wf(
        events=ev,
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: None,
            steer_prompt=lambda: "/pin " + "y" * 100,
        ),
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=3, state=st) is None
    assert len(st.pins) == 1  # the oversized pin was NOT recorded
    text = messages[0]["content"][0]["text"]
    assert "OPERATOR STEERING" in text and "y" * 100 in text
    assert "not pinned" in text  # the refusal is visible on every surface
    assert "PINNED" not in text
    refused = [e for e in ev.events if e["type"] == "loop.pin.refused"]
    assert refused and refused[-1]["limit"] == PINS_MAX_CHARS


def test_steer_bare_pin_answers_with_feedback() -> None:
    st = _state()
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True, steer_clear=lambda: None, steer_prompt=lambda: "/pin   "
        ),
    )
    messages: list[dict[str, Any]] = []
    assert _steer_via_wire(wf, messages, iteration=1, state=st) is None
    assert st.pins == []
    assert messages and "nothing pinned" in messages[0]["content"][0]["text"]


def test_steer_abort_signal() -> None:
    """Operator typed 'abort' (case-insensitive) -> returns 'abort'."""
    for typed in ("abort", "ABORT", "Abort"):
        cleared: list[bool] = []

        def _record(c: list[bool] = cleared) -> None:
            c.append(True)

        def _typed(t: str = typed) -> str:
            return t

        wf = _wf(
            bridge=OperatorBridge(
                steer_requested=lambda: True, steer_clear=_record, steer_prompt=_typed
            ),
        )
        messages: list[dict[str, Any]] = []
        result = _steer_via_wire(wf, messages, iteration=5, state=_state())
        assert result == "abort", f"typed={typed!r}"
        assert cleared == [True]
        assert messages == [], "abort must not inject a message"


def test_steer_detach_signal() -> None:
    """Operator chose 'detach' -> returns 'detach' (the caller backgrounds the run)."""
    cleared: list[bool] = []
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=lambda: "detach",
        ),
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=4, state=_state())
    assert result == "detach"
    assert cleared == [True]
    assert messages == [], "detach must not inject a message"


def test_steer_clear_called_even_when_prompt_raises() -> None:
    """A misbehaving steer_prompt must not leave the flag set."""
    cleared: list[bool] = []

    def boom() -> str | None:
        raise RuntimeError("input EOF")

    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=boom,
        ),
    )
    messages: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="input EOF"):
        _steer_via_wire(wf, messages, iteration=1, state=_state())
    assert cleared == [True], "finally must run steer_clear even on prompt failure"


# --- resume: snapshot save/load and resume() behaviour ------------


def test_save_resume_snapshot_noop_when_path_unset(tmp_path: Path) -> None:
    """resume_state_path=None -> no file written, no exception."""
    wf = _wf()
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(system="s", tool_calls=0, root_task_id=None), [], next_iteration=1
    )
    # tmp_path should still be empty.
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_run_snapshot_round_trip(tmp_path: Path) -> None:
    """Snapshot written by _save_resume_snapshot loads back identically."""
    from agent6.harness.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
    ]
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(tool_calls=3, system="SYSTEM PROMPT", root_task_id="task-abc"),
        msgs,
        next_iteration=7,
    )
    assert snap_path.is_file()
    loaded = load_session_snapshot(snap_path)
    assert loaded.system == "SYSTEM PROMPT"
    assert loaded.messages == msgs
    assert loaded.tool_calls == 3
    assert loaded.next_iteration == 7
    assert loaded.root_task_id == "task-abc"


def test_save_resume_snapshot_atomic_no_partial_tmp(tmp_path: Path) -> None:
    """After save, no .tmp file remains: the snapshot and its checkpoint are written atomically."""
    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(system="s", tool_calls=0, root_task_id=None),
        [],
        next_iteration=1,
        write_checkpoint=True,
    )
    assert snap_path.is_file()
    # The per-turn checkpoint lands under checkpoints/; nothing else (no .tmp).
    assert (tmp_path / "checkpoints" / "0001.json").is_file()
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != snap_path.name)
    assert leftovers == ["checkpoints"], f"unexpected leftover files: {leftovers}"
    cp_leftovers = [p.name for p in (tmp_path / "checkpoints").iterdir()]
    assert cp_leftovers == ["0001.json"], f"unexpected checkpoint leftovers: {cp_leftovers}"


def test_save_resume_snapshot_uses_durable_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume state must go through the durable writer, not plain write_text."""
    writes: list[Path] = []

    def _fake_atomic_write(path: Path, data: str | bytes) -> None:
        writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")

    monkeypatch.setattr("agent6.harness.loop.atomic_write", _fake_atomic_write)
    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)

    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(system="s", tool_calls=0, root_task_id=None),
        [],
        next_iteration=9,
        write_checkpoint=True,
    )

    assert writes == [tmp_path / "checkpoints" / "0009.json", snap_path]


def test_load_run_snapshot_rejects_version_mismatch(tmp_path: Path) -> None:
    """A snapshot with a wrong version must raise ValueError."""
    import json as _json

    from agent6.harness.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap_path = tmp_path / "loop_state.json"
    snap_path.write_text(
        _json.dumps(
            {
                "version": 999,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=f"is version 999, not {SNAPSHOT_VERSION}"):
        load_session_snapshot(snap_path)


def test_resume_raises_when_path_unset() -> None:
    """resume() with resume_state_path=None must raise ResumeError."""
    from agent6.harness.loop import ResumeError

    wf = _wf()
    with pytest.raises(ResumeError, match="resume_state_path"):
        wf.resume()


def test_resume_raises_on_missing_snapshot(tmp_path: Path) -> None:
    """resume() with a nonexistent snapshot file must raise ResumeError."""
    from agent6.harness.loop import ResumeError

    wf = _wf(resume_state_path=tmp_path / "nope.json")
    with pytest.raises(ResumeError, match="failed to load"):
        wf.resume()


def test_resume_drives_loop_from_snapshot(tmp_path: Path) -> None:
    """resume() loads snapshot, calls provider once, finishes via silent_finish."""
    snap_path = tmp_path / "loop_state.json"
    # The snapshot as a prior run left it: iter 4 done, iter 5 about to start.
    snap_path.write_text(
        f'{{"version": {SNAPSHOT_VERSION}, "system": "S", "messages": [{{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 2, '
        '"next_iteration": 5, "root_task_id": null, "original_task": "go", '
        '"verify_command": []}',
        encoding="utf-8",
    )
    seeded_mtime_ns = snap_path.stat().st_mtime_ns

    provider = MagicMock()
    provider.call.return_value = _resp("all done")  # no tool_uses -> silent_finish

    dispatcher = MagicMock()
    dispatcher.set_run_root_node_id = MagicMock()

    wf = _wf(provider=provider, dispatcher=dispatcher, resume_state_path=snap_path)
    result = wf.resume()

    assert result.completed is True
    assert result.reason == "silent_finish"
    assert result.iterations == 5, "must resume at snapshot's next_iteration"
    assert result.tool_calls == 2, "must carry forward snapshot's tool_calls"
    # The pre-call save REWROTE the seeded snapshot (not merely left it there).
    assert snap_path.stat().st_mtime_ns > seeded_mtime_ns


def test_resume_restores_root_task_id_on_dispatcher(tmp_path: Path) -> None:
    """A non-null root_task_id in the snapshot must be re-set on dispatcher."""
    snap_path = tmp_path / "loop_state.json"
    snap_path.write_text(
        f'{{"version": {SNAPSHOT_VERSION}, "system": "S", "messages": [{{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 0, '
        '"next_iteration": 1, "root_task_id": "task-xyz", "original_task": "go", '
        '"verify_command": []}',
        encoding="utf-8",
    )
    provider = MagicMock()
    provider.call.return_value = _resp("done")
    dispatcher = MagicMock()
    wf = _wf(provider=provider, dispatcher=dispatcher, resume_state_path=snap_path)
    wf.resume()
    dispatcher.set_run_root_node_id.assert_called_once_with("task-xyz")


# --- crash-and-resume: snapshot survives a provider crash mid-run ---


def test_crash_mid_run_then_resume_continues_from_snapshot(tmp_path: Path) -> None:
    """A snapshot is written before each LLM call, so a crash mid-loop resumes at that turn.

    A fake provider raises on the first call; a fresh one on resume drives the loop to a clean
    finish.
    """
    import subprocess as _sp

    # Real git repo so load_repo_summary() succeeds.
    repo = tmp_path / "repo"
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n")
    _sp.run(["git", "add", "x.txt"], cwd=repo, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    snap_path = repo / "loop_state.json"

    # First "process": crash on the first LLM call.
    crashing_provider = MagicMock()
    crashing_provider.call.side_effect = ProviderError("simulated network drop / SIGKILL window")
    dispatcher = MagicMock()
    dispatcher.set_run_root_node_id = MagicMock()
    wf1 = _wf(
        root=repo,
        provider=crashing_provider,
        dispatcher=dispatcher,
        resume_state_path=snap_path,
        call=CallSettings(retry_count=0, retry_delay_s=0.01),  # don't mask the crash with a retry
    )
    # The first run ends with provider_error; the pre-call snapshot makes that iteration resumable.
    result1 = wf1.run("do a thing")
    assert result1.completed is False
    assert result1.reason == "provider_error"

    # Snapshot must exist after the crash and be loadable.
    assert snap_path.is_file(), "snapshot must be written before every LLM call"
    from agent6.harness.loop import load_session_snapshot  # pyright: ignore[reportPrivateUsage]

    snap = load_session_snapshot(snap_path)
    # The user's task message survived in the snapshot.
    user_text = "".join(
        block.get("text", "")
        for msg in snap.messages
        if msg["role"] == "user"
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "do a thing" in user_text, "user task message must be preserved in the snapshot"

    # Second "process": new provider, drives to silent_finish.
    fresh_provider = MagicMock()
    fresh_provider.call.return_value = _resp("done now")
    wf2 = _wf(
        root=repo,
        provider=fresh_provider,
        dispatcher=dispatcher,
        resume_state_path=snap_path,
    )
    result = wf2.resume()
    assert result.completed is True
    assert result.reason == "silent_finish"


# Tier-2 summarise-and-restart, driven past compact_summarise_at_chars.


def _ctx_chars(messages: list[dict[str, Any]]) -> int:
    from agent6.harness._compaction import context_chars

    return context_chars(Conversation.from_wire(messages))


def _big_text_history(task: str, *, blocks: int, block_chars: int) -> list[dict[str, Any]]:
    # Assistant text accumulates and tier-1 never elides it, so it is what tier-2 must catch.
    big = "x" * block_chars
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": task}]}]
    for _ in range(blocks):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": big}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": "keep going"}]})
    return msgs


def test_tier2_summarise_fires_and_restarts_past_threshold(tmp_path: Path) -> None:
    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("PROGRESS SUMMARY: explored modules, applied 3 patches.")

    summ = SummariserStub()
    wf = _wf(
        root=tmp_path,
        compaction=CompactionSettings(
            summariser=summ, drop_at_chars=256_000, summarise_at_chars=500_000
        ),
    )
    messages = _big_text_history("TASK: optimize the kernel", blocks=8, block_chars=100_000)
    assert _ctx_chars(messages) > 500_000  # over the tier-2 threshold

    _compact_via_wire(wf, messages)

    assert summ.calls == 1  # tier-2 summariser ran
    # Restarted to [task, restart+summary, recent tail]: the trailing small turn survives verbatim.
    assert len(messages) == 3
    assert messages[0]["content"][0]["text"] == "TASK: optimize the kernel"
    assert "PROGRESS SUMMARY" in messages[1]["content"][0]["text"]
    assert messages[2]["content"][0]["text"] == "keep going"
    assert _ctx_chars(messages) < 500_000  # context actually shrank


def test_tier2_summarise_failsafe_keeps_context_on_empty_summary(tmp_path: Path) -> None:
    class EmptySummariser:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _resp("")  # empty -> fail-safe: keep the (tier-1-elided) context

    wf = _wf(
        root=tmp_path,
        compaction=CompactionSettings(summariser=EmptySummariser(), summarise_at_chars=500_000),
    )
    messages = _big_text_history("TASK", blocks=8, block_chars=100_000)
    n_before = len(messages)

    _compact_via_wire(wf, messages)

    assert len(messages) == n_before  # unchanged; the run continues on tier-1 elision


def test_drive_loop_summarises_midrun_then_completes(tmp_path: Path) -> None:
    import json

    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls >= 6:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id=f"f{self.calls}")
            # Large assistant text accumulates each turn; tier-1 can't elide it.
            big = "y" * 3000
            tid = f"t{self.calls}"
            return ProviderResponse(
                text=big,
                tool_uses=({"id": tid, "name": "noop", "input": {}},),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={
                    "content": [
                        {"type": "text", "text": big},
                        {"type": "tool_use", "id": tid, "name": "noop", "input": {}},
                    ]
                },
            )

    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("SUMMARY of progress so far")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})
            return RawResult({"ok": True})

    events = EventSink(tmp_path / "logs.jsonl")
    summ = SummariserStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
        prompt=SimpleNamespace(decompose=False),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        compaction=CompactionSettings(
            summariser=summ, drop_at_chars=256_000, summarise_at_chars=5_000
        ),
        events=events,  # low so it fires mid-run
        budget=None,
        max_iterations=30,
    )
    wf.config = _knobs(wf.config, loop_guard_kill_threshold=0)
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: optimize"}]}]

    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )

    assert result.completed is True
    assert result.reason == "finish_session"
    assert summ.calls >= 1  # tier-2 fired mid-run
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # summarise-and-restart happened cleanly


def test_pass_pending_root_tasks_passes_only_pending_roots() -> None:
    """`_pass_pending_root_tasks` passes pending root tasks and leaves everything else alone.

    A finish_session-only ask or run then reads N/N, not 0/1.
    """

    class _FakeClient:
        def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
            self._nodes = nodes
            self.passed: list[str] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def cursor(self) -> str | None:
            return None

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

    nodes: dict[str, dict[str, Any]] = {
        "root1": {"parent_id": None, "status": "pending"},
        "root2": {"parent_id": None, "status": "passed"},  # already done -> skip
        "child": {"parent_id": "root1", "status": "pending"},  # not a root -> skip
        "root3": {"parent_id": None, "status": "in_progress"},
        "root4": {"parent_id": None, "status": "failed"},  # failed -> leave honest
    }
    fake = _FakeClient(nodes)
    wf = _wf(curator=fake)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]
    assert set(fake.passed) == {"root1", "root3"}


def test_pass_pending_root_tasks_noop_without_curator() -> None:
    """No curator wired (e.g. ask without a DAG) -> the auto-pass is a no-op."""
    wf = _wf(curator=None)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]  (must not raise)


def test_drive_loop_gateless_settles_after_commit(tmp_path: Path) -> None:
    """A gateless run settles as 'settled' once an edit is committed and the worker goes idle.

    Never 'verify_settled': nothing was verified.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                # an edit -> gateless auto-commit -> seeds gateless_ever_edited
                return _tool_resp("apply_edit", {"path": "x", "edits": []}, tool_id="e1")
            # then spin on read-only commands (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "settled"
    assert result.completed is True
    assert provider.calls < 30  # stopped well before max_iterations, not burned to the cap


def test_resume_snapshot_carries_verify_command(tmp_path: Path) -> None:
    """The snapshot stores the run's resolved verify_command so resume reuses it.

    Re-inferring could diverge from the frozen prompt; a gateless run stores [] and loads back as
    ().
    """
    from agent6.harness._snapshot import load_session_snapshot

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("pytest", "-q"),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(system="s", tool_calls=0, root_task_id=None), [], next_iteration=1
    )
    assert load_session_snapshot(snap).verify_command == ("pytest", "-q")

    config.harness.verify_command = ()  # gateless run -> stored as [] -> loads as ()
    wf = _wf(resume_state_path=snap, config=config)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        _state(system="s", tool_calls=0, root_task_id=None), [], next_iteration=1
    )
    assert load_session_snapshot(snap).verify_command == ()


def test_provider_error_hint_for_auth_and_quota() -> None:
    from agent6.harness.loop import provider_error_hint  # pyright: ignore[reportPrivateUsage]

    assert "agent6 connect" in provider_error_hint(401)
    assert "agent6 connect" in provider_error_hint(403)
    # The failing provider's own config key, when the wrapper stamped it.
    assert "[providers.openrouter].api_key_env" in provider_error_hint(401, "openrouter")
    assert "[providers.<name>].api_key_env" in provider_error_hint(401)
    assert "credits" in provider_error_hint(402).lower()
    # Transient / unknown statuses get no hint (don't mislead).
    assert provider_error_hint(429) == ""
    assert provider_error_hint(500) == ""
    assert provider_error_hint(None) == ""


def test_save_resume_snapshot_degrades_on_unwritable_state_dir(tmp_path: Path) -> None:
    # A read-only state dir disables resume but must not abort the run: mkdir raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    snap = blocker / "loop_state.json"  # parent "blocker" is a file -> mkdir fails
    logs: list[str] = []
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(resume_state_path=snap, config=config, logger=logs.append)
    # Must not raise, twice (the second call must not re-warn).
    for _ in range(2):
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            _state(system="s", tool_calls=0, root_task_id=None), [], next_iteration=1
        )
    warnings = [m for m in logs if "could not persist resume snapshot" in m]
    assert len(warnings) == 1, "warn exactly once, then stay quiet"
    assert not snap.exists()


def test_run_result_docstring_enumerates_every_loop_reason() -> None:
    # The reason glossary drifted to omit five reasons; pinned to the literals the loop constructs.
    import ast
    import inspect

    import agent6.harness._guards as guardsmod
    import agent6.harness.loop as loopmod
    from agent6.harness._snapshot import SessionResult

    reasons: set[str] = set()
    source = inspect.getsource(loopmod) + inspect.getsource(guardsmod)
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ("SessionResult", "End"):
            continue
        literal = [*node.args[:1], *(kw.value for kw in node.keywords if kw.arg == "reason")]
        for value in literal:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                reasons.add(value.value)
    # `reason=finish.kind` is the one non-literal construction; its Literal covers these two.
    reasons |= {"finish_session", "finish_planning"}
    assert reasons >= {
        "loop_guard_killed",
        "verify_settled",
        "verify_command_unexecutable",
        "interactive_stop",
        "finish_planning",
    }  # the five the docstring omitted
    doc = SessionResult.__doc__ or ""
    undocumented = {r for r in reasons if r not in doc}
    assert not undocumented, f"SessionResult docstring omits reasons: {sorted(undocumented)}"


def test_question_nudge_then_accept(tmp_path: Path) -> None:
    """A prose question with no tool call is nudged once toward ask_user, then silently finished."""
    from agent6.harness._nudges import QUESTION_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.turns = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.turns += 1
            # After the nudge, the last user message carries _QUESTION_NUDGE.
            msgs = kwargs.get("messages", [])
            last = msgs[-1] if msgs else {}
            blocks = last.get("content", []) if isinstance(last, dict) else []
            text = " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            if QUESTION_NUDGE in text:
                self.saw_nudge = True
                return _tool_resp("ask_user", {"questions": [{"question": "A?"}]}, tool_id="q1")
            if self.turns == 1:
                return _resp("Which theme should I add?")  # prose question, no tool
            return _resp("Anything else you want?")  # would-be second question

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "ask_user":
                return RawResult({"answers": ["dracula"]})
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nadd a theme"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    # Turn 3 asked again, but the one-shot nudge is spent, so it silently finished.
    assert provider.saw_nudge
    assert result.reason == "silent_finish"


def test_ends_with_question_detection() -> None:
    from agent6.harness._nudges import ends_with_question

    assert ends_with_question("I found two options.\nWhich do you prefer?")
    assert not ends_with_question("Done. All tests pass.")
    assert not ends_with_question("")
    # A ruling the recorder must not miss: decoration after the '?' and the options close.
    assert ends_with_question("Should I proceed? (y/n)")
    assert ends_with_question("**Which one?**")
    assert ends_with_question("Which shape do you want?\n1. the modal\n2) the inline row")
    assert ends_with_question("Keep or drop?\n- keep\n- drop")
    assert not ends_with_question("Which one?\nI went with the modal. Done.")
    assert not ends_with_question("Options considered?\nA long explanation follows here.")
    assert ends_with_question("Should I proceed?  ")  # trailing space tolerated


def test_drive_loop_no_progress_nudges_on_identical_failures(tmp_path: Path) -> None:
    """Identical verify failures draw a root-cause nudge at the 4th and an escalation at the 7th.

    The signature ignores cosmetic drift like line numbers.
    """
    from agent6.harness._nudges import (
        NO_PROGRESS_ESCALATION,
        NO_PROGRESS_NUDGE,
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0
            self.escalations = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if NO_PROGRESS_NUDGE[:28] in last:
                self.nudges += 1
            if NO_PROGRESS_ESCALATION[:28] in last:
                self.escalations += 1
            if self.calls >= 18:
                return _tool_resp("finish_session", {"summary": "stuck"}, tool_id="f")
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.verifies = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                self.verifies += 1
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr=f'File "t.py", line {40 + self.verifies}\nAssertionError: want 3 got 2',
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 1
    assert provider.escalations == 1
    assert result.completed is True


def test_drive_loop_no_progress_silent_when_failures_differ(tmp_path: Path) -> None:
    """Distinct failures mean real progress through the error list; the guard must stay quiet."""
    from agent6.harness._nudges import NO_PROGRESS_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if NO_PROGRESS_NUDGE[:28] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 20:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.verifies = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                self.verifies += 1
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr=f"AssertionError: case {self.verifies} failed",
                    duration_s=0.1,
                    exec_failed=False,
                )
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=30,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0


def test_verify_failure_signature_normalizes_cosmetics() -> None:
    from agent6.harness._nudges import verify_failure_signature

    a = verify_failure_signature("", 'File "t.py", line 41\nAssertionError: want 3 got 2')
    b = verify_failure_signature("", 'File "t.py", line 97\nAssertionError: want 3 got 2')
    c = verify_failure_signature("", 'File "t.py", line 41\nAssertionError: want 5 got 1')
    assert a == b
    assert a != c
    d = verify_failure_signature("ran in 3.21s at 0x7f01ab", "")
    e = verify_failure_signature("ran in 0.07s at 0x9921cd", "")
    assert d == e


def test_drive_loop_no_progress_stops_after_unheeded_interventions(tmp_path: Path) -> None:
    """Ten identical failures with both nudges delivered end the run as no_progress."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr="AssertionError: want 3 got 2",
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=60,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is False
    assert result.reason == "no_progress"
    assert result.iterations < 30  # stopped well before the 60-iteration cap


def test_drive_loop_silent_finish_on_untouched_tree_is_nudged(tmp_path: Path) -> None:
    """A prose-only turn before any edit or green verify is a stall, not an implicit finish.

    Two nudges steer back to the tools; a third prose turn is honoured as silent_finish.
    """
    from agent6.harness._nudges import SILENT_NO_WORK_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if SILENT_NO_WORK_NUDGE[:22] in str(kwargs["messages"][-1]):
                self.nudges += 1
            return _resp("Here is my analysis of the problem.")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=MagicMock(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 2
    assert result.reason == "silent_finish"
    assert result.completed is True


def test_drive_loop_silent_finish_after_real_work_is_honored(tmp_path: Path) -> None:
    """A prose wrap-up after an edit is the normal implicit finish; the no-work gate lets it by."""
    from agent6.harness._nudges import SILENT_NO_WORK_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if SILENT_NO_WORK_NUDGE[:22] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id="e1",
                )
            return _resp("Done: applied the fix.")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return RawResult({"applied": True, "path": "f.py"})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason == "silent_finish"


def test_drive_loop_no_progress_defers_to_metric_runs(tmp_path: Path) -> None:
    """On a metric run repeated identical verify failures are search; no no-progress stop."""
    from agent6.harness._nudges import NO_PROGRESS_NUDGE

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if NO_PROGRESS_NUDGE[:28] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 18:
                return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")
            if self.calls % 2 == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "f.py", "edits": [{"old_string": "a", "new_string": "b"}]},
                    tool_id=f"e{self.calls}",
                )
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=1,
                    stdout="",
                    stderr="AssertionError: want 3 got 2",
                    duration_s=0.1,
                    exec_failed=False,
                )
            if name == "apply_edit":
                return RawResult({"applied": True, "path": "f.py"})
            return RawResult({"ok": True})

    provider = ProviderStub()
    # metric configured -> this is an optimization run
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason != "no_progress"


def test_drive_loop_dedupes_identical_back_to_back_tool_results(tmp_path: Path) -> None:
    """A repeated identical call with an identical result is served a short stub.

    The call still dispatches; a changed result is served in full.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls <= 3:
                return _tool_resp("read_file", {"path": "big.py"}, tool_id=f"r{self.calls}")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="f")

    big = "界" * 4000

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "read_file":
                return RawResult({"content": big, "size": len(big)})
            return RawResult({"ok": True})

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=10,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nread"}]}]
    conversation = Conversation.from_wire(messages)
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    # collect the served tool_result contents for read_file
    served = []
    for m in conversation.to_wire():
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    served.append(b["content"])
    # first read served in full (contains the payload); the 2nd/3rd deduped
    full = [c for c in served if big[:200] in c]
    stubs = [c for c in served if "identical" in c.lower() and big[:200] not in c]
    assert len(full) == 1, f"expected exactly one full payload, got {len(full)}"
    assert len(stubs) >= 1, f"expected the repeats deduped to a stub, got {stubs}"
    assert f"{len(full[0].encode())} bytes elided" in stubs[0]


def test_drive_loop_tool_error_ladder_nudges_then_stops(tmp_path: Path) -> None:
    """A call failing with the same error is nudged, escalated, then stopped as tool_error_stuck."""
    from agent6.harness._nudges import (
        TOOL_ERROR_ESCALATION,
        TOOL_ERROR_NUDGE,
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0
            self.escs = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if TOOL_ERROR_NUDGE[:26] in last:
                self.nudges += 1
            if TOOL_ERROR_ESCALATION[:26] in last:
                self.escs += 1
            # The same tool with a runaway arg each time: same error signature, different args.
            return _tool_resp("read_file", {"path": "x/" * self.calls}, tool_id=f"g{self.calls}")

    from agent6.tools.errors import ToolError as _ToolError

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            raise _ToolError("read_file: the arguments were not valid JSON. Resend the call.")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nsearch"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 1
    assert provider.escs == 1
    assert result.reason == "tool_error_stuck"
    assert result.completed is False
    assert result.iterations < 20


def test_drive_loop_denial_streak_gets_policy_nudge_not_malformed(tmp_path: Path) -> None:
    """A streak of policy refusals is nudged as refused, never as malformed.

    A stale binary a real exec failure recorded first is not resurfaced by what is pure policy.
    """
    from agent6.harness._nudges import (
        TOOL_DENIED_NUDGE,
        TOOL_ERROR_NUDGE,
    )
    from agent6.tools.errors import ToolDeniedError as _ToolDenied

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.denial_nudges = 0
            self.malformed_nudges = 0
            self.reach_notes = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            last = str(kwargs["messages"][-1])
            if TOOL_DENIED_NUDGE[:30] in last:
                self.denial_nudges += 1
            if TOOL_ERROR_NUDGE[:26] in last:
                self.malformed_nudges += 1
            if "installed on this machine" in last:
                self.reach_notes += 1
            return _tool_resp(
                "run_command",
                {"argv": ["git", "status", f"-{self.calls}"]},
                tool_id=f"c{self.calls}",
            )

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.calls += 1
            if self.calls == 1:
                # A real exec failure records argv[0]="git"; a ToolError never entered the jail.
                return ExecResult(
                    returncode=127,
                    stdout="",
                    stderr="git: command not found or not executable",
                    duration_s=0.0,
                    exec_failed=True,
                )
            raise _ToolDenied("run_command not approved (sandbox.run_commands='ask')")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nship"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.denial_nudges >= 1  # the policy wording reached the model
    assert provider.malformed_nudges == 0  # never told its call shape is wrong
    assert provider.reach_notes == 0  # no stale-binary jail misdiagnosis
    assert result.reason == "tool_error_stuck"


def test_drive_loop_tool_error_streak_resets_on_success(tmp_path: Path) -> None:
    """A successful tool call between errors clears the streak; intermittent errors never trip."""
    from agent6.harness._nudges import TOOL_ERROR_NUDGE
    from agent6.tools.errors import ToolError as _ToolError

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudges = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if TOOL_ERROR_NUDGE[:26] in str(kwargs["messages"][-1]):
                self.nudges += 1
            if self.calls >= 12:
                return _tool_resp("finish_session", {"summary": "ok"}, tool_id="f")
            return _tool_resp("read_file", {"path": "p"}, tool_id=f"g{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.n = 0

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            self.n += 1
            if self.n % 2 == 0:  # alternate error / success
                return RawResult({"content": "ok"})
            raise _ToolError("read_file: bad path")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.nudges == 0
    assert result.reason != "tool_error_stuck"


def test_note_verify_result_flags_a_dead_verify(tmp_path: Path) -> None:
    """A verify that failed instantly for a missing runner draws the verify-broken nudge once.

    A legitimate slow test failure is not flagged.
    """
    from agent6.harness._nudges import VERIFY_BROKEN_NUDGE

    wf = _wf(root=tmp_path, config=MagicMock(), provider=MagicMock(), dispatcher=MagicMock())
    st = _state()
    turn = _turn(iteration=1)
    # dead verify: instant, "No module named pytest"
    wf.gate.note_result(
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.02,
            exec_failed=False,
        ),
    )
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert any(VERIFY_BROKEN_NUDGE[:24] in t for t in texts)
    assert st.verify.broken_warned is True

    # a second dead verify does not re-warn
    turn2 = _turn(iteration=2)
    wf.gate.note_result(
        st,
        turn2,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.02,
            exec_failed=False,
        ),
    )
    assert not any(
        isinstance(it, Notice) and "verify-broken" in it.text for it in turn2.tool_results
    )


def test_note_verify_result_does_not_flag_real_failure(tmp_path: Path) -> None:
    from agent6.harness._nudges import VERIFY_BROKEN_NUDGE

    wf = _wf(root=tmp_path, config=MagicMock(), provider=MagicMock(), dispatcher=MagicMock())
    st = _state()
    turn = _turn(iteration=1)
    # a real test failure: took real time, ordinary assertion output
    wf.gate.note_result(
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="5 failed, 200 passed",
            stderr="AssertionError: x != y",
            duration_s=12.4,
            exec_failed=False,
        ),
    )
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert not any(VERIFY_BROKEN_NUDGE[:24] in t for t in texts)
    assert st.verify.broken_warned is False


def test_tool_error_spiral_stops_without_blaming_the_sandbox(tmp_path: Path) -> None:
    """A run_command ToolError spiral climbs the ladder and stops, with no reachability note.

    A ToolError never entered the jail, so it says nothing about reachability.
    """

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.reach_hits = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "sandbox-\n" in str(kwargs["messages"][-1]) or "reachability" in str(
                kwargs["messages"][-1]
            ):
                self.reach_hits += 1
            return _tool_resp(
                "run_command", {"argv": ["python3", "-c", "x"]}, tool_id=f"c{self.calls}"
            )

    from agent6.tools.errors import ToolError as _ToolError

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            raise _ToolError("python3: boom in the sandbox")

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=20,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert provider.reach_hits == 0  # no sandbox blame for a non-jail failure
    assert result.reason == "tool_error_stuck"


def test_drive_loop_gateless_settle_never_claims_verify_passed(tmp_path: Path) -> None:
    """A gateless run that commits and goes idle settles with all_passed=False, said plainly."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                    tool_id="e1",
                )
            # Then the worker goes idle: read-only calls, no edits, no finish.
            return _tool_resp("read_file", {"path": "a.py"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),  # GATELESS
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "settled"
    assert "no verify" in result.summary
    assert "verify passed" not in result.summary
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "settled" and ends[-1]["all_passed"] is False


def test_drive_loop_interactive_stop_never_ends_passed(tmp_path: Path) -> None:
    """The REPL hook's "stop" ends the run `interactive_stop` with all_passed=False."""

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            return _tool_resp(
                "apply_edit",
                {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                tool_id="e1",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    def _stop_hook(_i: int, _sha: str) -> Literal["continue", "stop"]:
        return "stop"

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        events=_Events(),
        bridge=OperatorBridge(after_auto_commit=_stop_hook),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert result.reason == "interactive_stop"
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "interactive_stop"
    assert ends[-1]["all_passed"] is False  # a stop is deliberate, never "passed"


def test_drive_loop_interactive_exit_ends_steer_exit(tmp_path: Path) -> None:
    """`/exit` from the REPL ends the run `steer_exit`: stopped, all_passed=False, no "next:"."""

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            return _tool_resp(
                "apply_edit",
                {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                tool_id="e1",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    def _exit_hook(_i: int, _sha: str) -> Literal["continue", "exit"]:
        return "exit"

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        events=_Events(),
        bridge=OperatorBridge(after_auto_commit=_exit_hook),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is False  # an operator exit is a stop, not a completion
    assert result.reason == "steer_exit"
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "steer_exit"
    assert ends[-1]["all_passed"] is False  # a stop is deliberate, never "passed"


def test_drive_loop_repl_undo_takes_the_steer_undo_path(tmp_path: Path) -> None:
    """The REPL hook's "undo" is the loop's own /undo: the run ends `undone` naming the fork."""

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            return _tool_resp(
                "apply_edit",
                {"path": "a.py", "edits": [{"kind": "create", "new_string": "x = 1\n"}]},
                tool_id="e1",
            )

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    def _undo_hook(_i: int, _sha: str) -> Literal["continue", "stop", "undo"]:
        return "undo"

    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        events=_Events(),
        bridge=OperatorBridge(
            after_auto_commit=_undo_hook, undo_forker=lambda: ("forked-child-ID", "t")
        ),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "undone"
    assert "forked-child-ID" in result.summary
    undone = [e for e in events if e["type"] == "session.undone"]
    assert undone and undone[-1]["new_session_id"] == "forked-child-ID"
    # An undo journals a session.end (undone -> "stopped"); the run read "stale" before.
    ends = [e for e in events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "undone" and ends[-1]["all_passed"] is False


def test_drive_loop_gateless_run_adopts_verify_when_the_repo_materializes(
    tmp_path: Path,
) -> None:
    """A verify inferred at a later auto-commit is adopted by config, dispatcher and model.

    Preflight on an empty repo finds nothing; the run then creates a recognizable project.
    """
    from agent6.config import Config

    # What the run "just created" before its first commit.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.adoption_notices = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "verify command was adopted" in str(kwargs["messages"][-1]):
                self.adoption_notices += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "pyproject.toml", "edits": [{"kind": "create", "new_string": "x"}]},
                    tool_id="e1",
                )
            return _tool_resp("read_file", {"path": "pyproject.toml"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.adopted: tuple[str, ...] | None = None
            self.gate_runs = 0

        def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
            self.adopted = argv
            return True

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

        def run_verify(self, extra_argv: tuple[str, ...] = ()) -> ExecResult:
            self.gate_runs += 1
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    wf = _wf(
        root=tmp_path,
        config=Config(),  # real config: verify_command defaults empty (gateless)
        mode="run",
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert dispatcher.adopted is not None  # the dispatcher gates run_verify now
    assert provider.adoption_notices >= 1  # the gate flip was said to the model
    assert result.completed is True
    # The settled end runs the adopted verify the worker never ran, so the run ends verified.
    assert dispatcher.gate_runs == 1
    assert result.reason == "verify_settled"
    assert result.verified == "passed"


def test_drive_loop_gateless_adoption_declines_an_unexecutable_verify(
    tmp_path: Path,
) -> None:
    """An inferred command the dispatcher refuses leaves the run gateless.

    Adopting a gate the sandbox cannot execute would turn the honest settle into an abort.
    """
    from agent6.config import Config

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.adoption_notices = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "verify command was adopted" in str(kwargs["messages"][-1]):
                self.adoption_notices += 1
            if self.calls == 1:
                return _tool_resp(
                    "apply_edit",
                    {"path": "pyproject.toml", "edits": [{"kind": "create", "new_string": "x"}]},
                    tool_id="e1",
                )
            return _tool_resp("read_file", {"path": "pyproject.toml"}, tool_id=f"r{self.calls}")

    class DispatcherStub(_StubDispatcher):
        def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
            return False  # the jail cannot execute the inferred runner

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        config=Config(),
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=40,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nbuild"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert tuple(wf.config.harness.verify_command) == ()  # still gateless
    assert provider.adoption_notices == 0  # no false gate-flip message
    assert result.reason == "settled"
    assert "no verify command existed" in result.summary


def _run_command_provider(calls_before_idle: int) -> Any:
    """Return a provider that runs commands then goes read-only, counting reachability notes."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.reachability_notes = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if "the sandbox cannot execute it" in str(kwargs["messages"][-1]):
                self.reachability_notes += 1
            if self.calls <= calls_before_idle:
                return _tool_resp(
                    "run_command", {"argv": ["sh", "-c", "true"]}, tool_id=f"c{self.calls}"
                )
            return _tool_resp("read_file", {"path": "a"}, tool_id=f"r{self.calls}")

    return ProviderStub()


def test_reachability_note_fires_on_repeated_jail_exec_failure(tmp_path: Path) -> None:
    """Two exec failures of a host-present binary emit the unreachable event once."""

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_command":
                return ExecResult(
                    returncode=127,
                    stdout="",
                    stderr="sh: command not found or not executable",
                    duration_s=0.0,
                    exec_failed=True,
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.0, exec_failed=False
            )

    provider = _run_command_provider(calls_before_idle=4)
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    unreachable = [e for e in events if e["type"] == "loop.sandbox_tool_unreachable"]
    assert [e["binary"] for e in unreachable] == ["sh"]  # once, at the 2nd failure
    assert provider.reachability_notes >= 1  # the model was told


def test_reachability_note_never_fires_on_a_validation_error(tmp_path: Path) -> None:
    """A run_command rejected at input validation never entered the jail and seeds no diagnosis."""
    from agent6.tools.errors import ToolError as _ToolError

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_command":
                raise _ToolError("1 validation error for RunCommandInput: env extra_forbidden")
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.0, exec_failed=False
            )

    provider = _run_command_provider(calls_before_idle=4)
    events: list[dict[str, Any]] = []

    class _Events:
        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=(),
            verify_infer=True,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=8,
        events=_Events(),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert not any(e["type"] == "loop.sandbox_tool_unreachable" for e in events)
    assert provider.reachability_notes == 0


def test_load_repo_summary_tolerates_a_broken_agents_md(tmp_path: Path) -> None:
    """A non-UTF-8 or unreadable AGENTS.md degrades instead of raising after session.start."""
    from agent6.harness._context import load_repo_summary

    (tmp_path / "AGENTS.md").write_bytes(b"Style: use \x93smart quotes\x94\n")
    summary = load_repo_summary(tmp_path)
    assert "smart quotes" in summary.agents_md  # lossy, present, no crash


def test_refused_finish_tool_is_not_captured_as_a_finish() -> None:
    """A finish tool the dispatcher refused is not captured as a finish signal."""
    from agent6.harness._conversation import ToolUse
    from agent6.harness.loop import TurnState
    from agent6.tools.dispatch import ToolError

    dispatcher = MagicMock()
    dispatcher.dispatch.side_effect = ToolError("finish_planning is not available in run mode")
    wf = _wf(mode="run", dispatcher=dispatcher)
    turn = TurnState(
        iteration=1,
        resp=_resp(""),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(
                ToolUse(
                    id="tu1",
                    name="finish_planning",
                    input={"summary": "all done", "plan_markdown": "# x"},
                ),
            ),
        ),
    )
    out = wf._turn_dispatch_tools(_state(), turn, turn_context())  # pyright: ignore[reportPrivateUsage]
    assert out is None  # the refusal is served as an error result, not an abort
    assert turn.finish is None  # and never captured as a finish


def test_finish_dispatch_is_not_work_for_the_standing_streak() -> None:
    """A dispatched finish_session does not advance ok_tool_calls, so standing_patience engages."""
    from agent6.harness._conversation import ToolUse
    from agent6.harness.loop import TurnState
    from agent6.tools.results import FinishSessionResult

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = FinishSessionResult(summary_text="done", result=None)
    wf = _wf(mode="run", dispatcher=dispatcher)
    state = _state()
    turn = TurnState(
        iteration=1,
        resp=_resp(""),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="finish_session", input={"summary": "done"}),),
        ),
    )
    wf._turn_dispatch_tools(state, turn, turn_context())  # pyright: ignore[reportPrivateUsage]
    assert state.ok_tool_calls == 0  # a control verb is not work

    worked = TurnState(
        iteration=2,
        resp=_resp(""),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu2", name="read_file", input={"path": "x"}),),
        ),
    )
    wf._turn_dispatch_tools(state, worked, turn_context())  # pyright: ignore[reportPrivateUsage]
    assert state.ok_tool_calls == 1


def test_stop_request_honored_after_a_prose_turn(tmp_path: Path) -> None:
    """A stop after this step is honoured at the end of every completed iteration, prose too."""
    calls = {"n": 0}

    class ProviderStub:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            calls["n"] += 1
            return ProviderResponse(
                text="still thinking about the approach",
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={"content": [{"type": "text", "text": "still thinking about the approach"}]},
            )

    cleared = {"n": 0}
    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=MagicMock(),
        max_iterations=5,
        bridge=OperatorBridge(
            stop_requested=lambda: True,
            stop_clear=lambda: cleared.__setitem__("n", cleared["n"] + 1),
        ),
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        conversation=Conversation.from_wire(messages),
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
        original_task="t",
    )
    assert result.reason == "steer_abort"
    assert result.completed is False
    assert calls["n"] == 1  # stopped at the FIRST boundary; no further provider calls
    assert cleared["n"] == 1  # the marker was consumed, not left pending


def test_metric_plateau_over_a_stale_verify_is_not_passed() -> None:
    """The plateau stop grounds all_passed on the final tree, like its sibling clean ends."""
    from agent6.harness._conversation import ToolUse
    from agent6.harness.loop import TurnState

    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev, root=Path("/tmp"))
    turn = TurnState(
        iteration=7,
        resp=_resp(""),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="apply_edit", input={}),),
        ),
    )
    turn.stops.append(_plateau_stop())
    state = _state(
        ever_edited=True,
        # The green verify predates the last edit.
        verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=True),
    )
    with patch.object(RunChain, "dirty", return_value=False):
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.reason == "metric_plateau"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is False


def test_metric_plateau_over_a_green_tree_stays_passed() -> None:
    """The mirror: a verified-green tree at the plateau still ends passed."""
    from agent6.harness._conversation import ToolUse
    from agent6.harness.loop import TurnState

    ev = _EventCapture()
    wf = _wf(mode="run", config=_cfg_with_verify(), events=ev, root=Path("/tmp"))
    turn = TurnState(
        iteration=7,
        resp=_resp(""),
        assistant=AssistantTurn(
            raw_content=(),
            tool_uses=(ToolUse(id="tu1", name="run_verify_command", input={}),),
        ),
    )
    turn.stops.append(_plateau_stop())
    state = _state(
        ever_edited=True, verify=VerifyVerdict(ever_passed=True, last_ok=True, edited_since=False)
    )
    with patch.object(RunChain, "dirty", return_value=False):
        result = wf._turn_stop_checks(state, turn, Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.reason == "metric_plateau"
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["all_passed"] is True


def test_a_red_verify_finish_still_passes_its_root_tasks() -> None:
    """Every deliberate end passes its root tasks the same way, whatever the verify truth."""

    class _FakeClient:
        def __init__(self) -> None:
            self._nodes: dict[str, dict[str, Any]] = {
                "root1": {"parent_id": None, "status": "pending"}
            }
            self.passed: list[str] = []

        def nodes(self) -> dict[str, Any]:
            return _typed(self._nodes)

        def cursor(self) -> str | None:
            return None

        def update_status(self, intent: Any) -> None:
            self.passed.append(intent.id)
            self._nodes[intent.id]["status"] = intent.new_status

    fake = _FakeClient()
    wf = _wf(
        curator=fake,
        config=MagicMock(
            git=_GIT_STUB,
            budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=MagicMock(system_prompt_file=""),
            harness=MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=("false",),
                verify_when="never",
                verify_retries=2,
            ),
        ),
    )
    state = _state()
    state.verify.last_ok = False  # red tree: all_passed must stay False
    state.verify.edited_since = True
    events: list[dict[str, Any]] = []
    wf._emit = lambda event_type, **fields: events.append({"type": event_type, **fields})  # pyright: ignore[reportPrivateUsage]

    wf._finish(  # pyright: ignore[reportPrivateUsage]
        state,
        End("finish_session", "", completed=True, verdict="grounded", checkpoint=False),
        iteration=3,
    )

    (end,) = [e for e in events if e["type"] == "session.end"]
    assert end["all_passed"] is False  # the verify truth is unchanged...
    assert fake.passed == ["root1"]  # ...and the work item is no longer pending


def test_an_operator_stop_names_the_worktree_it_leaves_dirty(tmp_path: Path) -> None:
    """An operator stop names the uncommitted work it leaves; a clean tree adds nothing."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    wf = _wf(root=tmp_path, mode="run")
    assert wf._dirty_tree_note() == ""  # pyright: ignore[reportPrivateUsage]

    (tmp_path / "a.txt").write_text("edited\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("untracked\n", encoding="utf-8")
    note = wf._dirty_tree_note()  # pyright: ignore[reportPrivateUsage]
    assert "worktree left dirty" in note
    assert "2 file" in note  # the real count, not a capped one


def test_parallel_group_counter_reaches_disk_before_the_group_runs(tmp_path: Path) -> None:
    """The parallel group counter is bumped before the snapshot; a crash never reuses p1."""
    import json

    from agent6.directive import Segment
    from agent6.harness.subrun import LaneResult, LaneSpec

    snap = tmp_path / "loop_state.json"
    at_spawn: dict[str, Any] = {}

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        at_spawn["group"] = group
        at_spawn["persisted"] = json.loads(snap.read_text(encoding="utf-8"))[
            "parallel_groups_dispatched"
        ]
        return [
            LaneResult(
                spec=LaneSpec(lane=i, session_id=f"run-{group}-l{i}", workdir=tmp_path, route=None),
                session_dir=tmp_path,
                branch="b",
                ok=False,
                error="lane failed",
            )
            for i in range(1, len(lanes) + 1)
        ]

    wf = _wf(
        root=tmp_path,
        mode="run",
        bridge=OperatorBridge(lane_spawner=spawner),
        resume_state_path=snap,
    )
    conversation = Conversation.from_wire(
        [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    )
    wf.parallel.dispatch(conversation, 3, _state(), [Segment(spec="", task="do the thing")])
    assert at_spawn["group"] == "p1"
    assert at_spawn["persisted"] == 1, "the bump must be on disk before the group blocks"


def test_steer_abort_names_the_dirty_worktree_like_its_siblings(tmp_path: Path) -> None:
    """A boundary Stop names the uncommitted worktree, as a mid-stream Stop does."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    (tmp_path / "a.txt").write_text("edited after the last checkpoint\n", encoding="utf-8")

    wf = _wf(root=tmp_path, mode="run")
    res = wf._steer_outcome("abort", 4, _state())  # pyright: ignore[reportPrivateUsage]
    assert res is not None
    assert res.reason == "steer_abort"
    assert "worktree left dirty" in res.summary


def test_a_second_restart_carries_the_first_summary_forward() -> None:
    """The prior restart's summary reaches the summariser out-of-band, surviving the tail clip."""
    from agent6.prompts.revision import context_restart_notice

    summariser = MagicMock()
    summariser.call.return_value = _resp("second summary")
    wf = _wf(
        compaction=CompactionSettings(
            summariser=summariser, drop_at_chars=10**9, summarise_at_chars=10**9
        ),
        bridge=OperatorBridge(compact_requested=lambda: ""),
    )
    # One restart already, then enough new work for the tail clip to prefer over the notice.
    restart = context_restart_notice("run") + "SUMMARY-1: found the parser bug in a.md"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]},
        {"role": "user", "content": [{"type": "text", "text": restart}]},
    ]
    # Enough work to overflow the summariser's 60k tail clip, which drops the notice at the head.
    history += _long_history(120)[1:]

    assert _compact_via_wire(wf, history) is True
    sent = str(summariser.call.call_args)
    assert "SUMMARY-1" in sent, "the first restart's summary never reached the summariser"


def test_the_frontier_executes_the_order_the_children_list_shows() -> None:
    """`list_tasks` and every task tree render a parent's children in its `children` order."""
    from agent6.harness._dag_focus import (
        first_ready_subtask,  # pyright: ignore[reportPrivateUsage]
    )

    nodes = _typed(
        {
            "root": {"parent_id": None, "status": "in_progress", "children": ("b", "a")},
            "a": {"parent_id": "root", "status": "pending"},
            "b": {"parent_id": "root", "status": "pending"},
        }
    )
    assert first_ready_subtask(nodes) == "b", "the frontier ignored the children order"


def test_the_frontier_walks_depth_first_through_children() -> None:
    """A decomposed child's own leaves come before its later siblings, as the tree shows them."""
    from agent6.harness._dag_focus import (
        first_ready_subtask,  # pyright: ignore[reportPrivateUsage]
    )

    nodes = _typed(
        {
            "root": {"parent_id": None, "status": "in_progress", "children": ("p", "z")},
            "p": {"parent_id": "root", "status": "in_progress", "children": ("p2", "p1")},
            "p1": {"parent_id": "p", "status": "pending"},
            "p2": {"parent_id": "p", "status": "pending"},
            "z": {"parent_id": "root", "status": "pending"},
        }
    )
    assert first_ready_subtask(nodes) == "p2"


def test_steer_undo_signal() -> None:
    """`/undo` typed as a steer -> the "undo" sentinel, no injected message."""
    cleared: list[bool] = []
    wf = _wf(
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_clear=lambda: cleared.append(True),
            steer_prompt=lambda: "/undo",
        ),
    )
    messages: list[dict[str, Any]] = []
    result = _steer_via_wire(wf, messages, iteration=4, state=_state())
    assert result == "undo"
    assert cleared == [True]
    assert messages == [], "/undo must not inject a message"


def _standing_nodes() -> Any:
    """Root -> one standing child, ready (the queue is empty)."""
    return _typed(
        {
            "a": {"children": ("b",)},
            "b": {"parent_id": "a", "standing": True},
        }
    )


def test_standing_default_never_self_quits_and_escalates() -> None:
    """At standing_patience -1 a fruitless quiet round never ends the run by itself.

    Fruitless re-entries carry the escalating dig-deeper nudge; a refused tool call is not work, an
    executed one resets the streak.
    """
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    first = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=3), _ctx(wf, _state()))  # pyright: ignore[reportPrivateUsage]
    assert first is None
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]
    # Quiet again with no executed call: still absorbed, nudge escalates.
    state.tool_calls += 1  # a REFUSED call is not work
    second = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=4), _ctx(wf, _state()))  # pyright: ignore[reportPrivateUsage]
    assert second is None
    text = conv.to_wire()[-1]["content"][0]["text"]
    assert "different approach" in text and "fruitless round 1" in text
    third = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=5), _ctx(wf, _state()))  # pyright: ignore[reportPrivateUsage]
    assert third is None
    assert "fruitless round 2" in conv.to_wire()[-1]["content"][0]["text"]
    # Work landing resets the streak: the next absorb is the plain nudge.
    state.ok_tool_calls += 1
    fourth = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=6), _ctx(wf, _state()))  # pyright: ignore[reportPrivateUsage]
    assert fourth is None
    assert "fruitless" not in conv.to_wire()[-1]["content"][0]["text"]
    assert state.standing.fruitless == 0


def test_standing_patience_bounds_fruitless_reentries() -> None:
    """standing_patience = N absorbs N fruitless rounds then honours the end; 0 gives up first."""
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    wf.config.harness.standing_patience = 1
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    ctx = _ctx(wf, state)
    assert wf._handle_silent_finish("Done.", conv, state, _turn(iteration=3), ctx) is None  # pyright: ignore[reportPrivateUsage]
    assert wf._handle_silent_finish("Done.", conv, state, _turn(iteration=4), ctx) is None  # pyright: ignore[reportPrivateUsage]
    ended = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=5), ctx)  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "silent_finish"

    wf0 = _wf(mode="run", curator=curator, budget=None)
    wf0.config.harness.standing_patience = 0
    state0 = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    ctx0 = _ctx(wf0, state0)
    quiet = _turn(iteration=3)
    assert wf0._handle_silent_finish("Done.", Conversation(), state0, quiet, ctx0) is None  # pyright: ignore[reportPrivateUsage]
    quiet = _turn(iteration=4)
    ended0 = wf0._handle_silent_finish("Done.", Conversation(), state0, quiet, ctx0)  # pyright: ignore[reportPrivateUsage]
    assert ended0 is not None and ended0.reason == "silent_finish"


def test_standing_task_gates_finish_session_and_soft_stops() -> None:
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf = _wf(mode="run", curator=curator, budget=None)
    state = _state()
    turn = _turn(iteration=2)
    turn.finish = FinishCall("finish_session", "all done")
    wf._turn_finish_gates(state, turn, _ctx(wf, state, 2))  # pyright: ignore[reportPrivateUsage]
    assert turn.finish is None  # revoked: the goal continues
    assert any("standing task" in getattr(n, "text", "") for n in turn.tool_results)
    # Soft stop: verify_settled absorbs and clears its streak.
    state.ok_tool_calls += 1
    turn2 = _turn(iteration=3)
    ctx = _ctx(wf, state, 3)
    turn2.stops.append(Stop(lambda: settled_end(state, ctx), soft="verify_settled"))
    state.settled.idle = 9
    conv = Conversation()
    wf.standing.absorb_soft_stop(state, turn2, conv)  # pyright: ignore[reportPrivateUsage]
    assert turn2.stops == []
    assert state.settled.idle == 0
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]


def test_standing_absorb_refuses_without_a_ready_standing_task() -> None:
    curator = MagicMock()
    curator.nodes.return_value = _typed({"a": {}})  # no standing node
    wf = _wf(mode="run", curator=curator, budget=None)
    assert wf.standing.absorb(_state(), reason="silent_finish", iteration=1) is None  # pyright: ignore[reportPrivateUsage]


def test_standing_goal_seeds_a_standing_child_under_the_root() -> None:
    """`run --standing` reaches the graph as one standing child under the root, as steering."""
    curator = MagicMock()
    root = _tn("a")
    curator.add_subtask.side_effect = [root, _tn("b", parent_id="a", standing=True)]
    curator.nodes.return_value = _typed({"a": {}})
    provider = MagicMock()
    provider.call.return_value = _resp("done")
    wf = _wf(
        mode="run",
        curator=curator,
        standing_goal="keep hunting bugs",
        budget=None,
        provider=provider,
    )
    wf.run("t")
    drafts = [c.args[0].draft for c in curator.add_subtask.call_args_list]
    assert len(drafts) == 2  # the root, then the standing goal
    assert drafts[1].standing is True
    assert drafts[1].title == "keep hunting bugs"
    assert drafts[1].created_by == "steering"


def test_session_start_carries_the_operators_words_under_a_seed() -> None:
    """`run --from` composes a `<prior-run>` digest ahead of the operator's task.

    The session.start event carries the operator's words, not the digest's opening tag.
    """
    events: list[dict[str, Any]] = []

    class _Events:
        path = Path("/tmp/x/logs.jsonl")

        def emit(self, event_type: str, /, **fields: Any) -> None:
            events.append({"type": event_type, **fields})

    provider = MagicMock()
    provider.call.return_value = _resp("done")
    wf = _wf(mode="run", budget=None, provider=provider, events=_Events())
    composed = (
        '<prior-run id="agile-echo-H2EWX5">\nThis question is about a PRIOR agent6 run.\n'
        "## Run task\nhow many functions?\n</prior-run>\n\n"
        "add a module docstring to calc.py"
    )
    wf.run(composed)
    start = next(e for e in events if e["type"] == "session.start")
    assert start["user_task"] == "add a module docstring to calc.py"


def test_interactive_quiet_turn_parks_and_a_steer_continues_the_conversation() -> None:
    """Interactively, going quiet is a turn boundary: the run parks and a steer continues it.

    An "abort" steer ends it as steer_abort.
    """
    steers = iter(["keep going: also cover sub()"])
    wf = _wf(
        mode="run",
        interactive=True,
        bridge=OperatorBridge(steer_requested=lambda: True, steer_prompt=lambda: next(steers)),
    )
    conv = Conversation()
    state = _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True))
    parked = wf._handle_silent_finish("Done.", conv, state, _turn(iteration=4), _ctx(wf, _state()))  # pyright: ignore[reportPrivateUsage]
    assert parked is None  # steered onward, same conversation
    wire = conv.to_wire()
    assert "keep going: also cover sub()" in wire[-1]["content"][0]["text"]

    aborts = iter(["abort"])
    wf2 = _wf(
        mode="run",
        interactive=True,
        bridge=OperatorBridge(steer_requested=lambda: True, steer_prompt=lambda: next(aborts)),
    )
    ended = wf2._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        _turn(iteration=4),
        _ctx(wf2, _state()),
    )  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "steer_abort"


def test_non_interactive_quiet_turn_still_ends_and_standing_outranks_the_park() -> None:
    # Non-interactive: unchanged silent_finish end.
    wf = _wf(mode="run", interactive=False)
    ended = wf._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        Conversation(),
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        _turn(iteration=4),
        _ctx(wf, _state()),
    )  # pyright: ignore[reportPrivateUsage]
    assert ended is not None and ended.reason == "silent_finish"
    # A standing goal outranks the park: the absorb nudge continues the run.
    curator = MagicMock()
    curator.nodes.return_value = _standing_nodes()
    wf2 = _wf(mode="run", interactive=True, curator=curator, budget=None)
    conv = Conversation()
    out = wf2._handle_silent_finish(  # pyright: ignore[reportPrivateUsage]
        "Done.",
        conv,
        _state(ever_edited=True, verify=VerifyVerdict(ever_passed=True)),
        _turn(iteration=4),
        _ctx(wf2, _state()),
    )  # pyright: ignore[reportPrivateUsage]
    assert out is None
    assert "standing task" in conv.to_wire()[-1]["content"][0]["text"]


def test_tier2_growth_floor_prevents_zero_growth_refire(tmp_path: Path) -> None:
    """A restart that lands above the threshold does not re-summarise every iteration.

    Tier-2 re-fires only after the context grew 25% past the last restart's size.
    """

    class SummariserStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _resp("PROGRESS SUMMARY: " + "s" * 4_000)  # bigger than the threshold

    summ = SummariserStub()
    wf = _wf(
        root=tmp_path,
        compaction=CompactionSettings(
            summariser=summ, drop_at_chars=2_000, summarise_at_chars=3_000
        ),
    )
    state = _state()
    messages = _big_text_history("TASK: t", blocks=4, block_chars=1_000)
    assert _compact_via_wire(wf, messages, state=state) is True
    assert summ.calls == 1
    # The context exceeds the threshold, but with zero growth the next pass must not re-summarise.
    assert _compact_via_wire(wf, messages, state=state) is False
    assert summ.calls == 1
    # Real growth past the floor re-arms tier-2.
    for _ in range(6):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "y" * 1_000}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})
    assert _compact_via_wire(wf, messages, state=state) is True
    assert summ.calls == 2


def test_auto_commit_with_nothing_changed_emits_no_event(tmp_path: Path) -> None:
    """A green verify with no new edits emits no commit event: chain_commit returns ""."""
    events: list[dict[str, Any]] = []
    wf = _wf(root=tmp_path, mode="run", per_step=True)

    def _capture(_type: str, **f: Any) -> None:
        events.append({"type": _type, **f})

    wf.events = MagicMock()
    wf.events.emit = _capture  # type: ignore[method-assign]
    turn = _turn(iteration=3)
    turn.verify_just_passed = True
    turn.edit_since_verify_pass = False
    with patch.object(RunChain, "commit", return_value=""):
        wf._turn_auto_commit_and_metric(_state(), turn)  # pyright: ignore[reportPrivateUsage]
    assert [e for e in events if e["type"] == "loop.auto_commit"] == []
    assert turn.committed is False


def test_auto_commit_failure_surface_tells_the_truth(tmp_path: Path) -> None:
    """The failure reporter is silent on the benign nothing-changed case and reports a GitError."""
    from agent6.git_ops import GitError

    events: list[dict[str, Any]] = []
    wf = _wf(root=tmp_path, mode="run", per_step=True)

    def _capture(_type: str, **f: Any) -> None:
        events.append({"type": _type, **f})

    wf.events = MagicMock()
    wf.events.emit = _capture  # type: ignore[method-assign]

    for benign in ("nothing to commit, working tree clean", "no changes added to commit"):
        wf.checkpoints.report_failure(GitError(benign), "s", iteration=1)
    assert events == []  # a non-failure never claims to be one

    wf.checkpoints.report_failure(
        GitError("fatal: unable to write new index file"), "agent6 iter 2: fix", iteration=2
    )
    (evt,) = [e for e in events if e["type"] == "loop.auto_commit.failed"]
    assert evt["iteration"] == 2
    assert "unable to write" in evt["error"]
    assert evt["commit_subject"] == "agent6 iter 2: fix"


def test_turn_marker_covers_dispatch_and_clears_after_the_snapshot(tmp_path: Path) -> None:
    """The mid-turn-crash marker is on disk while tools dispatch and gone after the snapshot."""
    from agent6.harness._snapshot import TURN_IN_FLIGHT_NAME, read_turn_marker

    marker = tmp_path / TURN_IN_FLIGHT_NAME
    seen: list[tuple[int, tuple[str, ...]] | None] = []

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub(_StubDispatcher):
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            seen.append(read_turn_marker(marker))
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            return RawResult({"acknowledged": True, "summary": raw_input.get("summary", "")})

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=None,
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=3,
        resume_state_path=tmp_path / "loop_state.json",
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nt"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.completed is True
    assert seen[0] == (1, ("run_verify_command",))  # live during dispatch
    assert seen[-1] is not None and seen[-1][0] == 2  # second turn's own marker
    assert not marker.exists()  # a clean end leaves nothing


def test_the_old_crash_marker_survives_the_replayed_provider_call(tmp_path: Path) -> None:
    """The replayed turn's own marker write or snapshot supersedes the old marker, nothing else."""
    from agent6.harness._snapshot import (
        TURN_IN_FLIGHT_NAME,
        SessionSnapshot,
        read_turn_marker,
        write_turn_marker,
    )

    class ReplayStarted(Exception):  # noqa: N818  # a signal, not an error  # a signal, not an error
        pass

    marker = tmp_path / TURN_IN_FLIGHT_NAME
    snapshot_path = tmp_path / "loop_state.json"
    snapshot_path.write_text(
        SessionSnapshot(
            system="system",
            messages=[],
            tool_calls=0,
            next_iteration=1,
            root_task_id=None,
            original_task="t",
            verify_command=(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    write_turn_marker(marker, 1, ("run_command",))
    seen: list[tuple[int, tuple[str, ...]] | None] = []

    class ProviderStub:
        def call(self, **_kwargs: Any) -> ProviderResponse:
            seen.append(read_turn_marker(marker))
            raise ReplayStarted

    wf = _wf(root=tmp_path, provider=ProviderStub(), resume_state_path=snapshot_path)
    with pytest.raises(ReplayStarted):
        wf.resume()
    assert seen == [(1, ("run_command",))]


def test_turn_replay_allowed_marker_semantics(tmp_path: Path) -> None:
    """No marker proceeds; a stale one clears silently; a matching one asks and stays put.

    Cleared on approval, a later preflight refusal replayed the turn on the next attempt and its
    tools' side effects happened twice.
    """
    from agent6.app.resume import turn_replay_allowed
    from agent6.harness._snapshot import TURN_IN_FLIGHT_NAME, write_turn_marker

    marker = tmp_path / TURN_IN_FLIGHT_NAME
    asked: list[tuple[int, tuple[str, ...]]] = []

    def _no(iteration: int, tools: tuple[str, ...]) -> bool:
        asked.append((iteration, tools))
        return False

    def _yes(iteration: int, tools: tuple[str, ...]) -> bool:
        asked.append((iteration, tools))
        return True

    assert turn_replay_allowed(tmp_path, 5, _no) is True  # no marker, never asks
    write_turn_marker(marker, 3, ("apply_patch",))
    assert turn_replay_allowed(tmp_path, 5, _no) is True  # stale: cleared, no ask
    assert not marker.exists()
    assert asked == []
    write_turn_marker(marker, 5, ("run_command",))
    assert turn_replay_allowed(tmp_path, 5, _no) is False  # matching + decline
    assert marker.exists()  # stays for the next resume to ask again
    assert turn_replay_allowed(tmp_path, 5, _yes) is True  # matching + accept
    assert marker.exists(), "the answer is spent by the execution, not by the question"
    assert asked == [(5, ("run_command",)), (5, ("run_command",))]


def test_steer_exit_ends_steer_exit_and_suppresses_the_follow_up() -> None:
    """The pause menu's /exit stops the run with its own end reason, skipping the "next:" prompt."""
    import json

    from agent6.ui.cli._session_prompt import follow_up_on_offer
    from agent6.viewmodel.listing import status_word

    ev = _EventCapture()
    wf = _wf(
        mode="run",
        events=ev,
        bridge=OperatorBridge(steer_requested=lambda: True, steer_prompt=lambda: "exit"),
    )
    result = wf.steering.handle(Conversation(), 3, _state())
    assert result == "exit"
    out = wf._steer_outcome("exit", 3, _state())  # pyright: ignore[reportPrivateUsage]
    assert out is not None and out.reason == "steer_exit" and out.completed is False
    ends = [e for e in ev.events if e["type"] == "session.end"]
    assert ends and ends[-1]["reason"] == "steer_exit"
    assert status_word(finished=True, all_passed=False, end_reason="steer_exit") == ("stopped", "")

    # The log-derived follow-up gate: steer_exit never re-opens the prompt.
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as td:
        d = _Path(td)
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "steer_exit", "all_passed": False})
            + "\n",
            encoding="utf-8",
        )
        assert follow_up_on_offer(d) is False
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "steer_abort", "all_passed": False})
            + "\n",
            encoding="utf-8",
        )
        assert follow_up_on_offer(d) is True


def test_an_adopted_gate_that_cannot_run_is_un_adopted(tmp_path: Path) -> None:
    """An adopted verify whose first failure is an unrunnable signature is dropped again.

    Exit 127, or the adopted `-m` module missing: the model is told, the manifest re-pins gateless,
    and that argv is never re-adopted; a configured gate stays red.
    """
    from agent6.config import Config
    from agent6.harness._nudges import VERIFY_UNADOPTED_NOTICE

    argv = ("python3", "-m", "pytest", "-q")
    events: list[dict[str, Any]] = []

    def emit(kind: str, **kw: Any) -> None:
        events.append({"type": kind, **kw})

    dispatcher = MagicMock()
    wf = _wf(
        root=tmp_path,
        config=Config(),  # gateless: the argv below is the adopted one, not a configured gate
        provider=MagicMock(),
        dispatcher=dispatcher,
        events=MagicMock(emit=emit),
    )
    st = _state()
    st.verify.adopted = argv
    turn = _turn(iteration=4)
    wf.gate.note_result(
        st,
        turn,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="/usr/bin/python3: No module named pytest",
            duration_s=0.03,
            exec_failed=False,
        ),
    )
    assert wf.gate.command(st.verify) == ()
    dispatcher.drop_verify_command.assert_called_once()
    assert st.verify.adopted == () and argv in st.verify.unadoptable
    texts = [it.text for it in turn.tool_results if isinstance(it, Notice)]
    assert any(t.startswith(VERIFY_UNADOPTED_NOTICE[:30]) for t in texts)
    assert not st.verify.broken_warned
    # No verdict: the turn is not "verify failed" (the panel and checkpoint logic key on the flag).
    assert turn.verify_just_failed is False
    assert any(
        e.get("command") == [] and e.get("source") == "unadopted" and e.get("adopted_at") == 4
        for e in events
    )

    # A configured gate with the same failure stays a red verify; nothing is un-adopted.
    wf2 = _wf(
        root=tmp_path,
        config=Config().with_verify_command(argv),
        provider=MagicMock(),
        dispatcher=MagicMock(),
    )
    st2 = _state()
    turn2 = _turn(iteration=2)
    wf2.gate.note_result(
        st2,
        turn2,
        ExecResult(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
            duration_s=0.03,
            exec_failed=False,
        ),
    )
    assert wf2.config.harness.verify_command == argv and st2.verify.broken_warned


def test_operator_answers_become_recorded_rulings(tmp_path: Path) -> None:
    """The decisions file is written by the harness, not the model.

    An ask_user answer lands as a ruling with its question, a steer answering the model's trailing
    question lands with it, an ordinary steer does not.
    """
    from agent6.harness._conversation import Conversation
    from agent6.memory import decisions_path
    from agent6.tools.results import AnswersResult

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    events = MagicMock(path=tmp_path / "sessions" / "runs" / "tidy-fox-1" / "logs.jsonl")
    prompts = iter(["Use the inline item.", "unrelated instruction"])
    wf = _wf(
        root=tmp_path,
        state_dir=state_dir,
        provider=MagicMock(),
        dispatcher=MagicMock(),
        events=events,
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_prompt=lambda: next(prompts),
            steer_clear=lambda: None,
        ),
    )
    st = _state()
    turn = _turn(iteration=1)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        st,
        turn,
        "ask_user",
        AnswersResult(answers=("8931", "no"), asked=("Which port?", "Keep the modal?")),
        {"questions": [{"question": "Which port?"}, {"question": "Keep the modal?"}]},
    )
    conv = Conversation()
    conv.notice("task")
    conv.assistant([{"type": "text", "text": "Two shapes fit.\nDrop the modal or keep it?"}])
    assert wf.steering.handle(conv, 2, st) is None
    conv.assistant([{"type": "text", "text": "Done with the item."}])
    assert wf.steering.handle(conv, 3, st) is None
    text = decisions_path(state_dir).read_text(encoding="utf-8")
    assert text.count("[tidy-fox-1]") == 3
    assert "Q: Which port?\n  A: 8931\n" in text and "Q: Keep the modal?\n  A: no\n" in text
    assert "Q: Drop the modal or keep it?\n  A: Use the inline item.\n" in text
    assert "unrelated instruction" not in text
    assert len(st.decisions_recorded) == 3
    # The check reads the file, not the capped injection view.
    with decisions_path(state_dir).open("a", encoding="utf-8") as fh:
        fh.write("- 2026-08-23T00:00:00Z [other] Q: pad\n  A: " + "x" * 5000 + "\n")
    wf._check_decisions_recorded(st)  # pyright: ignore[reportPrivateUsage]
    assert not any(
        c.kwargs.get("missing")
        for c in events.emit.call_args_list
        if c.args[:1] == ("loop.decision.unrecorded",)
    )


def test_a_skill_command_steer_expands_in_the_loop(tmp_path: Path) -> None:
    """`/<skill> [args]` from any composer injects the skill's full text as the instruction.

    A slash word that is no skill stays an ordinary steer.
    """
    from agent6.skills import ResolvedSkills, Skill

    skill = Skill(name="caveman", description="Use when grunting.", dir=tmp_path, text="GRUNT")
    prompts = iter(["/caveman lite", "/nosuch thing"])
    dispatcher = MagicMock()
    dispatcher.resolved_skills.return_value = ResolvedSkills(
        enabled=(skill,), always=(), warnings=("one skill dir is unreadable",)
    )
    events = MagicMock()
    wf = _wf(
        root=tmp_path,
        provider=MagicMock(),
        dispatcher=dispatcher,
        events=events,
        bridge=OperatorBridge(
            steer_requested=lambda: True,
            steer_prompt=lambda: next(prompts),
            steer_clear=lambda: None,
        ),
    )
    wf.mode = "run"
    st = _state()
    conv = MagicMock()
    assert wf.steering.handle(conv, 1, st) is None
    # The steer reads the cached resolution: the assembly warnings are not re-emitted per steer.
    assert not [c for c in events.emit.call_args_list if c.args[:1] == ("loop.skills.warning",)]
    injected = conv.notice.call_args.args[0]
    assert "Apply the operator-installed skill 'caveman'" in injected
    assert (
        "Skill arguments: lite" in injected
        and '<skill name="caveman">\nGRUNT\n</skill>' in injected
    )
    assert wf.steering.handle(conv, 2, st) is None
    assert "/nosuch thing" in conv.notice.call_args.args[0]


def _metric_repo(repo: Path) -> str:
    """A one-commit git repo; returns HEAD's sha (a run's chain_fallback_parent)."""
    repo.mkdir(parents=True)
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    (repo / "x.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "x.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _metric_wf(
    repo: Path, base: str, dispatcher: MagicMock, *, commit_per_step: bool = True
) -> Harness:
    """A gateless run with an operator metric on a chain."""
    return Harness(
        chain=RunChain(
            repo, ref="refs/agent6/metric-run/head", fallback_parent=base, per_step=commit_per_step
        ),
        config=Config.model_validate(
            {
                "harness": {
                    "verify_command": [],
                    "metric": {
                        "command": ["score.sh"],
                        "pattern": r"SCORE: ([0-9.]+)",
                        "goal": "maximize",
                    },
                }
            }
        ),
        provider=MagicMock(),
        dispatcher=dispatcher,
        logger=_silent,
        mode="run",
    )


def _metric_result(score: float) -> MetricResult:
    return MetricResult(
        returncode=0,
        stdout=f"SCORE: {score}\n",
        stderr="",
        duration_s=0.1,
        exec_failed=False,
        score=score,
    )


def _edited_turn(iteration: int) -> TurnState:
    turn = TurnState(
        iteration=iteration, resp=_resp(""), assistant=AssistantTurn(raw_content=(), tool_uses=())
    )
    turn.edited = True
    return turn


def test_the_workers_own_metric_call_is_not_re_run_by_the_harness(tmp_path: Path) -> None:
    """The worker's manual metric reading stamps the tree it covers, so the auto path skips it.

    Otherwise the operator's benchmark ran twice per turn over one tree, and the duplicate read as
    "not a new best".
    """
    repo = tmp_path / "repo"
    base = _metric_repo(repo)
    dispatched: list[str] = []
    dispatcher = MagicMock(operator_wait_s=0.0)

    def dispatch(name: str, _args: dict[str, Any]) -> MetricResult:
        dispatched.append(name)
        return _metric_result(42.0)

    dispatcher.dispatch.side_effect = dispatch
    wf = _metric_wf(repo, base, dispatcher, commit_per_step=False)
    state = LoopState(original_task="t", tool_calls=0)
    turn = _edited_turn(1)
    (repo / "x.txt").write_text("an improvement\n", encoding="utf-8")

    wf._note_tool_effects(state, turn, "run_metric_command", _metric_result(42.0), {})  # pyright: ignore[reportPrivateUsage]
    wf._turn_auto_commit_and_metric(state, turn)  # pyright: ignore[reportPrivateUsage]

    assert dispatched == [], "the harness re-ran the operator's metric over an unchanged tree"
    assert [s.score for s in state.metric.history] == [42.0]
    assert "not a new best" not in (turn.metric_feedback or "")
    # The next turn reads the same tree: still one reading per state of it.
    later = TurnState(
        iteration=2, resp=_resp(""), assistant=AssistantTurn(raw_content=(), tool_uses=())
    )
    wf._turn_auto_commit_and_metric(state, later)  # pyright: ignore[reportPrivateUsage]
    assert dispatched == []
    assert [s.score for s in state.metric.history] == [42.0]


def test_three_real_improvements_do_not_read_as_a_plateau(tmp_path: Path) -> None:
    """Three strictly better readings are not a plateau: a sample never ties with itself."""
    repo = tmp_path / "repo"
    base = _metric_repo(repo)
    score = {"v": 41.0}
    dispatcher = MagicMock(operator_wait_s=0.0)

    def dispatch(_name: str, _args: dict[str, Any]) -> MetricResult:
        return _metric_result(score["v"])

    dispatcher.dispatch.side_effect = dispatch
    wf = _metric_wf(repo, base, dispatcher)
    state = LoopState(original_task="t", tool_calls=0)
    plateaus: list[str] = []
    for i in (1, 2, 3):
        score["v"] = 41.0 + i
        (repo / "x.txt").write_text(f"improvement {i}\n", encoding="utf-8")
        turn = _edited_turn(i)
        wf._note_tool_effects(state, turn, "run_metric_command", _metric_result(score["v"]), {})  # pyright: ignore[reportPrivateUsage]
        wf._turn_auto_commit_and_metric(state, turn)  # pyright: ignore[reportPrivateUsage]
        if turn.metric_plateau_finish is not None:
            plateaus.append(turn.metric_plateau_finish)

    assert [s.score for s in state.metric.history] == [42.0, 43.0, 44.0]
    assert plateaus == []


def _settles_at(wf: Harness, repo: Path, state: LoopState, *, gated: bool) -> int | None:
    """The iteration the verify-settled stop fires at, or None within a generous window."""
    from agent6.harness._nudges import VERIFY_SETTLED_STOP_AFTER

    (repo / "x.txt").write_text("the worker's finished work\n", encoding="utf-8")
    first = _edited_turn(1)
    if gated:
        state.verify.note_pass()
        first.verify_just_passed = True
    wf._turn_auto_commit_and_metric(state, first)  # pyright: ignore[reportPrivateUsage]
    _settle(wf, state, first)
    for i in range(2, VERIFY_SETTLED_STOP_AFTER * 3):
        turn = TurnState(
            iteration=i, resp=_resp(""), assistant=AssistantTurn(raw_content=(), tool_uses=())
        )
        wf._turn_auto_commit_and_metric(state, turn)  # pyright: ignore[reportPrivateUsage]
        _settle(wf, state, turn)
        if turn.stops:
            return i
    return None


@pytest.mark.parametrize("gated", [True, False])
def test_the_settled_stop_still_fires_without_per_step_commits(tmp_path: Path, gated: bool) -> None:
    """With `[git].commit_per_step = false` progress is a changed tree; an edit step seeds.

    The chain never advances, so a dirty worktree is not progress and a gateless run still settles.
    """

    def wf_for(repo: Path, base: str, *, commit_per_step: bool) -> Harness:
        return Harness(
            chain=RunChain(
                repo,
                ref="refs/agent6/settled-run/head",
                fallback_parent=base,
                per_step=commit_per_step,
            ),
            config=Config.model_validate(
                {"harness": {"verify_command": ["true"] if gated else []}}
            ),
            provider=MagicMock(),
            dispatcher=MagicMock(),
            logger=_silent,
            mode="run",
        )

    repo = tmp_path / "repo"
    on = wf_for(repo, _metric_repo(repo), commit_per_step=True)
    baseline = _settles_at(on, repo, LoopState(original_task="t", tool_calls=0), gated=gated)
    assert baseline is not None

    repo2 = tmp_path / "repo2"
    off = wf_for(repo2, _metric_repo(repo2), commit_per_step=False)
    state = LoopState(original_task="t", tool_calls=0)

    assert _settles_at(off, repo2, state, gated=gated) == baseline


def _ruling_wf(tmp_path: Path) -> Harness:
    """Return a run with a state dir whose steer bridge always answers "keep squash"."""

    def steer_prompt() -> str | None:
        return "keep squash"

    return Harness(
        chain=RunChain(tmp_path),
        config=Config(),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=_silent,
        mode="run",
        state_dir=tmp_path / "state",
        bridge=OperatorBridge(
            steer_requested=lambda: True, steer_clear=lambda: None, steer_prompt=steer_prompt
        ),
    )


def test_a_flat_ask_user_call_records_its_ruling(tmp_path: Path) -> None:
    """`AskUserInput` accepts one flat question and records the ruling from its result.

    The result carries what was asked, so nothing parses the raw dict a second time.
    """
    from agent6.memory import decisions_path
    from agent6.tools.results import AnswersResult

    question = "Should the default merge strategy stay squash?"
    wf = _ruling_wf(tmp_path)
    state = LoopState(original_task="t", tool_calls=0)
    turn = TurnState(
        iteration=1, resp=_resp(""), assistant=AssistantTurn(raw_content=(), tool_uses=())
    )
    flat = {"question": question, "options": ["yes", "no"]}

    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "ask_user", AnswersResult(answers=("yes",), asked=(question,)), flat
    )

    assert len(state.decisions_recorded) == 1
    assert question in decisions_path(tmp_path / "state").read_text(encoding="utf-8")


def test_ask_user_args_the_dispatcher_coerced_still_record_their_ruling(tmp_path: Path) -> None:
    """A `questions` sent as a JSON string is coerced once; nothing re-parses the input."""
    import json

    from agent6.memory import decisions_path
    from agent6.tools.dispatch import ToolDispatcher
    from agent6.tools.operator_prompts import OperatorPrompts, QuestionAnswer, QuestionRequest

    def questioner(request: QuestionRequest, /) -> QuestionAnswer:
        return QuestionAnswer(answers=tuple("yes" for _ in request.questions), source="stdin")

    dispatcher = ToolDispatcher(
        root=tmp_path, config=Config(), prompts=OperatorPrompts(questioner=questioner), mode="run"
    )
    question = "Ship it?"
    raw = {"questions": json.dumps([{"question": question, "options": ["yes", "no"]}])}
    result = dispatcher.dispatch("ask_user", raw)
    wf = _ruling_wf(tmp_path)
    state = LoopState(original_task="t", tool_calls=0)
    turn = TurnState(
        iteration=1, resp=_resp(""), assistant=AssistantTurn(raw_content=(), tool_uses=())
    )
    wf._note_tool_effects(state, turn, "ask_user", result, raw)  # pyright: ignore[reportPrivateUsage]
    assert len(state.decisions_recorded) == 1
    assert question in decisions_path(tmp_path / "state").read_text(encoding="utf-8")


def test_a_steer_answering_a_prose_question_records_after_the_nudge(tmp_path: Path) -> None:
    """The nudge for a prose question pairs the operator's answer with the prose, not the notice."""
    from agent6.harness._nudges import QUESTION_NUDGE
    from agent6.memory import decisions_path

    question = "Should the default merge strategy stay squash?"
    wf = _ruling_wf(tmp_path)
    conversation = Conversation()
    conversation.notice("TASK: do the thing")
    conversation.assistant([{"type": "text", "text": f"I need a ruling.\n{question}"}])
    conversation.notice(QUESTION_NUDGE)
    state = LoopState(original_task="t", tool_calls=0)

    wf.steering.handle(conversation, 4, state)

    assert len(state.decisions_recorded) == 1
    assert question in decisions_path(tmp_path / "state").read_text(encoding="utf-8")


def test_a_steer_answering_an_optioned_question_records_the_question(tmp_path: Path) -> None:
    """Options after a prose question do not become the recorded question."""
    from agent6.harness._nudges import QUESTION_NUDGE
    from agent6.memory import decisions_path

    question = "Which merge strategy should remain the default?"
    wf = _ruling_wf(tmp_path)
    conversation = Conversation()
    conversation.notice("TASK: do the thing")
    conversation.assistant([{"type": "text", "text": f"{question}\n1. merge\n2. squash"}])
    conversation.notice(QUESTION_NUDGE)
    state = LoopState(original_task="t", tool_calls=0)

    wf.steering.handle(conversation, 4, state)

    text = decisions_path(tmp_path / "state").read_text(encoding="utf-8")
    assert f"Q: {question}\n  A: keep squash\n" in text
    assert "Q: 2. squash" not in text


@pytest.mark.parametrize(
    ("label", "standing", "created_by"),
    [("standing run", True, "steering"), ("patience finish", False, "planner")],
)
def test_the_root_passes_with_an_open_child(
    tmp_path: Path, label: str, standing: bool, created_by: str
) -> None:
    """A completed run marks its root passed; the container rule exempts the root.

    A `--standing` goal never passes by design, and a finish can leave a subtask open.
    """
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import AddSubtaskIntent, NodeActor, TaskNodeDraft
    from agent6.sessions.layout import SessionLayout

    actor: NodeActor = "steering" if created_by == "steering" else "planner"
    curator = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = curator.add_subtask(
        AddSubtaskIntent(
            parent_id=None,
            draft=TaskNodeDraft(title="run root", depends_on=(), created_by="planner"),
        )
    )
    curator.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(title=label, depends_on=(), created_by=actor, standing=standing),
        )
    )
    wf = Harness(
        chain=RunChain(tmp_path),
        config=Config(),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=_silent,
        mode="run",
        curator=curator,
    )

    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]

    assert curator.nodes()[root.id].status == "passed"


def test_the_focus_surface_fits_a_standing_task(tmp_path: Path) -> None:
    """With nothing ordinary ready the focus falls back to the standing goal, unpassable."""
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
    from agent6.harness._conversation import UserTurn
    from agent6.harness._dag_focus import STUCK_ON_TASK_AFTER
    from agent6.sessions.layout import SessionLayout

    curator = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = curator.add_subtask(
        AddSubtaskIntent(
            parent_id=None,
            draft=TaskNodeDraft(title="run root", depends_on=(), created_by="planner"),
        )
    )
    standing = curator.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(
                title="keep the fleet green", depends_on=(), created_by="steering", standing=True
            ),
        )
    )
    wf = Harness(
        chain=RunChain(tmp_path),
        config=Config(),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=_silent,
        mode="run",
        curator=curator,
    )
    state = LoopState(original_task="t", tool_calls=0)
    conversation = Conversation()
    conversation.notice("TASK: keep the fleet green")

    for _ in range(STUCK_ON_TASK_AFTER + 2):
        wf._maybe_surface_current_task(conversation, state)  # pyright: ignore[reportPrivateUsage]

    texts = [
        item.text
        for turn in conversation.turns
        if isinstance(turn, UserTurn)
        for item in turn.items
        if isinstance(item, Notice)
    ]
    banner = next(t for t in texts if t.startswith("[harness focus]"))
    assert standing.id in banner and "standing task" in banner
    assert "mark it passed with update_task" not in banner
    assert "only the operator retires it" in banner
    assert not any(t.startswith("[harness] You have spent") for t in texts)


def test_a_turn_declaring_two_ends_seats_the_panel_once(tmp_path: Path) -> None:
    """One turn's end is reviewed once: a surviving finish skips the plateau's end gates."""
    from unittest.mock import patch

    from agent6.harness._reviewer import CritiqueResult
    from agent6.tools.results import FinishSessionResult

    def _tool_use(name: str, call_id: str, args: dict[str, Any]) -> ProviderResponse:
        block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
        return ProviderResponse(
            text="",
            tool_uses=({"id": call_id, "name": name, "input": args},),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [block]},
        )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls == 8:
                blocks = [
                    {"type": "tool_use", "id": "v8", "name": "run_verify_command", "input": {}},
                    {
                        "type": "tool_use",
                        "id": "f8",
                        "name": "finish_session",
                        "input": {"summary": "done"},
                    },
                ]
                return ProviderResponse(
                    text="",
                    tool_uses=tuple(
                        {"id": b["id"], "name": b["name"], "input": b["input"]} for b in blocks
                    ),
                    stop_reason="tool_use",
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    raw={"content": blocks},
                )
            return _tool_use("run_verify_command", f"v{self.calls}", {})

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "run_verify_command":
                return ExecResult(
                    returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
                )
            if name == "run_metric_command":
                score = next(self.scores)
                return MetricResult(
                    returncode=0,
                    stdout=f"CYCLES: {score:g}\n",
                    stderr="",
                    duration_s=0.1,
                    exec_failed=False,
                    score=score,
                )
            if name == "finish_session":
                return FinishSessionResult(
                    summary_text=str(raw_input.get("summary", "")), result=None
                )
            raise AssertionError(f"unexpected tool: {name}")

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="never",
            verify_retries=2,
            verify_command=("true",),
            metric=SimpleNamespace(goal="minimize"),
            verify_timeout_s=60.0,
            verify_infer=True,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=10,
        review=ReviewSettings(
            trigger="before_finish",
            seats=[object()],  # pyright: ignore[reportArgumentType]
        ),
    )
    panels: list[str] = []

    def fake_panel(self: Reviewer, state: Any, *, trigger: str, iteration: int) -> CritiqueResult:
        del self, state, iteration
        panels.append(trigger)
        return CritiqueResult(text="No blocking findings.", satisfied=True)

    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]
    with (
        patch("agent6.harness._chain.chain_commit", side_effect=[f"sha{i}" for i in range(1, 20)]),
        patch.object(Reviewer, "critique", fake_panel),
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "metric_plateau"
    assert panels == ["before_finish"]


def test_a_gate_nobody_may_run_leaves_the_run_gateless_for_commits(tmp_path: Path) -> None:
    """`run_commands = "no"` withholds the verify gate from the harness and the model alike.

    Gate presence has one owner, so under `verify_when = "step"` editing turns still commit.
    """
    from unittest.mock import patch

    from agent6.tools.results import FinishSessionResult

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("apply_edit", {"path": "x", "edits": []}, tool_id="e1")
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="f1")

    class DispatcherStub(_StubDispatcher):
        def command_policy(self) -> str:
            return "no"

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return FinishSessionResult(
                    summary_text=str(raw_input.get("summary", "")), result=None
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    config = SimpleNamespace(
        git=_GIT_STUB,
        budget=SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
        harness=SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_when="step",
            verify_retries=2,
            verify_command=("true",),
            verify_infer=False,
            metric=SimpleNamespace(goal=None),
            verify_timeout_s=60.0,
        ),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        mode="run",
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        max_iterations=5,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    with patch("agent6.harness._chain.chain_commit", return_value="sha1") as commit:
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "finish_session"
    assert commit.call_count >= 1  # the edit landed as a checkpoint


def test_a_denied_gate_is_never_replaced_by_an_adopted_one(tmp_path: Path) -> None:
    """A configured gate the operator denied makes the run gateless for its commits, unadopted."""
    from unittest.mock import patch

    from agent6.events import EventSink
    from agent6.tools.errors import ToolDeniedError
    from agent6.tools.results import FinishSessionResult

    (tmp_path / "AGENTS.md").write_text(
        "# AGENTS.md\n\n## Verify command\n\n```bash\necho ok\n```\n", encoding="utf-8"
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls <= 3:
                return _tool_resp(
                    "apply_edit", {"path": "x", "edits": []}, tool_id=f"e{self.calls}"
                )
            return _tool_resp("finish_session", {"summary": "done"}, tool_id="f1")

    class DispatcherStub(_StubDispatcher):
        def __init__(self) -> None:
            self.adopted: list[tuple[str, ...]] = []

        def run_verify(self, *, extra_argv: tuple[str, ...] = ()) -> ExecResult:
            del extra_argv
            raise ToolDeniedError("operator denied the verify gate")

        def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
            self.adopted.append(tuple(argv))
            return True

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
            if name == "finish_session":
                return FinishSessionResult(
                    summary_text=str(raw_input.get("summary", "")), result=None
                )
            return ExecResult(
                returncode=0, stdout="ok", stderr="", duration_s=0.1, exec_failed=False
            )

    cfg = Config.model_validate(
        {
            "harness": {
                "verify_command": ["configured-gate"],
                "verify_when": "step",
                "verify_infer": True,
            }
        }
    )
    dispatcher = DispatcherStub()
    events_path = tmp_path / "logs.jsonl"
    wf = _wf(
        root=tmp_path,
        config=cfg,
        provider=ProviderStub(),
        dispatcher=dispatcher,
        events=EventSink(events_path),
        max_iterations=8,
        mode="run",
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    shas = iter(f"sha{i}" for i in range(1, 20))

    def _next_sha(*_args: object, **_kwargs: object) -> str:
        return next(shas)

    with patch("agent6.harness._chain.chain_commit", side_effect=_next_sha):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            conversation=Conversation.from_wire(messages),
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
            original_task="t",
        )
    assert result.reason == "finish_session"
    assert dispatcher.adopted == []
    assert wf.config.harness.verify_command == ("configured-gate",)
    assert "loop.verify_inferred" not in events_path.read_text(encoding="utf-8")
