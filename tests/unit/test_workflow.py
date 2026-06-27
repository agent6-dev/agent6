# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for -era Workflow additions.

Covers the helpers that loop.py landed at the same time as the audit pass:
* _call_with_retry  - ProviderError single-retry behaviour (finding #5)
* _maybe_handle_steer - operator steering between iterations (finding #32)

Termination-reason distinction (finding #1) is exercised end-to-end in the
integration suite; the helpers above are pure-Python so they're cheaper
to test directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent6.providers import ProviderError, ProviderResponse
from agent6.workflows.loop import Workflow


def _silent(_msg: str) -> None:
    return None


def _wf(**kw: Any) -> Workflow:
    """Construct a Workflow with mocks for everything not under test.

    Caller-supplied kwargs win over the defaults so a test can pass its
    own provider / steer callables without colliding on the keyword.
    """
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(prompt=MagicMock(system_prompt_file="")),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
        "provider_retry_delay_s": 0.01,  # keep tests fast
    }
    defaults.update(kw)
    return Workflow(**defaults)


def _state(**kw: Any) -> Any:
    """Minimal _LoopState for _save_resume_snapshot call sites."""
    from agent6.workflows.loop import _LoopState  # pyright: ignore[reportPrivateUsage]

    defaults: dict[str, Any] = {"original_task": "t", "tool_calls": 0}
    defaults.update(kw)
    return _LoopState(**defaults)


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


# --- _call_with_retry -----------------------------------------------------


def test_call_with_retry_first_try_returns() -> None:
    """No ProviderError -> single call, returns immediately."""
    provider = MagicMock()
    provider.call.return_value = _resp("first")
    wf = _wf(provider=provider)
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "first"
    assert provider.call.call_count == 1


def test_call_with_retry_succeeds_on_retry() -> None:
    """ProviderError on first call, success on retry -> returns the retry."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("transient 529"), _resp("retried")]
    wf = _wf(provider=provider, provider_retry_count=1)
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "retried"
    assert provider.call.call_count == 2


def test_call_with_retry_reraises_after_retries_exhausted() -> None:
    """Two ProviderErrors with retry_count=1 -> bubble the last error."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("flake 1"), ProviderError("flake 2")]
    wf = _wf(provider=provider, provider_retry_count=1)
    with pytest.raises(ProviderError, match="flake 2"):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 2


def test_call_with_retry_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 carrying retry_after_s waits at least that long, not the (shorter)
    self-computed backoff."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.workflows.loop.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429 rate limited", status_code=429, retry_after_s=50.0),
        _resp("ok"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)  # _wf backoff is 0.01s
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "ok"
    assert slept and slept[0] >= 50.0  # honored the server's window, not ~0.01


def test_call_with_retry_clamps_retry_after_to_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile/buggy Retry-After can't hang the run: clamp to the ceiling."""
    slept: list[float] = []
    monkeypatch.setattr("agent6.workflows.loop.time.sleep", slept.append)
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("429", status_code=429, retry_after_s=9999.0),
        _resp("ok"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)
    wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert slept and slept[0] <= 120.0  # _RETRY_AFTER_CEILING_S


def _empty_tool_call_resp() -> ProviderResponse:
    """A self-contradictory response: stop_reason=tool_calls but no tool_use/text
    (the GLM-via-OpenRouter post-restart flake)."""
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="tool_calls",
        input_tokens=1,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def test_call_with_retry_retries_empty_tool_call_response() -> None:
    """An empty finish=tool_calls response (no tool_use, no text) is retried; the
    recovered real response is returned."""
    provider = MagicMock()
    provider.call.side_effect = [_empty_tool_call_resp(), _tool_resp("read_file", {"path": "x"})]
    wf = _wf(provider=provider, provider_retry_count=4, provider_retry_delay_s=0.001)
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.tool_uses  # recovered to a real tool call
    assert provider.call.call_count == 2


def test_call_with_retry_returns_last_empty_after_exhausting() -> None:
    """If every attempt is empty, return the last empty response (the loop's
    went_quiet handler takes over) -- never raise / assert-fail."""
    provider = MagicMock()
    provider.call.return_value = _empty_tool_call_resp()
    wf = _wf(provider=provider, provider_retry_count=2, provider_retry_delay_s=0.001)
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.stop_reason == "tool_calls" and not out.tool_uses
    assert provider.call.call_count == 3  # 1 initial + 2 retries


def test_is_empty_tool_call_response_discriminates() -> None:
    from agent6.workflows.loop import (
        _is_empty_tool_call_response,  # pyright: ignore[reportPrivateUsage]
    )

    assert _is_empty_tool_call_response(_empty_tool_call_resp())
    assert not _is_empty_tool_call_response(_resp("hi"))  # has text -> a silent finish
    assert not _is_empty_tool_call_response(_tool_resp("read_file"))  # has a tool_use
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
    assert not _is_empty_tool_call_response(starved)


def test_call_with_retry_default_rides_out_multiple_flaps() -> None:
    """The default retry budget survives more than one consecutive transient
    disconnect. Regression: a single retry (the old default) aborted long,
    expensive runs on a multi-second Anthropic 'Server disconnected' flap."""
    provider = MagicMock()
    disconnect = ProviderError("Server disconnected without sending a response")
    provider.call.side_effect = [disconnect, disconnect, disconnect, _resp("recovered")]
    wf = _wf(provider=provider)  # uses the default provider_retry_count
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "recovered"
    assert provider.call.call_count == 4


def test_call_with_retry_zero_retries_no_retry() -> None:
    """provider_retry_count=0 -> single attempt, no retry on error."""
    provider = MagicMock()
    provider.call.side_effect = [ProviderError("nope")]
    wf = _wf(provider=provider, provider_retry_count=0)
    with pytest.raises(ProviderError, match="nope"):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_does_not_swallow_non_provider_errors() -> None:
    """RuntimeError (etc.) must propagate without retry."""
    provider = MagicMock()
    provider.call.side_effect = [RuntimeError("not a provider error")]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(RuntimeError, match="not a provider error"):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_skips_retry_on_permanent_status() -> None:
    """A permanent client error (402 insufficient credits) re-raises on the
    first failure without consuming a retry. Observed live: a 402 was
    otherwise retried on every remaining turn, burning wall-time."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 402: Insufficient credits", status_code=402),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderError, match="402"):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_call_with_retry_skips_retry_on_all_permanent_statuses(status: int) -> None:
    """Every status in _NON_RETRYABLE_HTTP_STATUSES re-raises on the first
    failure without consuming a retry (not just the 402 observed live)."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError(f"provider error {status}", status_code=status),
        _resp("should-never-be-reached"),
    ]
    wf = _wf(provider=provider, provider_retry_count=3)
    with pytest.raises(ProviderError, match=str(status)):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1


def test_call_with_retry_still_retries_transient_5xx() -> None:
    """A 503 carries a status_code but is NOT in the permanent set, so the
    normal single-retry path still applies."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("OpenAI API error 503: upstream", status_code=503),
        _resp("recovered"),
    ]
    wf = _wf(provider=provider, provider_retry_count=1)
    out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "recovered"
    assert provider.call.call_count == 2


# --- exponential backoff with jitter -------------------------------------


def test_call_with_retry_exponential_backoff() -> None:
    """Retry delays grow exponentially: attempt N sleeps
    provider_retry_delay_s * 2 ** (attempt - 1), scaled by the jitter factor."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("flake 1"),
        ProviderError("flake 2"),
        ProviderError("flake 3"),
        _resp("success"),
    ]
    wf = _wf(
        provider=provider,
        provider_retry_count=3,
        provider_retry_delay_s=2.0,
        provider_retry_max_delay_s=30.0,
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=0.75),
    ):
        out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
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
        provider_retry_count=4,
        provider_retry_delay_s=2.0,
        provider_retry_max_delay_s=5.0,
    )
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        patch("random.uniform", return_value=1.0),
    ):
        out = wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert out.text == "success"
    assert provider.call.call_count == 5
    assert sleep_calls[0] == pytest.approx(2.0)  # min(2.0 * 2**0, 5.0)
    assert sleep_calls[1] == pytest.approx(4.0)  # min(2.0 * 2**1, 5.0)
    assert sleep_calls[2] == pytest.approx(5.0)  # min(2.0 * 2**2, 5.0) capped
    assert sleep_calls[3] == pytest.approx(5.0)  # min(2.0 * 2**3, 5.0) capped


def test_call_with_retry_backoff_skips_sleep_on_permanent_status() -> None:
    """A permanent status re-raises immediately with no sleep at all,
    even though provider_retry_count would otherwise allow retries."""
    provider = MagicMock()
    provider.call.side_effect = [
        ProviderError("Insufficient credits", status_code=402),
    ]
    wf = _wf(provider=provider, provider_retry_count=3, provider_retry_delay_s=10.0)
    sleep_calls: list[float] = []
    with (
        patch("time.sleep", side_effect=sleep_calls.append),
        pytest.raises(ProviderError, match="Insufficient credits"),
    ):
        wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_count == 1
    assert sleep_calls == []


# --- temperature wiring (Amp 2) -----------------------------------


def test_call_with_retry_pins_default_temperature_to_zero() -> None:
    """Default Workflow.temperature is 0.0; every provider.call must
    receive it. agent6 used to pass temperature=None
    so OpenRouter routed to the model's (often high) provider default,
    which produced observable degeneration on Kimi K2.6."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider)
    wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] == 0.0


def test_call_with_retry_honours_overridden_temperature() -> None:
    """Operators who set `[models.worker].temperature = 0.7` get it
    threaded through verbatim."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, temperature=0.7)
    wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] == 0.7


def test_call_with_retry_passes_through_none_temperature() -> None:
    """Explicit `temperature = None` reverts to the previous behaviour
    (let the provider pick), for operators who specifically want it."""
    provider = MagicMock()
    provider.call.return_value = _resp("ok")
    wf = _wf(provider=provider, temperature=None)
    wf._call_with_retry(system="s", messages=[], tools=[])  # pyright: ignore[reportPrivateUsage]
    assert provider.call.call_args.kwargs["temperature"] is None


# --- automatic metric feedback ------------------------------------------


def test_drive_loop_auto_runs_metric_after_verify_pass(tmp_path: Path) -> None:
    """Metric-configured runs should not rely on the worker remembering to
    call run_metric_command. After a green verify, the harness runs it and
    injects a compact history block into the next worker turn.
    """

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
            self.saw_metric_feedback = (
                "[harness metric]" in rendered
                and "score=42" in rendered
                and "first parsed metric sample" in rendered
            )
            return _tool_resp("finish_run", {"summary": "done"}, tool_id="tool-2")

    class DispatcherStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(name)
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                return {
                    "returncode": 0,
                    "stdout": "CYCLES: 42\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": 42.0,
                }
            if name == "finish_run":
                return {"acknowledged": True, "summary": raw_input["summary"]}
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch("agent6.workflows.loop.commit_all", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    assert result.completed is True
    assert result.reason == "finish_run"
    assert provider.saw_metric_feedback is True
    assert dispatcher.calls == ["run_verify_command", "run_metric_command", "finish_run"]


def test_drive_loop_finishes_on_metric_plateau(tmp_path: Path) -> None:
    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub:
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Improves to 50, then ties it. The plateau detector fires once
            # >=5 parsed samples exist, but the loop now answers the first
            # _METRIC_PLATEAU_PATIENCE (3) plateaus with a pivot nudge and
            # only stops on the 4th, so we need four tied samples at the end.
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                score = next(self.scores)
                return {
                    "returncode": 0,
                    "stdout": f"CYCLES: {score:g}\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": score,
                }
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
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
        "agent6.workflows.loop.commit_all",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5", "sha6", "sha7", "sha8"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    assert result.completed is True
    assert result.reason == "metric_plateau"
    assert "performance per dollar" in result.summary
    # 8 verify+metric pairs: samples 5-7 each draw a pivot nudge, sample 8 stops.
    assert dispatcher.calls == ["run_verify_command", "run_metric_command"] * 8


def test_drive_loop_plateau_nudges_before_stopping(tmp_path: Path) -> None:
    """The first plateau should not stop the run: the loop injects a pivot
    nudge and keeps going, so a worker that changes strategy can recover the
    remaining budget instead of quitting at a local optimum."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_plateau_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness plateau]" in rendered:
                self.saw_plateau_nudge = True
                return _tool_resp("finish_run", {"summary": "pivoted"}, tool_id="fin")
            return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")

    class DispatcherStub:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.scores = iter([100.0, 80.0, 60.0, 50.0, 50.0])

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(name)
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                score = next(self.scores)
                return {
                    "returncode": 0,
                    "stdout": f"CYCLES: {score:g}\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": score,
                }
            if name == "finish_run":
                return {"acknowledged": True, "summary": raw_input["summary"]}
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
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
        "agent6.workflows.loop.commit_all",
        side_effect=["sha1", "sha2", "sha3", "sha4", "sha5"],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    # The plateau at the 5th sample injected a pivot nudge instead of
    # stopping; the worker saw it and finished on its own terms.
    assert provider.saw_plateau_nudge is True
    assert result.reason == "finish_run"


def test_drive_loop_plateau_final_nudge_fires_in_final_budget_slice(tmp_path: Path) -> None:
    """On a REAL-budget run, ties while budget is high must not exhaust the
    plateau patience: the escalating FINAL ("make your one best bet") nudge has
    to still fire once the budget enters the final slice. Pins the bug where
    `plateau_nudges_used` accrued on high-budget ties, so the run stopped the
    instant the budget crossed the threshold and the FINAL nudge never showed."""
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_FINAL as _METRIC_PLATEAU_NUDGE_FINAL,
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_final_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _METRIC_PLATEAU_NUDGE_FINAL in str(kwargs["messages"][-1]):
                self.saw_final_nudge = True
            # Vary the call signature each turn so the repeat-loop-guard (which
            # kills at 10 identical back-to-back calls) does not fire; a real
            # worker varies its edits between verifies. This isolates the plateau
            # logic under test from the orthogonal loop-guard.
            return _tool_resp(
                "run_verify_command", {"n": self.calls}, tool_id=f"verify-{self.calls}"
            )

    class DispatcherStub:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.metric_count = 0
            # Improve to 50, then tie it for many rounds. Plateau fires from the
            # 5th sample; ties 5-8 land while budget is high (runway), 9+ land in
            # the final slice. With the fix, runway ties do not consume patience,
            # so the FINAL nudge fires on samples 9/10/11 and the run stops on 12.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 8)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                self.metric_count += 1
                score = next(self.scores)
                return {
                    "returncode": 0,
                    "stdout": f"CYCLES: {score:g}\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": score,
                }
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        max_iterations=20,
    )
    # Drive the budget fraction off the dispatcher's measurement count (robust to
    # _budget_fraction_remaining being read more than once per iteration): samples
    # 5-8 see 80% left (runway), 9+ see 10% left (final slice, FINAL nudge tier).
    wf._budget_fraction_remaining = lambda: 0.8 if dispatcher.metric_count <= 8 else 0.1  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.commit_all",
        side_effect=[f"sha{i}" for i in range(20)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    # The FINAL nudge must have fired in the final slice before the run stopped.
    assert provider.saw_final_nudge is True
    assert result.reason == "metric_plateau"
    # Runway ties (samples 5-8) did not consume patience, so the run kept going
    # well past the point the old code stopped (sample 9): >=12 metric samples.
    assert dispatcher.metric_count >= 12


def test_drive_loop_plan_finish_nudge_fires_once_at_iter_cap(tmp_path: Path) -> None:
    """A verbose planner that never calls finish_planning gets a single harness
    'finish now' nudge once it hits the plan turn cap -- not before, not again.
    This is the lever that makes Kimi K2.6 actually land a plan; pins the
    off-by-one (iteration - start + 1 >= cap) and the one-shot latch."""
    from agent6.workflows.loop import (
        _PLAN_BUDGET_NUDGE,  # pyright: ignore[reportPrivateUsage]
        _PLAN_NUDGE_AFTER_ITERS,  # pyright: ignore[reportPrivateUsage]
    )

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            # never finish on our own -> the loop must force the issue
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            assert name == "read_file"
            return {"content": "..."}

    provider = ProviderStub()
    wf = _wf(
        root=tmp_path,
        mode="plan",
        provider=provider,
        dispatcher=DispatcherStub(),
        max_iterations=_PLAN_NUDGE_AFTER_ITERS + 3,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nplan a feature"}]}]
    wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=messages, tools=[], tool_calls=0, start_iteration=1, root_task_id=None
    )
    # Injected exactly once, on the turn-cap iteration (mode stays "plan" on
    # every later turn, so the latch is what keeps it to one).
    assert provider.nudged_on == [_PLAN_NUDGE_AFTER_ITERS]


def test_drive_loop_plan_finish_nudge_fires_on_low_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nudge also fires early when the token budget runs low (not only on
    the turn cap) -- e.g. a planner reading large files burns budget fast."""
    from agent6.workflows import loop as loopmod
    from agent6.workflows.loop import _PLAN_BUDGET_NUDGE  # pyright: ignore[reportPrivateUsage]

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Workflow, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _PLAN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r-{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"content": "..."}

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
        system="s", messages=messages, tools=[], tool_calls=0, start_iteration=1, root_task_id=None
    )
    # Budget already below the threshold -> nudge on the very first turn, once.
    assert provider.nudged_on == [1]


def test_drive_loop_run_budget_nudge_forces_verify_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-metric `run` gets a one-shot wrap-up nudge when budget runs low.
    Observed live: the worker solves the task but never re-verifies or calls
    finish_run, so the budget dies on read-only commands."""
    from agent6.workflows import loop as loopmod
    from agent6.workflows.loop import _RUN_BUDGET_NUDGE  # pyright: ignore[reportPrivateUsage]

    def _low_budget(_self: object) -> float:
        return 0.2

    monkeypatch.setattr(loopmod.Workflow, "_budget_fraction_remaining", _low_budget)

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.nudged_on: list[int] = []

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _RUN_BUDGET_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.nudged_on.append(self.calls)
            return _tool_resp("list_dir", {"path": "."}, tool_id=f"l-{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"content": "..."}

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
        system="s", messages=messages, tools=[], tool_calls=0, start_iteration=1, root_task_id=None
    )
    # fires once, on the first turn at/below the threshold, and only once.
    assert provider.nudged_on == [1]


def test_drive_loop_verify_settled_nudges_then_stops(tmp_path: Path) -> None:
    """A run-mode worker that keeps spinning after verify already passed (no new
    commit, no edit) gets one finish nudge, then the loop stops it with
    reason='verify_settled' — the positive completion signal a non-metric run
    otherwise lacks (Kimi K2.6 observed running 128 iters when done at ~45)."""
    from agent6.workflows.loop import _VERIFY_SETTLED_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # -> verify passes
            # then spin on read-only commands forever (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.1}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal=None))
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
    with patch("agent6.workflows.loop.commit_all", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )
    assert provider.saw_nudge is True
    assert result.reason == "verify_settled"
    assert result.completed is True


def test_drive_loop_verify_settled_does_not_fire_before_first_verify(tmp_path: Path) -> None:
    """The settled detector must stay dormant until verify has passed at least
    once — a worker still reading toward its first green build must not be
    stopped early."""
    from agent6.workflows.loop import _VERIFY_SETTLED_NUDGE  # pyright: ignore[reportPrivateUsage]

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.saw_nudge = False

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if _VERIFY_SETTLED_NUDGE[:24] in str(kwargs["messages"][-1]):
                self.saw_nudge = True
            if self.calls >= 6:
                return _tool_resp("finish_run", {"summary": "done"}, tool_id="fin")
            return _tool_resp("read_file", {"path": f"f{self.calls}.py"}, tool_id=f"r{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            if name == "finish_run":
                return {"acknowledged": True, "summary": raw_input["summary"]}
            return {"content": "..."}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal=None))
    )
    wf = _wf(
        root=tmp_path, config=config, mode="run", provider=provider, dispatcher=DispatcherStub()
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\ndo it"}]}]
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=messages, tools=[], tool_calls=0, start_iteration=1, root_task_id=None
    )
    # never verified -> never nudged/stopped by the settled detector
    assert provider.saw_nudge is False
    assert result.reason == "finish_run"


def test_drive_loop_verify_settled_neutral_on_reverify(tmp_path: Path) -> None:
    """Re-running verify on an already-green tree (which the prompt encourages
    between reads) is active work, not idle — it must NOT accrue toward the
    verify-settled hard-stop, or a legit run gets truncated."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            return _tool_resp("run_verify_command", tool_id=f"v{self.calls}")  # always re-verify

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.1}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal=None))
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
    with patch("agent6.workflows.loop.commit_all", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )
    assert result.reason != "verify_settled"


def test_drive_loop_verify_settled_dormant_on_metric_runs(tmp_path: Path) -> None:
    """On a metric run, post-verify measure/analyse/read iterations legitimately
    make no commit; completion is owned by the metric early-finish + plateau
    logic, so the verify-settled detector must NOT hard-stop them."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id="v1")  # verify passes
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.1}

    provider = ProviderStub()
    # goal set -> this is a metric run (still mode=="run")
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize"))
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
    with patch("agent6.workflows.loop.commit_all", return_value=""):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )
    # would have been killed at idle 6 without the metric gate
    assert result.reason != "verify_settled"


def test_metric_plateau_nudge_escalates_with_budget_pressure() -> None:
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_EXPLORE as _METRIC_PLATEAU_NUDGE_EXPLORE,
    )
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_FINAL as _METRIC_PLATEAU_NUDGE_FINAL,
    )
    from agent6.workflows._metric import (
        METRIC_PLATEAU_NUDGE_PIVOT as _METRIC_PLATEAU_NUDGE_PIVOT,
    )
    from agent6.workflows._metric import (
        metric_plateau_nudge as _metric_plateau_nudge,
    )

    # No budget signal -> explore tier (keep trying new directions).
    assert _metric_plateau_nudge(None) is _METRIC_PLATEAU_NUDGE_EXPLORE
    # Plenty of runway -> explore.
    assert _metric_plateau_nudge(0.80) is _METRIC_PLATEAU_NUDGE_EXPLORE
    # Boundary at 0.5 is still "more than half" only when strictly above.
    assert _metric_plateau_nudge(0.50) is _METRIC_PLATEAU_NUDGE_PIVOT
    # Mid budget -> decisive pivot.
    assert _metric_plateau_nudge(0.40) is _METRIC_PLATEAU_NUDGE_PIVOT
    # Final slice -> single best bet.
    assert _metric_plateau_nudge(0.20) is _METRIC_PLATEAU_NUDGE_FINAL
    # Every tier keeps the greppable marker.
    for tier in (
        _METRIC_PLATEAU_NUDGE_EXPLORE,
        _METRIC_PLATEAU_NUDGE_PIVOT,
        _METRIC_PLATEAU_NUDGE_FINAL,
    ):
        assert tier.startswith("[harness plateau]")


def test_drive_loop_plateau_keeps_nudging_while_budget_high(tmp_path: Path) -> None:
    """With most of the budget unspent, a metric plateau must NOT terminate
    the run even after the fixed nudge patience is exhausted — the loop keeps
    pivoting until the budget enters its final slice."""
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

    class DispatcherStub:
        def __init__(self) -> None:
            self.calls: list[str] = []
            # Plateaus at the 5th sample and stays flat thereafter.
            self.scores = iter([100.0, 80.0, 60.0, 50.0] + [50.0] * 20)

        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del raw_input
            self.calls.append(name)
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                score = next(self.scores)
                return {
                    "returncode": 0,
                    "stdout": f"CYCLES: {score:g}\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": score,
                }
            raise AssertionError(f"unexpected tool: {name}")

    provider = ProviderStub()
    dispatcher = DispatcherStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    # Fresh budget with huge ceilings -> fraction_remaining stays ~1.0, well
    # above the final-slice threshold, so the plateau never becomes terminal.
    budget = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
    max_iters = 12
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=dispatcher,
        budget=budget,
        max_iterations=max_iters,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.commit_all",
        side_effect=[f"sha{i}" for i in range(1, max_iters + 2)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    # Ran out the iteration cap rather than stopping on the plateau, and
    # kept nudging past the fixed patience of 3.
    assert result.reason == "max_iterations"
    assert provider.plateau_nudges_seen > 3


def test_drive_loop_rejects_early_finish_while_budget_high(tmp_path: Path) -> None:
    """A finish_run on a metric run with most of the budget unspent is rejected
    and nudged a few times before the loop honours it."""
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
                "finish_run",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del name, raw_input
            return {"ok": True}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    # Huge ceilings keep fraction_remaining ~1.0, well above the final slice.
    budget = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        messages=messages,
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
    )

    # Rejected for the fixed patience of 3, then honoured on the 4th call.
    assert result.reason == "finish_run"
    assert provider.finish_nudges_seen == 3
    assert provider.calls == 4


def test_drive_loop_honors_finish_without_budget_signal(tmp_path: Path) -> None:
    """With no budget tracker wired in, an early finish_run is honoured at once
    so the guard can never deadlock a run that lacks a budget signal."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0
            self.finish_nudges_seen = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            rendered = str(kwargs["messages"][-1])
            if "[harness budget]" in rendered:
                self.finish_nudges_seen += 1
            return _tool_resp("finish_run", {"summary": "done"}, tool_id=f"finish-{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del name, raw_input
            return {"ok": True}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=None,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="system",
        messages=messages,
        tools=[],
        tool_calls=0,
        start_iteration=1,
        root_task_id=None,
    )

    assert result.reason == "finish_run"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 1


def test_metric_at_fraction_ceiling_detects_maxed_score() -> None:
    from agent6.workflows._metric import (
        metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
    )

    # Maxed-out fraction: numerator == score == denominator.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 27.0) is True
    assert _metric_at_fraction_ceiling("passed 5 / 5 checks", 5.0) is True
    # Partial score is not the ceiling.
    assert _metric_at_fraction_ceiling("SCORE: 26/27\n", 26.0) is False
    # Score that does not match the numerator is ignored.
    assert _metric_at_fraction_ceiling("SCORE: 27/27\n", 26.0) is False
    # Unbounded metric (raw count, no denominator) never trips the ceiling.
    assert _metric_at_fraction_ceiling("CYCLES: 1487\n", 1487.0) is False


def test_drive_loop_honors_finish_at_metric_ceiling(tmp_path: Path) -> None:
    """A finish_run on a maximize metric that is already at its provable
    ceiling (SCORE: N/N) is honoured immediately — even with most of the
    budget unspent — instead of being rejected and nudged. This is the guard
    against weak models burning their whole budget re-deriving a solved task.
    """
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
            # First turn: pass verify (auto-metric will report the ceiling).
            # Subsequent turns: try to finish.
            if self.calls == 1:
                return _tool_resp("run_verify_command", tool_id=f"verify-{self.calls}")
            return _tool_resp(
                "finish_run",
                {"summary": f"done-{self.calls}"},
                tool_id=f"finish-{self.calls}",
            )

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            del raw_input
            if name == "run_verify_command":
                return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
            if name == "run_metric_command":
                return {
                    "returncode": 0,
                    "stdout": "SCORE: 27/27\n",
                    "stderr": "",
                    "duration_s": 0.1,
                    "score": 27.0,
                }
            return {"ok": True}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="maximize")),
    )
    # Huge ceilings keep fraction_remaining ~1.0: without the ceiling guard
    # the early-finish guard would reject the finish here.
    budget = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=provider,
        dispatcher=DispatcherStub(),
        budget=budget,
        max_iterations=20,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK:\noptimize"}]}]

    with patch(
        "agent6.workflows.loop.commit_all",
        side_effect=[f"sha{i}" for i in range(1, 22)],
    ):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    # Honoured on the very first finish_run, with no budget nudges.
    assert result.reason == "finish_run"
    assert provider.finish_nudges_seen == 0
    assert provider.calls == 2


# --- tier-aware metric targets --------------------------------------------


def test_extract_metric_targets_minimize_picks_upper_bounds() -> None:
    from agent6.workflows._metric import (
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
    from agent6.workflows._metric import (
        extract_metric_targets as _extract_metric_targets,
    )

    text = "assert score > 0.80\nassert score >= 0.95\nassert other < 5\n"
    targets = _extract_metric_targets(text, goal="maximize")
    assert targets == (0.80, 0.95)


def test_next_metric_target_minimize_returns_nearest_unmet() -> None:
    from agent6.workflows._metric import next_metric_target as _next_metric_target

    targets = (147734.0, 18532.0, 1579.0, 1487.0)
    # At 8256 we've cleared 18532/147734; nearest unmet is the largest
    # threshold still below the current score.
    assert _next_metric_target(targets, 8256.0, "minimize") == 1579.0
    # Once under everything, no target remains.
    assert _next_metric_target(targets, 1000.0, "minimize") is None


def test_next_metric_target_maximize_returns_nearest_unmet() -> None:
    from agent6.workflows._metric import next_metric_target as _next_metric_target

    targets = (0.50, 0.80, 0.95)
    assert _next_metric_target(targets, 0.83, "maximize") == 0.95
    assert _next_metric_target(targets, 0.99, "maximize") is None


def test_format_metric_feedback_shows_next_target() -> None:
    from agent6.workflows._metric import (
        MetricSample as _MetricSample,
    )
    from agent6.workflows._metric import (
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
    assert "next target: drive the metric below 1579" in text
    assert "current 8256" in text


def test_worker_max_tokens_lifts_cap_on_metric_runs() -> None:
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    wf = _wf(
        config=config,
        mode="run",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens() == 32768  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_without_metric() -> None:
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal=None))
    )
    wf = _wf(
        config=config,
        mode="run",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens() == 16384  # pyright: ignore[reportPrivateUsage]


def test_worker_max_tokens_keeps_default_in_plan_mode() -> None:
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("true",), metric=SimpleNamespace(goal="minimize")),
    )
    wf = _wf(
        config=config,
        mode="plan",
        per_call_max_tokens=16384,
        metric_task_max_tokens=32768,
    )
    assert wf._worker_max_tokens() == 16384  # pyright: ignore[reportPrivateUsage]


# --- tier-2 summarise-and-restart compaction ------------------------------


def _long_history(n_pairs: int) -> list[dict[str, Any]]:
    """An original task message followed by ``n_pairs`` assistant tool_use /
    user tool_result turns with bulky payloads."""
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


def test_summarise_and_restart_replaces_history() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("done: tried A (kept), B (reverted); best=42 at sha9")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(6)
    original = messages[0]

    wf._summarise_and_restart(messages)  # pyright: ignore[reportPrivateUsage]

    # Collapsed to (original task, restart-with-summary).
    assert len(messages) == 2
    assert messages[0] is original
    text = messages[1]["content"][0]["text"]
    assert "[harness context restart]" in text
    assert "best=42 at sha9" in text
    # The summariser saw the worker provider's content, not the worker itself.
    summariser.call.assert_called_once()


def test_summarise_and_restart_applies_dag_checkoff() -> None:
    """At tier-2 compaction agent6 asks the summariser which tasks finished and
    applies it to the curator (passes completed, queues discovered), strips the
    bookkeeping block from the restart, and ignores hallucinated task ids."""

    class _FakeClient:
        def __init__(self) -> None:
            self._nodes = {
                "01ROOT": {"parent_id": None, "status": "in_progress", "title": "review repo"},
                "01DONE": {"parent_id": "01ROOT", "status": "pending", "title": "audit providers"},
                "01OPEN": {"parent_id": "01ROOT", "status": "pending", "title": "audit sandbox"},
            }
            self.passed: list[str] = []
            self.added: list[tuple[str | None, str]] = []

        def get_state(self) -> dict[str, Any]:
            return {"nodes": self._nodes}

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
    wf = _wf(summariser_provider=summariser, graph_client=fake)
    messages = _long_history(6)
    wf._summarise_and_restart(messages)  # pyright: ignore[reportPrivateUsage]

    assert fake.passed == ["01DONE"]  # valid completed id passed; hallucinated id ignored
    assert fake.added == [("01ROOT", "fix the budget rounding bug")]  # queued under the root
    restart_text = messages[1]["content"][0]["text"]
    assert "providers audit" in restart_text
    assert "checkoff" not in restart_text  # bookkeeping block stripped from the restart


class _FakeGraph:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self._nodes = nodes

    def get_state(self) -> dict[str, Any]:
        return {"nodes": self._nodes}


def test_task_finish_gate_nudges_open_subtasks_then_caps() -> None:
    """The finish-gate nudges while a SUBTASK is open, naming it, and stops after
    _TASK_FINISH_PATIENCE so a stuck worker can't bounce the loop forever."""
    from agent6.workflows.loop import _TASK_FINISH_PATIENCE  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "sub1": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "sub2": {"parent_id": "root", "status": "passed", "title": "audit sandbox"},  # done
    }
    wf = _wf(graph_client=_FakeGraph(nodes))
    st = _state()
    for i in range(1, _TASK_FINISH_PATIENCE + 1):
        nudge = wf._task_finish_gate_nudge(st)  # pyright: ignore[reportPrivateUsage]
        assert nudge is not None and "audit providers" in nudge
        assert "audit sandbox" not in nudge  # passed subtask not listed
        assert st.task_finish_nudges_used == i
    # Cap reached -> finish is honoured (no further nudges).
    assert wf._task_finish_gate_nudge(st) is None  # pyright: ignore[reportPrivateUsage]


def test_task_finish_gate_allows_finish_without_open_subtasks() -> None:
    """Only SUBTASKS gate. The always-pending auto-root alone must NOT block a
    finish (else every run deadlocks); no curator -> no gate either."""
    root_only = _FakeGraph({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    assert _wf(graph_client=root_only)._task_finish_gate_nudge(_state()) is None  # pyright: ignore[reportPrivateUsage]
    assert _wf(graph_client=None)._task_finish_gate_nudge(_state()) is None  # pyright: ignore[reportPrivateUsage]


# --- surface-current-task -------------------------------------------------


def test_current_task_id_prefers_open_cursor() -> None:
    """The cursor wins when it still points at an open subtask, even if an
    earlier subtask is also open (the worker's explicit focus choice is kept)."""
    from agent6.workflows.loop import _current_task_id  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "in_progress", "title": "b"},
    }
    assert _current_task_id(nodes, "b") == "b"  # cursor respected
    assert _current_task_id(nodes, None) == "a"  # no cursor -> first open subtask
    # Stale cursor (points at a closed task) -> recompute the frontier.
    nodes["b"]["status"] = "passed"
    assert _current_task_id(nodes, "b") == "a"
    # Cursor on the auto-root is not a focus target -> first open subtask.
    assert _current_task_id(nodes, "root") == "a"


def test_first_ready_subtask_respects_deps_and_order() -> None:
    """The frontier skips a subtask whose dependency is not yet done, and a
    passed/obsolete dependency unblocks it; roots and done tasks never surface."""
    from agent6.workflows.loop import _first_ready_subtask  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "passed", "title": "a"},  # done
        "b": {"parent_id": "root", "status": "pending", "title": "b", "depends_on": ["c"]},
        "c": {"parent_id": "root", "status": "pending", "title": "c"},
    }
    # b is blocked on c (pending) -> c is the first ready subtask.
    assert _first_ready_subtask(nodes) == "c"
    # Once c is done, b unblocks.
    nodes["c"]["status"] = "obsolete"
    assert _first_ready_subtask(nodes) == "b"
    # Everything done -> nothing ready (the finish-gate, not this, ends the run).
    nodes["b"]["status"] = "passed"
    assert _first_ready_subtask(nodes) is None


def test_first_ready_subtask_prefers_leaf_over_decomposed_parent() -> None:
    """A subtask with open children is a container -- the frontier surfaces its
    first ready leaf, not the parent, so a decompose moves focus forward. A cursor
    still pointing at the parent falls through to the leaf too."""
    from agent6.workflows.loop import (
        _current_task_id,  # pyright: ignore[reportPrivateUsage]
        _first_ready_subtask,  # pyright: ignore[reportPrivateUsage]
    )

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a", "b"]},
        "a": {"parent_id": "root", "status": "in_progress", "title": "a", "children": ["a1", "a2"]},
        "a1": {"parent_id": "a", "status": "pending", "title": "a1"},
        "a2": {"parent_id": "a", "status": "pending", "title": "a2"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    assert _first_ready_subtask(nodes) == "a1"  # the parent 'a' is skipped as a container
    assert _current_task_id(nodes, "a") == "a1"  # stale cursor on the parent falls through
    # Once the children are done, the parent becomes a focusable leaf again.
    nodes["a1"]["status"] = "passed"
    nodes["a2"]["status"] = "passed"
    assert _first_ready_subtask(nodes) == "a"


def test_current_task_banner_carries_title_acceptance_paths() -> None:
    from agent6.workflows.loop import _current_task_banner  # pyright: ignore[reportPrivateUsage]

    banner = _current_task_banner(
        "01TASK",
        {"title": "audit providers", "acceptance": "no bugs left", "relevant_paths": ["a.py"]},
    )
    assert "Current task (01TASK): audit providers" in banner
    assert "Acceptance: no bugs left" in banner
    assert "Relevant paths: a.py" in banner
    assert "ONE task to completion" in banner
    # Absent acceptance/paths are simply omitted, not rendered empty.
    bare = _current_task_banner("01X", {"title": "t"})
    assert "Acceptance:" not in bare and "Relevant paths:" not in bare


class _FakeCurator:
    """In-memory GraphClient stand-in: get_state / set_cursor / update_status."""

    def __init__(self, nodes: dict[str, dict[str, Any]], cursor: str | None = None) -> None:
        self._nodes = nodes
        self._cursor = cursor
        self.cursor_sets: list[str | None] = []
        self.status_sets: list[tuple[str, str]] = []

    def get_state(self) -> dict[str, Any]:
        return {"nodes": self._nodes, "cursor": self._cursor}

    def set_cursor(self, intent: Any) -> None:
        self._cursor = intent.id
        self.cursor_sets.append(intent.id)

    def update_status(self, intent: Any) -> None:
        self.status_sets.append((intent.id, intent.new_status))
        self._nodes[intent.id]["status"] = intent.new_status


def _surface(wf: Workflow, st: Any, messages: list[dict[str, Any]]) -> None:
    wf._maybe_surface_current_task(messages, st)  # pyright: ignore[reportPrivateUsage]


def test_surface_current_task_surfaces_advances_then_quiets() -> None:
    """First call surfaces the focus banner, advances the cursor onto the task,
    and marks it in_progress; a repeat call with the same focus stays quiet (the
    banner survives tier-1 elision); marking it passed advances to the next."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "review repo"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        "b": {"parent_id": "root", "status": "pending", "title": "audit sandbox"},
    }
    cur = _FakeCurator(nodes)
    wf = _wf(graph_client=cur)
    st = _state()
    messages: list[dict[str, Any]] = []

    _surface(wf, st, messages)
    assert len(messages) == 1
    assert "audit providers" in messages[0]["content"][0]["text"]
    assert cur.cursor_sets == ["a"]  # cursor advanced onto the focus task
    assert cur.status_sets == [("a", "in_progress")]  # reflected as being worked
    assert st.surfaced_task_id == "a"

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
    assert st.surfaced_task_id == "b"


def test_surface_current_task_skips_status_write_when_already_in_progress() -> None:
    """The in_progress-only guard: a current task already in_progress is surfaced
    WITHOUT a redundant update_status write (only pending -> in_progress writes).
    Pins the negative branch of the sole conditional curator write."""
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "in_progress", "title": "audit providers"},
        },
        cursor="a",
    )
    wf = _wf(graph_client=cur)
    messages: list[dict[str, Any]] = []
    _surface(wf, _state(), messages)
    assert len(messages) == 1  # banner still surfaced
    assert cur.status_sets == []  # already in_progress -> no redundant status write
    assert cur.cursor_sets == []  # cursor already on it -> no redundant set_cursor


def test_surface_current_task_resurfaces_after_compaction_reset() -> None:
    """A tier-2 restart resets surfaced_task_id to None; the next surface call
    re-injects the focus banner into the fresh context."""
    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
    }
    wf = _wf(graph_client=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    _surface(wf, st, messages)
    assert len(messages) == 1
    st.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert len(messages) == 2  # re-surfaced after the restart wiped the banner


def test_surface_current_task_noop_cases() -> None:
    """No-op without open subtasks (root only), without a curator, or outside run
    mode -- nothing is appended and no cursor/status write happens."""
    root_only = _FakeCurator({"root": {"parent_id": None, "status": "pending", "title": "t"}})
    msgs: list[dict[str, Any]] = []
    _surface(_wf(graph_client=root_only), _state(), msgs)
    assert msgs == [] and root_only.cursor_sets == []

    _surface(_wf(graph_client=None), _state(), msgs)
    assert msgs == []

    open_sub = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "pending", "title": "t"},
            "a": {"parent_id": "root", "status": "pending", "title": "a"},
        }
    )
    _surface(_wf(graph_client=open_sub, mode="plan"), _state(), msgs)
    assert msgs == [] and open_sub.cursor_sets == []  # plan mode does not surface


def _stuck_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if "without concluding it" in m["content"][0]["text"])


def test_surface_current_task_stuck_nudge_fires_periodically_then_caps() -> None:
    """The split/pass/skip nudge re-fires every _STUCK_ON_TASK_AFTER turns on the
    same stuck task (a weak model ignored a single nudge live), but caps at
    _STUCK_NUDGE_MAX so it cannot nag forever."""
    from agent6.workflows.loop import (
        _STUCK_NUDGE_MAX,  # pyright: ignore[reportPrivateUsage]
        _STUCK_ON_TASK_AFTER,  # pyright: ignore[reportPrivateUsage]
    )

    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "r"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    wf = _wf(graph_client=cur)
    st = _state()
    messages: list[dict[str, Any]] = []
    # One nudge after the first period, but not before it.
    for _ in range(_STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0  # turns_on_task is _STUCK_ON_TASK_AFTER-1 here
    _surface(wf, st, messages)
    assert _stuck_count(messages) == 1  # crossed the first period
    # Keep grinding well past the cap; it re-fires periodically then stops.
    for _ in range((_STUCK_NUDGE_MAX + 2) * _STUCK_ON_TASK_AFTER):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == _STUCK_NUDGE_MAX
    assert st.stuck_nudges_fired == _STUCK_NUDGE_MAX


def test_surface_current_task_stuck_nudge_resets_on_progress() -> None:
    """Forward motion (a task marked passed -> focus advances) resets the grind
    counter, so the stuck nudge does not fire."""
    from agent6.workflows.loop import _STUCK_ON_TASK_AFTER  # pyright: ignore[reportPrivateUsage]

    nodes = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r"},
        "a": {"parent_id": "root", "status": "pending", "title": "a"},
        "b": {"parent_id": "root", "status": "pending", "title": "b"},
    }
    wf = _wf(graph_client=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(_STUCK_ON_TASK_AFTER - 1):  # grind almost to the threshold on a
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    nodes["a"]["status"] = "passed"  # progress -> focus advances to b
    for _ in range(3):
        _surface(wf, st, messages)
    assert _stuck_count(messages) == 0
    assert st.last_focus_id == "b" and st.turns_on_task < _STUCK_ON_TASK_AFTER


def test_surface_current_task_stuck_counter_survives_compaction() -> None:
    """A tier-2 restart resets the banner (surfaced_task_id) but NOT the grind
    counter -- compaction is not progress on the task."""
    wf = _wf(
        graph_client=_FakeCurator(
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
    assert st.turns_on_task == 4
    st.surfaced_task_id = None  # what the loop does on a tier-2 restart
    _surface(wf, st, messages)
    assert st.turns_on_task == 5  # kept climbing across the restart
    assert st.last_focus_id == "a"


def test_surface_decompose_resets_grind_counter() -> None:
    """Obeying the nudge -- decomposing the focus task with add_task -- moves focus
    to the first new leaf and resets the grind counter (the fix for the
    self-defeating-nudge bug)."""
    nodes: dict[str, dict[str, Any]] = {
        "root": {"parent_id": None, "status": "in_progress", "title": "r", "children": ["a"]},
        "a": {"parent_id": "root", "status": "pending", "title": "a", "children": []},
    }
    wf = _wf(graph_client=_FakeCurator(nodes))
    st = _state()
    messages: list[dict[str, Any]] = []
    for _ in range(5):
        _surface(wf, st, messages)
    assert st.last_focus_id == "a" and st.turns_on_task == 4
    # Worker splits 'a' into a child -> 'a' becomes a container, focus moves to a1.
    nodes["a"]["status"] = "in_progress"
    nodes["a"]["children"] = ["a1"]
    nodes["a1"] = {"parent_id": "a", "status": "pending", "title": "a1"}
    _surface(wf, st, messages)
    assert st.last_focus_id == "a1"  # focus advanced to the new leaf
    assert st.turns_on_task == 0  # grind counter reset by the decompose


def test_maybe_compact_returns_restart_signal() -> None:
    """_maybe_compact returns True only when a tier-2 restart actually replaced
    the history (the loop's cue to re-surface the focus banner)."""
    summariser = MagicMock()
    summariser.call.return_value = _resp("progress summary")
    wf = _wf(summariser_provider=summariser, compact_summarise_at_chars=500_000)
    # Below the tier-2 threshold -> no restart, returns False.
    short = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert wf._maybe_compact(short) is False  # pyright: ignore[reportPrivateUsage]
    # Over the threshold -> restart, returns True.
    big = _big_text_history("TASK: x", blocks=8, block_chars=100_000)
    assert wf._maybe_compact(big) is True  # pyright: ignore[reportPrivateUsage]
    assert len(big) == 2  # history was replaced


def test_drive_loop_resurfaces_current_task_after_compaction(tmp_path: Path) -> None:
    """Integration: a tier-2 restart mid-run wipes the focus banner, and the loop's
    `if self._maybe_compact(messages): state.surfaced_task_id = None` edge makes the
    next nudge pass RE-SURFACE the current task into the fresh context. Pins that
    edge -- dropping the reset (or inverting the _maybe_compact bool) leaves no
    loop.task.surfaced after the restart, which is exactly the regression the
    surface/check-off/finish-gate trio exists to prevent."""
    import json

    from agent6.events import EventSink

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            self.calls += 1
            if self.calls >= 6:
                return _tool_resp("finish_run", {"summary": "done"}, tool_id=f"f{self.calls}")
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

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            if name == "finish_run":
                return {"acknowledged": True, "summary": raw_input.get("summary", "")}
            return {"ok": True}

    events = EventSink(tmp_path / "logs.jsonl")
    cur = _FakeCurator(
        {
            "root": {"parent_id": None, "status": "in_progress", "title": "review"},
            "a": {"parent_id": "root", "status": "pending", "title": "audit providers"},
        }
    )
    config = SimpleNamespace(workflow=SimpleNamespace(verify_command=("true",), metric=None))
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        summariser_provider=SummariserStub(),
        events=events,
        graph_client=cur,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=5_000,  # low so tier-2 fires mid-run
        budget=None,
        max_iterations=30,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: review"}]}]
    with patch("agent6.workflows.loop.commit_all", return_value="abc1234567890"):
        wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id="root",
        )
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # tier-2 restart happened
    assert "loop.task.surfaced" in types
    # The focus banner re-surfaces AFTER the restart wiped it (the load-bearing edge).
    restart_at = types.index("loop.compact.summarise.done")
    assert "loop.task.surfaced" in types[restart_at + 1 :]


def test_summarise_and_restart_falls_back_to_worker_provider() -> None:
    worker = MagicMock()
    worker.call.return_value = _resp("summary text")
    wf = _wf(provider=worker, summariser_provider=None)
    messages = _long_history(4)

    wf._summarise_and_restart(messages)  # pyright: ignore[reportPrivateUsage]

    assert len(messages) == 2
    worker.call.assert_called_once()


def test_summarise_and_restart_keeps_history_on_empty_summary() -> None:
    summariser = MagicMock()
    summariser.call.return_value = _resp("   ")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(5)
    before = list(messages)

    wf._summarise_and_restart(messages)  # pyright: ignore[reportPrivateUsage]

    # Empty summary -> message list untouched (fail-safe).
    assert messages == before


def test_summarise_and_restart_keeps_history_on_provider_error() -> None:
    summariser = MagicMock()
    summariser.call.side_effect = ProviderError("boom")
    wf = _wf(summariser_provider=summariser)
    messages = _long_history(5)
    before = list(messages)

    wf._summarise_and_restart(messages)  # pyright: ignore[reportPrivateUsage]

    assert messages == before


# --- _maybe_handle_steer --------------------------------------------------


def test_steer_noop_when_not_requested() -> None:
    """steer_requested() returns False -> _maybe_handle_steer is a no-op."""
    wf = _wf()  # default steer_requested = lambda: False
    messages: list[dict[str, Any]] = []
    result = wf._maybe_handle_steer(messages, iteration=1)  # pyright: ignore[reportPrivateUsage]
    assert result is None
    assert messages == []


def test_steer_injects_instruction() -> None:
    """Requested + non-empty prompt text -> instruction appended to messages."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "focus on perf_takehome.py first",
    )
    messages: list[dict[str, Any]] = []
    result = wf._maybe_handle_steer(messages, iteration=3)  # pyright: ignore[reportPrivateUsage]
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
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: "   ",
    )
    messages: list[dict[str, Any]] = []
    result = wf._maybe_handle_steer(messages, iteration=2)  # pyright: ignore[reportPrivateUsage]
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_none_text_continues_without_inject() -> None:
    """Operator EOF'd (None) -> continue with no message."""
    cleared: list[bool] = []
    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=lambda: None,
    )
    messages: list[dict[str, Any]] = []
    result = wf._maybe_handle_steer(messages, iteration=2)  # pyright: ignore[reportPrivateUsage]
    assert result is None
    assert cleared == [True]
    assert messages == []


def test_steer_abort_signal() -> None:
    """Operator typed 'abort' (case-insensitive) -> returns 'abort'."""
    for typed in ("abort", "ABORT", "Abort"):
        cleared: list[bool] = []

        def _record(c: list[bool] = cleared) -> None:
            c.append(True)

        def _typed(t: str = typed) -> str:
            return t

        wf = _wf(
            steer_requested=lambda: True,
            steer_clear=_record,
            steer_prompt=_typed,
        )
        messages: list[dict[str, Any]] = []
        result = wf._maybe_handle_steer(messages, iteration=5)  # pyright: ignore[reportPrivateUsage]
        assert result == "abort", f"typed={typed!r}"
        assert cleared == [True]
        assert messages == [], "abort must not inject a message"


def test_steer_clear_called_even_when_prompt_raises() -> None:
    """A misbehaving steer_prompt must not leave the flag set."""
    cleared: list[bool] = []

    def boom() -> str | None:
        raise RuntimeError("input EOF")

    wf = _wf(
        steer_requested=lambda: True,
        steer_clear=lambda: cleared.append(True),
        steer_prompt=boom,
    )
    messages: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="input EOF"):
        wf._maybe_handle_steer(messages, iteration=1)  # pyright: ignore[reportPrivateUsage]
    assert cleared == [True], "finally must run steer_clear even on prompt failure"


# --- resume: snapshot save/load and resume() behaviour ------------


def test_save_resume_snapshot_noop_when_path_unset(tmp_path: Path) -> None:
    """resume_state_path=None -> no file written, no exception."""
    wf = _wf()
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    # tmp_path should still be empty.
    assert list(tmp_path.iterdir()) == []


def test_save_and_load_resume_snapshot_round_trip(tmp_path: Path) -> None:
    """Snapshot written by _save_resume_snapshot loads back identically."""
    from agent6.workflows.loop import _load_resume_snapshot  # pyright: ignore[reportPrivateUsage]

    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
    ]
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="SYSTEM PROMPT",
        messages=msgs,
        tool_calls=3,
        next_iteration=7,
        root_task_id="task-abc",
        state=_state(tool_calls=3),
    )
    assert snap_path.is_file()
    loaded = _load_resume_snapshot(snap_path)
    assert loaded.system == "SYSTEM PROMPT"
    assert loaded.messages == msgs
    assert loaded.tool_calls == 3
    assert loaded.next_iteration == 7
    assert loaded.root_task_id == "task-abc"


def test_save_resume_snapshot_atomic_no_partial_tmp(tmp_path: Path) -> None:
    """After save, no .tmp file remains: the final snapshot + its per-turn
    checkpoint are the only artifacts (both written atomically)."""
    snap_path = tmp_path / "loop_state.json"
    wf = _wf(resume_state_path=snap_path)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    assert snap_path.is_file()
    # The per-turn checkpoint lands under checkpoints/; nothing else (no .tmp).
    assert (tmp_path / "checkpoints" / "0001.json").is_file()
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != snap_path.name)
    assert leftovers == ["checkpoints"], f"unexpected leftover files: {leftovers}"
    cp_leftovers = [p.name for p in (tmp_path / "checkpoints").iterdir()]
    assert cp_leftovers == ["0001.json"], f"unexpected checkpoint leftovers: {cp_leftovers}"


def test_load_resume_snapshot_rejects_version_mismatch(tmp_path: Path) -> None:
    """A snapshot with a wrong version must raise ValueError."""
    import json as _json

    from agent6.workflows.loop import _load_resume_snapshot  # pyright: ignore[reportPrivateUsage]

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
    with pytest.raises(ValueError, match="version mismatch"):
        _load_resume_snapshot(snap_path)


def test_resume_raises_when_path_unset() -> None:
    """resume() with resume_state_path=None must raise ResumeError."""
    from agent6.workflows.loop import ResumeError

    wf = _wf()
    with pytest.raises(ResumeError, match="resume_state_path"):
        wf.resume()


def test_resume_raises_on_missing_snapshot(tmp_path: Path) -> None:
    """resume() with a nonexistent snapshot file must raise ResumeError."""
    from agent6.workflows.loop import ResumeError

    wf = _wf(resume_state_path=tmp_path / "nope.json")
    with pytest.raises(ResumeError, match="failed to load"):
        wf.resume()


def test_resume_drives_loop_from_snapshot(tmp_path: Path) -> None:
    """resume() loads snapshot, calls provider once, finishes via silent_finish."""
    snap_path = tmp_path / "loop_state.json"
    # Pre-seed the snapshot as if a prior run had just completed iter 4
    # and was about to start iter 5.
    snap_path.write_text(
        '{"version": 1, "system": "S", "messages": [{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 2, '
        '"next_iteration": 5, "root_task_id": null}',
        encoding="utf-8",
    )

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
    # Snapshot was rewritten before the (single) call this run made.
    assert snap_path.is_file()


def test_resume_restores_root_task_id_on_dispatcher(tmp_path: Path) -> None:
    """A non-null root_task_id in the snapshot must be re-set on dispatcher."""
    snap_path = tmp_path / "loop_state.json"
    snap_path.write_text(
        '{"version": 1, "system": "S", "messages": [{"role": "user", '
        '"content": [{"type": "text", "text": "go"}]}], "tool_calls": 0, '
        '"next_iteration": 1, "root_task_id": "task-xyz"}',
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
    """Simulate a provider crash mid-loop: snapshot must allow a clean resume.

    The v2 contract is: a snapshot is written BEFORE each LLM call, so a
    crash at any point (network, OOM, SIGKILL) leaves the run resumable
    from exactly that iteration with the prior turn's messages intact.
    Here we use a fake provider that raises on the first call to simulate
    the crash, then a fresh provider on resume that drives the loop to a
    clean finish.
    """
    import subprocess as _sp

    # Real git repo so _load_repo_summary() succeeds.
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
        provider_retry_count=0,  # don't mask the crash with a retry
    )
    # The first .run() ends with provider_error (v2's clean-shutdown path
    # for provider crashes). The snapshot was written BEFORE the call, so
    # the run is resumable from exactly that iteration.
    result1 = wf1.run("do a thing")
    assert result1.completed is False
    assert result1.reason == "provider_error"

    # Snapshot must exist after the crash and be loadable.
    assert snap_path.is_file(), "snapshot must be written before every LLM call"
    from agent6.workflows.loop import _load_resume_snapshot  # pyright: ignore[reportPrivateUsage]

    snap = _load_resume_snapshot(snap_path)
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


# --- tier-2 summarise-and-restart compaction -------------------------------
# Synthetic exercise driving context past compact_summarise_at_chars to confirm
# tier-2 actually summarises-and-restarts (the path that was unreachable before
# it measured the whole context via _context_chars).


def _ctx_chars(messages: list[dict[str, Any]]) -> int:
    from agent6.workflows._compaction import context_chars

    return context_chars(messages)


def _big_text_history(task: str, *, blocks: int, block_chars: int) -> list[dict[str, Any]]:
    # Assistant TEXT accumulates across a long run and tier-1 never elides it
    # (it only drops tool_results), so this is exactly what tier-2 must catch.
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
        summariser_provider=summ,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=500_000,
    )
    messages = _big_text_history("TASK: optimize the kernel", blocks=8, block_chars=100_000)
    assert _ctx_chars(messages) > 500_000  # over the tier-2 threshold

    wf._maybe_compact(messages)  # pyright: ignore[reportPrivateUsage]

    assert summ.calls == 1  # tier-2 summariser ran
    assert len(messages) == 2  # restarted to [original task, restart+summary]
    assert messages[0]["content"][0]["text"] == "TASK: optimize the kernel"
    assert "PROGRESS SUMMARY" in messages[1]["content"][0]["text"]
    assert _ctx_chars(messages) < 500_000  # context actually shrank


def test_tier2_summarise_failsafe_keeps_context_on_empty_summary(tmp_path: Path) -> None:
    class EmptySummariser:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return _resp("")  # empty -> fail-safe: keep the (tier-1-elided) context

    wf = _wf(
        root=tmp_path,
        summariser_provider=EmptySummariser(),
        compact_summarise_at_chars=500_000,
    )
    messages = _big_text_history("TASK", blocks=8, block_chars=100_000)
    n_before = len(messages)

    wf._maybe_compact(messages)  # pyright: ignore[reportPrivateUsage]

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
                return _tool_resp("finish_run", {"summary": "done"}, tool_id=f"f{self.calls}")
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

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            if name == "finish_run":
                return {"acknowledged": True, "summary": raw_input.get("summary", "")}
            return {"ok": True}

    events = EventSink(tmp_path / "logs.jsonl")
    summ = SummariserStub()
    config = SimpleNamespace(workflow=SimpleNamespace(verify_command=("true",), metric=None))
    wf = _wf(
        root=tmp_path,
        config=config,
        provider=ProviderStub(),
        dispatcher=DispatcherStub(),
        summariser_provider=summ,
        events=events,
        compact_drop_at_chars=256_000,
        compact_summarise_at_chars=5_000,  # low so it fires mid-run
        budget=None,
        max_iterations=30,
        loop_guard_kill_threshold=0,
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "TASK: optimize"}]}]

    with patch("agent6.workflows.loop.commit_all", return_value="abc1234567890"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="system",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )

    assert result.completed is True
    assert result.reason == "finish_run"
    assert summ.calls >= 1  # tier-2 fired mid-run
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "loop.compact.summarise.done" in types  # summarise-and-restart happened cleanly


def test_pass_pending_root_tasks_passes_only_pending_roots() -> None:
    """_pass_pending_root_tasks marks pending/in-progress ROOT tasks passed and
    leaves everything else (already-terminal roots, non-root subtasks) alone --
    so a finish_run-only ask/run reads N/N, not 0/1."""

    class _FakeClient:
        def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
            self._nodes = nodes
            self.passed: list[str] = []

        def get_state(self) -> dict[str, Any]:
            return {"nodes": self._nodes}

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
    wf = _wf(graph_client=fake)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]
    assert set(fake.passed) == {"root1", "root3"}


def test_pass_pending_root_tasks_noop_without_graph_client() -> None:
    """No curator wired (e.g. ask without a DAG) -> the auto-pass is a no-op."""
    wf = _wf(graph_client=None)
    wf._pass_pending_root_tasks()  # pyright: ignore[reportPrivateUsage]  (must not raise)


def test_drive_loop_gateless_settles_after_commit(tmp_path: Path) -> None:
    """A GATELESS run (no verify_command) has no green verify to seed the
    idle-stop net. Once an edit is committed it must still settle: spinning on
    read-only commands after the commit stops the run (reason='verify_settled'),
    so a gateless run can't burn budget to exhaustion when the worker is done."""

    class ProviderStub:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kwargs: Any) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                # an edit -> gateless auto-commit -> seeds gateless_ever_committed
                return _tool_resp("apply_edit", {"path": "x", "edits": []}, tool_id="e1")
            # then spin on read-only commands (no edit, no commit)
            return _tool_resp("run_command", {"cmd": f"ls {self.calls}"}, tool_id=f"c{self.calls}")

    class DispatcherStub:
        def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
            return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.1}

    provider = ProviderStub()
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=(), metric=SimpleNamespace(goal=None))
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
    with patch("agent6.workflows.loop.commit_all", return_value="sha1"):
        result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=messages,
            tools=[],
            tool_calls=0,
            start_iteration=1,
            root_task_id=None,
        )
    assert result.reason == "verify_settled"
    assert result.completed is True
    assert provider.calls < 30  # stopped well before max_iterations, not burned to the cap


def test_resume_snapshot_carries_verify_command(tmp_path: Path) -> None:
    """The snapshot stores the run's resolved verify_command so resume reuses it
    rather than re-inferring (which could diverge from the frozen prompt). A
    gateless run stores []; an older snapshot without the field loads as None."""
    import json as _json

    from agent6.workflows._run_state import load_resume_snapshot

    snap = tmp_path / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=("pytest", "-q"), metric=SimpleNamespace(goal=None))
    )
    wf = _wf(resume_state_path=snap, config=config)
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    assert load_resume_snapshot(snap).verify_command == ("pytest", "-q")

    config.workflow.verify_command = ()  # gateless run -> stored as [] -> loads as ()
    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=1, root_task_id=None, state=_state()
    )
    assert load_resume_snapshot(snap).verify_command == ()

    raw = _json.loads(snap.read_text(encoding="utf-8"))  # older snapshot: no field
    del raw["verify_command"]
    snap.write_text(_json.dumps(raw), encoding="utf-8")
    assert load_resume_snapshot(snap).verify_command is None


def test_provider_error_hint_for_auth_and_quota() -> None:
    from agent6.workflows.loop import _provider_error_hint  # pyright: ignore[reportPrivateUsage]

    assert "agent6 connect" in _provider_error_hint(401)
    assert "agent6 connect" in _provider_error_hint(403)
    assert "credits" in _provider_error_hint(402).lower()
    # Transient / unknown statuses get no hint (don't mislead).
    assert _provider_error_hint(429) == ""
    assert _provider_error_hint(500) == ""
    assert _provider_error_hint(None) == ""


def test_save_resume_snapshot_degrades_on_unwritable_state_dir(tmp_path: Path) -> None:
    # A full disk / read-only state dir disables resume/fork but must not abort
    # the run. Simulate by pointing the snapshot under a path whose parent is a
    # regular file, so mkdir raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    snap = blocker / "loop_state.json"  # parent "blocker" is a file -> mkdir fails
    logs: list[str] = []
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=(), metric=None),
    )
    wf = _wf(resume_state_path=snap, config=config, logger=logs.append)
    # Must not raise, twice (the second call must not re-warn).
    for _ in range(2):
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=[],
            tool_calls=0,
            next_iteration=1,
            root_task_id=None,
            state=_state(),
        )
    warnings = [m for m in logs if "could not persist resume snapshot" in m]
    assert len(warnings) == 1, "warn exactly once, then stay quiet"
    assert not snap.exists()


def test_open_tasks_for_checkoff_excludes_auto_root() -> None:
    # The tier-2 compaction check-off must never offer the auto-root (parent_id
    # is None): a summariser listing it would mark the whole run passed mid-run.
    graph_client = MagicMock()
    graph_client.get_state.return_value = {
        "nodes": {
            "root": {"status": "in_progress", "title": "the whole run", "parent_id": None},
            "01A": {"status": "pending", "title": "subtask A", "parent_id": "root"},
            "01B": {"status": "in_progress", "title": "subtask B", "parent_id": "root"},
            "01C": {"status": "passed", "title": "done subtask", "parent_id": "root"},
        }
    }
    wf = _wf(graph_client=graph_client)
    ids = {nid for nid, _ in wf._open_tasks_for_checkoff()}  # pyright: ignore[reportPrivateUsage]
    assert ids == {"01A", "01B"}  # root excluded; passed subtask excluded
