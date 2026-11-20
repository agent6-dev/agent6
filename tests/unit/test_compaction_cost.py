from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import budget
from agent6.app import finalize, reporter, role_providers
from agent6.providers import openai_parse, responses_codec
from agent6.sessions import layout
from agent6.viewmodel import receipt


def _call(call_id: str, *, role: str = "worker", **usage: Any) -> list[dict[str, Any]]:
    identity = {"role": role, "model": "test", "provider": "test", "model_call_id": call_id}
    return [
        {"type": "model.call", **identity},
        {"type": "model.result", "ok": True, **identity, **usage},
    ]


def _write(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_receipt_reports_cache_share_and_only_the_next_worker_call(tmp_path: pathlib.Path) -> None:
    """Helper calls and several compaction events cannot multiply a cold worker call."""
    rows = _call("one", input_tokens=100, cache_read_tokens=900)
    rows += [{"type": "compact.dropped", "count": 4}]
    rows += _call("helper", role="summariser", input_tokens=5000, cache_read_tokens=0)
    rows += [{"type": "compact.thinking_dropped", "messages": 2}]
    rows += _call("two", input_tokens=800, cache_read_tokens=200, cache_write_tokens=100)
    rows += _call("three", input_tokens=100, cache_read_tokens=900)
    rows += [{"type": "compact.summarise.failed", "error": "no summary"}]
    rows += _call("four", input_tokens=100, cache_read_tokens=900)
    _write(tmp_path / "logs.jsonl", rows)
    view = receipt.read_receipt(tmp_path, tmp_path)
    text = receipt.render(view)
    assert "Worker input cache (session): 70.7% (2,900/4,100 tokens)" in text
    assert "After compaction (session): 1 worker call, 900 uncached input tokens" in text
    assert receipt.as_dict(view)["cache_usage"]["post_compaction_calls"] == 1


@pytest.mark.parametrize(
    "event",
    [
        {"type": "compact.deduped", "count": 1},
        {"type": "compact.gist.done", "gisted": 0, "demoted": 1},
        {"type": "compact.thinking_dropped", "messages": 1},
        {"type": "compact.summarise.done", "summary_chars": 5},
    ],
)
def test_each_history_rewrite_counts_once_across_resumes(
    tmp_path: pathlib.Path, event: dict[str, Any]
) -> None:
    rows = [event, *_call("one", input_tokens=40, cache_read_tokens=60)]
    rows += [event, {"type": "session.resume", "execution_id": "new"}]
    rows += _call("two", input_tokens=40, cache_read_tokens=60)
    _write(tmp_path / "logs.jsonl", rows)
    text = receipt.render(receipt.read_receipt(tmp_path, tmp_path))
    assert "60.0% (120/200 tokens)" in text
    assert "1 worker call, 40 uncached input tokens" in text


@pytest.mark.parametrize("cached", [None, 0])
def test_unreported_cache_is_not_a_measured_zero(
    tmp_path: pathlib.Path, cached: int | None
) -> None:
    _write(tmp_path / "logs.jsonl", _call("one", input_tokens=100, cache_read_tokens=cached))
    text = receipt.render(receipt.read_receipt(tmp_path, tmp_path))
    assert ("Worker input cache" in text) is (cached is not None)
    assert ("After compaction" in text) is (cached is not None)


def test_partial_usage_and_failed_calls_do_not_transfer_compaction_cost(
    tmp_path: pathlib.Path,
) -> None:
    rows = [{"type": "compact.requested"}, {"type": "compact.dropped", "count": 0}]
    rows += _call("first", input_tokens=100, cache_read_tokens=900)
    rows += [{"type": "compact.dropped", "count": 1}]
    rows += _call("failed", ok=False)
    rows += [{"type": "model.attempt", "role": "worker", "input_tokens": 999999}]
    rows += _call("last", input_tokens=50, cache_read_tokens=50)
    _write(tmp_path / "logs.jsonl", rows)
    text = receipt.render(receipt.read_receipt(tmp_path, tmp_path))
    assert "86.4% (950/1,100 tokens); 2/3 calls reported" in text
    assert "0 worker calls, 0 uncached input tokens (reported calls only)" in text


def test_run_end_uses_the_same_journal_summary(tmp_path: pathlib.Path) -> None:
    saved = layout.SessionLayout(tmp_path, "cache-test")
    saved.ensure()
    _write(saved.logs_path, _call("one", input_tokens=100, cache_read_tokens=900))
    output: list[str] = []
    finalize.print_interrupt_end(
        layout=saved,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        reporter=reporter.Reporter(out=output.append, err=output.append),
    )
    assert any("Worker input cache (session): 90.0% (900/1,000 tokens)" in s for s in output)


def test_receipt_names_both_summary_attempts_and_their_usage(tmp_path: pathlib.Path) -> None:
    rows = [
        {"type": "compact.summarise.call", "writer": "worker", "model_call_id": "summary1"},
        *_call("summary1", input_tokens=50, cache_read_tokens=950, cache_write_tokens=10),
        {
            "type": "compact.summarise.failed",
            "writer": "worker",
            "error": "summary contains a tool call",
        },
        {
            "type": "compact.summarise.call",
            "writer": "helper",
            "model_call_id": "summary2",
            "fallback_reason": "summary contains a tool call",
        },
        *_call("summary2", role="summariser", input_tokens=300, cache_read_tokens=0),
        {
            "type": "compact.summarise.done",
            "writer": "helper",
            "summary_chars": 10,
            "fallback_reason": "summary contains a tool call",
        },
        *_call("ordinary", input_tokens=200, cache_read_tokens=800),
    ]
    _write(tmp_path / "logs.jsonl", rows)
    view = receipt.read_receipt(tmp_path, tmp_path)
    text = receipt.render(view)
    assert (
        "Summary attempt 1: worker; 60 uncached, 950 cached input tokens; "
        "rejected: summary contains a tool call" in text
    )
    assert (
        "Summary attempt 2: helper; 300 uncached, 0 cached input tokens; "
        "accepted; fallback: summary contains a tool call" in text
    )
    assert "Worker input cache (session): 80.0% (800/1,000 tokens)" in text
    assert "After compaction (session): 1 worker call, 200 uncached input tokens" in text
    assert len(receipt.as_dict(view)["cache_usage"]["summaries"]) == 2


def test_unreported_summary_usage_stays_unknown(tmp_path: pathlib.Path) -> None:
    _write(
        tmp_path / "logs.jsonl",
        [
            {"type": "compact.summarise.call", "writer": "worker", "model_call_id": "s"},
            *_call("s", input_tokens=10),
            {"type": "compact.summarise.done", "writer": "worker", "summary_chars": 10},
        ],
    )
    text = receipt.render(receipt.read_receipt(tmp_path, tmp_path))
    assert (
        "Summary attempt 1: worker; 10 uncached, unreported cached input tokens; accepted" in text
    )
    assert "Worker input cache" not in text


@pytest.mark.parametrize("details", [{}, {"cached_tokens": 0}])
def test_openai_usage_keeps_absent_cache_counts_absent(details: dict[str, int]) -> None:
    parsed = openai_parse.parse_response(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": details,
            },
        }
    )
    fields = role_providers._usage_fields(parsed.usage)
    assert (fields.get("cache_read_tokens") is not None) is bool(details)


def test_responses_usage_keeps_absent_cache_counts_absent() -> None:
    parsed = responses_codec.parse_output_items(
        [],
        usage={"input_tokens": 100, "output_tokens": 5},
        stop_reason="end_turn",
        label="Responses",
        route="test/test",
    )
    assert role_providers._usage_fields(parsed.usage).get("cache_read_tokens") is None


def test_cold_idle_notice_is_part_of_the_receipt(tmp_path: pathlib.Path) -> None:
    message = (
        "providers.p.idle_compact_s = 240 is longer than the cache lasts. "
        "Idle summary 241.0s after the last request began: 600 uncached, 0 cached input tokens."
    )
    session = layout.SessionLayout(state_dir=tmp_path, session_id="cold")
    session.session_dir.mkdir(parents=True)
    _write(
        session.session_dir / "logs.jsonl",
        [{"type": "compact.idle.cache_cold", "message": message}],
    )
    assert message in receipt.render(receipt.read_receipt(session.session_dir, tmp_path))
    lines: list[str] = []
    finalize.print_session_usage(session, emit=lines.append)
    assert not any(message in line for line in lines), "the live run already logged the notice"
