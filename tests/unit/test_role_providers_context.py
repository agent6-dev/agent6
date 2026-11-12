from __future__ import annotations

import contextlib
import json
import pathlib
from collections.abc import Generator
from typing import Any

import httpx2
import pytest

from agent6 import budget
from agent6.app import conversation_state, role_providers
from agent6.app.session import runtime as session_runtime
from agent6.config import ReviewerConfig
from agent6.config import model as config_model
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel
from agent6.providers import provider_types, stream_lifecycle, transport
from agent6.sessions import event_log
from tests import review_lane_env, review_traces, subordinate_env


def _session(
    tmp_path: pathlib.Path, *, streaming: bool = False
) -> session_runtime.SessionProviders:
    cfg = config_model.Config.model_validate(
        {
            "providers": {
                "work": {
                    "backend": "openai",
                    "auth_style": "none",
                    "extra_headers": {
                        "x-conversation": {"template": "{conversation_id}"},
                        "x-first": {"template": "{is_first_turn}"},
                    },
                }
            },
            "models": {"default": {"model": "work/shared"}},
            "review": {
                "when": "finish",
                "reviewers": [{"focus": "security"}, {"focus": "security"}],
            },
        }
    )
    providers = session_runtime.build_session_providers(
        cfg,
        role="worker",
        events=event_log.EventSink(tmp_path / "logs.jsonl"),
        transcript_sink=provider_types.TranscriptSink(
            tmp_path / "transcripts", objects_dir=tmp_path / "objects"
        ),
        stream_text=streaming,
        conversations=conversation_state.Conversations(
            path=tmp_path / conversation_state.FILE_NAME
        ),
    )
    return providers


def _mock_http(monkeypatch: pytest.MonkeyPatch, *, reject_temperature: bool = False) -> None:
    def response(content: bytes, *, streaming: bool) -> httpx2.Response:
        body = json.loads(content)
        if reject_temperature and "temperature" in body:
            return httpx2.Response(400, json={"error": "temperature is unsupported"})
        text = '{"verdict":"pass","findings":[],"summary":"ok"}'
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        if streaming:
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": usage},
            ]
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx2.Response(200, text=data + "data: [DONE]\n\n")
        return httpx2.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    def post(_url: str, *, content: bytes, **_kwargs: Any) -> httpx2.Response:
        return response(content, streaming=False)

    @contextlib.contextmanager
    def stream(
        _method: str, _url: str, *, content: bytes, **_kwargs: Any
    ) -> Generator[httpx2.Response]:
        yield response(content, streaming=True)

    monkeypatch.setattr(transport, "http_post", post)
    monkeypatch.setattr(stream_lifecycle, "http_stream", stream)


def _events(tmp_path: pathlib.Path, kind: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (tmp_path / "logs.jsonl").read_text().splitlines()]
    return [row for row in rows if row["type"] == kind]


def _transcripts(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in (tmp_path / "transcripts").glob("*.json")]


def test_review_requests_belong_to_real_distinct_seat_sessions(
    short_repo: pathlib.Path,
) -> None:
    """Duplicate-focus seats have distinct sessions, conversations and bound call transcripts."""
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    with review_traces.server("pass") as (url, calls):
        cfg = subordinate_env.config(url)
        cfg = cfg.model_copy(
            update={
                "review": cfg.review.model_copy(
                    update={
                        "reviewers": tuple(ReviewerConfig(focus="security") for _ in range(2)),
                    }
                )
            }
        )
        executor = subordinate_env.dispatcher(
            repo,
            directory,
            cfg,
            budget.BudgetTracker(max_usd=10, max_tokens_fallback=10000, max_plan_points=20),
        )
        target = executor.chain.review_target()
        assert target is not None
        target = executor.chain.retain_review_target(target, directory.name)
        seats = role_providers.resolve_review_seats(cfg)
        result = review_gate.run_panel(
            seats,
            review_panel.ReviewContext(task="review", diff=""),
            target=target,
            executor=executor,
            block_on="none",
            min_blocking_models=2,
            panel_id="panel-a",
            max_turns=2,
            tools=review_lane_env.READING_TOOLS,
            concurrency=2,
        )
        assert [s.verdict for s in result.per_seat] == ["pass", "pass"], result
        assert len(calls) == 2
        child_dirs = [directory.parent.parent / "read_only" / s.session_id for s in result.per_seat]
        requests = [r for child in child_dirs for r in _events(child, "provider.request")]
        logical = [r for child in child_dirs for r in _events(child, "model.call")]
        transcripts = [r for child in child_dirs for r in _transcripts(child)]
        assert len(requests) == len(logical) == len(transcripts) == 2
        assert {r["seat_index"] for r in requests} == {0, 1}
        assert {r["panel_id"] for r in requests} == {"panel-a"}
        assert len({r["model_call_id"] for r in requests}) == 2
        by_id = {r["request_id"]: r for r in requests}
        for transcript in transcripts:
            request = by_id[transcript["request_id"]]
            assert transcript["model_call_id"] == request["model_call_id"]
        assert len({r["execution_id"] for r in logical}) == 2
        # A repeated identity never buys replacement children or another grant.
        repeated = review_gate.run_panel(
            seats,
            review_panel.ReviewContext(task="review", diff=""),
            target=target,
            executor=executor,
            block_on="none",
            min_blocking_models=2,
            panel_id="panel-a",
            max_turns=2,
            tools=review_lane_env.READING_TOOLS,
            concurrency=2,
        )
        assert repeated.n_abstain == 2 and len(calls) == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_request_and_transcript_carry_call_id_without_seat_fields(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The ordinary worker uses the same join without inventing a review identity."""
    _mock_http(monkeypatch)
    providers = _session(tmp_path, streaming=streaming)
    providers.provider.call(system="s", messages=[])
    (call,) = _events(tmp_path, "model.call")
    (request,) = _events(tmp_path, "provider.request")
    (transcript,) = _transcripts(tmp_path)
    assert request.get("model_call_id") == call["model_call_id"]
    assert transcript["model_call_id"] == call["model_call_id"]
    assert transcript["request_id"] == request["request_id"]
    assert "panel_id" not in request and "seat_index" not in request
    providers.close()
