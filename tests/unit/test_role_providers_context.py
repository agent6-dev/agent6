from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Generator
from typing import Any

import httpx2
import pytest

from agent6.app import conversation_state, role_providers, session_runtime
from agent6.config import model as config_model
from agent6.harness import review_gate, review_panel
from agent6.providers import provider_types, stream_lifecycle, transport
from agent6.sessions import event_log
from agent6.tools import results


def _session(
    tmp_path: pathlib.Path, *, scope: str = "diff", streaming: bool = False
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
                "trigger": "before_finish",
                "scope": scope,
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
    for index, seat in enumerate(providers.review_seats):
        assert isinstance(seat.provider, role_providers.InstrumentedProvider)
        providers.review_seats[index] = dataclasses.replace(
            seat, provider=dataclasses.replace(seat.provider, stream_text=streaming)
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


def _panel(providers: session_runtime.SessionProviders, panel_id: str) -> None:
    def dispatch(_name: str, _args: dict[str, Any]) -> results.ToolResult:
        raise AssertionError("the response requests no tools")

    result = review_gate.run_panel(
        providers.review_seats,
        review_panel.ReviewContext(task="review", diff=""),
        block_on="none",
        min_blocking_models=2,
        panel_id=panel_id,
        repo_max_turns=2,
        concurrency=2,
        tools=[],
        dispatch=dispatch,
    )
    assert [seat.verdict for seat in result.per_seat] == ["pass", "pass"]
    assert result.panel_id == panel_id


def _events(tmp_path: pathlib.Path, kind: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (tmp_path / "logs.jsonl").read_text().splitlines()]
    return [row for row in rows if row["type"] == kind]


def _transcripts(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in (tmp_path / "transcripts").glob("*.json")]


def _assert_joins(tmp_path: pathlib.Path, *, attempts: int) -> None:
    calls = _events(tmp_path, "role.call")
    requests = _events(tmp_path, "provider.request")
    transcripts = _transcripts(tmp_path)
    assert len(calls) == 2
    assert len(requests) == len(transcripts) == attempts
    assert len({call["model_call_id"] for call in calls}) == 2
    assert {row.get("model_call_id") for row in requests} == {
        call["model_call_id"] for call in calls
    }
    by_request = {row["request_id"]: row for row in requests}
    assert len(by_request) == attempts
    for transcript in transcripts:
        request = by_request[transcript["request_id"]]
        assert transcript["model_call_id"] == request["model_call_id"]
        call = next(row for row in calls if row["model_call_id"] == request["model_call_id"])
        assert request["panel_id"] == "panel-a"
        assert request["seat_index"] == call["seat_index"]
    assert {row["seat_index"] for row in requests} == {0, 1}


@pytest.mark.parametrize("scope", ["diff", "repo"])
@pytest.mark.parametrize("streaming", [False, True])
def test_review_requests_join_calls_and_transcripts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, scope: str, streaming: bool
) -> None:
    """Identical seats remain addressable by call and request ids, not their shared label."""
    _mock_http(monkeypatch)
    providers = _session(tmp_path, scope=scope, streaming=streaming)
    _panel(providers, "panel-a")
    _assert_joins(tmp_path, attempts=2)
    providers.close()


@pytest.mark.parametrize("scope", ["diff", "repo"])
def test_review_conversations_belong_to_panel_seats(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    """Same-focus seats are separate; repeating a panel continues only its own seats."""
    _mock_http(monkeypatch)
    providers = _session(tmp_path, scope=scope)
    _panel(providers, "panel-a")
    first = _transcripts(tmp_path)
    conversations = {row["request"]["headers"]["x-conversation"] for row in first}
    assert len(conversations) == 2
    assert {row["request"]["headers"]["x-first"] for row in first} == {"true"}
    providers.close()

    resumed = _session(tmp_path, scope=scope)
    _panel(resumed, "panel-a")
    repeated = [row for row in _transcripts(tmp_path) if row["seq"] > 2]
    assert {row["request"]["headers"]["x-conversation"] for row in repeated} == conversations
    assert {row["request"]["headers"]["x-first"] for row in repeated} == {"false"}
    _panel(resumed, "panel-b")
    other = [row for row in _transcripts(tmp_path) if row["seq"] > 4]
    assert len({row["request"]["headers"]["x-conversation"] for row in other} - conversations) == 2
    assert {row["request"]["headers"]["x-first"] for row in other} == {"true"}
    resumed.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_retried_requests_keep_the_call_id_and_get_distinct_transcript_ids(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """A body adaptation keeps its parent call but records each HTTP attempt separately."""
    _mock_http(monkeypatch, reject_temperature=True)
    providers = _session(tmp_path, streaming=streaming)
    _panel(providers, "panel-a")
    _assert_joins(tmp_path, attempts=4)
    assert sorted(row["response"]["status"] for row in _transcripts(tmp_path)) == [
        200,
        200,
        400,
        400,
    ]
    providers.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_request_and_transcript_carry_call_id_without_seat_fields(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The ordinary worker uses the same join without inventing a review identity."""
    _mock_http(monkeypatch)
    providers = _session(tmp_path, streaming=streaming)
    providers.provider.call(system="s", messages=[])
    (call,) = _events(tmp_path, "role.call")
    (request,) = _events(tmp_path, "provider.request")
    (transcript,) = _transcripts(tmp_path)
    assert request.get("model_call_id") == call["model_call_id"]
    assert transcript["model_call_id"] == call["model_call_id"]
    assert transcript["request_id"] == request["request_id"]
    assert "panel_id" not in request and "seat_index" not in request
    providers.close()
