from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from bench.review import recordings

from agent6.prompts import review
from agent6.providers import provider_types
from agent6.sessions import completion, objects
from tests import review_replay_env as env


def test_only_finished_panel_children_supply_their_first_reviewer_request(
    tmp_path: pathlib.Path,
) -> None:
    """A helper, a later tool turn, a judge and a worker are not reviewer openings."""
    env.record_review(tmp_path, "run-rp1-l1", env.responses_body())
    judge = env.record_review(tmp_path, "run-p2-l1", {})
    completion.write_completion(
        judge.session_dir,
        completion.CompletionRecord(text="order", stop_reason="answered", calls=1, output_tokens=3),
    )
    unfinished = env.record_review(tmp_path, "run-rp3-l1", {})
    (unfinished.session_dir / completion.COMPLETION_NAME).unlink()
    (unfinished.session_dir.parent / "sample.txt").write_text("selected reviewers")
    (tmp_path / "sessions/runs/run/transcripts").mkdir(parents=True)
    (tmp_path / "sessions/runs/run/transcripts/worker.json").write_text("not a reviewer")

    (request,) = recordings.read_reviews(tmp_path)
    assert request.session_id == "run-rp1-l1"
    assert request.recorded == env.SYSTEM
    assert request.tree == review.panel_prompt("security", max_turns=7, commands=True)
    assert request.messages == [
        {"role": "user", "content": env.OPENING},
        {"role": "user", "content": env.NOTICE},
    ]
    assert request.tools == [
        provider_types.ToolDefinition("run_command", "Recorded command description.", env.SCHEMA)
    ]
    assert request.temperature == 0.25
    assert request.max_tokens == 18000


@pytest.mark.parametrize("wire", ["chat", "messages"])
def test_other_http_wires_keep_the_recorded_message_boundaries(
    tmp_path: pathlib.Path, wire: str
) -> None:
    """The tree uses this reviewer's focus and cap even when commands were not offered."""
    last = (
        "[harness] reply 1 of 1, the last: reply with your verdict now."
        " A tool call in this reply is not run."
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": t} for t in (env.OPENING, last)]
    tool = {"name": "read_file", "description": "Read the cited line."}
    if wire == "chat":
        body = {
            "messages": [{"role": "system", "content": env.SYSTEM}, *messages],
            "tools": [{"type": "function", "function": {**tool, "parameters": env.SCHEMA}}],
            "max_completion_tokens": 4321,
        }
    else:
        for msg in messages:
            msg["content"] = [
                {"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}
            ]
        body = {
            "system": [
                {"type": "text", "text": env.SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": messages,
            "tools": [{**tool, "input_schema": env.SCHEMA}],
            "max_tokens": 4321,
        }
    env.record_review(tmp_path, "run-rp1-l2", body, focus="callers: UI")
    (request,) = recordings.read_reviews(tmp_path)
    assert request.tree == review.panel_prompt("callers: UI", max_turns=1, commands=False)
    assert request.recorded == env.SYSTEM and request.messages == messages
    assert request.tools == [
        provider_types.ToolDefinition("read_file", tool["description"], env.SCHEMA)
    ]
    assert request.max_tokens == 4321


@pytest.mark.parametrize("limit", ["You have at most 4 replies", "You have 4 replies"])
def test_a_limit_in_the_system_prompt_needs_no_synthetic_reply_notice(
    tmp_path: pathlib.Path,
    limit: str,
) -> None:
    body = env.responses_body()
    body["input"].pop()
    body["instructions"] = f"{limit}, including tool calls."
    env.record_review(tmp_path, "run-rp1-l1", body)
    (request,) = recordings.read_reviews(tmp_path)
    assert request.messages == [{"role": "user", "content": env.OPENING}]
    assert request.tree == review.panel_prompt("security", max_turns=4, commands=True)


def test_a_reviewer_actor_without_a_focus_is_reported_not_missing(tmp_path: pathlib.Path) -> None:
    child = env.record_review(tmp_path, "no-focus", env.responses_body())
    for path in child.transcripts_dir.glob("*.json"):
        transcript = json.loads(path.read_text())
        if transcript["actor"].startswith("reviewer:"):
            transcript["actor"] = "reviewer"
            path.write_text(json.dumps(transcript))
    with pytest.raises(recordings.RecordingError, match="actor 'reviewer' has no recorded focus"):
        recordings.read_reviews(tmp_path)


@pytest.mark.parametrize("damage", ["object", "pruned", "tool history", "limit", "local process"])
def test_incomplete_openings_cannot_be_replaced_by_a_guess(
    tmp_path: pathlib.Path, damage: str
) -> None:
    body = env.responses_body()
    if damage == "tool history":
        body["input"].insert(1, {"type": "function_call_output", "output": "historical result"})
    elif damage == "limit":
        body["input"].pop()
    elif damage == "local process":
        body = {"stdin": [], "system": "old", "messages": []}
    child = env.record_review(tmp_path, "run-rp1-l1", body)
    path = sorted(child.transcripts_dir.glob("*.json"))[1]
    if damage == "object":
        digest = json.loads(path.read_text())["request"]["body"]["input"][0]["$object"]
        (child.objects_dir / f"{digest}.json").unlink()
    elif damage == "pruned":
        path.unlink()
    with pytest.raises((recordings.RecordingError, objects.ObjectStoreError), match="run-rp1-l1"):
        recordings.read_reviews(tmp_path)
