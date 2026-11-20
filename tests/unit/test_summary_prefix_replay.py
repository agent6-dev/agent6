"""Pin the offline prefix check to the recorded request and intervening exchange."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from bench import summary_prefix_replay as replay


def point() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    task = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "goal"}]}
    first: dict[str, Any] = {
        "model": "model",
        "instructions": "original system",
        "prompt_cache_key": "recorded",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "read",
                "parameters": {},
                "strict": False,
            }
        ],
        "reasoning": {"effort": "low", "summary": "auto"},
        "input": [task],
        "stream": True,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    reply = [
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "function_call", "call_id": "x", "name": "read_file", "arguments": "{}"},
    ]
    following = copy.deepcopy(first)
    following["input"] = [
        task,
        *reply,
        {"type": "function_call_output", "call_id": "x", "output": "last read"},
    ]
    return first, reply, following


def test_replay_includes_the_unseen_reply_and_result_before_the_instruction() -> None:
    before, reply, after = point()
    got = replay.build(before, reply, after, task="goal")
    assert got["input"][:-1] == after["input"]
    assert got["input"][-1]["content"][0]["text"].startswith("Write the summary now.")
    replay.check_prefix(before, got)


@pytest.mark.parametrize("key", ["instructions", "input", "tools", "reasoning", "prompt_cache_key"])
def test_a_changed_prefix_fails_the_replay(key: str) -> None:
    before, reply, after = point()
    got = replay.build(before, reply, after, task="goal")
    if key == "input":
        got[key].pop(0)
    else:
        got[key] = None
    with pytest.raises(replay.ReplayError, match=key):
        replay.check_prefix(before, got)


def test_omitted_images_keep_their_position_and_identity_not_their_bytes() -> None:
    before, reply, after = point()
    ref = {
        "source_path": "p.png",
        "media_type": "image/png",
        "sha256": "aa" * 32,
        "byte_size": 42,
        "payload_omitted": True,
    }
    before["input"] += [
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "p.png"}',
            "call_id": "img",
        },
        {"type": "function_call_output", "call_id": "img", "output": "attached"},
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "image from tool call img: p.png"},
                {"type": "input_image", "source": ref},
            ],
        },
    ]
    got = replay.build(before, reply, after, task="goal")
    replay.check_prefix(before, got)
    got["input"][3]["content"].pop()
    with pytest.raises(replay.ReplayError, match="input"):
        replay.check_prefix(before, got)
