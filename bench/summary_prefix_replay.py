"""Rebuild the worker summary request at recorded ChatGPT tier-2 points without sending it."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import itertools
import json
import pathlib
import sys
from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

from agent6.harness import loop_conversation
from agent6.harness.compaction import calls, window, writers
from agent6.providers import chatgpt, provider_types, responses_codec, transport
from agent6.sessions import objects


class ReplayError(ValueError):
    """A recording is incomplete or the rebuilt prefix differs."""


type Body = dict[str, Any]


def _group(item: Body) -> str:
    if item["type"] == "function_call_output":
        return "results"
    if item.get("role") == "user":
        return "notices"
    if item["type"] in ("reasoning", "function_call", "message"):
        return "assistant"
    raise ReplayError(f"unsupported input item: {item['type']}")


def _symbol(source: Body) -> str:
    """Represent omitted bytes by their recorded identity, never by reading the source path."""
    if source.get("payload_omitted") is not True:
        raise ReplayError("expected omitted media")
    return base64.b64encode(json.dumps(source, sort_keys=True).encode()).decode("ascii")


def _images(items: list[Body]) -> Iterator[tuple[str, Body]]:
    for item in items:
        parts = item.get("content", ())
        for i, part in enumerate(parts):
            if part["type"] == "input_image":
                label = parts[i - 1]["text"]
                if not label.startswith("image from tool call "):
                    raise ReplayError("image has no tool call label")
                call_id, _, path = label.removeprefix("image from tool call ").partition(": ")
                if path != part["source"]["source_path"]:
                    raise ReplayError("image path differs from its label")
                yield call_id, part["source"]


def _result(item: Body, images: dict[str, Body]) -> Body:
    content = item["output"]
    if source := images.get(item["call_id"]):
        content = [
            {"type": "text", "text": content},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source["media_type"],
                    "data": _symbol(source),
                },
            },
        ]
    return {"type": "tool_result", "tool_use_id": item["call_id"], "content": content}


def _messages(items: list[Body], route: str) -> Iterator[Body]:
    images = dict(_images(items))
    for group, values in itertools.groupby(items, key=_group):
        if group == "assistant":
            parsed = responses_codec.parse_output_items(
                list(values), usage={}, stop_reason="end_turn", label="replay", route=route
            )
            yield {"role": "assistant", "content": parsed.raw["content"]}
        elif group == "results":
            yield {
                "role": "user",
                "content": [_result(item, images) for item in values],
            }
        else:
            for item in values:
                if any(p["type"] == "input_image" for p in item["content"]):
                    # The codec emits labelled media from the paired tool result.
                    continue
                if any(p["type"] != "input_text" for p in item["content"]):
                    raise ReplayError("non-text user input")
                yield {
                    "role": "user",
                    "content": [{"type": "text", "text": p["text"]} for p in item["content"]],
                }


def _last_exchange(reply: list[Body], following: Body) -> list[Body]:
    ids = [x["call_id"] for x in reply if x["type"] == "function_call"]
    if not ids:
        raise ReplayError("previous reply has no tool call")
    tail = following["input"]
    start = next(
        (
            i
            for i, x in enumerate(tail)
            if x["type"] == "function_call_output" and x["call_id"] == ids[0]
        ),
        None,
    )
    if start is None or [x.get("call_id") for x in tail[start : start + len(ids)]] != ids:
        raise ReplayError("retained tail is missing the last tool results")
    end = start + len(ids)
    notices: list[Body] = []
    for item in tail[end:]:
        if _group(item) != "notices":
            raise ReplayError("unexpected exchange after the last result")
        # The following worker request carries a fresh focus banner after the restart.
        if not item["content"][0]["text"].startswith("[harness focus]"):
            notices.append(item)
    return [*reply, *tail[start:end], *notices]


def build(
    before: Body,
    reply: list[Body],
    following: Body,
    *,
    task: str,
    open_tasks: Sequence[tuple[str, str]] = (),
    pins: Sequence[str] = (),
    focus: str = "",
    base_url: str = "https://chatgpt.com/backend-api/codex",
) -> Body:
    """Build the summary using the worker call builder and the ChatGPT adapter.

    Args:
        before: The last ordinary request before the summary.
        reply: Its recorded output items.
        following: The first worker request after the restart, including its retained tail.
        task: The recorded goal.
        open_tasks: Open subtasks at the summary point.
        pins: Operator pins at the summary point.
        focus: Any manual summary focus.
        base_url: The recorded route, used for reasoning replay only.

    Returns:
        The serialized request captured before transport or authentication.
    """
    model = following["model"]
    route = responses_codec.replay_route("chatgpt", base_url, model)
    messages = [
        *_messages(before["input"], route),
        *_messages(_last_exchange(reply, following), route),
    ]
    conversation = loop_conversation.Conversation.from_wire(messages)
    conversation.roll_cache_marks()
    tools = [
        provider_types.ToolDefinition(t["name"], t["description"], t["parameters"])
        for t in following["tools"]
    ]
    sink = mock.Mock()
    prepared = calls.Calls(
        sink,
        window.CompactionSettings(),
        conversation,
        following["instructions"],
        tools,
        None,
        "replay",
        None,
        lambda: None,
        lambda: False,
        lambda: False,
    )
    instruction = writers.instruction(
        task=task, open_tasks=list(open_tasks), pins=pins, focus=focus, messages=()
    )
    prepared.call_worker(instruction, "summary")
    kwargs = sink.call.call_args.kwargs
    kwargs["context"] = dataclasses.replace(
        kwargs["context"], conversation_id=following["prompt_cache_key"]
    )
    adapter = chatgpt.ChatGPTProvider(
        model=model,
        credential=mock.Mock(),
        account_id="",
        base_url=base_url,
        reasoning_effort=following["reasoning"]["effort"],
    )
    # No budget preflight, token lookup or HTTP request: capture the constructed body.
    with mock.patch.object(transport.ProviderCall, "run", autospec=True) as send:
        adapter.call(**kwargs)
    body = send.call_args.args[0].body
    final = body["input"][-1]
    if final != {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": instruction}],
    }:
        raise ReplayError("last input is not the summary instruction")
    if body["input"][:-1] != responses_codec.responses_input(messages, route=route):
        raise ReplayError("worker builder changed the prepared conversation")
    return body


def _reference(item: Body) -> Body:
    if not any(p["type"] == "input_image" for p in item.get("content", ())):
        return item
    parts = [
        {
            "type": "input_image",
            "image_url": f"data:{p['source']['media_type']};base64,{_symbol(p['source'])}",
        }
        if p["type"] == "input_image"
        else p
        for p in item["content"]
    ]
    return {**item, "content": parts}


def check_prefix(before: Body, candidate: Body) -> None:
    """Compare all recorded body fields and the ordered prefix of input items.

    Args:
        before: The recorded previous request.
        candidate: The rebuilt summary request.

    Raises:
        ReplayError: A field differs or the summary has no new input.
    """
    for key in sorted(before.keys() | candidate.keys()):
        if key == "input":
            n = len(before[key])
            if len(candidate[key]) <= n or candidate[key][:n] != [
                _reference(x) for x in before[key]
            ]:
                raise ReplayError("input prefix differs")
        elif key not in candidate or key not in before or candidate[key] != before[key]:
            raise ReplayError(f"{key} differs")


def replay(session: pathlib.Path) -> bool:
    """Compare requests adjacent to each journaled summary call.

    Args:
        session: A session with logs.jsonl, transcripts and objects.

    Returns:
        True when every recorded point matches.
    """
    frames: dict[str, pathlib.Path] = {}
    for path in sorted((session / "transcripts").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("seat") == "worker":
            frames[record["model_call_id"]] = path
    rows = [json.loads(line) for line in (session / "logs.jsonl").read_text().splitlines()]

    def worker(row: Body) -> bool:
        return row["type"] == "model.result" and row["role"] == "worker" and row["ok"]

    def frame(row: Body) -> Body:
        path = frames[row["model_call_id"]]
        return objects.materialize(
            json.loads(path.read_text()), session / "objects", record_path=path
        )

    task = ""
    focus = ""
    pins: list[str] = []
    nodes: dict[str, Body] = {}
    ok = True
    for i, row in enumerate(rows):
        match row["type"]:
            case "session.start":
                task = row["user_task"]
            case "graph.update":
                nodes = row["nodes"]
            case "compact.requested":
                focus = row["focus"]
            case "pin.added":
                pins.append(row["text"])
            case "pin.restored":
                pins = row["pins"]
            case "compact.summarise.call":
                try:
                    prev = frame(next(r for r in reversed(rows[:i]) if worker(r)))
                    after = frame(next(r for r in rows[i + 1 :] if worker(r)))
                    before = prev["request"]["body"]
                    base_url = after["request"]["url"].removesuffix("/responses")
                    if prev["request"]["url"] != after["request"]["url"]:
                        raise ReplayError("worker route differs")
                    tasks = [
                        (k, v["title"][:120])
                        for k, v in nodes.items()
                        if v["parent_id"]
                        and not v["standing"]
                        and v["status"] in ("pending", "in_progress")
                    ]
                    candidate = build(
                        before,
                        prev["response"]["body"]["output"],
                        after["request"]["body"],
                        task=task,
                        open_tasks=tasks,
                        pins=pins,
                        focus=focus,
                        base_url=base_url,
                    )
                    check_prefix(before, candidate)
                    result = f"{len(before['input'])} items match"
                    if omitted := len(list(_images(before["input"]))):
                        result += (
                            f" on recorded fields; full prefix null ({omitted} images omitted)"
                        )
                except (
                    ReplayError,
                    ValueError,
                    objects.ObjectStoreError,
                    OSError,
                    KeyError,
                    StopIteration,
                ) as exc:
                    result = f"unavailable or different: {exc}"
                    ok = False
                print(f"| {session.name} | {row['ts'][11:19]} | {result} |")
                focus = ""
            case _:
                pass
    return ok


def main() -> int:
    """Report the recorded points without making any model calls.

    Returns:
        Zero for a complete comparison, one for a mismatch or unavailable recording.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=pathlib.Path)
    sessions = sorted(d for d in parser.parse_args().recordings.iterdir() if d.is_dir())
    if not sessions:
        parser.error("no recorded sessions")
    print("| Session | Summary time (UTC) | Previous request prefix |\n| --- | --- | --- |")
    passed = True
    for session in sessions:
        try:
            passed = replay(session) and passed
        except (OSError, KeyError, ValueError) as exc:
            print(f"{session.name}: {exc}", file=sys.stderr)
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
