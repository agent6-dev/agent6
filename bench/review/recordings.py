"""Read the first provider request of each finished panel reviewer."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from typing import Any

from agent6.prompts import review
from agent6.providers import provider_types
from agent6.sessions import completion, layout, objects


class RecordingError(ValueError):
    """The recording cannot supply a reviewer opening."""


@dataclasses.dataclass(frozen=True, slots=True)
class Opening:
    """Hold both system prompts and the inputs shared by their calls."""

    session_id: str
    transcript: str
    recorded: str
    tree: str
    messages: list[dict[str, Any]]
    tools: list[provider_types.ToolDefinition]
    max_tokens: int | None
    temperature: float | None


def _messages(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Decode the three HTTP wires without accepting tool history or omitted media."""
    match body:
        case {"instructions": str(system), "input": list(items)}:
            messages: list[dict[str, Any]] = []
            for item in items:
                match item:
                    case {"role": "user", "content": [{"type": "input_text", "text": str(text)}]}:
                        messages.append({"role": "user", "content": text})
                    case _:
                        raise RecordingError("expected a Responses user text item")
        case {"messages": [{"role": "system", "content": str(system)}, *messages]}:
            pass
        case {"system": [{"type": "text", "text": str(system)}], "messages": list(messages)}:
            pass
        case _:
            raise RecordingError("expected an agent6 HTTP reviewer request")
    for message in messages:
        match message:
            case {"role": "user", "content": str()}:
                pass
            case {"role": "user", "content": list(blocks)} if blocks and all(
                isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
                for b in blocks
            ):
                pass
            case _:
                raise RecordingError("expected user text, not history or media")
    if not system or not messages:
        raise RecordingError("missing reviewer system prompt or opening")
    return system, messages


def _tool(definition: object) -> provider_types.ToolDefinition:
    """Recover the function schema the reviewer saw, not today's tool schema."""
    if isinstance(definition, dict) and definition.get("type") == "function":
        definition = definition.get("function", definition)
    match definition:
        case {"name": str(name), "description": str(description), "parameters": dict(schema)} | {
            "name": str(name),
            "description": str(description),
            "input_schema": dict(schema),
        }:
            return provider_types.ToolDefinition(name, description, schema)
        case _:
            raise RecordingError("expected a function tool definition")


def _opening(path: pathlib.Path, body: dict[str, Any], focus: str) -> Opening:
    system, messages = _messages(body)
    content = messages[-1]["content"]
    text = content if isinstance(content, str) else content[-1]["text"]
    limit = (
        re.match(r"\[harness\] reply 1 of ([1-9]\d*)[.,]", text)
        if len(messages) > 1
        else re.search(r"\bYou have at most ([1-9]\d*) replies\b", system)
    )
    if limit is None:
        raise RecordingError("missing recorded reply limit for the first turn")
    tools = [_tool(t) for t in body.get("tools", [])]
    tree = review.panel_prompt(
        focus, max_turns=int(limit[1]), commands=any(t.name == "run_command" for t in tools)
    )
    cap = body.get("max_output_tokens", body.get("max_completion_tokens", body.get("max_tokens")))
    if cap is not None and (type(cap) is not int or cap < 1):
        raise RecordingError("expected a positive output token cap")
    temperature = body.get("temperature")
    if temperature is not None and type(temperature) not in (float, int):
        raise RecordingError("expected a numeric temperature")
    return Opening(
        path.parent.parent.name, path.name, system, tree, messages, tools, cap, temperature
    )


def _sequence(path: pathlib.Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError as exc:
        raise RecordingError(f"{path}: expected a transcript sequence number") from exc


def _first(child: layout.SessionLayout) -> Opening:
    paths = sorted(child.transcripts_dir.glob("*.json"), key=_sequence)
    for seq, path in enumerate(paths, 1):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["seq"] != seq:
                raise RecordingError(f"missing transcript sequence {seq}")
            actor = record["actor"]
            if isinstance(actor, str) and actor.startswith("reviewer:"):
                body = objects.materialize(
                    record["request"]["body"], child.objects_dir, record_path=path
                )
                return _opening(path, body, actor.removeprefix("reviewer:"))
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordingError(f"{path}: {exc}") from exc
    raise RecordingError(f"{child.session_dir}: no recorded first reviewer request")


def read_reviews(state_dir: pathlib.Path) -> list[Opening]:
    """Read all openings before spending on the first pair.

    Args:
        state_dir: A repository's state directory containing `sessions/reviewers`.

    Returns:
        One opening per child with a bound panel completion, in session id order.

    Raises:
        RecordingError: No panel children were found or an opening is incomplete.
    """
    bucket = layout.bucket_dir(state_dir, layout.EVIDENCE_BUCKET)
    reviews: list[Opening] = []
    for directory in sorted(bucket.glob("*")):
        if not directory.is_dir():
            continue
        record = completion.read_completion(directory)
        if record is None or record.review_binding is None:
            continue
        if record.session_id != directory.name:
            raise RecordingError(f"{directory}: completion belongs to {record.session_id!r}")
        reviews.append(_first(layout.layout_of(directory)))
    if not reviews:
        raise RecordingError(f"{bucket}: no finished panel reviewers")
    return reviews
