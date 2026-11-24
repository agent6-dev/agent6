"""Count the journal text each keep arm retains. No checkout or provider is used."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any

from bench import compaction_arms as arms
from bench import compaction_replay

from agent6.harness import loop_conversation as history
from agent6.harness.compaction import retention, window

type Row = dict[str, Any]


def _project_turn(rows: list[Row]) -> Iterator[history.Message]:
    """Project one journal turn onto operator notices, settled text and paired tool previews."""
    calls = {r.get("call_id"): r for r in rows if r.get("type") == "tool.call"}
    blocks: list[Row] = []
    results: list[history.ToolResultItem] = []
    for r in rows:
        match r.get("type"):
            case "session.start" | "steer.injected" | "session.resume":
                key = {"session.start": "user_task", "session.resume": "steer"}.get(
                    r["type"], "text"
                )
                if text := r.get(key):
                    yield history.UserMessage((history.Notice(text, "operator"),))
            case "question.answer" if r.get("answers"):
                yield history.UserMessage((history.Notice("\n".join(r["answers"]), "operator"),))
            case "model.result" if r.get("actor") == "worker" and r.get("text"):
                blocks.append({"type": "text", "text": r["text"]})
            case "tool.result":
                call = calls.get(r.get("call_id"), {})
                args = call.get("args_preview", {})
                name = r["name"]
                uid = str(r.get("provider_tool_call_id") or r.get("call_id", len(results)))
                blocks.append({"type": "tool_use", "name": name, "id": uid, "input": args})
                text = "\n".join(
                    r[k] for k in ("summary", "head_tail", "stdout_tail", "stderr_tail") if r.get(k)
                )
                results.append(history.ToolResultItem(uid, text, history.ToolUse(uid, name, args)))
            case _:
                pass
    if blocks:
        yield history.AssistantMessage(tuple(blocks), tuple(item.for_call for item in results))
    if results:
        yield history.UserMessage(tuple(results))


def projected(rows: Iterable[Row]) -> Iterator[history.Message]:
    turn: list[Row] = []
    for r in rows:
        if r.get("type") == "budget.turn":
            yield from _project_turn(turn)
            turn.clear()
        else:
            turn.append(r)
    yield from _project_turn(turn)


def sizes(messages: tuple[history.Message, ...]) -> Iterator[Row]:
    """Separate retained source text from replacement characters."""
    source = sum(map(window.message_chars, messages))
    for arm, caps in arms.KEEP.items():
        settings = window.CompactionSettings(
            keep_recent_chars=caps.chars,
            keep_recent_turns=caps.turns,
            keep_operator_messages=caps.operator,
        )
        kept = retention.select(messages, settings)
        quotes = sum(map(len, kept.quotes))
        original = quotes + sum(map(window.message_chars, kept.tail))
        pointers = 0
        if arm == "pointers":
            outputs = [
                it
                for m in kept.tail
                if isinstance(m, history.UserMessage)
                for it in m.items
                if isinstance(it, history.ToolResultItem) and it.for_call.name != "ask_user"
            ]
            original -= sum(len(it.content) for it in outputs)
            pointers = (
                sum(map(window.message_chars, arms.pointer_tail(messages, settings).tail))
                + quotes
                - original
            )
        yield dict(
            arm=arm,
            basis="journal_previews",
            source_chars=source,
            kept_chars=original,
            dropped_chars=source - original,
            quote_chars=quotes,
            pointer_chars=pointers,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=pathlib.Path, help="recorded logs.jsonl")
    parser.add_argument("--turn", type=int, help="stop before this unique budget.turn")
    args = parser.parse_args()
    try:
        if args.turn is None:
            with args.journal.open("rb") as stream:
                messages = tuple(projected(json.loads(line) for line in stream))
        else:
            prefix = compaction_replay.journal_prefix(args.journal, args.turn)
            messages = tuple(projected(json.loads(line) for line in prefix.splitlines()))
        if not messages:
            raise compaction_replay.ReplayError("no countable journal text")
        for row in sizes(messages):
            print(json.dumps(row))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
