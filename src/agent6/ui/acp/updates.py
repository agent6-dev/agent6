# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project the shared transcript fold into ACP `session/update` notifications.

The fold (`viewmodel.transcript`) is what the CLI, the TUI and the web already
render. Projecting it -- rather than reading the journal again with ACP's own
rules -- is what keeps a fourth surface from disagreeing with the other three
about what happened in a run.

Pure: events in, notification bodies out. Nothing here touches the wire, so a
test can assert the exact JSON an editor would receive.
"""

from __future__ import annotations

from typing import Any

from agent6.viewmodel.transcript import TranscriptFold, TranscriptItem

# Which ACP update a fold item becomes. `thinking` is the model's reasoning and
# ACP has a distinct channel for it; an editor renders it collapsed rather than
# as the answer. `operator` is the human's own words -- a steer, or the
# follow-up a resume began with -- so it echoes back as a user message, not as
# something the agent said.
_CHUNK_KIND = {
    "thinking": "agent_thought_chunk",
    "text": "agent_message_chunk",
    "operator": "user_message_chunk",
    # Harness prose: a compaction, a btw answer, an operator notice. Not the
    # model speaking, but it IS what the run said.
    "marker": "agent_message_chunk",
}


def updates_for(item: TranscriptItem, *, session_id: str) -> list[dict[str, Any]]:
    """The `session/update` notifications one fold item becomes.

    A tool becomes TWO: the call, then its outcome. ACP models a tool call as a
    thing with a lifecycle, and an editor that only ever sees the finished one
    cannot show work in progress -- which for a long verify is the whole point.
    """
    if item.kind == "done":
        return [
            _update(
                session_id,
                {"sessionUpdate": "agent_message_chunk", "content": _text(_ending(item))},
            )
        ]
    if item.kind == "commit":
        # `body` is empty on a commit; the sha and the line count live in
        # `detail`. Keying on body alone dropped every auto-commit.
        text = " ".join(part for part in ("committed", item.arg, item.detail) if part)
        return [
            _update(session_id, {"sessionUpdate": "agent_message_chunk", "content": _text(text)})
        ]
    if item.kind == "tool":
        return [
            _update(session_id, {"sessionUpdate": "tool_call", **_tool_call(item)}),
            _update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": _tool_id(item),
                    "status": "completed" if item.ok is not False else "failed",
                    **({"content": [_text(item.detail)]} if item.detail else {}),
                },
            ),
        ]
    chunk = _CHUNK_KIND.get(item.kind)
    body = item.body.strip()
    if chunk is None or not body:
        return []
    return [_update(session_id, {"sessionUpdate": chunk, "content": _text(body)})]


def _ending(item: TranscriptItem) -> str:
    """How a run ended, in words.

    The fold sets `body` only for a clean `finish_run`, carrying everything
    else in `ok`/`name`/`detail`. Reading `body` alone made a provider error, a
    budget stop and an iteration cap render as SILENCE -- an editor watching a
    run that simply stops -- and made a finish over a red gate look identical
    to a green one.
    """
    verdict = "passed" if item.ok else "did not pass"
    parts = [f"Run {verdict}"]
    if item.name:
        parts.append(f"({item.name})")
    if item.detail:
        parts.append(f"- {item.detail}")
    ending = " ".join(parts)
    return f"{item.body}\n\n{ending}" if item.body.strip() else ending


def updates_for_events(events: list[dict[str, Any]], *, session_id: str) -> list[dict[str, Any]]:
    """Every notification a run's events produce, in order.

    One fold instance across the whole sequence, because the fold is stateful:
    deltas accumulate and flush at a turn boundary. Feeding events to fresh
    folds would emit each partial message as if it were whole.
    """
    fold = TranscriptFold()
    out: list[dict[str, Any]] = []
    for event in events:
        for item in fold.feed(event):
            out.extend(updates_for(item, session_id=session_id))
    return out


def _update(session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_id(item: TranscriptItem) -> str:
    """The provider's stamped call id, so every call is its own entity.

    Reconstructing one from name+arg made two identical calls share an id, and
    an editor keyed on it (ACP models a tool call as one thing with a
    lifecycle) overwrote the first call's FAILURE with the second's success --
    the red run vanished from the editor's view. The fall-back is for historical
    events with no stamped id.
    """
    return item.call_id or (f"{item.name}:{item.arg}" if item.arg else item.name)


def _tool_call(item: TranscriptItem) -> dict[str, Any]:
    title = f"{item.name} {item.arg}".strip()
    return {
        "toolCallId": _tool_id(item),
        "title": title,
        # ACP's `kind` drives the editor's icon. agent6's own tool names are
        # the honest source; guessing a finer category from them would be a
        # second vocabulary to keep in sync.
        "kind": "other",
        "status": "pending",
    }
