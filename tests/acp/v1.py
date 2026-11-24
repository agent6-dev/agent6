# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render the projection as the frames a v1 editor receives, for tests that read frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from agent6.ui.acp import model, updates, wire
from agent6.viewmodel import transcript_items


def frames(acp_session_id: str, many: Iterable[model.Update]) -> list[dict[str, Any]]:
    """Return the `session/update` frames v1 sends for the updates, in order."""
    return [
        wire.notification(acp_session_id, body)
        for update in many
        if (body := wire.v1(update)) is not None
    ]


def updates_for(
    item: transcript_items.TranscriptItem, *, acp_session_id: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Return the frames v1 sends for one fold item."""
    return frames(acp_session_id, updates.updates_for(item, **kwargs))


def collector(acp_session_id: str, sent: list[dict[str, Any]]) -> Callable[[model.Update], None]:
    """Return a replay's `notify` that appends each update's v1 frame to `sent`."""
    return lambda update: sent.extend(frames(acp_session_id, [update]))
