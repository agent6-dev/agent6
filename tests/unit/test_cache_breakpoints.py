# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The provider-side cache_control strip (prompt_caching = false).

The rolling breakpoint semantics live on Conversation.roll_cache_marks and
are covered in test_conversation.py; this pins the anthropic provider's
copy-on-write strip of the marks the conversation stamped into the wire.
"""

from __future__ import annotations

from typing import Any

from agent6.providers.anthropic import strip_cache_control_messages
from agent6.workflows._conversation import Conversation


def _marked_wire() -> list[dict[str, Any]]:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()
    return conv.to_wire()


def test_strip_cache_control_is_copy_on_write() -> None:
    messages = _marked_wire()
    stripped = strip_cache_control_messages(messages)
    assert stripped is not messages
    assert "cache_control" not in stripped[0]["content"][0]
    # The original (loop-owned, snapshot-shared) list keeps its marker.
    assert "cache_control" in messages[0]["content"][0]


def test_strip_cache_control_passthrough_when_unmarked() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "TASK"}]},
        {"role": "user", "content": "plain"},
    ]
    assert strip_cache_control_messages(messages) is messages
