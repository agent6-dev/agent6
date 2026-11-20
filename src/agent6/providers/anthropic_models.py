"""Model support for signed compaction on Anthropic Messages."""

from __future__ import annotations

import re

_PREFIX_CHECK_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5-2",
        "claude-haiku-5-5",
        "claude-mythos-5-1",
        "claude-mythos-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    }
)


def has_signed_compaction(model: str) -> bool:
    """Check a model id against the signed compaction table.

    Args:
        model: The id with or without a direct or Vertex release date.

    Returns:
        Whether the model supports signed summaries and checks thinking prefixes.
    """
    return re.sub(r"[-@][0-9]{8}$", "", model) in _PREFIX_CHECK_MODELS
