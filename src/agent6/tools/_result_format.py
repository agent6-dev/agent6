# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Post-dispatch helpers: the jail passthrough env, metric parsing and argument previews."""

from __future__ import annotations

import os
import re
from typing import Any

PASSTHROUGH_ENV_KEYS = ("LANG", "LC_ALL", "TERM", "CI")


def passthrough_env() -> dict[str, str]:
    """Return the environment variables a jailed command inherits from the agent."""
    return {k: os.environ[k] for k in PASSTHROUGH_ENV_KEYS if k in os.environ}


def parse_metric_score(stdout: str, stderr: str, *, pattern: str) -> float | None:
    """Apply the metric regex to the combined output and read its first capture group.

    The harness and the tool handler score the same command output through this one parser.

    Args:
        stdout: The command's stdout.
        stderr: The command's stderr.
        pattern: The regex whose first group is the score.

    Returns:
        The score, or None when the regex does not compile, does not match, or captures a
        non-number; the caller treats that as no score this turn.
    """
    combined = f"{stdout}\n{stderr}"
    try:
        m = re.search(pattern, combined)
    except re.error:
        return None
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError, TypeError):
        # TypeError: a capture group that did not take part in the match yields None.
        return None


def truncate_args(raw: dict[str, Any], *, max_value_chars: int = 200) -> dict[str, Any]:
    """Return an argument preview for telemetry, clipped at every depth.

    Strings longer than `max_value_chars` and lists longer than 10 items are clipped (an
    apply_edit's `edits` is a short list of dicts whose strings hold whole files).

    Args:
        raw: The tool call's arguments.
        max_value_chars: The longest string kept whole.

    Returns:
        The clipped arguments.
    """
    return {k: _clip_value(v, max_value_chars) for k, v in raw.items()}


def _clip_value(v: Any, max_value_chars: int) -> Any:
    if isinstance(v, str):
        return v if len(v) <= max_value_chars else v[:max_value_chars] + f"… ({len(v)} chars)"
    if isinstance(v, dict):
        return {k: _clip_value(x, max_value_chars) for k, x in v.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(v, list | tuple):
        items = [_clip_value(x, max_value_chars) for x in v[:10]]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return [*items, f"… ({len(v)} items)"] if len(v) > 10 else items
    return v
