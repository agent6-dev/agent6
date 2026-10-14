# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Post-dispatch helpers: the jail env, metric parsing, argument previews and result shaping."""

from __future__ import annotations

import os
import re
from typing import Any

from agent6 import kinds
from agent6.tools import results, schema

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


# Tools whose tool.result event carries a capped output tail, so logs.jsonl shows the output.
_EXEC_OUTPUT_TOOLS = frozenset({schema.RunCommandInput.TOOL_NAME, schema.RunMetricInput.TOOL_NAME})
_TOOL_OUTPUT_TAIL = 2000  # chars, matching verify.end's stdout_tail/stderr_tail

_READ_HEAD_LINES = 6
_READ_HEAD_CHARS = 300


def output_tails(name: str, result: results.ToolResult) -> dict[str, Any]:
    """Return the excerpts a tool's result carries into its tool.result event, else {}.

    A command gets its output tails; read_file a head preview and the line count; an edit or
    patch the paths it wrote, so a resume can tell the run's own untracked files from the
    operator's.
    """
    if isinstance(result, results.EditResult):
        return {"paths": [result.path]}
    if isinstance(result, results.PatchResult):
        written = [p for p, _ in result.files] or [result.path]
        return {"paths": [p for p in written if p not in result.deleted]}
    if isinstance(result, results.ExecResult | results.MetricResult) and name in _EXEC_OUTPUT_TOOLS:
        tails: dict[str, Any] = {
            "stdout_tail": result.stdout[-_TOOL_OUTPUT_TAIL:],
            "stderr_tail": result.stderr[-_TOOL_OUTPUT_TAIL:],
        }
        if isinstance(result, results.ExecResult) and result.stopped_by_operator:
            tails["stopped_by_operator"] = True
        return tails
    if isinstance(result, results.ReadFileResult):
        head = "\n".join(result.content.splitlines()[:_READ_HEAD_LINES])
        return {
            "head_tail": head[:_READ_HEAD_CHARS],
            "lines_total": result.lines_total,
        }
    return {}


def clip_tail(text: str, limit: int = 20_000) -> str:
    """Return the last `limit` chars behind a marker naming what was dropped."""
    if len(text) <= limit:
        return text
    return f"... {len(text) - limit} earlier chars clipped ...\n" + text[-limit:]


def exec_result(res: kinds.CommandResult, *, timeout_s: float = 0.0) -> results.ExecResult:
    """Return the model's view of a finished command: the tails of both streams."""
    return results.ExecResult(
        returncode=res.returncode,
        stdout=clip_tail(res.stdout),
        stderr=clip_tail(res.stderr),
        duration_s=res.duration_s,
        exec_failed=res.exec_failed,
        timeout_s=timeout_s,
    )
