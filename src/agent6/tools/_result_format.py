# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Post-dispatch helpers: the jail passthrough env, metric-score parsing, and
compacting tool args/results for the event log.
"""

from __future__ import annotations

import os
import re
from typing import Any

PASSTHROUGH_ENV_KEYS = ("LANG", "LC_ALL", "TERM", "CI")


def passthrough_env() -> dict[str, str]:
    return {k: os.environ[k] for k in PASSTHROUGH_ENV_KEYS if k in os.environ}


def parse_metric_score(res: dict[str, Any], *, pattern: str) -> float | None:
    """Apply the metric ``pattern`` regex to combined stdout+stderr.

    Shared metric parser; centralised so the workflow and tool handler
    scores from the same command output. Returns ``None`` on regex compile
    failure, no-match, or non-numeric capture group - the caller treats
    that as "no score this turn" and falls back to raw stdout inspection.
    """
    combined = f"{res.get('stdout', '')}\n{res.get('stderr', '')}"
    try:
        m = re.search(pattern, combined)
    except re.error:
        return None
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError, TypeError):
        # TypeError: an optional/alternation capture group that did not
        # participate in the match yields None, and float(None) raises it.
        return None


def truncate_args(raw: dict[str, Any], *, max_value_chars: int = 200) -> dict[str, Any]:
    """Cheap argument preview for telemetry; truncates strings longer than
    *max_value_chars* and lists longer than 10 items."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str) and len(v) > max_value_chars:
            out[k] = v[:max_value_chars] + f"… ({len(v)} chars)"
        elif isinstance(v, list | tuple) and len(v) > 10:
            out[k] = [*list(v[:10]), f"… ({len(v)} items)"]
        else:
            out[k] = v
    return out


def summarize_result(name: str, result: dict[str, Any]) -> str:  # noqa: PLR0911, PLR0912
    """One-line human-readable summary for the TUI / log tail."""
    if "skill" in result and "content" in result:
        return f"skill {result['skill']}/{result.get('file')} ({len(str(result['content']))} chars)"
    if "size" in result:
        return f"{result['size']} bytes"
    if "entries" in result and isinstance(result["entries"], list):
        return f"{len(result['entries'])} entries"
    if "hits" in result and isinstance(result["hits"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['hits'])} matches{more}"
    if "symbols" in result and isinstance(result["symbols"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['symbols'])} symbols{more}"
    if "definitions" in result and isinstance(result["definitions"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['definitions'])} definitions{more}"
    if "references" in result and isinstance(result["references"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['references'])} references{more}"
    if "applied" in result:
        return f"applied={result['applied']} path={result.get('path')}"
    if "bytes_written" in result:
        return f"patched path={result.get('path')} bytes={result['bytes_written']}"
    if "returncode" in result:
        return f"exit={result['returncode']} in {result.get('duration_s', 0):.1f}s"
    if "title" in result and "status" in result:
        # DAG task tools: the title is what a human scans for, not the ULID.
        return f"{result['status']}: {str(result['title'])[:60]}"
    if "tasks" in result and isinstance(result["tasks"], list):
        return f"{len(result['tasks'])} tasks"
    if "answers" in result and isinstance(result["answers"], list):
        answered = sum(1 for a in result["answers"] if str(a).strip())
        return f"{answered}/{len(result['answers'])} answered"
    # Nothing structured to say: "ok" -- echoing the tool name back (the old
    # fallback) doubled the name column with noise on every DAG/finish call.
    return "ok"
