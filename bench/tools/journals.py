"""Read one benchmark run and its children from their private state directory."""

from __future__ import annotations

import collections
import json
import pathlib
from typing import Any


class MeasurementError(Exception):
    """A run lacks the evidence needed for a measurement."""


def measure(state: pathlib.Path, root_id: str | None) -> dict[str, Any]:
    """Count completed edits and model usage across sessions, and read the root's outcome."""
    logs = sorted(state.glob("agent6/*/sessions/*/*/logs.jsonl"))
    roots = [log for log in logs if root_id is None or log.parent.name == root_id]
    if len(roots) != 1:
        raise MeasurementError(f"expected one root journal, found {len(roots)}")
    root = roots[0]
    stop: dict[str, Any] = {}
    counts: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    inputs = outputs = 0
    usage_known = True
    for log in logs:
        with log.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] == "session.stop" and log == root:
                    stop = event
                elif event["type"] == "tool.result" and event["name"] in {
                    "apply_edit",
                    "apply_patch",
                }:
                    name = event["name"].removeprefix("apply_")
                    counts[f"{name}_calls"] += 1
                    if not event["ok"]:
                        counts[f"{name}_failures"] += 1
                elif event["type"] == "model.result" and event["model_call_id"] not in seen:
                    seen.add(event["model_call_id"])
                    if event.get("input_tokens") is None or event.get("output_tokens") is None:
                        usage_known = False
                    else:
                        inputs += (
                            event["input_tokens"]
                            + (event.get("cache_read_tokens") or 0)
                            + (event.get("cache_write_tokens") or 0)
                        )
                        outputs += event["output_tokens"]
    if not stop:
        raise MeasurementError(f"no session.stop in {root}")
    return {
        "session_id": root.parent.name,
        "journal": str(root.relative_to(state.parent)),
        "stop_reason": stop["reason"],
        "turns": stop["turns"],
        "answer": stop.get("summary") if stop["reason"] == "answered" else None,
        "input_tokens": inputs if seen and usage_known else None,
        "output_tokens": outputs if seen and usage_known else None,
        **{
            key: counts[key]
            for key in ("patch_calls", "patch_failures", "edit_calls", "edit_failures")
        },
    }
