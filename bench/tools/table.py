"""Print replicate counts and spread from tool benchmark JSONL files."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import statistics
from collections.abc import Sequence
from typing import Any


def spread(rows: list[dict[str, Any]], key: str) -> str:
    """Show the median and range, with coverage when a reading is missing."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return "?"
    coverage = f" ({len(values)}/{len(rows)})" if len(values) != len(rows) else ""
    return f"{statistics.median(values):g} [{min(values)},{max(values)}]{coverage}"


def edit_rate(rows: list[dict[str, Any]], name: str) -> str:
    """Pool failed and completed calls only across runs with journal evidence."""
    known = [r for r in rows if r.get(f"{name}_calls") is not None]
    if not known:
        return "?"
    failures = f"{name}_failures"
    failed = sum(r[failures] for r in known)
    calls = sum(r[f"{name}_calls"] for r in known)
    return f"{failed}/{calls}; {spread(rows, failures)}"


def main(argv: Sequence[str] | None = None) -> None:
    """Print one table per tree, route, effort, plan cap and isolation level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=pathlib.Path, nargs="+", metavar="RESULTS.jsonl")
    cells: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for path in parser.parse_args(argv).files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                cells[
                    tuple(
                        str(row[k])
                        for k in (
                            "revision",
                            "model",
                            "effort",
                            "max_plan_points",
                            "isolation",
                            "split",
                            "task",
                            "arm",
                        )
                    )
                ].append(row)
    for group, keys in itertools.groupby(sorted(cells), key=lambda k: k[:-3]):
        tree, model, effort, cap, isolation = group
        print(f"\n{tree}  {model}  effort={effort}  plan cap={cap}  isolation={isolation}\n")
        print(
            "split | task | arm | n | pass/scored | capped | errors | patch failed/calls; failed/run | edit failed/calls; failed/run | turns | input tokens | output tokens"
        )
        print(" | ".join(["---"] * 12))
        for key in keys:
            rows = cells[key]
            scored = [r["passed"] for r in rows if r["passed"] is not None]
            print(
                *key[-3:],
                len(rows),
                f"{sum(scored)}/{len(scored)}",
                sum(r.get("stop_reason") == "budget_exhausted" for r in rows),
                sum(r["error"] is not None for r in rows),
                edit_rate(rows, "patch"),
                edit_rate(rows, "edit"),
                *(spread(rows, k) for k in ("turns", "input_tokens", "output_tokens")),
                sep=" | ",
            )


if __name__ == "__main__":
    main()
