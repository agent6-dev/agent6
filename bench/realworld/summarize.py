#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Aggregate one or more realworld bench BENCH_ROOTs into a markdown table.

Usage:
  bench/realworld/summarize.py BENCH_ROOT [BENCH_ROOT ...]

Reads ``<root>/_logs/<task>_<toolset>/result.json`` for each task that
ran in each root and emits a markdown table to stdout. Multiple roots
make it easy to compare configurations side-by-side (e.g. one root with
worker_loop=true and one with worker_loop=false).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _scan_root(root: Path) -> list[dict[str, object]]:
    """Return one dict per task that produced a result.json in *root*."""
    rows: list[dict[str, object]] = []
    logs = root / "_logs"
    if not logs.is_dir():
        return rows
    for d in sorted(logs.iterdir()):
        rj = d / "result.json"
        if not rj.is_file():
            continue
        try:
            payload = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["_dir"] = d.name
        rows.append(payload)
    return rows


def _format_markdown(roots: list[Path]) -> str:
    out: list[str] = []
    out.append("| Root | Task | Verify | Metric | Cost | Wall | In | Out |")
    out.append("|------|------|--------|-------:|-----:|-----:|---:|----:|")
    grand_pass = 0
    grand_total = 0
    grand_cost = 0.0
    for root in roots:
        rows = _scan_root(root)
        for r in rows:
            verdict = "PASS" if r.get("verify_pass") else "FAIL"
            if r.get("verify_pass"):
                grand_pass += 1
            grand_total += 1
            cost = float(r.get("cost_usd", 0.0) or 0.0)
            grand_cost += cost
            wall = float(r.get("wall_seconds", 0.0) or 0.0)
            in_tok = int(r.get("input_tokens", 0) or 0)
            out_tok = int(r.get("output_tokens", 0) or 0)
            metric_raw = r.get("metric_score")
            if metric_raw is None or metric_raw == "null":
                metric_disp = "-"
            else:
                try:
                    metric_disp = f"{float(metric_raw):.3f}"
                except (TypeError, ValueError):
                    metric_disp = str(metric_raw)
            out.append(
                f"| {root.name} | {r.get('task')} | {verdict} | {metric_disp} | "
                f"${cost:.4f} | {wall:.1f}s | {in_tok} | {out_tok} |"
            )
    out.append("")
    out.append(f"**Total: {grand_pass}/{grand_total} pass, ${grand_cost:.4f}**")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    roots = [Path(a).resolve() for a in argv]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        for r in missing:
            print(f"error: not a directory: {r}", file=sys.stderr)
        return 2
    print(_format_markdown(roots))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
