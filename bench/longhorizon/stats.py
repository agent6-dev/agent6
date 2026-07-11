#!/usr/bin/env python3
"""Summarize longhorizon benchmark results.

Per (model, task, leg, condition) cell: n, mean score +/- stderr, solved
rate, tier-1/2 compaction activity, re-reads (total and post-first-drop),
memory/dependency tool usage, iterations, usd, wall. Then paired
condition-vs-baseline deltas on score and iterations. `--components` prints
mean per-component scores per cell: for stylebook that is the per-rule
retention curve.

Usage: python3 stats.py results/wave1.jsonl [more.jsonl ...] [--components]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(paths: list[str]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    mu = sum(xs) / n
    if n == 1:
        return (mu, 0.0)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return (mu, math.sqrt(var / n))


def _num(rs: list[dict[str, Any]], field: str) -> list[float]:
    return [float(r[field]) for r in rs if isinstance(r.get(field), (int, float))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--group", default="model,task,leg,condition")
    ap.add_argument("--components", action="store_true", help="mean per-component scores")
    args = ap.parse_args()
    keys = args.group.split(",")
    recs = _load(args.paths)

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        cells[tuple(r.get(k) for k in keys)].append(r)

    print(" | ".join(keys))
    print(
        "  n   score(mean+/-se)  solved  drops  compact  rr/rr_pd  memw  deps"
        "  iters   usd     wall  tamper"
    )
    for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
        rs = cells[key]
        mu, se = _mean_se(_num(rs, "score"))
        solved = sum(1 for r in rs if float(r.get("score") or 0) >= 0.999) / len(rs)
        drops = _mean_se(_num(rs, "drops_total"))[0]
        compact = _mean_se(_num(rs, "compactions"))[0]
        rr = _mean_se(_num(rs, "redundant_reads"))[0]
        rr_pd = _mean_se(_num(rs, "redundant_reads_post_drop"))[0]
        memw = _mean_se(_num(rs, "memory_writes"))[0]
        deps = _mean_se(_num(rs, "deps_added"))[0]
        iters = _mean_se(_num(rs, "iterations"))[0]
        usd = _mean_se(_num(rs, "usd"))[0]
        wall = _mean_se(_num(rs, "wall_s"))[0]
        tamper = sum(1 for r in rs if r.get("tampered")) / len(rs)
        print(
            f"  {' | '.join(str(x) for x in key)}\n"
            f"    {len(rs):<3} {mu:.3f}+/-{se:.3f}     {solved:.2f}  {drops:6.1f}"
            f"  {compact:5.1f}  {rr:5.1f}/{rr_pd:<5.1f} {memw:4.1f}  {deps:4.1f}"
            f"  {iters or float('nan'):5.1f}  {usd or float('nan'):.3f}"
            f"  {wall:5.0f}s  {tamper:.2f}"
        )
        if args.components:
            comp_sums: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                for cname, cscore in (r.get("component_scores") or {}).items():
                    comp_sums[cname].append(float(cscore))
            if comp_sums:
                parts = [f"{c}={_mean_se(v)[0]:.2f}" for c, v in sorted(comp_sums.items())]
                print(f"      components: {' '.join(parts)}")

    if "condition" in keys:
        ci = keys.index("condition")
        print("\n=== condition deltas vs baseline (same other keys) ===")
        base_score: dict[tuple[Any, ...], float] = {}
        base_iters: dict[tuple[Any, ...], float] = {}
        for key, rs in cells.items():
            if key[ci] == "baseline":
                rest = tuple(x for i, x in enumerate(key) if i != ci)
                base_score[rest] = _mean_se(_num(rs, "score"))[0]
                base_iters[rest] = _mean_se(_num(rs, "iterations"))[0]
        for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
            if key[ci] == "baseline":
                continue
            rest = tuple(x for i, x in enumerate(key) if i != ci)
            if rest not in base_score:
                continue
            mu = _mean_se(_num(cells[key], "score"))[0]
            it = _mean_se(_num(cells[key], "iterations"))[0]
            d = mu - base_score[rest]
            di = it - base_iters[rest] if not math.isnan(base_iters[rest]) else float("nan")
            tag = "  WIN" if d > 0.02 else "  loss" if d < -0.02 else "  flat"
            print(f"  {' | '.join(str(x) for x in key)}: dscore={d:+.3f} diters={di:+.1f}{tag}")


if __name__ == "__main__":
    main()
