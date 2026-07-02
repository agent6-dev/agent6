#!/usr/bin/env python3
"""Summarize coreagent benchmark results.

Reads one or more results/*.jsonl files and prints, per (model, task,
condition) cell: n, mean score +/- stderr, pass@compile rate, mean subtasks
created, mean compactions, mean iters, mean usd, mean wall. Designed for
adopt/scrap calls: the per-cell mean+stderr and the paired baseline-vs-treatment
delta are what separate signal from noise.

Usage: python3 stats.py results/screen1.jsonl [more.jsonl ...]
       python3 stats.py --group model,condition results/*.jsonl
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--group", default="model,task,condition")
    args = ap.parse_args()
    keys = args.group.split(",")
    recs = _load(args.paths)

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        cells[tuple(r.get(k) for k in keys)].append(r)

    def num(rs: list[dict[str, Any]], field: str) -> list[float]:
        return [float(r[field]) for r in rs if isinstance(r.get(field), (int, float))]

    hdr = " | ".join(keys)
    print(f"{hdr}")
    print(f"{'':<2}n   score(mean+/-se)   solved  subtasks  compact  rereads  iters   usd     wall")
    for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
        rs = cells[key]
        scores = num(rs, "score")
        mu, se = _mean_se(scores)
        solved = sum(1 for r in rs if float(r.get("score") or 0) >= 0.999) / len(rs)
        subtasks = _mean_se(num(rs, "n_subtasks"))[0]
        compact = _mean_se(num(rs, "compactions"))[0]
        rereads = _mean_se(num(rs, "redundant_reads"))[0]
        iters = _mean_se(num(rs, "iterations"))[0]
        usd = _mean_se(num(rs, "usd"))[0]
        wall = _mean_se(num(rs, "wall_s"))[0]
        tamper = sum(1 for r in rs if r.get("tampered")) / len(rs)
        print(
            f"  {' | '.join(str(x) for x in key)}\n"
            f"    {len(rs):<3} {mu:.3f}+/-{se:.3f}      {solved:.2f}   "
            f"{subtasks:5.1f}    {compact:4.1f}    {rereads:5.1f}   {iters or float('nan'):5.1f}  "
            f"{usd or float('nan'):.3f}  {wall:5.0f}s  tamper={tamper:.2f}"
        )

    # Paired baseline-vs-treatment delta when conditions are grouped.
    if "condition" in keys:
        ci = keys.index("condition")
        print("\n=== treatment deltas vs baseline (same other keys) ===")
        base: dict[tuple[Any, ...], float] = {}
        for key, rs in cells.items():
            if key[ci] == "baseline":
                base[tuple(x for i, x in enumerate(key) if i != ci)] = _mean_se(num(rs, "score"))[0]
        for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
            if key[ci] == "baseline":
                continue
            other = tuple(x for i, x in enumerate(key) if i != ci)
            if other in base:
                mu = _mean_se(num(cells[key], "score"))[0]
                d = mu - base[other]
                tag = "  WIN" if d > 0.02 else "  loss" if d < -0.02 else "  flat"
                print(f"  {' | '.join(str(x) for x in key)}: {d:+.3f}{tag}")


if __name__ == "__main__":
    main()
