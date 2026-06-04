#!/usr/bin/env python3
"""Summarise n samples of bench/perf JSON results.

Usage: summarize_samples.py <label> [<label>...]
  Each label is the prefix of files under $BENCH_ROOT/_samples/<label>_s*.json
  (default $BENCH_ROOT = /tmp/agent6-perf).

Prints per-sample lines plus mean / median / min / max for speedup and cost,
plus the geometric mean of speedup (the right central tendency for ratios).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path


def load(label: str, root: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for p in sorted(root.glob(f"{label}_s*.json")):
        d = json.loads(p.read_text())
        out.append(
            {
                "name": p.stem,
                "speedup": float(d.get("speedup_over_baseline") or 0.0),
                "cost": float(d.get("cost_usd") or 0.0),
                "wall": float(d.get("wall_seconds") or 0.0),
                "in": float(d.get("input_tokens") or 0.0),
                "out": float(d.get("output_tokens") or 0.0),
                "exit": int(d.get("agent_exit") or 0),
            }
        )
    return out


def gmean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def summarise(label: str, rows: list[dict[str, float]]) -> None:
    print(f"\n=== {label} (n={len(rows)}) ===")
    for r in rows:
        print(
            f"  {r['name']}: speedup={r['speedup']:.2f}x "
            f"cost=${r['cost']:.2f} wall={r['wall']:.0f}s "
            f"in={int(r['in'])} out={int(r['out'])} exit={r['exit']}"
        )
    if not rows:
        return
    sp = [r["speedup"] for r in rows]
    co = [r["cost"] for r in rows]
    cpx = [r["cost"] / r["speedup"] if r["speedup"] > 0 else float("inf") for r in rows]
    print(
        f"  speedup: mean={statistics.mean(sp):.2f}x "
        f"gmean={gmean(sp):.2f}x "
        f"median={statistics.median(sp):.2f}x "
        f"min={min(sp):.2f}x max={max(sp):.2f}x"
    )
    print(f"  cost:    mean=${statistics.mean(co):.2f} median=${statistics.median(co):.2f}")
    finite_cpx = [x for x in cpx if math.isfinite(x)]
    if finite_cpx:
        print(
            f"  $/x:     mean=${statistics.mean(finite_cpx):.2f} "
            f"median=${statistics.median(finite_cpx):.2f}"
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(os.environ.get("BENCH_ROOT", "/tmp/agent6-perf")) / "_samples"
    for label in sys.argv[1:]:
        summarise(label, load(label, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
