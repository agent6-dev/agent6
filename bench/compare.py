#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Side-by-side head-to-head report: agent6 vs another runner (claude-code).

Given two parallel bench roots (the agent6 baseline and a runner-specific
mirror), this script:

  1. Computes the composite Q score for each task on each side using
     bench/quality.py's `score_task` (re-imported, not shelled out).
  2. Times the verify command (`python3 -m unittest -v`) on each side
     three times and records the minimum wall in milliseconds. This
     gives a "solution performance" metric independent of agent wall
     time.
  3. Counts the unified-diff line totals (added) for each side.
  4. Emits a side-by-side markdown table and a JSON dump.

Usage:
  python3 bench/compare.py \
      --agent6-root /tmp/agent6-bench \
      --claude-root /tmp/agent6-bench-claude-t1 \
      --label tier1 \
      --out /tmp/h2h-tier1.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import score_task


@dataclass(frozen=True, slots=True)
class SidePerf:
    verify_pass: bool
    cost_usd: float
    agent_wall_s: float
    perf_ms: float | None
    diff_lines: int
    composite_q: float


@dataclass(frozen=True, slots=True)
class TaskCompare:
    task: str
    agent6: SidePerf
    claude: SidePerf


def _time_verify(task_dir: Path, runs: int = 3) -> float | None:
    """Time `python3 -m unittest -v` and return the minimum wall in ms.

    Returns None if the verify command errors out (we don't want timing
    on a failing solution to skew the report).
    """
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["python3", "-m", "unittest", "-v"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(task_dir),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            return None
        times.append(elapsed_ms)
    return min(times)


def _added_lines(task_dir: Path) -> int:
    out = subprocess.run(
        ["git", "-C", str(task_dir), "diff", "--shortstat", "master"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    added = 0
    for token in out.split(","):
        token = token.strip()
        if token.endswith("insertion(+)") or token.endswith("insertions(+)"):
            try:
                added = int(token.split()[0])
            except (ValueError, IndexError):
                pass
    return added


def _read_result(task_dir: Path) -> tuple[bool, float, float]:
    """Returns (verify_pass, cost_usd, wall_seconds) from result.json.

    For agent6 result.json the cost is buried in `cost_summary` as a string
    like 'TOTAL: in=... out=... cost~$0.0123'. Parse it out.
    """
    rp = task_dir / "result.json"
    if not rp.is_file():
        return False, 0.0, 0.0
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, 0.0, 0.0
    verify = bool(data.get("verify_pass"))
    wall = float(data.get("wall_seconds", 0.0))
    if "cost_usd" in data:
        cost = float(data["cost_usd"])
    else:
        summary = data.get("cost_summary", "") or ""
        cost = 0.0
        if "cost~$" in summary:
            try:
                cost = float(summary.split("cost~$", 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
    return verify, cost, wall


def _side(task_dir: Path) -> SidePerf:
    verify, cost, wall = _read_result(task_dir)
    perf = _time_verify(task_dir) if verify else None
    diff_lines = _added_lines(task_dir)
    composite = score_task(task_dir).composite
    return SidePerf(
        verify_pass=verify,
        cost_usd=cost,
        agent_wall_s=wall,
        perf_ms=perf,
        diff_lines=diff_lines,
        composite_q=composite,
    )


def compare_one(name: str, agent6_dir: Path, claude_dir: Path) -> TaskCompare:
    return TaskCompare(
        task=name,
        agent6=_side(agent6_dir),
        claude=_side(claude_dir),
    )


def _fmt_perf(p: float | None) -> str:
    if p is None:
        return "  n/a"
    return f"{p:6.1f}"


def _fmt_q(s: SidePerf) -> str:
    verify = "PASS" if s.verify_pass else "FAIL"
    return (
        f"{verify} ${s.cost_usd:.4f} {s.agent_wall_s:5.1f}s "
        f"Q={s.composite_q:.3f} ({s.diff_lines:>3d}L,{_fmt_perf(s.perf_ms)}ms)"
    )


def render_markdown(label: str, rows: list[TaskCompare]) -> str:
    lines: list[str] = []
    lines.append(f"# Head-to-head: agent6 vs claude-code — {label}")
    lines.append("")
    lines.append(
        "Per-task verify, agent cost, agent wall, composite Q, "
        "diff size (added lines), and solution perf (min of 3 verify "
        "runs, ms). Solution perf is the time the agent's *output code* "
        "takes to run the test suite — independent of how long the "
        "agent itself took. n/a = verify failed."
    )
    lines.append("")
    lines.append(
        "| Task | agent6 verify | a6 cost | a6 wall | a6 Q | a6 diff | a6 perf(ms) | "
        "claude verify | cl cost | cl wall | cl Q | cl diff | cl perf(ms) |"
    )
    lines.append(
        "|------|--------------|---------|---------|------|---------|-------------|"
        "---------------|---------|---------|------|---------|-------------|"
    )
    a6_pass = cl_pass = 0
    a6_cost = cl_cost = 0.0
    a6_wall = cl_wall = 0.0
    a6_q: list[float] = []
    cl_q: list[float] = []
    a6_perf: list[float] = []
    cl_perf: list[float] = []
    for r in rows:
        a6 = r.agent6
        cl = r.claude
        a6_pass += int(a6.verify_pass)
        cl_pass += int(cl.verify_pass)
        a6_cost += a6.cost_usd
        cl_cost += cl.cost_usd
        a6_wall += a6.agent_wall_s
        cl_wall += cl.agent_wall_s
        a6_q.append(a6.composite_q)
        cl_q.append(cl.composite_q)
        if a6.perf_ms is not None:
            a6_perf.append(a6.perf_ms)
        if cl.perf_ms is not None:
            cl_perf.append(cl.perf_ms)
        lines.append(
            f"| {r.task} | "
            f"{'PASS' if a6.verify_pass else 'FAIL'} | ${a6.cost_usd:.4f} | "
            f"{a6.agent_wall_s:.1f}s | {a6.composite_q:.3f} | {a6.diff_lines} | "
            f"{_fmt_perf(a6.perf_ms)} | "
            f"{'PASS' if cl.verify_pass else 'FAIL'} | ${cl.cost_usd:.4f} | "
            f"{cl.agent_wall_s:.1f}s | {cl.composite_q:.3f} | {cl.diff_lines} | "
            f"{_fmt_perf(cl.perf_ms)} |"
        )
    n = len(rows)
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(
        f"- agent6: {a6_pass}/{n} verify, ${a6_cost:.4f}, {a6_wall:.1f}s wall, "
        f"mean Q={statistics.mean(a6_q):.3f}"
    )
    lines.append(
        f"- claude: {cl_pass}/{n} verify, ${cl_cost:.4f}, {cl_wall:.1f}s wall, "
        f"mean Q={statistics.mean(cl_q):.3f}"
    )
    if a6_perf and cl_perf:
        lines.append(
            f"- solution-perf medians: agent6 {statistics.median(a6_perf):.1f}ms, "
            f"claude {statistics.median(cl_perf):.1f}ms "
            f"(across {len(a6_perf)} / {len(cl_perf)} passing tasks)"
        )
    if a6_cost > 0 and cl_cost > 0:
        lines.append(f"- cost ratio: agent6 / claude = {a6_cost / cl_cost:.2f}x")
    if a6_wall > 0 and cl_wall > 0:
        lines.append(f"- wall ratio: agent6 / claude = {a6_wall / cl_wall:.2f}x")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent6-root", required=True, type=Path)
    ap.add_argument("--claude-root", required=True, type=Path)
    ap.add_argument("--label", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.agent6_root.is_dir():
        print(f"missing {args.agent6_root}", file=sys.stderr)
        return 1
    if not args.claude_root.is_dir():
        print(f"missing {args.claude_root}", file=sys.stderr)
        return 1

    rows: list[TaskCompare] = []
    for sub in sorted(args.agent6_root.iterdir()):
        if not sub.is_dir() or sub.name == "logs":
            continue
        claude_dir = args.claude_root / sub.name
        if not claude_dir.is_dir():
            print(f"skip {sub.name}: no claude mirror at {claude_dir}", file=sys.stderr)
            continue
        rows.append(compare_one(sub.name, sub, claude_dir))

    md = render_markdown(args.label, rows)
    args.out.write_text(md, encoding="utf-8")

    # JSON dump alongside markdown.
    json_out = args.out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            [
                {"task": r.task, "agent6": asdict(r.agent6), "claude": asdict(r.claude)}
                for r in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(md)
    print(f"\nWrote {args.out} and {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
