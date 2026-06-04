#!/usr/bin/env python3
"""Aggregator for /tmp/agent6-overnight-full/ — Phase 4 results.

Prints a markdown report combining realworld pass/fail/cost and perf
cycle scores for the 3 candidate models. Run after run_overnight_full.sh
completes.
"""

from __future__ import annotations

import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agent6-overnight-full"


def fmt_model(slug: str) -> str:
    return slug.replace("_", "/", 1).replace("_", ".")


def main() -> None:
    print("# overnight full bench — Phase 4 results")
    print()
    print("## Realworld (11 tasks per model)")
    print()
    print("| model | task | pass | wall (s) | tok_in | tok_out |")
    print("|---|---|---|---|---|---|")
    per_model: dict[str, dict[str, float | int]] = {}
    for slug in sorted(os.listdir(ROOT)):
        if not os.path.isdir(os.path.join(ROOT, slug)):
            continue
        d = os.path.join(ROOT, slug, "realworld")
        if not os.path.isdir(d):
            continue
        agg = per_model.setdefault(slug, {"tasks": 0, "pass": 0, "wall_s": 0.0, "in": 0, "out": 0})
        for r in sorted(glob.glob(os.path.join(d, "_logs", "*", "result.json"))):
            try:
                j = json.load(open(r))
            except Exception:
                continue
            task = os.path.basename(os.path.dirname(r)).rsplit("_", 1)[0]
            print(
                f"| {fmt_model(slug)} | {task} | "
                f"{'PASS' if j['verify_pass'] else 'FAIL'} | "
                f"{j['wall_seconds']:.1f} | {j['input_tokens']} | "
                f"{j['output_tokens']} |"
            )
            agg["tasks"] = int(agg["tasks"]) + 1
            agg["pass"] = int(agg["pass"]) + (1 if j["verify_pass"] else 0)
            agg["wall_s"] = float(agg["wall_s"]) + float(j["wall_seconds"])
            agg["in"] = int(agg["in"]) + int(j["input_tokens"])
            agg["out"] = int(agg["out"]) + int(j["output_tokens"])

    print()
    print("### Realworld totals")
    print()
    print("| model | pass/total | total wall (s) | total tok_in | total tok_out |")
    print("|---|---|---|---|---|")
    for slug, a in sorted(per_model.items()):
        print(
            f"| {fmt_model(slug)} | {a['pass']}/{a['tasks']} | "
            f"{a['wall_s']:.1f} | {a['in']} | {a['out']} |"
        )

    print()
    print("## Perf takehome (single task per model)")
    print()
    print(
        "| model | start_cycles | end_cycles | speedup | wall (s) | tok_in | tok_out | cost_usd |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for slug in sorted(os.listdir(ROOT)):
        if not os.path.isdir(os.path.join(ROOT, slug)):
            continue
        p = os.path.join(ROOT, slug, "perf", "result_agent6.json")
        if not os.path.isfile(p):
            print(f"| {fmt_model(slug)} | — | — | — | (no result_agent6.json) | — | — | — |")
            continue
        try:
            j = json.load(open(p))
        except Exception as e:
            print(f"| {fmt_model(slug)} | — | — | — | (parse error: {e}) | — | — | — |")
            continue
        sc = j.get("start_cycles") or j.get("baseline_cycles") or "?"
        ec = j.get("final_cycles") or j.get("end_cycles") or j.get("best_cycles") or "?"
        sp = j.get("speedup_over_baseline") or j.get("speedup") or "?"
        print(
            f"| {fmt_model(slug)} | {sc} | {ec} | "
            f"{sp if isinstance(sp, str) else f'{sp:.2f}x'} | "
            f"{j.get('wall_seconds', '?')} | {j.get('input_tokens', '?')} | "
            f"{j.get('output_tokens', '?')} | "
            f"${j.get('cost_usd', 0)} |"
        )

    print()
    print("## Loop-guard activity")
    print()
    print("Triggers per model (counts of `v2.loop_guard.triggered` events):")
    print()
    for slug in sorted(os.listdir(ROOT)):
        if not os.path.isdir(os.path.join(ROOT, slug)):
            continue
        n = 0
        for log in glob.glob(os.path.join(ROOT, slug, "**", "logs.jsonl"), recursive=True):
            try:
                with open(log) as f:
                    for ln in f:
                        if '"v2.loop_guard.triggered"' in ln:
                            n += 1
            except Exception:
                pass
        print(f"- {fmt_model(slug)}: {n}")


if __name__ == "__main__":
    main()
