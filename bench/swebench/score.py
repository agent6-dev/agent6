#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Aggregate agent6 SWE-bench predictions per model and run the official
`swebench.harness.run_evaluation`, then print resolve rates.

Predictions are produced by run_sweep.py (source-only). Scoring uses the
unmodified SWE-bench evaluator (gold FAIL_TO_PASS / PASS_TO_PASS), so the
numbers are directly comparable to the SWE-bench Verified leaderboard.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import subprocess
from pathlib import Path

SWEPY = "/tmp/swebench-venv/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/a6swebench"))
    ap.add_argument("--run-id", default="a6_pilot")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for f in glob.glob(str(args.out / "preds" / "*.json")):
        p = json.loads(Path(f).read_text())
        by_model[p["model_name_or_path"]].append(p)

    summary: dict[str, dict] = {}
    for model, plist in sorted(by_model.items()):
        combined = args.out / f"preds_{model}.json"
        combined.write_text(json.dumps(plist))
        run_id = f"{args.run_id}_{model}"
        print(f"\n=== {model}: {len(plist)} predictions -> swebench eval ===", flush=True)
        subprocess.run(
            [
                "sudo", SWEPY, "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Verified",
                "--predictions_path", str(combined.resolve()),
                "--max_workers", str(args.max_workers),
                "--run_id", run_id, "--cache_level", "instance",
            ],
            cwd=str(args.out), check=False,
        )
        report = args.out / f"{model}.{run_id}.json"
        if report.exists():
            r = json.loads(report.read_text())
            resolved = r.get("resolved_instances", 0)
            total = len(plist)  # we submitted len(plist), not the full dataset
            summary[model] = {
                "resolved": resolved, "total": total,
                "rate": round(resolved / total, 3) if total else 0.0,
                "empty_patches": r.get("empty_patch_instances", 0),
                "errors": r.get("error_instances", 0),
            }

    print("\n===== SWE-bench Verified resolve rates =====")
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["rate"]):
        print(f"  {model:36} {s['resolved']:3}/{s['total']:<3} "
              f"= {s['rate']*100:4.1f}%  (empty={s['empty_patches']}, err={s['errors']})")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
