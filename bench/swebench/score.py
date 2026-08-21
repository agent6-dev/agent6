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
import os
import subprocess
from pathlib import Path

# On disk, never /tmp: tmpfiles aged the venv out mid-campaign and the
# harness died at import. AGENT6_SB_SWEPY overrides.
SWEPY = os.environ.get(
    "AGENT6_SB_SWEPY", str(Path.home() / "agent6-work" / "swebench-venv" / "bin" / "python")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/a6swebench"))
    ap.add_argument("--run-id", default="a6_pilot")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="HF dataset the predictions target; SWE-rebench: "
        "nebius/SWE-rebench-leaderboard (needs --split, --namespace swerebench, "
        "and AGENT6_SB_SWEPY pointing at a venv of the SWE-rebench/SWE-bench-fork "
        "harness, which reads each task's install_config)",
    )
    ap.add_argument("--split", default=None, help="dataset split (rebench: the monthly split)")
    ap.add_argument(
        "--namespace",
        default=None,
        help="Docker Hub namespace for prebuilt eval images (rebench: swerebench)",
    )
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
                "sudo",
                SWEPY,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                args.dataset,
                *(["--split", args.split] if args.split else []),
                *(["--namespace", args.namespace] if args.namespace else []),
                "--predictions_path",
                str(combined.resolve()),
                "--max_workers",
                str(args.max_workers),
                "--run_id",
                run_id,
                # none: each eval removes its image after running; the
                # instance level kept ~3G per pred and filled the 147G mount.
                "--cache_level",
                "none",
            ],
            cwd=str(args.out),
            check=False,
        )
        report = args.out / f"{model}.{run_id}.json"
        if report.exists():
            r = json.loads(report.read_text())
            resolved = r.get("resolved_instances", 0)
            total = len(plist)  # denominator: submitted predictions, not the full dataset
            summary[model] = {
                "resolved": resolved,
                "total": total,
                "rate": round(resolved / total, 3) if total else 0.0,
                "empty_patches": r.get("empty_patch_instances", 0),
                "errors": r.get("error_instances", 0),
            }

    print(f"\n===== {args.dataset} resolve rates =====")
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["rate"]):
        print(
            f"  {model:36} {s['resolved']:3}/{s['total']:<3} "
            f"= {s['rate'] * 100:4.1f}%  (empty={s['empty_patches']}, err={s['errors']})"
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
