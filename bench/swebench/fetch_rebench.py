#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fetch one SWE-rebench leaderboard split for the sweep harness.

SWE-rebench (nebius) publishes fresh, contamination-resistant tasks in
SWE-bench format as monthly splits of `nebius/SWE-rebench-leaderboard`,
with prebuilt per-instance images under the `swerebench/` Docker Hub
namespace (named in each row's `image_name`). This writes the two files
`run_sweep.py` takes:

    instances_rebench_<split>.json   full rows (--instances)
    rebench_<split>.json             {"source", "split", "sample_ids"} (--sample)

Rows come from the HF datasets-server rows API (no login, 100 rows per
page). Splits publish AFTER the leaderboard window scores, so the newest
split is the freshest fair comparison available.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "nebius/SWE-rebench-leaderboard"
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def fetch_split(split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        q = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=60) as resp:
            page = json.load(resp)
        batch = [r["row"] for r in page.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(page.get("num_rows_total", 0)):
            return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="monthly split, e.g. 2026_03")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "agent6-work")
    args = ap.parse_args()

    rows = fetch_split(args.split)
    if not rows:
        raise SystemExit(f"split {args.split!r} returned no rows")
    missing = [r["instance_id"] for r in rows if not r.get("image_name")]
    if missing:
        raise SystemExit(f"{len(missing)} rows carry no image_name (first: {missing[0]})")

    inst_path = args.out_dir / f"instances_rebench_{args.split}.json"
    sample_path = args.out_dir / f"rebench_{args.split}.json"
    inst_path.write_text(json.dumps(rows), encoding="utf-8")
    sample_path.write_text(
        json.dumps(
            {
                "source": DATASET,
                "split": args.split,
                "sample_ids": [r["instance_id"] for r in rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{len(rows)} instances -> {inst_path}")
    print(f"sample -> {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
