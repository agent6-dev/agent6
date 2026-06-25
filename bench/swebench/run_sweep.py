#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 x SWE-bench Verified harness.

For each (model, instance): pull the official SWE-bench instance image, run
agent6 inside it (uv-managed Python 3.14 + a locally-built wheel, hardened, the
repo's conda env granted read+exec via sandbox.extra_read_paths), and take the
git diff as the model's prediction. Predictions are SOURCE-ONLY (test-file
changes stripped) so the agent can never touch the gold grading tests. Then the
official `swebench.harness.run_evaluation` scores resolve/unresolve.

This drives the AGENT's capability; SWE-bench's containers provide the per-repo
environment + the FAIL_TO_PASS/PASS_TO_PASS oracle. Nothing here is privileged.

Usage:
  python3 bench/sweep.py ... see --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IN_CONTAINER = HERE / "in_container.sh"
SECRETS = Path.home() / ".config" / "agent6" / "secrets.toml"

# test-file paths whose diffs are stripped from predictions (source-only).
_TESTPAT = re.compile(r"(^|/)(tests?/|conftest\.py|test_[^/]*\.py$|[^/]*_test\.py$)")


def _docker(*args: str, check: bool = False, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sudo", "docker", *args], text=True, check=check, **kw)


def _image_for(instance_id: str) -> str:
    # Docker Hub encodes "__" as "_1776_" in the tag.
    tag = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{tag}:latest"


def _strip_test_files(patch: str) -> str:
    out: list[str] = []
    keep = True
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1].strip()
            keep = not _TESTPAT.search(path)
        if keep:
            out.append(line)
    return "".join(out)


def run_one(inst: dict, model: str, wheel: Path, out_dir: Path, *, max_usd: float,
            timeout_s: int) -> dict:
    iid = inst["instance_id"]
    pred_path = out_dir / "preds" / f"{model_label(model)}__{iid}.json"
    if pred_path.exists():
        return {"instance_id": iid, "model": model, "status": "cached"}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_dir / "runs" / model_label(model) / iid
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "problem.txt").write_text(inst["problem_statement"], encoding="utf-8")

    image = _image_for(iid)
    # ensure image present (pull is a no-op if cached)
    _docker("pull", "-q", image, capture_output=True)

    uv = shutil.which("uv") or "/usr/local/bin/uv"
    # Forward optional review-panel env (Fugu dimension) into the container.
    review_env = [
        flag
        for k in ("AGENT6_SB_VERIFY", "AGENT6_SB_STRUCTURAL_PRIORS", "AGENT6_SB_REVIEW_SEATS", "AGENT6_SB_REVIEW_DECISION", "AGENT6_SB_REVIEW_QUORUM")
        if (v := os.environ.get(k))
        for flag in ("-e", f"{k}={v}")
    ]
    cmd = [
        "run", "--rm",
        "-e", f"AGENT6_SB_MODEL={model}",
        "-e", f"AGENT6_SB_MAX_USD={max_usd}",
        "-e", f"AGENT6_SB_TIMEOUT={timeout_s}",
        *review_env,
        "-v", f"{uv}:/usr/local/bin/uv:ro",
        "-v", f"{wheel.parent}:/mnt/wheel:ro",
        "-v", f"{SECRETS}:/root/.config/agent6/secrets.toml:ro",
        "-v", f"{IN_CONTAINER}:/mnt/in_container.sh:ro",
        "-v", f"{work / 'problem.txt'}:/mnt/problem.txt:ro",
        "-v", f"{work}:/out",
        image, "bash", "/mnt/in_container.sh",
    ]
    log = (work / "run.log").open("w", encoding="utf-8")
    try:
        subprocess.run(
            ["sudo", "docker", *cmd], stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout_s + 600, check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    log.close()

    raw = (work / "patch.diff")
    patch = _strip_test_files(raw.read_text(encoding="utf-8")) if raw.exists() else ""
    pred_path.write_text(
        json.dumps(
            {
                "instance_id": iid,
                "model_name_or_path": model_label(model),
                "model_patch": patch,
            }
        ),
        encoding="utf-8",
    )
    return {
        "instance_id": iid, "model": model,
        "status": "timeout" if timed_out else ("patch" if patch.strip() else "empty"),
        "patch_bytes": len(patch),
    }


def model_label(model: str) -> str:
    return "agent6-" + model.rsplit("/", maxsplit=1)[-1].replace(":", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/a6swebench"))
    ap.add_argument("--instances", type=Path, default=HERE / "instances_verified.json")
    ap.add_argument("--sample", type=Path, default=HERE / "sample_50.json")
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model slugs")
    ap.add_argument("--n", type=int, default=0, help="first N sample instances (0 = all)")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--max-usd", type=float, default=1.0,
                    help="per-instance USD cap (SWE-agent uses $1)")
    ap.add_argument("--timeout", type=int, default=1200, help="per-instance agent timeout (s)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N sample instances")
    ap.add_argument("--wheel", type=Path, default=None,
                    help="explicit agent6 wheel (default: newest in dist/); use to A/B two builds")
    args = ap.parse_args()

    rows = {r["instance_id"]: r for r in json.loads(args.instances.read_text())}
    sample = json.loads(args.sample.read_text())
    ids = sample["sample_ids"][args.skip:]
    ids = ids[: args.n] if args.n else ids
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = [(rows[i], m) for m in models for i in ids if i in rows]

    if args.plan:
        print(f"{len(jobs)} runs: {len(models)} models x {len(ids)} instances "
              f"(seed {sample.get('seed')}, per-instance cap ${args.max_usd})")
        print(f"est max spend: ${args.max_usd * len(jobs):.0f} (cap x runs; real spend is usually lower)")
        return 0

    wheel = args.wheel or sorted((REPO / "dist").glob("agent6-*.whl"))[-1]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} runs, wheel={wheel.name}, conc={args.conc}, cap=${args.max_usd}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = {
            ex.submit(run_one, inst, m, wheel, args.out,
                      max_usd=args.max_usd, timeout_s=args.timeout): (inst["instance_id"], m)
            for inst, m in jobs
        }
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {r['model'].split('/')[-1]} / {r['instance_id']} "
                  f"-> {r['status']} ({r.get('patch_bytes', 0)}b)", flush=True)
    print(f"predictions in {args.out / 'preds'}. Score with bench/swebench/score.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
