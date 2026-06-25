#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-model benchmark sweep for agent6.

Runs the bench/realworld task set (restore a stubbed function in a pinned OSS
repo until its upstream tests pass) across several models, with N independent
repetitions per (model, task) cell, and writes one sample JSON per run for
bench/sweep/stats.py to summarise.

Design notes:
- Execution reuses the proven bench/realworld/run_realworld.sh (it clones, sets
  up a venv, applies the breakage, runs agent6, and scores verify_pass by
  RE-RUNNING the verify command out-of-band). We only orchestrate: per-cell env,
  bounded concurrency, resumability, disk cleanup, sample tagging.
- API keys are resolved by agent6 from ~/.config/agent6/secrets.toml (no key
  ever touches this process's argv/env or the samples).
- Clone caching: each upstream repo is mirrored locally once and git's
  ``insteadOf`` redirects every per-run clone to the mirror, so hundreds of runs
  never hit GitHub (no rate-limiting, fast setup). The user's ~/.gitconfig is
  included, not replaced.
- Each run's BENCH_ROOT (venv + clone, ~80 MB) is deleted after the sample is
  extracted unless --keep is given, so disk stays bounded across the sweep.

Usage:
    python3 bench/sweep/run_sweep.py --plan
    python3 bench/sweep/run_sweep.py [--models kimi-k2.7,glm-5.2] [--tasks ...]
                                     [--conc 6] [--out /tmp/a6sweep]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_REALWORLD = REPO / "bench" / "realworld" / "run_realworld.sh"
TASKS_DIR = REPO / "bench" / "realworld" / "tasks"
JAIL_BIN = REPO / "src" / "agent6" / "jail" / "target" / "release" / "agent6-jail"


@dataclass(frozen=True)
class Model:
    label: str  # short, filesystem-safe label used in sample names
    slug: str  # provider model id
    provider: str  # "openrouter" | "anthropic"
    reps: int  # repetitions per task


# Lineup. Open-weights via OpenRouter; frontier reference via Anthropic.
# reps scale down with per-run cost (Anthropic models cost more).
MODELS: list[Model] = [
    Model("kimi-k2.7", "moonshotai/kimi-k2.7-code-20260612", "openrouter", 8),
    Model("kimi-k2.6", "moonshotai/kimi-k2.6-20260420", "openrouter", 8),
    Model("glm-5.2", "z-ai/glm-5.2-20260616", "openrouter", 8),
    # local-runnable Mac tier; two Qwen3.6 of the same generation to isolate the
    # MoE-vs-dense agentic-reliability tradeoff (35B-A3B sparse vs 27B dense).
    Model("qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b-20260415", "openrouter", 8),
    Model("qwen3.6-27b-dense", "qwen/qwen3.6-27b-20260422", "openrouter", 8),
    Model("deepseek-v4-flash", "deepseek/deepseek-v4-flash-20260423", "openrouter", 8),
    Model("sonnet-4-6", "claude-sonnet-4-6", "anthropic", 5),
    Model("opus-4-8", "claude-opus-4-8", "anthropic", 2),
]

# A difficulty-graded subset of the realworld suite (mix of cloned + self-contained).
DEFAULT_TASKS = [
    "werkzeug-safe-join",
    "tinydb-search",
    "click-unstyle",
    "csv-rfc4180",
    "html-strip",
    "url-rfc3986",
]

# Rough per-1k-token blended cost (prompt+completion midpoint, USD) for the
# --plan estimate only; the report uses each run's measured cost.
_EST_PER_RUN = {
    "openrouter": 0.04,
    "anthropic": {"claude-sonnet-4-6": 0.18, "claude-opus-4-8": 0.70},
}


@dataclass
class Job:
    model: Model
    task: str
    rep: int
    sample_path: Path
    bench_root: Path
    log_path: Path
    status: str = "pending"
    note: str = ""
    extra: dict = field(default_factory=dict)


def _git_cache_config(cache_dir: Path, repos: set[str]) -> Path:
    """Mirror each repo once and write a gitconfig that redirects clones to the
    local mirror (insteadOf), including the user's real ~/.gitconfig."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"[include]\n\tpath = {Path.home() / '.gitconfig'}\n"]
    for url in sorted(repos):
        name = url.rstrip("/").split("/")[-1]
        mirror = cache_dir / name
        if not mirror.exists():
            print(f"[cache] mirroring {url} -> {mirror}", flush=True)
            subprocess.run(["git", "clone", "--mirror", "--quiet", url, str(mirror)], check=True)
        lines.append(f'[url "{mirror}"]\n\tinsteadOf = "{url}"\n')
    cfg = cache_dir / "gitconfig"
    cfg.write_text("".join(lines), encoding="utf-8")
    return cfg


def _task_repos(tasks: list[str]) -> set[str]:
    repos: set[str] = set()
    for t in tasks:
        d = json.loads((TASKS_DIR / f"{t}.json").read_text())
        if d.get("repo_url"):
            repos.add(d["repo_url"])
    return repos


def build_jobs(models: list[Model], tasks: list[str], out: Path) -> list[Job]:
    jobs: list[Job] = []
    for m in models:
        for t in tasks:
            for rep in range(1, m.reps + 1):
                sample = out / "samples" / f"{m.label}__{t}__r{rep}.json"
                jobs.append(
                    Job(
                        model=m,
                        task=t,
                        rep=rep,
                        sample_path=sample,
                        bench_root=out / "runs" / m.label / t / f"r{rep}",
                        log_path=out / "logs" / f"{m.label}__{t}__r{rep}.log",
                    )
                )
    return jobs


def run_job(job: Job, gitconfig: Path, timeout_s: float, keep: bool) -> Job:
    if job.sample_path.exists():
        job.status = "cached"
        return job
    job.sample_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if job.bench_root.exists():
        shutil.rmtree(job.bench_root, ignore_errors=True)
    env = dict(os.environ)
    env.update(
        {
            "AGENT6_JAIL_BIN": str(JAIL_BIN),
            "AGENT6_REALWORLD_MODEL": job.model.slug,
            "AGENT6_REALWORLD_TASK_FILTER": job.task,
            "BENCH_ROOT": str(job.bench_root),
            "GIT_CONFIG_GLOBAL": str(gitconfig),
        }
    )
    if job.model.provider == "openrouter":
        env["AGENT6_REALWORLD_OPENROUTER"] = "1"
        env["AGENT6_FORCE_STREAM"] = "1"
    t0 = time.monotonic()
    try:
        with job.log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                ["bash", str(RUN_REALWORLD)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.monotonic() - t0

    # run_realworld.sh writes to $BENCH_ROOT/_logs/<task>_<toolset>/result.json;
    # glob is robust to the toolset label and the single-task filter.
    found = sorted((job.bench_root / "_logs").glob("*/result.json"))
    result_json = found[0] if found else job.bench_root / "_missing_"
    sample: dict = {
        "model": job.model.label,
        "model_slug": job.model.slug,
        "provider": job.model.provider,
        "task": job.task,
        "rep": job.rep,
    }
    if result_json.exists():
        try:
            r = json.loads(result_json.read_text())
            sample.update(
                {
                    "success": bool(r.get("verify_pass")),
                    "cost_usd": float(r.get("cost_usd") or 0.0),
                    "input_tokens": int(r.get("input_tokens") or 0),
                    "output_tokens": int(r.get("output_tokens") or 0),
                    "wall_seconds": float(r.get("wall_seconds") or wall),
                    "agent_exit": r.get("agent_exit"),
                    "metric_score": r.get("metric_score"),
                }
            )
            job.status = "pass" if sample["success"] else "fail"
        except (json.JSONDecodeError, ValueError) as exc:
            sample.update({"success": False, "cost_usd": 0.0, "error": f"parse: {exc}"})
            job.status = "error"
    else:
        sample.update(
            {
                "success": False,
                "cost_usd": 0.0,
                "wall_seconds": wall,
                "error": "timeout" if timed_out else "no result.json",
            }
        )
        job.status = "timeout" if timed_out else "error"

    job.sample_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    job.extra = {"cost": sample.get("cost_usd", 0.0), "wall": round(wall, 1)}
    if not keep:
        shutil.rmtree(job.bench_root, ignore_errors=True)
    return job


def plan(jobs: list[Job]) -> None:
    by_model: dict[str, int] = {}
    est = 0.0
    for j in jobs:
        by_model[j.model.label] = by_model.get(j.model.label, 0) + 1
        if j.model.provider == "openrouter":
            est += _EST_PER_RUN["openrouter"]
        else:
            est += _EST_PER_RUN["anthropic"].get(j.model.slug, 0.3)
    print(f"Sweep plan: {len(jobs)} runs")
    for label, n in by_model.items():
        print(f"  {label:22} {n} runs")
    print(f"Estimated API spend: ~${est:.2f} (very rough; report uses measured cost)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/a6sweep"))
    ap.add_argument("--models", default="", help="comma-separated labels (default: all)")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: subset)")
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--reps", type=int, default=0, help="override reps per model (testing)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-run timeout seconds")
    ap.add_argument("--keep", action="store_true", help="keep per-run BENCH_ROOT dirs")
    ap.add_argument("--plan", action="store_true", help="print plan + cost estimate, do not run")
    args = ap.parse_args()

    models = MODELS
    if args.models:
        want = {x.strip() for x in args.models.split(",")}
        models = [m for m in MODELS if m.label in want]
    if args.reps:
        models = [Model(m.label, m.slug, m.provider, args.reps) for m in models]
    tasks = [x.strip() for x in args.tasks.split(",")] if args.tasks else DEFAULT_TASKS

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(models, tasks, args.out)
    if args.plan:
        plan(jobs)
        return 0

    if not JAIL_BIN.exists():
        print(f"jail missing: {JAIL_BIN}", file=sys.stderr)
        return 1
    gitconfig = _git_cache_config(args.out / "_cache", _task_repos(tasks))

    todo = [j for j in jobs if not j.sample_path.exists()]
    print(
        f"{len(jobs)} total, {len(jobs) - len(todo)} cached, {len(todo)} to run, conc={args.conc}"
    )
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = {ex.submit(run_job, j, gitconfig, args.timeout, args.keep): j for j in todo}
        for fut in as_completed(futs):
            j = fut.result()
            done += 1
            spent = sum(
                float(json.loads(p.read_text()).get("cost_usd") or 0.0)
                for p in (args.out / "samples").glob("*.json")
            )
            el = time.monotonic() - t0
            print(
                f"[{done}/{len(todo)}] {j.model.label}/{j.task} r{j.rep} -> {j.status} "
                f"({j.extra.get('wall', '?')}s, ${j.extra.get('cost', 0):.4f}); "
                f"spend ${spent:.2f}, elapsed {el / 60:.1f}m",
                flush=True,
            )
    print(f"done in {(time.monotonic() - t0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
