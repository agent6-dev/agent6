"""Run paired handoff replicates with fresh checkouts and private session state."""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from typing import Any

from bench.handoff import job, journals, table
from bench.tools import run as tool_runs

from agent6 import paths
from agent6.sessions import ipc, layout

ROOT = pathlib.Path(__file__).resolve().parents[2]
SESSION = "handoff"
EFFORT = "medium"


def config(model: str, profile: pathlib.Path, *, scripted: bool) -> str:
    """Pin the queue policy and limits for the parent and its children."""
    sandbox = (
        'isolation = "auto"'
        if scripted
        else (f'isolation = "full"\nhide_paths = {json.dumps([str(profile)])}')
    )
    return f"""[models.default]
model = {json.dumps(model)}
effort = "{EFFORT}"
[git]
control = "model"
[budget]
max_usd = 1.0
max_plan_points = 2.0
max_tokens_fallback = 100000
[sandbox]
{sandbox}
run_commands = "deny"
[harness]
max_turns = 48
parallel.tool = true
parallel.max_children = {job.CHILDREN}
parallel.judges = []
verify.infer = false
verify.when = "never"
[review]
when = "never"
"""


def seed(repo: pathlib.Path, env: dict[str, str]) -> str:
    """Create the same starting commit regardless of when a replicate runs."""
    repo.mkdir()
    for i, name in enumerate(job.FILES):
        (repo / name).write_text(f"{name[:-4]}-{731 + 17 * i}\n", encoding="utf-8")
    stamp = "2000-01-01T00:00:00+0000"
    git_env = env | {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    for args in (["init", "-q", "-b", "main"], ["add", *job.FILES], ["commit", "-qm", "seed"]):
        subprocess.run(["git", *args], cwd=repo, env=git_env, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, env=env, text=True
    ).strip()


def sample(
    out: pathlib.Path, profile: pathlib.Path, model: str, arm: str, rep: int, *, scripted: bool
) -> dict[str, Any]:
    """Queue the operator's tasks before the parent takes its first turn."""
    trial = out / f"{arm}-{rep}"
    repo = trial / "repo"
    state = trial / "state" / "agent6" / paths.repo_id(repo)
    parent = layout.SessionLayout(state, SESSION)
    row: dict[str, Any] = dict(
        arm=arm,
        rep=rep,
        split="scripted" if scripted else "live",
        seed=None,
        model=model,
        effort=EFFORT,
        journal=str(parent.logs_path.relative_to(out)),
        exit_code=None,
    )
    issue = ""
    try:
        trial.mkdir(mode=0o700)
        with tool_runs.environment(trial, profile) as env:
            row["seed"] = seed(repo, env)
            cfg = trial / "run.toml"
            cfg.write_text(config(model, profile, scripted=scripted), encoding="utf-8")
            for title in job.TITLES:
                ipc.queue_request(parent.session_dir, "task", title)
            cli = pathlib.Path(sys.executable).with_name("agent6")
            with (trial / "agent.log").open("xb") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        cli,
                        "--config",
                        cfg,
                        "run",
                        "--session-id",
                        SESSION,
                        "--pin",
                        job.PIN,
                        job.prompt(arm),
                    ],
                    cwd=repo,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    check=False,
                )
            row["exit_code"] = proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        issue = str(exc)
    row.update(journals.measure(parent.logs_path, arm, model))
    if issue:
        row["issues"].append(issue)
    return row


def main(argv: Sequence[str] | None = None) -> int:
    """Run both arms and retain one row per attempted run, including failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument(
        "--scripted", action="store_true", help="Use fixed loopback replies without credentials"
    )
    route.add_argument(
        "--profile", type=pathlib.Path, help="Dedicated agent6 provider config directory"
    )
    parser.add_argument("--model", help="Live provider/model (default: chatgpt/gpt-6.2-sol)")
    parser.add_argument("--reps", type=int, default=3, help="Runs per arm (default: 3)")
    parser.add_argument(
        "--out", type=pathlib.Path, required=True, help="New directory outside the checkout"
    )
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.is_relative_to(ROOT) or out.exists() or args.reps < 1:
        parser.error("use a new output directory outside the checkout and at least one replicate")
    if args.scripted:
        if args.model:
            parser.error("--model selects a live route; omit it with --scripted")
        from bench.handoff import scripted

        source = scripted.profile()
        model = "scripted/worker"
    else:
        profile = args.profile.resolve()
        raw = tomllib.loads((profile / "config.toml").read_text(encoding="utf-8"))
        if raw.keys() - {"agent6", "providers"} or raw.get("agent6", {}).keys() - {
            "config_version"
        }:
            parser.error("the profile may contain only providers and agent6.config_version")
        source = contextlib.nullcontext(profile)
        model = args.model or "chatgpt/gpt-6.2-sol"
    out.mkdir(mode=0o700, parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ok = True
    with source as profile, (out / "results.jsonl").open("x", encoding="utf-8") as stream:
        for rep in range(1, args.reps + 1):
            for arm in ("text", "id") if rep % 2 else ("id", "text"):
                row = sample(out, profile, model, arm, rep, scripted=args.scripted)
                print(json.dumps(dict(revision=revision, **row)), file=stream, flush=True)
                if row["issues"]:
                    print(f"{arm}-{rep}: {'; '.join(row['issues'])}", file=sys.stderr)
                    ok = False
    print(out / "results.jsonl")
    table.print_set(out / "results.jsonl")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
