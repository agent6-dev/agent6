"""Build one tool arm and run its tasks in fresh repositories."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import uuid
from collections.abc import Generator, Sequence
from typing import Any

from bench.coreagent import run as coreagent
from bench.tools import journals

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARMS = ("control", "no-docs", "no-patch", "short")


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """One registered prompt, with an exact answer for ask mode."""

    split: str
    prompt: str
    answer: str | None = None


TASKS = {
    name: Task(split, coreagent.task_prompt(name, coreagent.TASKS[name]))
    for name, split in (("textkit", "dev"), ("bugs", "eval"), ("eventflow", "eval"))
} | {
    "machine-test": Task(
        "dev",
        "Which agent6 command runs a machine bundle's mock tests offline? Reply only with the command, without the bundle path.",
        "agent6 machine test",
    ),
    "plan-cap": Task(
        "eval",
        "Which agent6 run flag caps ChatGPT subscription plan usage in percentage points? Reply only with the flag name.",
        "--max-plan-points",
    ),
}


def archive(build: pathlib.Path, revision: str, arm: str, launcher: pathlib.Path) -> pathlib.Path:
    """Extract the named commit, apply its arm patch and stage the static launcher."""
    src = build / "src"
    src.mkdir()
    with tempfile.TemporaryFile(dir=build) as stream:
        subprocess.run(["git", "archive", revision], cwd=ROOT, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(src, filter="data")
    # An enclosing checkout would make git apply skip paths outside its working prefix.
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    subprocess.run(
        ["git", "apply", "--allow-empty", src / f"bench/tools/arms/{arm}.patch"],
        cwd=src,
        check=True,
    )
    dest = src / "src/agent6/sandbox/_bin/agent6-jail"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(launcher, dest)
    return src


def install(source: pathlib.Path, venv: pathlib.Path) -> pathlib.Path:
    """Install the arm into an offline Python 3.13 environment."""
    subprocess.run(["uv", "venv", "--offline", "--python", "3.13", venv], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--offline", "--python", venv / "bin/python", source],
        env=os.environ | {"AGENT6_SKIP_JAIL_BUILD": "1"},
        check=True,
    )
    return venv / "bin/agent6"


@contextlib.contextmanager
def environment(trial: pathlib.Path, profile: pathlib.Path) -> Generator[dict[str, str]]:
    """Give one run private homes and return any refreshed grant to its dedicated profile."""
    env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "AGENT6_DETACHED_AWAY": "deny"}
    if "AGENT6_ALLOW_ROOT" in os.environ:
        env["AGENT6_ALLOW_ROOT"] = os.environ["AGENT6_ALLOW_ROOT"]
    for role in ("AUTHOR", "COMMITTER"):
        env[f"GIT_{role}_NAME"] = "bench"
        env[f"GIT_{role}_EMAIL"] = "bench@localhost"
    for key, folder in (
        ("HOME", "home"),
        ("XDG_STATE_HOME", "state"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_DATA_HOME", "data"),
    ):
        (trial / folder).mkdir(mode=0o700)
        env[key] = str(trial / folder)
    # Keep the grant on its own filesystem, outside both the submission and the results.
    with tempfile.TemporaryDirectory(dir=profile, prefix="xdg-") as home:
        config = pathlib.Path(home) / "agent6"
        config.mkdir(mode=0o700)
        env["XDG_CONFIG_HOME"] = home
        shutil.copy2(profile / "config.toml", config / "config.toml")
        secret = profile / "secrets.toml"
        grant = config / secret.name
        if secret.exists():
            os.link(secret, grant, follow_symlinks=False)
        try:
            yield env
        finally:
            if grant.exists():
                grant.replace(secret)


def seed(repo: pathlib.Path, env: dict[str, str]) -> None:
    """Commit the starting tree, including an empty tree for ask."""
    for command in (["init", "-q"], ["add", "."], ["commit", "--allow-empty", "-qm", "seed"]):
        subprocess.run(["git", *command], cwd=repo, env=env, check=True)


def grade(
    binary: pathlib.Path,
    config: pathlib.Path,
    workspace: pathlib.Path,
    env: dict[str, str],
    scorer: pathlib.Path,
) -> dict[str, Any]:
    """Run the archived hidden grader inside a jail rooted above the submission."""
    with scorer.open("rb") as script:
        proc = subprocess.run(
            [binary, "--config", config, "exec", "--new", "python3", "-I", "-", "repo"],
            cwd=workspace,
            env=env,
            stdin=script,
            stdout=subprocess.PIPE,
            text=True,
            timeout=120,
            check=True,
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    passed, total = result["cases_passed"], result["cases_total"]
    if not (type(passed) is int and type(total) is int and 0 <= passed <= total and total > 0):
        raise journals.MeasurementError("grader returned invalid case counts")
    return {"passed": passed == total, "cases_passed": passed, "cases_total": total}


def sample(
    binary: pathlib.Path,
    source: pathlib.Path,
    args: argparse.Namespace,
    name: str,
    trial: pathlib.Path,
    row: dict[str, Any],
) -> None:
    """Run one task and attach its measured outcome to the row."""
    task = TASKS[name]
    workspace = trial / "grading"
    repo = workspace / "repo"
    fixture = source / "bench/coreagent/tasks" / name
    editing = task.answer is None
    repo.mkdir(parents=True)
    if editing:
        shutil.copytree(fixture / "repo", repo, dirs_exist_ok=True)
    config = trial / "run.toml"
    config.write_text(
        f"[models.default]\nmodel = {json.dumps(args.model)}\neffort = {json.dumps(args.effort)}\n"
        f"[budget]\nmax_plan_points = {args.max_plan_points}\nmax_usd = 1.0\nmax_tokens_fallback = 100000\n"
        f'[sandbox]\nisolation = "{args.isolation}"\nrun_commands = "{"allow" if editing else "deny"}"\n'
        + (
            '[harness]\nverify.command = ["bash", "verify.sh"]\nparallel.tool = true\n'
            if editing
            else ""
        ),
        encoding="utf-8",
    )
    with environment(trial, args.profile) as env:
        seed(repo, env)
        session = uuid.uuid4().hex if editing else None
        command = ["run", "--session-id", session] if session else ["ask", "query"]
        with (trial / "agent.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [binary, "--config", config, *command, task.prompt],
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                timeout=1800,
                check=False,
            )
        row["exit_code"] = proc.returncode
        row.update(journals.measure(trial / "state", session))
        if editing:
            # exec --new reads config from this pristine root, not from the edited repository.
            subprocess.run(["git", "init", "-q"], cwd=workspace, env=env, check=True)
            row.update(grade(binary, config, workspace, env, fixture / "grade.py"))
        else:
            row["passed"] = (row["answer"] or "").strip().strip("`") == task.answer


def main(argv: Sequence[str] | None = None) -> int:
    """Build once and append a journal-derived row for each task and replicate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--launcher", type=pathlib.Path, required=True)
    parser.add_argument(
        "--profile", type=pathlib.Path, required=True, help="Dedicated provider config directory"
    )
    parser.add_argument("--model", default="chatgpt/gpt-6.2-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-plan-points", type=float, default=2.0)
    parser.add_argument(
        "--out", type=pathlib.Path, required=True, help="New directory on a real disk"
    )
    args = parser.parse_args(argv)
    if args.reps < 1 or len(args.tasks) != len(set(args.tasks)):
        parser.error("give distinct tasks and at least one replicate")
    if not math.isfinite(args.max_plan_points) or args.max_plan_points < 0:
        parser.error("--max-plan-points needs a finite nonnegative cap")
    args.out = args.out.resolve()
    args.profile = args.profile.resolve()
    profile = tomllib.loads((args.profile / "config.toml").read_text(encoding="utf-8"))
    if (
        profile.keys() - {"agent6", "providers", "sandbox"}
        or profile.get("agent6", {}).keys() - {"config_version"}
        or profile.get("sandbox", {}).keys() - {"isolation"}
    ):
        parser.error(
            "the profile may contain only providers, agent6.config_version and sandbox.isolation"
        )
    args.isolation = profile.get("sandbox", {}).get("isolation", "full")
    if args.isolation not in {"full", "partial"}:
        parser.error("use full isolation, or explicitly select partial for a local scripted run")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    ok = True
    with tempfile.TemporaryDirectory(dir=args.out, prefix="build-") as build_dir:
        build = pathlib.Path(build_dir)
        src = archive(build, revision, args.arm, args.launcher)
        binary = install(src, build / "venv")
        with (args.out / "results.jsonl").open("x", encoding="utf-8") as results:
            for rep in range(1, args.reps + 1):
                for name in args.tasks:
                    trial = args.out / f"{name}-{rep}"
                    row: dict[str, Any] = dict(
                        revision=revision,
                        arm=args.arm,
                        task=name,
                        split=TASKS[name].split,
                        rep=rep,
                        model=args.model,
                        effort=args.effort,
                        max_plan_points=args.max_plan_points,
                        isolation=args.isolation,
                        error=None,
                        passed=None,
                    )
                    try:
                        sample(binary, src, args, name, trial, row)
                    except (
                        journals.MeasurementError,
                        OSError,
                        subprocess.SubprocessError,
                        ValueError,
                        KeyError,
                        IndexError,
                    ) as exc:
                        ok = False
                        row["error"] = str(exc)
                        print(f"{trial}: {exc}", file=sys.stderr)
                    print(json.dumps(row), file=results, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
