from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
FOCUSES = ("security", "correctness", "data loss")
CHECKS = {
    "traversal": 'assert ns["read_doc"]("hello.txt") == "hello\\n"',
    "clean-math": 'assert ns["add"](2, 3) == 5',
    "secret": 'assert "API_KEY" in ns',
    "cross-file": 'assert ns["parse"]("x", True) == "x"',
    "clean-token": 'assert ns["check"]("a", "a") and not ns["check"]("a", "b")',
    "clean-string": 'assert ns["trim"](" x ") == "x"',
}


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    workdir: pathlib.Path
    state_home: pathlib.Path
    config: pathlib.Path
    checks: pathlib.Path


class BenchmarkError(Exception):
    """The benchmark cannot produce trustworthy measurements."""


def git(workdir: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare(
    case: dict[str, Any], directory: pathlib.Path, model: str, effort: str, temperature: float
) -> Prepared:
    """Build one immutable case comparison, config and private state directory."""
    directory = directory.resolve()
    workdir = directory / "repo"
    workdir.mkdir(parents=True)
    state_home = directory / "state"
    state_home.mkdir()
    git(workdir, "init", "-q")
    git(workdir, "config", "user.name", "bench")
    git(workdir, "config", "user.email", "bench@bench")
    prefix = "import hmac\n\n" if case["id"] == "clean-token" else ""
    base = "def parse(text):\n    return text.strip()\n" if case["id"] == "cross-file" else ""
    (workdir / "case.py").write_text(prefix + base, encoding="utf-8")
    if case["id"] == "cross-file":
        (workdir / "main.py").write_text('from case import parse\n\nparse("x")\n', encoding="utf-8")
    if case["id"] == "traversal":
        (workdir / "files").mkdir()
        (workdir / "files/hello.txt").write_text("hello\n", encoding="utf-8")
    git(workdir, "add", ".")
    git(workdir, "commit", "-qm", "base")
    (workdir / "case.py").write_text(prefix + case["change"] + "\n", encoding="utf-8")
    git(workdir, "add", "case.py")
    git(workdir, "commit", "-qm", "case")
    checks = directory / "checks.py"
    checks.write_text(
        'from __future__ import annotations\nimport runpy\nns = runpy.run_path("case.py")\n'
        + CHECKS[case["id"]]
        + "\nprint("
        + repr(case["supplied_checks"])
        + ")\n",
        encoding="utf-8",
    )
    effort_line = f"effort = {json.dumps(effort)}\n" if effort else ""
    config = directory / "config.toml"
    config.write_text(
        f"[models.default]\nmodel = {json.dumps(model)}\n{effort_line}"
        f'temperature = {temperature}\n[sandbox]\nrun_commands = "deny"\n'
        "[harness]\nverify_infer = false\n"
        '[review]\ntrigger = "before_finish"\nblock_on = "any"\nscope = "diff"\nconcurrency = 3\n'
        + "".join(
            f"[[review.reviewers]]\nfocus = {json.dumps(focus)}\nmodel = {json.dumps(model)}\n"
            f"{effort_line}temperature = {temperature}\n"
            for focus in FOCUSES
        ),
        encoding="utf-8",
    )
    return Prepared(workdir, state_home, config, checks)


def measures(panel: dict[str, Any], *, clean: bool) -> dict[str, Any]:
    """Count grounded findings and seat outcomes without inventing oracle matches."""
    seats = panel["seats"]
    if len(seats) != 3 or panel.get("result") is None:
        raise BenchmarkError("expected one completed three-seat panel")
    blocks = [o for o in panel["result"]["findings"] if o["finding"]["severity"] == "block"]
    return {
        "grounded_blocks": len(blocks),
        "grounded_true_blocks": 0 if clean or not blocks else None,
        "false_blocks": len(blocks) if clean else 0,
        "abstentions": sum(s["verdict"] == "abstain" for s in seats),
        "missing_risk": sum(not s.get("risk", "").strip() for s in seats),
        "risk_length_violations": sum(bool(s.get("risk_truncated")) for s in seats),
        "block_findings": blocks,
    }


def read_measurements(state_home: pathlib.Path, *, clean: bool) -> dict[str, Any]:
    receipts = list(state_home.glob("agent6/*/sessions/runs/*/receipt.json"))
    if len(receipts) != 1:
        raise BenchmarkError(f"expected one receipt, found {len(receipts)} under {state_home}")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    panels = receipt["review"]["panels"]
    if len(panels) != 1:
        raise BenchmarkError(f"expected one panel, found {len(panels)}")
    journal = receipts[0].with_name("logs.jsonl")
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    usage = next((e for e in reversed(events) if e["type"] == "budget.update"), {})
    return {
        **measures(panels[0], clean=clean),
        "tokens_in": usage.get("input_total"),
        "tokens_out": usage.get("output_total"),
        "usd": usage.get("usd_total"),
        "usd_partial": usage.get("usd_partial"),
        "plan_points": usage.get("plan_consumed"),
        "receipt": str(receipts[0]),
        "panel": panels[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="B14 risk-field prompt A/B; no calls in --dry-run")
    ap.add_argument("--model", help="qualified provider/model route")
    ap.add_argument(
        "--effort", default="high", choices=("off", "low", "medium", "high", "xhigh", "max")
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="b14")
    ap.add_argument("--runs-root", type=pathlib.Path, default=ROOT / "results/runs")
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "results/b14.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--panel-driver",
        action="store_true",
        help="opt in to the bench-local panel driver instead of /review",
    )
    args = ap.parse_args()
    if not args.dry_run and not args.panel_driver:
        ap.error(
            'agent6 ask "/review <range>" produces markdown, not panel records; use --panel-driver explicitly'
        )
    if not args.model and not args.dry_run:
        ap.error("--model provider/model is required for a live run")
    model = args.model or "chatgpt/DRY-RUN"
    if "/" not in model or not all(model.split("/", 1)):
        ap.error("--model must be a provider/model route")
    cases_bytes = (ROOT / "cases.json").read_bytes()
    cases = json.loads(cases_bytes)["cases"]
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    failed = 0
    for case in cases:
        for rep in range(3):
            # Alternating order keeps one arm from always paying the first-call cost.
            for arm in ("A", "B") if rep % 2 == 0 else ("B", "A"):
                run_id = f"{case['id']}-{arm}-r{rep}-{uuid.uuid4().hex[:8]}"
                prepared = prepare(
                    case, args.runs_root / run_id, model, args.effort, args.temperature
                )
                count += 1
                env = dict(
                    os.environ,
                    XDG_STATE_HOME=str(prepared.state_home),
                    XDG_DATA_HOME=str(prepared.state_home / "data"),
                    AGENT6_REVIEW_RISK="off" if arm == "A" else "on",
                )
                command = [
                    sys.executable,
                    str(ROOT / "panel_driver.py"),
                    str(prepared.config),
                    str(prepared.checks),
                    case["supplied_checks"],
                ]
                print(f"{arm} {case['id']} r{rep}: {prepared.config}", flush=True)
                if args.dry_run:
                    continue
                row: dict[str, Any] = dict(
                    label=args.label,
                    run_id=run_id,
                    case=case["id"],
                    split=case["split"],
                    arm=arm,
                    rep=rep,
                    model=model,
                    effort=args.effort,
                    temperature=args.temperature,
                    seats=3,
                    max_tokens=1500,
                    scope="diff",
                    engine="panel-driver",
                    cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
                    oracle=case["oracle"],
                    supplied_checks=case["supplied_checks"],
                    config=str(prepared.config),
                )
                started = time.monotonic()
                try:
                    proc = subprocess.run(
                        command,
                        cwd=prepared.workdir,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=900,
                    )
                    (prepared.config.parent / "agent.log").write_text(
                        proc.stdout + proc.stderr, encoding="utf-8"
                    )
                    row["exit"] = proc.returncode
                    if proc.returncode:
                        raise BenchmarkError(
                            f"panel driver exited {proc.returncode}; see agent.log"
                        )
                    row.update(
                        read_measurements(
                            prepared.state_home, clean=case["id"].startswith("clean-")
                        )
                    )
                except (BenchmarkError, OSError, subprocess.TimeoutExpired) as exc:
                    row["error"] = str(exc)
                    failed += 1
                row["elapsed_s"] = round(time.monotonic() - started, 3)
                with args.output.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row) + "\n")
    print(
        f"{count} repos prepared; no model calls"
        if args.dry_run
        else f"{count} rows written; {failed} failed: {args.output}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
