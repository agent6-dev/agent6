from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time
import uuid

import pydantic

from agent6 import budget, config, evidence, paths, review_records
from agent6.app import role_providers
from agent6.config import layer
from agent6.harness import review_gate, review_history, review_panel
from agent6.providers import provider_types
from agent6.sessions import event_log, journal, layout, manifest
from agent6.vcs import git_read, review
from agent6.viewmodel import receipt


def run(
    cfg: config.Config, root: pathlib.Path, checks: pathlib.Path, supplied_checks: str
) -> pathlib.Path:
    """Run one diff-scope panel and retain the native journal and derived receipt."""
    compared = review.comparison(root, base="HEAD~1..HEAD", head="HEAD", paths=(), hide=())
    target = compared.target
    task = "Review the change to case.py. Supplied checks: " + supplied_checks
    execution_id = uuid.uuid4().hex
    panel_id = uuid.uuid4().hex
    saved = layout.SessionLayout(paths.state_dir(root), "b14-" + execution_id)
    saved.ensure()
    saved.manifest_path.write_text(
        manifest.SessionManifest(user_task=task).model_dump_json(), encoding="utf-8"
    )
    events = event_log.EventSink(saved.logs_path)
    events.emit(
        journal.SessionStart(
            session_id=saved.session_id,
            execution_id=execution_id,
            mode="run",
            user_task=task,
        )
    )
    verify_id = uuid.uuid4().hex
    command = (sys.executable, "-B", str(checks))
    tree_before = git_read.worktree_tree(root, target.head_commit, frozenset())
    started = time.monotonic()
    proc = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True, timeout=60)
    verified = evidence.VerificationEvidence(
        verify_id=verify_id,
        execution_id=execution_id,
        cmd=command,
        tree_before=tree_before,
        tree_after=git_read.worktree_tree(root, target.head_commit, frozenset()),
        scoped=False,
        selected_paths=(),
        outcome="passed",
        exit_code=0,
        duration_s=time.monotonic() - started,
        timeout_s=60,
        stdout_tail=proc.stdout,
        stderr_tail=proc.stderr,
        stdout_truncated=False,
        stderr_truncated=False,
    )
    events.emit(journal.VerifyEnd(**dataclasses.asdict(verified)))
    tracker = budget.BudgetTracker(
        max_usd=cfg.budget.max_usd,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
        max_plan_points=cfg.budget.max_plan_points,
        allow_plan_overage=cfg.budget.allow_plan_overage,
    )
    sink = provider_types.TranscriptSink(saved.transcripts_dir, objects_dir=saved.objects_dir)
    seats = role_providers.build_review_seats(
        cfg, transcript_sink=sink, budget=tracker, events=events
    )
    identity = dict(panel_id=panel_id, execution_id=execution_id, turn=0, trigger="benchmark")
    events.emit(
        journal.LoopReviewStart(
            **identity,
            target=dataclasses.asdict(target),
            rule_sources=[],
            rules_complete=True,
            rules_reason=None,
            seats=[
                dict(
                    seat_index=i,
                    focus=s.focus,
                    model_route=s.model,
                    scope=s.scope,
                    temperature=s.temperature,
                )
                for i, s in enumerate(seats)
            ],
        )
    )
    result = review_gate.run_panel(
        seats,
        review_panel.ReviewContext(
            task=task, diff=compared.diff, verification="passed", verify_output=proc.stdout
        ),
        block_on="any",
        min_blocking_models=cfg.review.min_blocking_models,
        panel_id=panel_id,
        repo_max_turns=cfg.review.repo_max_turns,
        concurrency=cfg.review.concurrency,
        emit=events.emit,
    )
    for verdict in result.per_seat:
        events.emit(
            journal.LoopReviewSeat(
                **identity,
                **pydantic.TypeAdapter(review_records.ReviewVerdict).dump_python(
                    verdict, mode="json"
                ),
            )
        )
    observations = review_history.observations(
        panel_id, result.per_seat, {}, root=root, target=target, hide=()
    )
    record = review_records.ReviewPanel(
        panel_id=panel_id,
        execution_id=execution_id,
        turn=0,
        trigger="benchmark",
        target=target,
        findings=observations,
        injected_finding_ids=(),
        block_on="any",
        raw_blocked=result.blocked,
        blocked=result.blocked,
        disarmed=False,
        n_block=result.n_block,
        n_abstain=result.n_abstain,
        completion="inconclusive" if review_panel.panel_is_inconclusive(result) else "complete",
    )
    events.emit(journal.LoopReviewPanel(**record.model_dump(mode="json")))
    tracker.settle()
    events.emit(journal.BudgetUpdate(**role_providers.budget_update_fields(tracker)))
    events.emit(
        journal.SessionStop(
            execution_id=execution_id,
            reason="benchmark_complete",
            turns=0,
            verification="passed",
            final_verify_id=verify_id,
            tree=target.head_tree,
            summary="Bench-local panel completed; no worker ran.",
        )
    )
    report = saved.session_dir / "receipt.json"
    report.write_text(
        json.dumps(receipt.as_dict(receipt.read_receipt(saved.session_dir, root))), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Bench-local driver, not the /review directive")
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("checks", type=pathlib.Path)
    ap.add_argument("supplied_checks")
    args = ap.parse_args()
    root = pathlib.Path.cwd()
    cfg = layer.load_effective(root, (args.config,)).config
    try:
        print(run(cfg, root, args.checks, args.supplied_checks))
    except (config.ConfigError, pydantic.ValidationError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
