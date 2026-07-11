# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` command (freeform review + the adversarial review panel)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import load_effective
from agent6.git_ops import DIFF_SHOW_SAFETY_FLAGS, git_hardening_flags
from agent6.providers import (
    ProviderError,
    TranscriptSink,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.ui.cli._common import _check_provider_keys, _state_dir
from agent6.ui.cli.providers import _build_role_provider, build_review_seats
from agent6.workflows._panel import ReviewContext, render_findings
from agent6.workflows._review import run_panel
from agent6.workflows.loop import build_readonly_review_tools
from agent6.workflows.review import CodeReviewError, run_review


def _collect_review_diff(
    git: str,
    root: Path,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    """Collect the diff `agent6 review` reviews, leaving the index untouched.

    With ``base``: a plain ``git diff base..head`` (read-only). Without it:
    working tree vs HEAD *including untracked files*. To make untracked files
    show up, git needs intent-to-add (``git add -N``) entries, but review is
    documented read-only, so we register ONLY the currently-untracked paths and
    ``git reset`` them afterward (in a ``finally``), restoring the index exactly
    as we found it. Staged/tracked changes are never touched.

    Every invocation carries git_ops' hardening flags plus ``--no-ext-diff
    --no-textconv``: without them, a checkout with a poisoned ``.git/config``
    (``diff.external``/``diff.*.textconv``/``core.fsmonitor``) would run its
    payload on the host the moment the operator reviews it.
    """
    hardening = git_hardening_flags()
    if base:
        diff_args = [git, *hardening, "diff", *DIFF_SHOW_SAFETY_FLAGS, f"{base}..{head}"]
        if paths:
            diff_args.extend(["--", *paths])
        return subprocess.run(diff_args, cwd=root, capture_output=True, text=True, check=False)

    status = subprocess.run(
        [git, *hardening, "status", "--porcelain", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked = [entry[3:] for entry in status.stdout.split("\0") if entry.startswith("?? ")]
    if untracked:
        subprocess.run([git, *hardening, "add", "-N", "--", *untracked], cwd=root, check=False)
    try:
        diff_args = [git, *hardening, "diff", *DIFF_SHOW_SAFETY_FLAGS, "HEAD"]
        if paths:
            diff_args.extend(["--", *paths])
        return subprocess.run(diff_args, cwd=root, capture_output=True, text=True, check=False)
    finally:
        if untracked:
            subprocess.run(
                [git, *hardening, "reset", "-q", "--", *untracked], cwd=root, check=False
            )


def _run_review_panel(
    cfg: Config,
    *,
    base: str,
    diff: str,
    agents_md: str,
    reviewers: int,
    personas: str,
    model_override: str,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
) -> int:
    """Run the grounded adversarial review panel over *diff* and print a verdict
    + merged findings. Read-only; no gating here (post-hoc), the verdict is
    informational. Per-seat status and budget go to stderr."""
    persona_tuple = tuple(p.strip() for p in personas.split(",") if p.strip())
    seats = build_review_seats(
        cfg,
        transcript_sink=transcript_sink,
        budget=budget,
        n=reviewers,
        personas=persona_tuple,
        model_override=model_override,
    )
    label = base or "working tree vs HEAD"
    ctx = ReviewContext(task=f"code review: {label}", agents_md=agents_md, diff=diff)
    # explore-tier seats need a read-only tool surface over the repo.
    tools = None
    dispatch = None
    if any(s.tier == "explore" for s in seats):
        disp = ToolDispatcher(root=Path.cwd(), config=cfg)
        tools, dispatch = build_readonly_review_tools(disp)
    print(
        f"[agent6] review panel: {len(seats)} seats"
        f" ({', '.join(s.persona for s in seats)}) | decision={cfg.review.decision}"
        f" | tier={cfg.review.tier}",
        file=sys.stderr,
    )
    try:
        result = run_panel(
            seats,
            ctx,
            decision=cfg.review.decision,
            quorum=cfg.review.quorum,
            panel_id="cli",
            concurrency=len(seats),  # one-shot CLI: run all seats in parallel
            tools=tools,
            dispatch=dispatch,
        )
    except BudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3
    if result.blocked:
        verdict = "BLOCK"
    elif result.merged_findings:
        verdict = "PASS (with findings)"
    else:
        verdict = "PASS"
    print(f"VERDICT: {verdict}")
    body = render_findings(result.merged_findings)
    if body:
        print(body)
    print(
        f"\nper-seat ({result.n_block} blocking model(s), {result.n_abstain} abstained):",
        file=sys.stderr,
    )
    for v in result.per_seat:
        status = f"abstain: {v.error}" if v.error else f"{v.verdict} ({len(v.findings)} findings)"
        print(f"  - {v.seat} [{v.model}]: {status}", file=sys.stderr)
    print(budget.format_summary(), file=sys.stderr)
    return 0


def _cmd_review(  # noqa: PLR0911
    config_path: Path | None,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    model_override: str = "",
    reviewers: int = 0,
    personas: str = "",
) -> int:
    """Print a code review of a diff to stdout. Read-only; no jail. With
    ``reviewers >= 1``, runs the grounded adversarial review PANEL instead of the
    single freeform review."""
    try:
        cfg = load_effective(Path.cwd(), config_path).config
        cfg.require_runnable("reviewer")
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    err = _check_provider_keys(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    root = Path.cwd()
    git = shutil.which("git")
    if git is None:
        print("ERROR: git not found on PATH.", file=sys.stderr)
        return 2

    diff_proc = _collect_review_diff(git, root, base=base, head=head, paths=paths)
    if diff_proc.returncode != 0:
        print(f"ERROR: git diff failed: {diff_proc.stderr.strip()}", file=sys.stderr)
        return 2
    diff = diff_proc.stdout
    if not diff.strip():
        print("(no diff to review)", file=sys.stderr)
        return 0

    log_proc = subprocess.run(
        [git, *git_hardening_flags(), "log", "-n", "10", "--oneline"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    recent_log = log_proc.stdout if log_proc.returncode == 0 else ""

    agents_md_path = root / "AGENTS.md"
    agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""

    # Reviewer-only: route the "reviewer" role per [models.reviewer]. Budget
    # is per-invocation since this command is a one-shot.
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
        max_usd=cfg.budget.best_effort_usd_limit,
    )
    layout_root = _state_dir(root) / "reviews"
    layout_root.mkdir(parents=True, exist_ok=True)
    transcript_sink = TranscriptSink(layout_root)

    if reviewers >= 1:
        return _run_review_panel(
            cfg,
            base=base,
            diff=diff,
            agents_md=agents_md,
            reviewers=reviewers,
            personas=personas,
            model_override=model_override,
            transcript_sink=transcript_sink,
            budget=budget,
        )

    try:
        reviewer = _build_role_provider(
            cfg,
            "reviewer",
            transcript_sink=transcript_sink,
            budget=budget,
            model_override=model_override,
        )
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    label = (
        "working tree vs HEAD"
        if not base
        else f"{base}..{head}" + (f" -- {' '.join(paths)}" if paths else "")
    )
    print(f"[agent6] reviewing: {label}", file=sys.stderr)
    try:
        text = run_review(
            reviewer,
            diff=diff,
            agents_md=agents_md,
            recent_log=recent_log,
        )
    except CodeReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3

    print(text)
    print(budget.format_summary(), file=sys.stderr)
    return 0
