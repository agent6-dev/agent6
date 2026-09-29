# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions review`: a read-only review of one finished session's
record on the reviewer role. Its own module, imported by the dispatcher for
this verb alone, so the provider stack loads for no other `sessions` verb."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app._setup import budget_tracker, check_provider_keys
from agent6.app.providers import build_role_provider
from agent6.budget import BudgetExceededError
from agent6.config import ConfigError
from agent6.harness._context import agents_md_text
from agent6.harness.run_review import RunReviewError, run_digest, run_review
from agent6.paths import state_dir
from agent6.providers import ProviderError, TranscriptSink
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import error, print_nothing_yet, resolve_or_newest_layout
from agent6.ui.cli.review_cmds import _reviewer_config, save_review
from agent6.viewmodel import session_is_live


def _cmd_sessions_review(  # noqa: PLR0911
    config_path: Path | None, *, session_id: str, model: str
) -> int:
    """Print a review of a session's record to stdout and save it under the
    state dir's reviews/. Read-only; no jail. *model* is `--model`,
    `[provider/]model`, applied to the reviewer route over every config
    layer. Exit 0 reviewed, 2 refused, 3 budget."""
    cwd = Path.cwd()
    # Any session by id, the newest across every bucket without one: the
    # review reads a journal, so a plan, an ask, a fan-out and a model-git run
    # are all records it can read (the git verbs' resolver refuses those).
    try:
        layout = resolve_or_newest_layout(cwd, session_id)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if layout is None:
        print_nothing_yet("sessions")
        return 2
    if not session_id:
        print(f"[agent6] reviewing the newest session: {layout.session_id}", file=sys.stderr)
    if session_is_live(layout.session_dir):
        error(
            f"{layout.session_id} is live; its record is not complete. Review it once it"
            f" has ended (`agent6 stop {layout.session_id}` ends it now)."
        )
        return 2
    try:
        cfg = _reviewer_config(config_path, model)
    except ConfigError as exc:
        error(str(exc))
        return 2
    cfg.require_runnable("reviewer")
    err = check_provider_keys(cfg)
    if err is not None:
        error(f"{err}")
        return 2
    budget = budget_tracker(cfg)
    reviews_dir = state_dir(cwd) / "reviews"
    transcript_sink = TranscriptSink(reviews_dir)
    try:
        reviewer = build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat="review:run"
        )
    except ProviderError as exc:
        error(f"provider init failed: {exc}")
        return 2
    digest = run_digest(layout)
    print(f"[agent6] reviewing run: {layout.session_id}", file=sys.stderr)
    try:
        text = run_review(reviewer, digest=digest.render(), agents_md=agents_md_text(cwd))
    except RunReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except BudgetExceededError as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3
    print(text, flush=True)
    saved = save_review(reviews_dir, label=f"run {layout.session_id}", body=text)
    print(f"[agent6] review saved: {saved}", file=sys.stderr)
    print(budget.format_summary(), file=sys.stderr)
    return 0
