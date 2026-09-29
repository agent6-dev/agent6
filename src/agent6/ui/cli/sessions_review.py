# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions review`: a read-only review of one finished session's record.

Its own module, imported by the dispatcher for this verb alone, so the provider stack loads
for no other `sessions` verb.
"""

from __future__ import annotations

import pathlib
import sys

from agent6 import budget as agent6_budget
from agent6 import paths
from agent6.app import _setup, providers
from agent6.config import ConfigError
from agent6.harness import _context, run_review
from agent6.providers import ProviderError, TranscriptSink
from agent6.sessions import id
from agent6.ui.cli import _common, review_cmds
from agent6.viewmodel import session_is_live


def _cmd_sessions_review(  # noqa: PLR0911
    config_path: pathlib.Path | None, *, session_id: str, model: str
) -> int:
    """Print a review of a session's record and save it under the state dir's reviews/.

    Read-only; no jail.

    Args:
        config_path: The `--config` file, if any.
        session_id: The session, or "" for the newest.
        model: The `--model` override, `[provider/]model`, applied to the reviewer route.

    Returns:
        The exit code: 0 reviewed, 2 refused, 3 budget exceeded.
    """
    cwd = pathlib.Path.cwd()
    # Every bucket: the review reads a journal, so a plan, an ask or a fan-out is a record too.
    try:
        layout = _common.resolve_or_newest_layout(cwd, session_id)
    except id.SessionIdError as exc:
        _common.error(f"{exc}")
        return 2
    if layout is None:
        _common.print_nothing_yet("sessions")
        return 2
    if not session_id:
        print(f"[agent6] reviewing the newest session: {layout.session_id}", file=sys.stderr)
    if session_is_live(layout.session_dir):
        _common.error(
            f"{layout.session_id} is live; its record is not complete. Review it once it"
            f" has ended (`agent6 stop {layout.session_id}` ends it now)."
        )
        return 2
    try:
        cfg = review_cmds._reviewer_config(config_path, model)
    except ConfigError as exc:
        _common.error(str(exc))
        return 2
    cfg.require_runnable("reviewer")
    err = _setup.check_provider_keys(cfg)
    if err is not None:
        _common.error(f"{err}")
        return 2
    budget = _setup.budget_tracker(cfg)
    reviews_dir = paths.state_dir(cwd) / "reviews"
    transcript_sink = TranscriptSink(reviews_dir)
    try:
        reviewer = providers.build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat="review:run"
        )
    except ProviderError as exc:
        _common.error(f"provider init failed: {exc}")
        return 2
    digest = run_review.run_digest(layout)
    print(f"[agent6] reviewing run: {layout.session_id}", file=sys.stderr)
    try:
        text = run_review.review_digest(
            reviewer, digest=digest.render(), agents_md=_context.agents_md_text(cwd)
        )
    except run_review.RunReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except agent6_budget.BudgetExceededError as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3
    print(text, flush=True)
    saved = review_cmds.save_review(reviews_dir, label=f"run {layout.session_id}", body=text)
    print(f"[agent6] review saved: {saved}", file=sys.stderr)
    print(budget.format_summary(), file=sys.stderr)
    return 0
