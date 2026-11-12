# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review a finished session's record with the reviewer model.

`save_review` writes the rendered reviews in `<state-dir>/reviews/`, beside the provider
transcripts the review's calls leave there.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6 import paths
from agent6.app import reporter as app_reporter
from agent6.app import role_providers
from agent6.app.session import setup as session_setup
from agent6.config import Config, ConfigError
from agent6.harness import repo_context
from agent6.harness.review import session as review_session
from agent6.providers import ProviderError, TranscriptSink
from agent6.sessions import layout as sessions_layout
from agent6.viewmodel import scan_session_log, session_is_live


def save_review(reviews_dir: pathlib.Path, *, label: str, body: str) -> pathlib.Path:
    """Write one rendered review under the reviews dir, beside the provider transcripts.

    The file is `<utc-stamp>-review.md`: a `# review: <label>` line, then the body.
    The operator finds a session's review by searching the directory.

    Args:
        reviews_dir: The `<state-dir>/reviews/` directory.
        label: What was reviewed.
        body: The review text.

    Returns:
        The file's path.
    """
    paths.mkdir_for_real_user(reviews_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    content = f"# review: {label}\n\n{body.rstrip()}\n".encode()
    n = 1
    while True:
        path = reviews_dir / (f"{stamp}-review.md" if n == 1 else f"{stamp}-{n}-review.md")
        # An exclusive create claims the name, so two reviews in one second never collide.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        return path


def _not_reviewable(layout: sessions_layout.SessionLayout) -> str | None:
    """Return why the session's record cannot be reviewed yet, or None when it can."""
    if session_is_live(layout.session_dir):
        return (
            f"{layout.session_id} is live; its record is not complete. Review it once it"
            f" has ended (`agent6 stop {layout.session_id}` ends it now)."
        )
    if not scan_session_log(layout.logs_path).saw_start:
        return f"{layout.session_id} never started; there is no record to review."
    return None


def _runnable_config(load: Callable[[], Config], reporter: app_reporter.Reporter) -> Config | None:
    """Return the config with a runnable reviewer, or None after reporting why there is none."""
    try:
        cfg = load()
    except ConfigError as exc:
        reporter.error(str(exc))
        return None
    cfg.require_runnable("reviewer")
    err = session_setup.check_provider_keys(cfg)
    if err is not None:
        reporter.error(f"{err}")
        return None
    return cfg


def recap(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    *,
    config: Callable[[], Config],
    reporter: app_reporter.Reporter,
) -> int:
    """Review a session's record, print the review and save it under the state dir's reviews/.

    Read-only; no jail. The record is checked before the config loads, so a live or empty
    session is refused without a provider in hand.

    Args:
        cwd: The repository.
        layout: The session to review.
        config: Loads the effective config, the reviewer route already overridden.
        reporter: Receives the status lines, the refusals and the review text.

    Returns:
        The exit code: 0 reviewed, 2 refused, 3 budget exceeded.
    """
    if (why := _not_reviewable(layout)) is not None:
        reporter.error(why)
        return 2
    if (cfg := _runnable_config(config, reporter)) is None:
        return 2
    budget = session_setup.budget_tracker(cfg)
    reviews_dir = paths.state_dir(cwd) / "reviews"
    transcript_sink = TranscriptSink(reviews_dir, objects_dir=reviews_dir / "objects")
    try:
        reviewer = role_providers.build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat="review:run"
        )
    except ProviderError as exc:
        reporter.error(f"provider init failed: {exc}")
        return 2
    digest = review_session.run_digest(layout)
    reporter.note(f"reviewing run: {layout.session_id}")
    try:
        text = review_session.review_digest(
            reviewer,
            digest=digest.render(),
            agents_md=repo_context.agents_md_text(cwd, cfg.sandbox.hide_paths),
        )
    except review_session.RunReviewError as exc:
        reporter.err(f"REVIEW FAILED: {exc}")
        return 2
    except agent6_budget.BudgetExceededError as exc:
        reporter.err(str(exc))  # the reason names the ledger that ran out
        return 3
    reporter.out(text)
    saved = save_review(reviews_dir, label=f"run {layout.session_id}", body=text)
    reporter.note(f"review saved: {saved}")
    budget.settle()
    reporter.err(budget.format_summary())
    return 0
