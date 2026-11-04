# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions review`: a read-only review of one finished session's record.

Its own module, imported by the dispatcher for this verb alone, so the provider stack loads
for no other `sessions` verb.
`save_review` writes the rendered reviews in `<state-dir>/reviews/`.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

from agent6 import budget as agent6_budget
from agent6 import paths
from agent6.app import role_providers, session_setup
from agent6.config import ConfigError, layer
from agent6.harness import repo_context, review_session
from agent6.providers import ProviderError, TranscriptSink
from agent6.sessions import id
from agent6.ui.cli import common
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


def _cmd_sessions_review(  # noqa: PLR0911
    config_paths: tuple[pathlib.Path, ...], *, session_id: str, model: str
) -> int:
    """Print a review of a session's record and save it under the state dir's reviews/.

    Read-only; no jail.

    Args:
        config_paths: The `--config` files in command-line order, if any.
        session_id: The session, or "" for the newest.
        model: The `--model` override, `[provider/]model`, applied to the reviewer route.

    Returns:
        The exit code: 0 reviewed, 2 refused, 3 budget exceeded.
    """
    cwd = pathlib.Path.cwd()
    # Every bucket: the review reads a journal, so a plan, an ask or a fan-out is a record too.
    try:
        layout = common.resolve_or_newest_layout(cwd, session_id)
    except id.SessionIdError as exc:
        common.error(f"{exc}")
        return 2
    if layout is None:
        common.print_nothing_yet("sessions")
        return 2
    if not session_id:
        print(f"[agent6] reviewing the newest session: {layout.session_id}", file=sys.stderr)
    if session_is_live(layout.session_dir):
        common.error(
            f"{layout.session_id} is live; its record is not complete. Review it once it"
            f" has ended (`agent6 stop {layout.session_id}` ends it now)."
        )
        return 2
    if not scan_session_log(layout.logs_path).saw_start:
        common.error(f"{layout.session_id} never started; there is no record to review.")
        return 2
    try:
        cfg = layer.load_effective(cwd, config_paths).config
        if model:
            cfg = cfg.with_model_route("reviewer", cfg.model_route("reviewer", model))
    except ConfigError as exc:
        common.error(str(exc))
        return 2
    cfg.require_runnable("reviewer")
    err = session_setup.check_provider_keys(cfg)
    if err is not None:
        common.error(f"{err}")
        return 2
    budget = session_setup.budget_tracker(cfg)
    reviews_dir = paths.state_dir(cwd) / "reviews"
    transcript_sink = TranscriptSink(reviews_dir)
    try:
        reviewer = role_providers.build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat="review:run"
        )
    except ProviderError as exc:
        common.error(f"provider init failed: {exc}")
        return 2
    digest = review_session.run_digest(layout)
    print(f"[agent6] reviewing run: {layout.session_id}", file=sys.stderr)
    try:
        text = review_session.review_digest(
            reviewer,
            digest=digest.render(),
            agents_md=repo_context.agents_md_text(cwd, cfg.sandbox.hide_paths),
        )
    except review_session.RunReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except agent6_budget.BudgetExceededError as exc:
        print(str(exc), file=sys.stderr)  # the reason names the ledger that ran out
        return 3
    print(text, flush=True)
    saved = save_review(reviews_dir, label=f"run {layout.session_id}", body=text)
    print(f"[agent6] review saved: {saved}", file=sys.stderr)
    budget.settle()
    print(budget.format_summary(), file=sys.stderr)
    return 0
