# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions review`: a read-only review of one finished session's record.

Its own module, imported by the dispatcher for this verb alone, so the provider stack loads
for no other `sessions` verb.
The recap itself is `app/review_recap.py`; this module resolves the target and the config.
"""

from __future__ import annotations

import pathlib
import sys

from agent6.app import reporter, review_recap
from agent6.config import Config, layer
from agent6.sessions import id
from agent6.ui.cli import common


def _cmd_sessions_review(
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

    def load() -> Config:
        cfg = layer.load_effective(cwd, config_paths).config
        if model:
            cfg = cfg.with_model_route("reviewer", cfg.model_route("reviewer", model))
        return cfg

    return review_recap.recap(cwd, layout, config=load, reporter=reporter.STDIO_REPORTER)
