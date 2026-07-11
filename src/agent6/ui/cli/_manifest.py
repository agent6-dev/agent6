# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The canonical manifest.json a run starts with (written by run/fork)."""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

from agent6 import __version__
from agent6.config import Config
from agent6.portable import atomic_write
from agent6.runs.layout import RunLayout


def _manifest_model_brief(rm: Any) -> dict[str, str] | None:
    """``{provider, model}`` for a resolved role, or None when unset."""
    if rm is None:
        return None
    return {"provider": rm.provider, "model": rm.model}


def write_run_manifest(
    layout: RunLayout,
    *,
    run_id: str,
    user_task: str,
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
    mode: str = "run",
    effective_profile: str = "",
    parent_run_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
) -> None:
    """Write the canonical manifest.json for a run.

    This is the only thing that reads/writes ``layout.manifest_path``.
    Format is JSON for the same reason logs.jsonl is JSON: trivially
    grep-able from a shell and easy to consume from any language. The
    on-disk shape is *liquid* until 1.0 - bump ``version`` only when
    the new shape genuinely improves a downstream consumer.

    ``parent_run_id`` / ``forked_from_turn`` / ``forked_from_sha`` are set only
    for a run created by ``agent6 fork``; they record the lineage (source run +
    the turn forked from + the workspace sha at that turn). A non-forked run
    leaves them out.
    """
    manifest: dict[str, Any] = {
        "version": 2,
        "agent6_version": __version__,
        "run_id": run_id,
        "mode": mode,  # run | plan (ask runs live under asks/, not here)
        "start_ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        "user_task": user_task[:4000],
        "base_sha": base_sha,
        "base_branch": base_branch,
        "run_branch": run_branch,
        "models": {
            "worker": _manifest_model_brief(cfg.models.resolve("worker")),
            "reviewer": _manifest_model_brief(cfg.models.resolve("reviewer")),
        },
        "workflow": {
            "critic": cfg.review.trigger,
            "revise_prompt": cfg.prompt.revise_prompt,
            # The profile the run actually used (--profile flag or top-level
            # `profile`), so `agent6 resume` re-applies the same strategy.
            "profile": effective_profile,
        },
    }
    if parent_run_id is not None:
        manifest["parent_run_id"] = parent_run_id
        manifest["forked_from_turn"] = forked_from_turn
        manifest["forked_from_sha"] = forked_from_sha
    # Durable temp+replace: the TUI hub and `runs show` poll this file on live
    # runs, and resume/fork need the manifest after a crash.
    atomic_write(layout.manifest_path, json.dumps(manifest, indent=2) + "\n")
