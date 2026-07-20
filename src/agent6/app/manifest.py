# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build + write the canonical manifest.json a run starts with (run/fork). The
reader and the on-disk shape (:class:`RunManifest`) live in ``runs.manifest``."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from agent6 import __version__
from agent6.config import Config
from agent6.portable import atomic_write
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import (
    MANIFEST_VERSION,
    ModelBrief,
    ModelsBrief,
    RunManifest,
    WorkflowStamp,
)


def _model_brief(rm: Any) -> ModelBrief | None:
    """A ``ModelBrief`` for a resolved role, or None when unset."""
    if rm is None:
        return None
    return ModelBrief(provider=rm.provider, model=rm.model)


def write_manifest(path: Path, m: RunManifest) -> None:
    """Serialize *m* to *path* (indent=2 + trailing newline), atomically.

    The one place a RunManifest reaches disk: the initial write below and the
    stamp rewrites (merge / lineage / compare) all route through here, so the
    format lives in one spot. Durable temp+replace: the TUI hub and `runs show`
    poll this file on live runs, and resume/fork need it after a crash.

    Re-stamps ``version``: a stamp-rewrite of a manifest written by a NEWER
    agent6 drops keys this version doesn't know (extra="ignore" on read), so
    the written file must claim the shape it actually has, not the one it lost.
    """
    if m.version != MANIFEST_VERSION:
        m = m.model_copy(update={"version": MANIFEST_VERSION})
    atomic_write(path, m.model_dump_json(indent=2) + "\n")


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
    parked_task: str = "",
    parent_run_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
) -> None:
    """Write the canonical manifest.json for a run.

    Format is JSON for the same reason logs.jsonl is JSON: trivially grep-able
    from a shell and easy to consume from any language. The on-disk shape is
    *liquid* until 1.0 - bump ``RunManifest.version`` only when the new shape
    genuinely improves a downstream consumer.

    ``parent_run_id`` / ``forked_from_turn`` / ``forked_from_sha`` are set only
    for a run created by ``agent6 fork``; they record the lineage (source run +
    the turn forked from + the workspace sha at that turn). A non-forked run
    leaves them null.
    """
    m = RunManifest(
        agent6_version=__version__,
        run_id=run_id,
        mode=mode,  # run | plan (ask runs live under asks/, not here)
        start_ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        # Display stamp only; RunSnapshot.original_task carries the verbatim
        # engine copy. Truncation here must never feed the engine.
        user_task=user_task[:4000],
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        models=ModelsBrief(
            worker=_model_brief(cfg.models.resolve("worker")),
            reviewer=_model_brief(cfg.models.resolve("reviewer")),
        ),
        workflow=WorkflowStamp(
            critic=cfg.review.trigger,
            revise_prompt=cfg.prompt.revise_prompt,
            # The profile the run actually used (--profile flag or top-level
            # `profile`), so `agent6 resume` re-applies the same strategy.
            profile=effective_profile,
        ),
        parked_task=parked_task,
        parent_run_id=parent_run_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
    )
    write_manifest(layout.manifest_path, m)
