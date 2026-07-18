# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read a run's manifest.json into the typed :class:`RunManifest`. The single
reader + the on-disk shape; the writer is ``app.manifest``.

A leaf beside ``layout.py``: pydantic + path arithmetic, no agent6 imports, so
app, the viewmodel, and the CLI parse a run's manifest through one owner and one
shape instead of each re-deriving the read + error-catch + stringly ``.get``.

manifest.json is persistent history: every run dir ever written must keep
rendering, so the model defaults every field and folds legacy shapes (``version:
1`` dirs, the pre-nesting flat ``merged_*`` keys). Reading is lenient
(``read_manifest`` degrades a corrupt file through ``ManifestError``, which the
render consumers already catch and degrade on); the ONE strict contract is
``strict_mode`` -- the fork/resume privilege gate, which refuses an unknown mode
rather than falling open to the write ("run") tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

_MODEL_CONFIG = ConfigDict(frozen=True, extra="ignore")


class ManifestError(Exception):
    """A run's manifest.json is missing, unreadable, corrupt, not a JSON object,
    does not validate, or (via ``strict_mode``) records an unknown privilege
    mode. Carries the underlying cause as its message, so a caller that wants to
    surface a detail can render it."""


class ModelBrief(BaseModel):
    """``{provider, model}`` for a resolved role."""

    model_config = _MODEL_CONFIG

    provider: str = ""
    model: str = ""


class ModelsBrief(BaseModel):
    """The worker/reviewer models the run resolved (a role is null when unset)."""

    model_config = _MODEL_CONFIG

    worker: ModelBrief | None = None
    reviewer: ModelBrief | None = None


class WorkflowStamp(BaseModel):
    """The in-loop strategy the run started with, so ``resume`` re-applies it."""

    model_config = _MODEL_CONFIG

    critic: str = ""
    revise_prompt: str = ""
    profile: str = ""


class MergeStamp(BaseModel):
    """Recorded once a run branch is merged, so later tooling tells a merged run
    branch from an unmerged one (nests the pre-v2 flat merged_into/_sha/_ts)."""

    model_config = _MODEL_CONFIG

    into: str = ""
    sha: str = ""
    ts: str = ""


class CompareStamp(BaseModel):
    """A fan-out lane's auto-compare placement. The fan-out id itself lives in the
    top-level ``parallel_id``; this stamp no longer duplicates it as ``group``."""

    model_config = _MODEL_CONFIG

    rank: int = 0
    of: int = 0
    winner: bool = False
    ranked_by: str = ""
    rationale: str = ""
    # The judge call's cost for the WHOLE group, recorded on every lane like
    # the rationale; summing it across lanes would double-count. 0.0 only when
    # no judge call was made (a failed judge that fell back mechanically still
    # spent); partial marks a lower bound (unpriced reviewer, no reported cost).
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


# The shape this binary writes. Stamp-rewrites re-stamp it (see write_manifest)
# so a manifest's version claim always matches the shape actually on disk.
MANIFEST_VERSION = 2


class RunManifest(BaseModel):
    """The typed manifest.json a run starts with (and later stamps).

    Every field defaults so ANY historical run dir on disk still parses (old
    ``version: 1`` dirs, dirs missing later-added fields). ``extra="ignore"`` on
    read drops keys this version dropped (the legacy ``compare.group``); the
    writer always emits the full shape. Known limitation: a stamp-rewrite by
    this version drops keys only a NEWER version knows (load -> model_copy ->
    dump cannot carry them), so the write path re-stamps ``version`` to keep
    the on-disk claim truthful.
    """

    model_config = _MODEL_CONFIG

    version: int = MANIFEST_VERSION
    agent6_version: str = ""
    run_id: str = ""
    mode: str = "run"  # run | plan; privilege-gated strictly via strict_mode()
    start_ts: str = ""
    user_task: str = ""
    base_sha: str = ""
    base_branch: str = ""
    run_branch: str | None = None
    models: ModelsBrief = ModelsBrief()
    workflow: WorkflowStamp = WorkflowStamp()
    # fork lineage (a non-forked run leaves these null)
    parent_run_id: str | None = None
    forked_from_turn: int | None = None
    forked_from_sha: str | None = None
    # merge stamp (null until the run branch is merged)
    merged: MergeStamp | None = None
    # parallel lineage + compare stamp (null outside a fan-out)
    parallel_id: str | None = None
    lane: int | None = None
    compare: CompareStamp | None = None

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_keys(cls, data: Any) -> Any:
        """Fold the pre-v2 flat merge keys (merged_into/merged_sha/merged_ts) into
        the nested ``merged`` stamp, so a run merged before this reshape still
        reads its merge record."""
        if not isinstance(data, dict) or data.get("merged"):
            return data
        if data.get("merged_into") or data.get("merged_sha"):
            data = dict(data)
            data["merged"] = {
                "into": data.get("merged_into", ""),
                "sha": data.get("merged_sha", ""),
                "ts": data.get("merged_ts", ""),
            }
        return data

    def strict_mode(self) -> Literal["run", "plan"]:
        """The privilege-gating mode for fork/resume. Refuses anything but the two
        known modes, so a damaged manifest never silently escalates a plan run to
        the more-privileged write ("run") tools. Pure-render consumers read the
        raw ``mode`` string for display instead."""
        if self.mode in ("run", "plan"):
            return self.mode  # type: ignore[return-value]
        raise ManifestError(f"unknown run mode {self.mode!r}")


def read_manifest(run_dir: Path) -> RunManifest:
    """Parse ``<run_dir>/manifest.json`` into a :class:`RunManifest`, or raise
    ``ManifestError``.

    Lenient by design: every field defaults, so any parseable historical manifest
    validates and renders. A file that cannot be read (``OSError``), is not JSON
    (any ``ValueError``: a truncated JSON is a ``JSONDecodeError`` and a
    torn-UTF-8 tail a ``UnicodeDecodeError``, both subclasses), is not a JSON
    object, or fails validation degrades through the one typed error the render
    consumers already catch; the fork/resume gate turns it into a loud refusal.
    """
    path = run_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    try:
        return RunManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(str(exc)) from exc
