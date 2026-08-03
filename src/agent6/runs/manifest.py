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
``validated_mode`` -- the fork/resume privilege gate, which refuses an unknown mode
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
    does not validate, or (via ``validated_mode``) records an unknown privilege
    mode. Carries the underlying cause as its message, so a caller that wants to
    surface a detail can render it."""


class ModelBrief(BaseModel):
    """``{provider, model}`` for a resolved role."""

    model_config = _MODEL_CONFIG

    provider: str = ""
    model: str = ""


class ModelsBrief(BaseModel):
    """The models the run resolved: the one that DROVE it (the worker, or the
    planner for a plan run) and the reviewer. Null when the role is unset."""

    model_config = _MODEL_CONFIG

    driver: ModelBrief | None = None
    reviewer: ModelBrief | None = None


class WorkflowStamp(BaseModel):
    """The in-loop strategy the run started with, so ``resume`` re-applies it."""

    model_config = _MODEL_CONFIG

    critic: str = ""
    revise_prompt: str = ""
    preset: str = ""
    # Whether `preset` was chosen by --preset rather than by a config file.
    # The name alone is half the fact: replaying a config-selected one as a flag
    # splices it ABOVE the repo config it originally lost to (see replay_preset).
    preset_from_flag: bool = False

    @property
    def replay_preset(self) -> str:
        """The ``--preset`` override a resumed or forked leg must re-apply.

        Only a FLAG-selected preset: a config-selected one re-resolves
        identically from the same config files, whereas handing its name back as
        an override makes `_select_preset` call it a flag, which outranks every
        config layer. A run whose repo config beat a global preset therefore
        came back from resume with the preset winning instead -- gaining, for
        example, a blocking review veto the original never had.
        """
        return self.preset if self.preset_from_flag else ""


class MergeStamp(BaseModel):
    """Recorded once a run branch is merged, so later tooling tells a merged run
    branch from an unmerged one (nests the pre-v2 flat merged_into/_sha/_ts)."""

    model_config = _MODEL_CONFIG

    into: str = ""
    sha: str = ""
    ts: str = ""
    # The RUN BRANCH tip that was merged (``sha`` is the commit in the base).
    # `runs prune --delete-squashed` force-deletes only when the branch still
    # points here: a resumed run keeps committing on the same branch under this
    # stamp, and those commits exist in no other ref.
    tip: str = ""


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
MANIFEST_VERSION = 3


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
    # No default mode: the field is the privilege gate's only input, and a
    # manifest that lost the key (truncated, hand-edited, foreign writer) must
    # not read as the more-privileged "run". Display consumers show "?" for it.
    mode: str = ""
    start_ts: str = ""
    user_task: str = ""
    base_sha: str = ""
    base_branch: str = ""
    run_branch: str | None = None
    models: ModelsBrief = ModelsBrief()
    workflow: WorkflowStamp = WorkflowStamp()
    # A parked run: submitted while another run-mode worker held the checkout
    # (the repo.lock refusal). Holds the VERBATIM task -- user_task above is
    # the truncated display twin -- and non-empty means the run never started:
    # `agent6 resume <id>` starts it fresh, whose manifest rewrite clears it.
    parked_task: str = ""
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

    def validated_mode(self) -> Literal["run", "plan"]:
        """The mode fork/resume may act on: anything but the two known ones is
        refused, so a damaged manifest never silently escalates a plan run to the
        more-privileged write ("run") tools. Pure-render consumers read the raw
        ``mode`` string for display instead."""
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
