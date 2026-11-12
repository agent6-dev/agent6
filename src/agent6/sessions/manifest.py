# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The manifest.json shape and its one reader; `app.stamps` is the writer.

Every field defaults and unknown keys are ignored, so a partial or foreign-keyed manifest
renders what it carries: lenience for damage, not a compatibility promise (the shape is
liquid until 1.0). The one strict contract is `session_mode`, the fork and resume privilege
gate, which refuses an unknown mode rather than falling open to the write tools.
"""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, Literal, cast

import pydantic

from agent6 import kinds

_MODEL_CONFIG = pydantic.ConfigDict(frozen=True, extra="ignore")


class ManifestError(Exception):
    """The manifest is missing, unreadable, corrupt, invalid, or names an unknown mode.

    The message is the underlying cause.
    """


class ModelBrief(pydantic.BaseModel):
    """The route and effective effort of a resolved role.

    Attributes:
        provider: The configured provider name.
        model: The provider's model identifier.
        effort: The effective effort; None sends no value.
        effort_source: The origin, including inheritance; empty when not recorded.
    """

    model_config = _MODEL_CONFIG

    provider: str = ""
    model: str = ""
    effort: str | None = None
    effort_source: str = ""


class ModelsBrief(pydantic.BaseModel):
    """The models the run resolved.

    Attributes:
        driver: The model that drove the run, the worker or a plan's planner; None unset.
        reviewer: The reviewer; None unset.
        helper: The helper; None unset.
        review_seats: Each panel seat's facts, in roster order, labelled by index and focus.
        driver_from_flag: The driver came from a `--model`, the run's or a resume's.
    """

    model_config = _MODEL_CONFIG

    driver: ModelBrief | None = None
    reviewer: ModelBrief | None = None
    helper: ModelBrief | None = None
    review_seats: dict[str, ModelBrief] = pydantic.Field(default_factory=dict)
    driver_from_flag: bool = False

    @property
    def replay_driver(self) -> ModelBrief | None:
        """The flag-selected driver a fork without its own `--model` re-applies."""
        return self.driver if self.driver_from_flag else None


class PolicyStamp(pydantic.BaseModel):
    """The policy the run launched under, so every surface and `agent6 exec` read one record.

    Attributes:
        run_commands: `[sandbox].run_commands` at launch.
        isolation: `[sandbox].isolation` at launch.
        network: The network the run's commands join, `session` or `host`; "" when the stamp
            names no level. A manifest an older agent6 wrote holds the configured word.
        commit_per_turn: `[git].commit_per_turn` for the live execution; False means a dirty
            tree at the end is the deliverable, never a stranded commit.
    """

    model_config = _MODEL_CONFIG

    run_commands: str = ""
    isolation: str = ""
    network: str = ""
    commit_per_turn: bool = True


GateSource = Literal["", "configured", "inferred", "adopted"]


class GateStamp(pydantic.BaseModel):
    """The pinned gate and its availability for this execution.

    Attributes:
        command: The pinned argv; only operator config can replace it on resume.
        source: `configured` from operator config, `inferred` from repo signals,
            `adopted` mid-run, or "" for no gate.
        availability: `ready` after resolution, `pending` before resolution, or `withheld`
            when this execution cannot run the command.
    """

    model_config = _MODEL_CONFIG

    command: tuple[str, ...] = ()
    source: GateSource = ""
    availability: Literal["ready", "pending", "withheld"] = "ready"


class HarnessStamp(pydantic.BaseModel):
    """The in-loop strategy the run started with, so `resume` re-applies it.

    Attributes:
        review_when: `[review].when` at launch.
        preset: The preset in force at launch.
        gate: The pinned gate, its source and this execution's availability.
        preset_from_flag: The preset came from `--preset` rather than a config file.
    """

    model_config = _MODEL_CONFIG

    review_when: str = ""
    preset: str = "none"
    gate: GateStamp = GateStamp(availability="pending")
    preset_from_flag: bool = False

    @property
    def replay_preset(self) -> str:
        """The `--preset` override a resume or fork re-applies: a flag-selected one only.

        A config-selected preset re-resolves from the same files; handed back as an override
        it would outrank every config layer, and a run whose repo config beat a global preset
        would resume with the preset winning.
        """
        return self.preset if self.preset_from_flag else ""


class BudgetStamp(pydantic.BaseModel):
    """The caps the budget flags set; a resume without a flag runs under the recorded caps.

    A cap set in config is not recorded: a resume reads it from config.

    Attributes:
        max_usd: `--max-usd`, None when no flag set it.
        max_tokens_fallback: `--max-tokens-fallback`, None when no flag set it.
        max_plan_points: `--max-plan-points`, None when no flag set it.
    """

    model_config = _MODEL_CONFIG

    max_usd: float | None = None
    max_tokens_fallback: int | None = None
    max_plan_points: float | None = None


# The `sha` of a merge that added no commit; the target's own tip is not the run's.
NO_MERGE_COMMIT = "0" * 40


class MergeStamp(pydantic.BaseModel):
    """The record of the run branch's merge.

    Attributes:
        into: The base branch.
        sha: The merge commit in the base, or NO_MERGE_COMMIT.
        ts: When it was merged.
        tip: The run branch tip that was merged. `sessions prune --delete-squashed` deletes
            only while the branch still points here: a resumed run's later commits exist in
            no other ref.
        into_tip: For a merge that added nothing, the base's own tip at the time, the commit
            that already held the run's content; the delete checks it as it checks `sha`.
        via: The run whose merge landed this one, when a later run took its work in under
            `dirty_tree = "include"`; "" for the run's own merge.
        method: The landing method, or null when unrecorded.
        target_before: The target commit before landing, or null when unrecorded.
    """

    model_config = _MODEL_CONFIG

    into: str = ""
    sha: str = ""
    ts: str = ""
    tip: str = ""
    into_tip: str = ""
    via: str = ""
    method: Literal["ff_only", "merge", "squash", "tasks"] | None = None
    target_before: str | None = None

    @property
    def commit(self) -> str:
        """The merge commit's abbreviated sha; "" for a record that names none."""
        return "" if not self.sha or self.sha == NO_MERGE_COMMIT else self.sha[:12]

    def landed(self) -> str:
        """Return where the merge put the run's work, one wording for every surface.

        Returns:
            `merged into <base> as <sha12>`, or `already on <base>, no merge commit`.
        """
        if self.commit:
            return f"merged into {self.into} as {self.commit}"
        return f"already on {self.into}, no merge commit"


class IncludedRun(pydantic.BaseModel):
    """An unmerged run whose whole work this run's start tree held and its commits carry.

    Attributes:
        session_id: That run.
        tip: Its chain tip at this run's start, the tree this run started on.
    """

    model_config = _MODEL_CONFIG

    session_id: str = ""
    tip: str = ""


class ParallelLineage(pydantic.BaseModel):
    """A fan-out lane's place.

    Attributes:
        group: The group it was dispatched in: the parent's id for `run --parallel`,
            `<parent>-p<n>` for a `/parallel` group.
        lane: Its number in the group.
        parent: The session that dispatched it, which every listing nests it under.
        config_notes: Resolved sampling values the lane's TOML snapshot could not preserve.
    """

    model_config = _MODEL_CONFIG

    group: str = ""
    lane: int = 0
    parent: str = ""
    config_notes: tuple[str, ...] = ()


class PlacementStamp(pydantic.BaseModel):
    """A fan-out lane's auto-compare placement.

    Attributes:
        rank: The lane's place, 1 first.
        of: How many lanes were ranked.
        winner: The lane ranked first.
        group: The comparison group recorded on its parent.
    """

    model_config = _MODEL_CONFIG

    rank: int = 0
    of: int = 0
    winner: bool = False
    group: str = ""


class ComparisonStamp(pydantic.BaseModel):
    """A group's ranking and judge spend, owned by its parent.

    Attributes:
        group: The comparison group id.
        ranking: The ranked lane ids, best first.
        ranked_by: The ranking method, `judge` or `mechanical`.
        rationale: The judge's reason, at most 2000 characters; empty for mechanical ranking.
        judge_cost_usd: The group's judge spend in USD, zero when no judge call was made.
        judge_cost_partial: The cost is a lower bound because some judge usage was unpriced.
    """

    model_config = _MODEL_CONFIG

    group: str = ""
    ranking: tuple[str, ...] = ()
    ranked_by: Literal["judge", "mechanical"] = "mechanical"
    rationale: str = ""
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


# Every stamp-rewrite re-stamps it, so the on-disk claim matches the shape on disk.
MANIFEST_VERSION = 13
MANIFEST_NAME = "manifest.json"


class SessionManifest(pydantic.BaseModel):
    """The typed manifest.json a session starts with and later stamps.

    A stamp-rewrite by this version drops keys only a newer version knows, so the write
    path re-stamps `version` to keep the on-disk claim truthful.

    Attributes:
        version: The manifest shape, MANIFEST_VERSION.
        agent6_version: The agent6 that wrote it.
        session_id: The session id.
        mode: The session mode. No default: it is the privilege gate's only input, and a
            manifest that lost the key must not read as the more-privileged `run`.
        start_ts: When the session started.
        user_task: A 4000-character fallback for listings whose journal is missing.
        continued_task: The latest resume steer, clipped the same way; the fallback for the
            continuation line.
        base_sha: The commit the run's un-landed work starts from: where it started, or the
            commit a resume continued from after its work landed.
        base_branch: The branch that commit was on.
        review_base_tree: The base tree of a `/review` session, whose AGENTS.md binds it.
        run_branch: The run's own branch, or None.
        git_control: `[git].control` at start; a `model` run has no per-turn chain or branch.
            The git surfaces refuse through `model_git_refusal`.
        models: The models the run resolved.
        harness: The strategy the run started with.
        policy: The policy the run launched under.
        execution_targets: Recorded target facts by reference, empty in historical manifests.
        budget: The caps the run's or a resume's budget flags set (`--max-usd` and its siblings).
        parked_task: The verbatim task of a run submitted but never started; `agent6 resume`
            starts it fresh and its manifest rewrite clears this.
        parked_operator_task: The parked task before composition, when recorded.
        parked_reason: Why it was parked: the checkout was busy, or the tree was dirty and
            the operator chose to wait.
        source_session_id: The `run --from` source, which contributes context but not this
            session's mode, checkout or history.
        parent_session_id: The fork parent, or None.
        forked_from_turn: The turn the fork rolled back to, or None.
        forked_from_sha: The commit the fork started on, or None.
        worktree: The absolute path of the linked worktree `agent6 fork` or `run --worktree`
            added; None for a session in the operator's checkout. An `/undo` fork names its
            source's.
        worktree_git_dir: The repository git dir granted to a linked checkout's jail, recorded
            before any model command, including for an operator-owned checkout. Resume and
            same-checkout forks inherit it; a writable `.git` pointer cannot replace it.
        workspace_set_up: Whether the setup command ran in the session's own worktree or
            clone; a resume runs it only while this is False. A fork's new worktree starts
            False, and an `/undo` fork carries its source's.
        merged: The merge record, None until the run branch is merged.
        includes: The unmerged run this run took in under `dirty_tree = "include"`, stamped
            merged when this run lands; None when it took none, or more than that run's tip.
        parallel: A fan-out lane's lineage, or None.
        placement: A fan-out lane's rank and group reference, or None.
        comparison: The group's ranking and judge spend on its parent, or None.
    """

    model_config = _MODEL_CONFIG

    version: int = MANIFEST_VERSION
    agent6_version: str = ""
    session_id: str = ""
    mode: str = ""
    start_ts: str = ""
    user_task: str = ""
    continued_task: str = ""
    base_sha: str = ""
    base_branch: str = ""
    review_base_tree: str = ""
    run_branch: str | None = None
    git_control: str = "agent6"
    models: ModelsBrief = ModelsBrief()
    harness: HarnessStamp = HarnessStamp()
    policy: PolicyStamp = PolicyStamp()
    execution_targets: dict[
        str,
        Annotated[kinds.ExecutionTargetStamp, pydantic.BeforeValidator(kinds.read_target_stamp)],
    ] = pydantic.Field(default_factory=dict)
    budget: BudgetStamp = BudgetStamp()
    parked_task: str = ""
    parked_operator_task: str | None = None
    parked_reason: str = ""
    source_session_id: str | None = None
    parent_session_id: str | None = None
    forked_from_turn: int | None = None
    forked_from_sha: str | None = None
    worktree: pathlib.Path | None = None
    worktree_git_dir: pathlib.Path | None = None
    workspace_set_up: bool = False
    merged: MergeStamp | None = None
    includes: IncludedRun | None = None
    parallel: ParallelLineage | None = None
    placement: PlacementStamp | None = None
    comparison: ComparisonStamp | None = None

    def session_mode(self) -> kinds.ResumableMode:
        """Return the session's mode, refusing one this agent6 does not know.

        Fork and resume act on this rather than the raw `mode`, so a damaged manifest never
        escalates a read-only session to the write tools; a render shows `mode` as is.

        Returns:
            The mode.

        Raises:
            ManifestError: The mode is unknown, or its kind is not resumable.
        """
        try:
            kind = kinds.session_kind(self.mode)
        except kinds.UnknownSessionKindError as exc:
            raise ManifestError(str(exc)) from exc
        if not kind.resumable:
            raise ManifestError(f"a {kind.name!r} session is not resumable")
        # Guarded by `resumable` above, which the type system cannot follow.
        return cast(kinds.ResumableMode, kind.name)


def model_git_refusal(manifest: SessionManifest, verb: str) -> str | None:
    """Return the refusal a git surface gives a model-controlled run, or None.

    Args:
        manifest: The session's manifest.
        verb: The surface, for the message.

    Returns:
        The refusal when `git_control` is `model`, whose record is the model's own commits.
    """
    if manifest.git_control != "model":
        return None
    return (
        f"{verb}: run {manifest.session_id or '?'} managed git itself"
        ' ([git].control = "model"); its record is the model\'s own commits,'
        " not an agent6 chain. Inspect it with plain git."
    )


def _newer_note(data: dict[str, object]) -> str | None:
    """Return the note for a record a newer agent6 wrote, or None."""
    version = data.get("version")
    if isinstance(version, int) and version > MANIFEST_VERSION:
        return f"newer record (manifest version {version} > {MANIFEST_VERSION}): upgrade agent6"
    return None


def newer_record(session_dir: pathlib.Path) -> str | None:
    """Return the note for a manifest a newer agent6 wrote; None for any other file.

    A version read only: the status decision asks this on every refresh, and a corrupt
    or missing file is the full reader's business.

    Args:
        session_dir: The session directory.

    Returns:
        The note naming both versions and the upgrade, or None.
    """
    try:
        data = json.loads((session_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _newer_note(data) if isinstance(data, dict) else None


def read_manifest(session_dir: pathlib.Path) -> SessionManifest:
    """Parse a session's manifest.json.

    Args:
        session_dir: The session directory.

    Returns:
        The manifest; every field defaults, so any parseable historical manifest validates.

    Raises:
        ManifestError: The file cannot be read, is not JSON (a truncated file, a torn UTF-8
            tail), is not a JSON object, was written by a newer agent6, or fails validation.
    """
    path = session_dir / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    if (newer := _newer_note(data)) is not None:
        # `extra="ignore"` would read a record it cannot interpret as a plain older one.
        raise ManifestError(newer)
    try:
        return SessionManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        raise ManifestError(str(exc)) from exc


def worktree_blocker(worktree: pathlib.Path | None) -> str:
    """Return the resume refusal for a recorded worktree absent from this home."""
    if worktree is not None and not (worktree / ".git").exists():
        return f"worktree {worktree} not found; resume needs it"
    return ""
