# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The manifest.json shape and its one reader; `app.manifest` is the writer.

Every field defaults and unknown keys are ignored, so a partial or foreign-keyed manifest
renders what it carries: lenience for damage, not a compatibility promise (the shape is
liquid until 1.0). The one strict contract is `session_mode`, the fork and resume privilege
gate, which refuses an unknown mode rather than falling open to the write tools.
"""

from __future__ import annotations

import json
import pathlib
from typing import cast

import pydantic

from agent6 import kinds

_MODEL_CONFIG = pydantic.ConfigDict(frozen=True, extra="ignore")


class ManifestError(Exception):
    """The manifest is missing, unreadable, corrupt, invalid, or names an unknown mode.

    The message is the underlying cause.
    """


class ModelBrief(pydantic.BaseModel):
    """The provider and model of a resolved role."""

    model_config = _MODEL_CONFIG

    provider: str = ""
    model: str = ""


class ModelsBrief(pydantic.BaseModel):
    """The models the run resolved.

    Attributes:
        driver: The model that drove the run, the worker or a plan's planner; None unset.
        reviewer: The reviewer; None unset.
        driver_from_flag: The driver came from a `--model`, the run's or a resume's.
    """

    model_config = _MODEL_CONFIG

    driver: ModelBrief | None = None
    reviewer: ModelBrief | None = None
    driver_from_flag: bool = False

    @property
    def replay_driver(self) -> ModelBrief | None:
        """The driver a resume or fork without its own `--model` re-applies: a flag-selected one."""
        return self.driver if self.driver_from_flag else None


class PolicyStamp(pydantic.BaseModel):
    """The policy the run launched under, so every surface and `agent6 exec` read one record.

    Attributes:
        run_commands: `[sandbox].run_commands` at launch.
        isolation: `[sandbox].isolation` at launch.
        network: `[sandbox].network` at launch.
        commit_per_step: `[git].commit_per_step` for the live execution; False means a dirty
            tree at the end is the deliverable, never a stranded commit.
    """

    model_config = _MODEL_CONFIG

    run_commands: str = ""
    isolation: str = ""
    network: str = ""
    commit_per_step: bool = True


class HarnessStamp(pydantic.BaseModel):
    """The in-loop strategy the run started with, so `resume` re-applies it.

    Attributes:
        review_trigger: `[review].trigger` at launch.
        revise_prompt: `[prompt].revise_prompt` at launch.
        preset: The preset in force at launch.
        verify_command: The gate the run is pinned to, so a mid-run edit to its source cannot
            move it; on a resumed execution only operator config outranks it.
        verify_origin: `configured` (a config file, which the model cannot write), `inferred`
            (repo signals, which it can), `adopted` (gained mid-run by a gateless run), or ""
            for no gate.
        preset_from_flag: The preset came from `--preset` rather than a config file.
    """

    model_config = _MODEL_CONFIG

    review_trigger: str = ""
    revise_prompt: str = ""
    preset: str = ""
    verify_command: tuple[str, ...] = ()
    verify_origin: str = ""
    preset_from_flag: bool = False

    @property
    def replay_preset(self) -> str:
        """The `--preset` override a resume or fork re-applies: a flag-selected one only.

        A config-selected preset re-resolves from the same files; handed back as an override
        it would outrank every config layer, and a run whose repo config beat a global preset
        would resume with the preset winning.
        """
        return self.preset if self.preset_from_flag else ""


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
    """

    model_config = _MODEL_CONFIG

    into: str = ""
    sha: str = ""
    ts: str = ""
    tip: str = ""
    into_tip: str = ""

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


class ParallelLineage(pydantic.BaseModel):
    """A fan-out lane's place.

    Attributes:
        group: The group it was dispatched in: the coordinator's id for `run --parallel`,
            `<coordinator>-p<n>` for a `/parallel` group.
        lane: Its number in the group.
        coordinator: The session that dispatched it, which every listing nests it under.
    """

    model_config = _MODEL_CONFIG

    group: str = ""
    lane: int = 0
    coordinator: str = ""


class FanoutStamp(pydantic.BaseModel):
    """The record a `run --parallel` fan-out leaves on its own session.

    A `/parallel` group's coordinator is an ordinary run and carries none.

    Attributes:
        lanes: How many lanes it dispatched.
        spec: The `--parallel` argument as typed.
    """

    model_config = _MODEL_CONFIG

    lanes: int = 0
    spec: str = ""


class CompareStamp(pydantic.BaseModel):
    """A fan-out lane's auto-compare placement.

    Attributes:
        rank: The lane's place, 1 first.
        of: How many lanes were ranked.
        winner: The lane ranked first.
        ranked_by: What ranked it.
        rationale: The judge's reason.
        judge_cost_usd: The judge call's cost for the whole group, recorded on every lane, so
            summing it across lanes double-counts; 0.0 only when no judge call was made.
        judge_cost_partial: The cost is a lower bound (an unpriced reviewer).
    """

    model_config = _MODEL_CONFIG

    rank: int = 0
    of: int = 0
    winner: bool = False
    ranked_by: str = ""
    rationale: str = ""
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


# Every stamp-rewrite re-stamps it, so the on-disk claim matches the shape on disk.
MANIFEST_VERSION = 4
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
        user_task: The task, truncated for display; `parked_task` holds it verbatim.
        base_sha: The commit the run started on.
        base_branch: The branch the run started on.
        run_branch: The run's own branch, or None.
        git_control: `[git].control` at start; a `model` run has no chain or branch, and the
            git surfaces refuse through `model_git_refusal`.
        models: The models the run resolved.
        harness: The strategy the run started with.
        policy: The policy the run launched under.
        parked_task: The verbatim task of a run submitted but never started; `agent6 resume`
            starts it fresh and its manifest rewrite clears this.
        parked_reason: Why it was parked: the checkout was busy, or the tree was dirty and
            the operator chose to wait.
        source_session_id: The `run --from` source, which contributes context but not this
            session's mode, checkout or history.
        parent_session_id: The fork parent, or None.
        forked_from_turn: The turn the fork rolled back to, or None.
        forked_from_sha: The commit the fork started on, or None.
        worktree: The absolute path of the linked worktree `agent6 fork` added; None for a
            session in the operator's checkout. An `/undo` fork names its source's.
        worktree_git_dir: The repository git dir that worktree points into: the one path a
            fork execution's jail grants beyond the workspace. Never read back from the
            worktree's own `.git` pointer, which a jailed command can rewrite under hardened.
        merged: The merge record, None until the run branch is merged.
        parallel: A fan-out lane's lineage, or None.
        compare: A fan-out lane's compare stamp, or None.
        fanout: A fan-out's own record on its coordinator, or None.
    """

    model_config = _MODEL_CONFIG

    version: int = MANIFEST_VERSION
    agent6_version: str = ""
    session_id: str = ""
    mode: str = ""
    start_ts: str = ""
    user_task: str = ""
    base_sha: str = ""
    base_branch: str = ""
    run_branch: str | None = None
    git_control: str = "agent6"
    models: ModelsBrief = ModelsBrief()
    harness: HarnessStamp = HarnessStamp()
    policy: PolicyStamp = PolicyStamp()
    parked_task: str = ""
    parked_reason: str = ""
    source_session_id: str | None = None
    parent_session_id: str | None = None
    forked_from_turn: int | None = None
    forked_from_sha: str | None = None
    worktree: pathlib.Path | None = None
    worktree_git_dir: pathlib.Path | None = None
    merged: MergeStamp | None = None
    parallel: ParallelLineage | None = None
    compare: CompareStamp | None = None
    fanout: FanoutStamp | None = None

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


def read_manifest(session_dir: pathlib.Path) -> SessionManifest:
    """Parse a session's manifest.json.

    Args:
        session_dir: The session directory.

    Returns:
        The manifest; every field defaults, so any parseable historical manifest validates.

    Raises:
        ManifestError: The file cannot be read, is not JSON (a truncated file, a torn UTF-8
            tail), is not a JSON object, or fails validation.
    """
    path = session_dir / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    if "harness" not in data and "workflow" in data:
        data["harness"] = data.pop("workflow")  # the stamp's key through version 3
    try:
        return SessionManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        raise ManifestError(str(exc)) from exc
