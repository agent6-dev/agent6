# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Write the manifest.json a run starts with, and every stamp that rewrites it.

The reader and the on-disk shape (`SessionManifest`) live in `sessions.manifest`.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import pathlib
from collections.abc import Mapping
from typing import Any

import pydantic

from agent6 import __version__, kinds, portable, task_text
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError, ReviewerConfig, RoleModel
from agent6.models import registry
from agent6.sessions import event_log, manifest
from agent6.sessions import layout as sessions_layout
from agent6.tools import target_policy

_LEVELS: dict[str, kinds.IsolationLevel] = {"full": "full", "partial": "partial", "none": "none"}


def _policy_stamp(cfg: Config, isolation: str) -> manifest.PolicyStamp:
    """Return the policy stamp: the resolved level and the network its commands join.

    Args:
        cfg: The run's config.
        isolation: The resolved isolation level; "" for a stamp naming no level (a fan-out
            coordinator's), which then knows no network either.

    Returns:
        The stamp, holding resolved words since `auto` degrades.
    """
    level = _LEVELS.get(isolation)
    return manifest.PolicyStamp(
        run_commands=cfg.sandbox.run_commands,
        isolation=isolation or str(cfg.sandbox.isolation),
        network=target_policy.resolve_network(cfg, level) if level is not None else "",
        commit_per_turn=cfg.git.commit_per_turn,
    )


def _model_brief(cfg: Config, rm: RoleModel | None, source: str) -> manifest.ModelBrief | None:
    """Return the effective effort and origin, or None without a route."""
    if rm is None:
        return None
    effort = registry.model_effort(cfg, rm)
    return manifest.ModelBrief(
        provider=rm.provider,
        model=rm.model,
        effort=effort,
        effort_source=(
            "provider default"
            if effort is None
            else "backend default"
            if rm.effort is None
            else source
        ),
    )


def _models_brief(
    cfg: Config, mode: str, driver_from_flag: bool, sources: Mapping[str, str]
) -> manifest.ModelsBrief:
    """Return the execution's roles and seats with each effort's config origin."""

    def role_source(role: kinds.RoleName) -> str:
        source_role = cfg.models.source_role(role, "effort")
        key = f"models.{source_role}.effort"
        origin = sources.get(key, "configured")
        return f"{key}, {origin}" if source_role != role else origin

    reviewer = cfg.models.resolve("reviewer")
    seats: dict[str, manifest.ModelBrief] = {}
    if cfg.review.trigger != "off":
        roster = cfg.review.reviewers or (ReviewerConfig(focus="security"),)
        for index, entry in enumerate(roster, 1):
            source = (
                sources.get("review.reviewers", "configured")
                if entry.effort is not None
                else f"models.reviewer.effort, {role_source('reviewer')}"
            )
            brief = _model_brief(cfg, entry.resolve(reviewer), source)
            if brief is not None:
                seats[f"review seat {index} ({entry.focus})"] = brief
    role = kinds.session_kind(mode).role
    return manifest.ModelsBrief(
        driver=_model_brief(cfg, cfg.models.resolve(role), role_source(role)),
        reviewer=_model_brief(cfg, reviewer, role_source("reviewer")),
        helper=_model_brief(cfg, cfg.models.resolve("helper"), role_source("helper")),
        review_seats=seats,
        driver_from_flag=driver_from_flag,
    )


def write_manifest(path: pathlib.Path, m: manifest.SessionManifest) -> None:
    """Serialize the manifest to disk atomically; the one place one reaches disk.

    An older manifest is upgraded to the shape written; a newer one is refused, since
    `extra="ignore"` would drop the keys this binary does not know.

    Args:
        path: The manifest file.
        m: The manifest to write.

    Raises:
        ManifestError: The manifest on disk is newer than this agent6 understands.
    """
    if m.version > manifest.MANIFEST_VERSION:
        raise manifest.ManifestError(
            f"refusing to rewrite {path}: it is version {m.version}, newer than this agent6 "
            f"understands (version {manifest.MANIFEST_VERSION}). Upgrade agent6 to stamp this run."
        )
    if m.version != manifest.MANIFEST_VERSION:
        m = m.model_copy(update={"version": manifest.MANIFEST_VERSION})
    portable.atomic_write(path, m.model_dump_json(indent=2) + "\n")


def write_session_manifest(
    layout: sessions_layout.SessionLayout,
    *,
    session_id: str,
    source_session_id: str | None = None,
    user_task: str,
    operator_task: str | None = None,
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
    config_sources: Mapping[str, str] | None = None,
    mode: str = "run",
    effective_preset: str = "none",
    preset_from_flag: bool = False,
    driver_from_flag: bool = False,
    gate: manifest.GateStamp | None = None,
    isolation: str = "",
    parent_session_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
    worktree: pathlib.Path | None = None,
    worktree_git_dir: pathlib.Path | None = None,
    review_base_tree: str = "",
    workspace_set_up: bool = False,
    fanout: manifest.FanoutStamp | None = None,
    budget: manifest.BudgetStamp | None = None,
) -> None:
    """Write the manifest.json a new run or fork starts with.

    Args:
        layout: The session's directory layout.
        session_id: The new session's id.
        source_session_id: The session whose context `--from` seeded into this one.
        user_task: The operator's task.
        operator_task: The words before composition, when available.
        base_sha: The commit the run started from.
        base_branch: The branch the run started from.
        run_branch: The branch the start cut, when one was.
        cfg: The resolved config.
        config_sources: The layer that supplied each config leaf.
        mode: run, plan or ask.
        effective_preset: The preset the run uses.
        preset_from_flag: The preset came from `--preset`, so a resume replays it.
        driver_from_flag: The driver model came from `--model`.
        gate: A fork's pinned gate record from its source.
        isolation: The resolved isolation level.
        review_base_tree: The base tree of a `/review` session; "" otherwise.
        parent_session_id: A fork's source run.
        forked_from_turn: The turn a fork was cut at.
        forked_from_sha: The workspace sha at that turn.
        worktree: The session's own checkout, a fork's or a `run --worktree`'s.
        worktree_git_dir: The repository git dir the worktree points into.
        workspace_set_up: Whether the checkout already ran its setup command (an `/undo`
            fork in its source's worktree).
        fanout: A `run --parallel` coordinator's own record.
        budget: The caps the run's `--max-*` flags set, so a resume replays them.
    """
    lineage = parallel_lineage()
    if gate is None:
        gate = manifest.GateStamp(
            command=cfg.harness.verify_command,
            source="configured" if cfg.harness.verify_command else "",
            availability="pending" if mode in ("run", "plan") else "ready",
        )
    m = manifest.SessionManifest(
        agent6_version=__version__,
        session_id=session_id,
        # `fork` and `resume` act on session_mode(), never on this string.
        mode=mode,
        start_ts=datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="microseconds"),
        user_task=(
            operator_task if operator_task is not None else task_text.operator_task_text(user_task)
        )[:4000],
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        git_control=cfg.git.control,
        models=_models_brief(cfg, mode, driver_from_flag, config_sources or {}),
        harness=manifest.HarnessStamp(
            review_trigger=cfg.review.trigger,
            preset=effective_preset,
            preset_from_flag=preset_from_flag,
            gate=gate,
        ),
        policy=_policy_stamp(cfg, isolation),
        budget=budget or manifest.BudgetStamp(),
        source_session_id=source_session_id,
        parent_session_id=parent_session_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
        worktree=worktree,
        worktree_git_dir=worktree_git_dir,
        review_base_tree=review_base_tree,
        workspace_set_up=workspace_set_up,
        parallel=lineage,
        fanout=fanout,
    )
    write_manifest(layout.manifest_path, m)


def lane_gate_source() -> manifest.GateSource:
    """Return the origin of the gate a lane's config carries, or "" outside a lane.

    `AGENT6_LANE_GATE_SOURCE`: the coordinator resolves the gate in the origin, where the
    repo signals are, and writes it into the lane's config; the lane alone would read it
    as configured.

    Returns:
        The gate's origin, "" outside a lane or when the coordinator passed none.
    """
    if parallel_lineage() is None:
        return ""
    source = os.environ.get("AGENT6_LANE_GATE_SOURCE", "")
    return source if source in ("configured", "inferred", "adopted") else ""


def parallel_lineage() -> manifest.ParallelLineage | None:
    """Return the fan-out lineage the spawner stamped into this lane's environment, or None.

    `AGENT6_PARALLEL_LINEAGE=<coordinator>:<group>:<lane>`. Read at the manifest's write so the
    grouping survives a coordinator death.

    Returns:
        The lineage, None outside a lane or when the value does not parse.

    Raises:
        ConfigError: The serialized lane config notes are not a JSON array of strings.
    """
    raw = os.environ.get("AGENT6_PARALLEL_LINEAGE", "")
    coordinator, _, rest = raw.partition(":")
    group, sep, lane = rest.rpartition(":")
    if not sep or not coordinator or not group or not lane.isdigit():
        return None
    try:
        notes = pydantic.TypeAdapter(tuple[str, ...]).validate_json(
            os.environ.get("AGENT6_LANE_CONFIG_NOTES", "[]"), strict=True
        )
    except pydantic.ValidationError as exc:
        raise ConfigError(f"invalid lane config notes: {exc}") from exc
    return manifest.ParallelLineage(
        group=group, lane=int(lane), coordinator=coordinator, config_notes=notes
    )


def stamp_parked(
    session_dir: pathlib.Path, *, task: str, reason: str, operator_task: str | None = None
) -> None:
    """Record that the run was submitted and never started; `unpark` replaces the stamp.

    Args:
        session_dir: The run.
        task: The verbatim task a resume starts fresh.
        operator_task: The words before composition, when available.
        reason: Why the run waits.
    """
    m = manifest.read_manifest(session_dir)
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(
            update={
                "parked_task": task,
                "parked_operator_task": operator_task,
                "parked_reason": reason,
                "run_branch": None,
            }
        ),
    )


def parked_stamp(session_dir: pathlib.Path) -> tuple[str, str, str | None] | None:
    """Return a parked submission's (task, reason, operator_task), or None without a park."""
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        return None
    return (m.parked_task, m.parked_reason, m.parked_operator_task) if m.parked_task else None


def unpark(session_dir: pathlib.Path, *, run_branch: str | None) -> None:
    """End the park at the execution's start and record the branch the start cut.

    A manifest carrying no park, or none readable, is left alone.

    Args:
        session_dir: The session directory.
        run_branch: The run's branch, None for a run without one.
    """
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        return
    if not m.parked_task:
        return
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(
            update={
                "parked_task": "",
                "parked_operator_task": None,
                "parked_reason": "",
                "run_branch": run_branch,
            }
        ),
    )


def stamp_execution(
    session_dir: pathlib.Path,
    cfg: Config,
    mode: str,
    isolation: str,
    *,
    config_sources: Mapping[str, str] | None = None,
) -> None:
    """Re-stamp what an execution owns: the models driving it and the policy it runs under.

    `agent6 exec` joins the recorded policy's jail and `sessions show` reads the recorded model.

    Args:
        session_dir: The session directory.
        cfg: The execution's config; its preset replaces the recorded one unless that came
            from `--preset`.
        mode: The session mode, which picks the driving role.
        isolation: The resolved isolation level.
        config_sources: The layer that supplied each config leaf.
    """
    m = manifest.read_manifest(session_dir)
    harness = m.harness
    if not harness.preset_from_flag:
        harness = harness.model_copy(update={"preset": cfg.preset})
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(
            update={
                "models": _models_brief(cfg, mode, m.models.driver_from_flag, config_sources or {}),
                "harness": harness,
                "policy": _policy_stamp(cfg, isolation),
            }
        ),
    )


def stamp_workspace_set_up(session_dir: pathlib.Path) -> None:
    """Record that the session's own worktree or clone ran its setup command."""
    m = manifest.read_manifest(session_dir)
    write_manifest(
        session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"workspace_set_up": True})
    )


def stamp_preset(session_dir: pathlib.Path, name: str) -> None:
    """Record the preset `--preset` set on a resume; a later resume without the flag replays it."""
    m = manifest.read_manifest(session_dir)
    harness = m.harness.model_copy(update={"preset": name, "preset_from_flag": True})
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"harness": harness}))


def stamp_model(session_dir: pathlib.Path, route: kinds.ModelRoute) -> None:
    """Record the route `--model` set on a resume; a later resume without the flag replays it."""
    m = manifest.read_manifest(session_dir)
    models = m.models.model_copy(
        update={
            "driver": manifest.ModelBrief(provider=route.provider, model=route.model),
            "driver_from_flag": True,
        }
    )
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"models": models}))


def stamp_budget(session_dir: pathlib.Path, budget: manifest.BudgetStamp) -> None:
    """Record the caps in force after a resume's `--max-*` flags; a later resume replays them."""
    m = manifest.read_manifest(session_dir)
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"budget": budget}))


def stamp_continued_from(session_dir: pathlib.Path, sha: str, branch: str) -> None:
    """Move the run's base to the commit its chain continues from after its work landed.

    The merge record's tip moves with it: the commit adds nothing over what landed, so the
    run still reads merged, and its next merge starts from there.

    Args:
        session_dir: The session's directory.
        sha: The commit the chain continues from.
        branch: The base branch from here on.
    """
    m = manifest.read_manifest(session_dir)
    merged = m.merged.model_copy(update={"tip": sha}) if m.merged is not None else None
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(update={"base_sha": sha, "base_branch": branch, "merged": merged}),
    )


def stamp_base(session_dir: pathlib.Path, sha: str) -> None:
    """Move the run's base to the commit its unlanded work starts at; its chain stays."""
    m = manifest.read_manifest(session_dir)
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"base_sha": sha}))


def stamp_includes(session_dir: pathlib.Path, included: manifest.IncludedRun) -> None:
    """Record the unmerged run whose work this run started on and carries in its commits."""
    m = manifest.read_manifest(session_dir)
    write_manifest(
        session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"includes": included})
    )


def stamp_fork_task(session_dir: pathlib.Path, steer: str, *, source_dir: pathlib.Path) -> None:
    """Record a fork's steer: its own task while it still carries the source's, else a continuation.

    A fork's journal opens with the steer that sent it elsewhere, so the fold names the fork by
    it; the manifest agrees. An unreadable manifest leaves the fork as it stands, and a pruned
    source makes the steer a continuation.

    Args:
        session_dir: The fork's session directory.
        steer: The steer that resumed the fork.
        source_dir: The source's session directory.
    """
    with contextlib.suppress(manifest.ManifestError, OSError):
        m = manifest.read_manifest(session_dir)
        try:
            source_task: str | None = manifest.read_manifest(source_dir).user_task
        except manifest.ManifestError:
            source_task = None
        field = (
            "user_task"
            if source_task is not None and m.user_task == source_task
            else "continued_task"
        )
        write_manifest(
            session_dir / manifest.MANIFEST_NAME,
            m.model_copy(update={field: task_text.operator_task_text(steer)[:4000]}),
        )


def stamp_continuation(session_dir: pathlib.Path, steer: str) -> None:
    """Record the steer as the run's latest continuation; the task stays as the run began.

    An empty steer clears it, as the journal's label clears its own on an unsteered resume.
    An unreadable manifest is left alone rather than failing the resume.

    Args:
        session_dir: The run's session directory.
        steer: The resume's steer, "" for an unsteered resume.
    """
    with contextlib.suppress(manifest.ManifestError, OSError):
        m = manifest.read_manifest(session_dir)
        write_manifest(
            session_dir / manifest.MANIFEST_NAME,
            m.model_copy(update={"continued_task": task_text.operator_task_text(steer)[:4000]}),
        )


def stamp_verify_gate(session_dir: pathlib.Path, gate: manifest.GateStamp) -> None:
    """Pin the gate record that viewers and the next execution read.

    Args:
        session_dir: The session directory.
        gate: The pinned command, source and execution availability.
    """
    m = manifest.read_manifest(session_dir)
    harness = m.harness.model_copy(update={"gate": gate})
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"harness": harness}))


def pin_gate(
    session_dir: pathlib.Path,
    gate: manifest.GateStamp,
    *,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter,
) -> None:
    """Pin the execution's gate, and re-pin it when the loop adopts one mid-run.

    A failed stamp is reported because viewers and the next execution read it.

    Args:
        session_dir: The run.
        gate: The pinned command, source and execution availability.
        events: The run's event sink, watched for an adopted gate.
        reporter: Where a failed stamp is reported.
    """

    def _stamp(record: manifest.GateStamp) -> None:
        try:
            stamp_verify_gate(session_dir, record)
        except (manifest.ManifestError, OSError) as exc:
            reporter.note(f"could not record this run's verify gate: {exc}")

    _stamp(gate)

    def _repin_adopted_gate(event: dict[str, Any]) -> None:
        """Re-pin on a `loop.verify_inferred` event that adopted a gate."""
        if event.get("type") == "loop.verify_inferred" and event.get("adopted_at") is not None:
            command = tuple(event.get("command", ()))
            _stamp(manifest.GateStamp(command=command, source="adopted" if command else ""))

    # EventSink swallows a listener's exceptions; _stamp reports for itself.
    events.subscribe(_repin_adopted_gate)
