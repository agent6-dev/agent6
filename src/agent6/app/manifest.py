# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Write the manifest.json a run starts with, and every stamp that rewrites it.

The reader and the on-disk shape (`SessionManifest`) live in `sessions.manifest`.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import pathlib
from collections.abc import Sequence
from typing import Any

from agent6 import __version__, kinds, portable, task_text
from agent6 import events as agent6_events
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest


def _policy_stamp(cfg: Config, isolation: str) -> manifest.PolicyStamp:
    """Return the policy stamp; `isolation` is the resolved level, since `auto` degrades."""
    return manifest.PolicyStamp(
        run_commands=cfg.sandbox.run_commands,
        isolation=isolation or str(cfg.sandbox.isolation),
        network=str(cfg.sandbox.network),
        commit_per_step=cfg.git.commit_per_step,
    )


def _model_brief(rm: Any) -> manifest.ModelBrief | None:
    """Return the brief for a resolved role, or None when unset."""
    if rm is None:
        return None
    return manifest.ModelBrief(provider=rm.provider, model=rm.model)


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
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
    mode: str = "run",
    effective_preset: str = "",
    preset_from_flag: bool = False,
    driver_from_flag: bool = False,
    gate: tuple[Sequence[str], str] | None = None,
    isolation: str = "",
    parent_session_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
    worktree: pathlib.Path | None = None,
    worktree_git_dir: pathlib.Path | None = None,
    fanout: manifest.FanoutStamp | None = None,
) -> None:
    """Write the manifest.json a new run or fork starts with.

    Args:
        layout: The session's directory layout.
        session_id: The new session's id.
        source_session_id: The session whose context `--from` seeded into this one.
        user_task: The operator's task.
        base_sha: The commit the run started from.
        base_branch: The branch the run started from.
        run_branch: The branch the start cut, when one was.
        cfg: The resolved config.
        mode: run, plan or ask.
        effective_preset: The preset the run uses.
        preset_from_flag: The preset came from `--preset`, so a resume replays it.
        driver_from_flag: The driver model came from `--model`.
        gate: A fork's pin of the source's (verify command, origin).
        isolation: The resolved isolation level.
        parent_session_id: A fork's source run.
        forked_from_turn: The turn a fork was cut at.
        forked_from_sha: The workspace sha at that turn.
        worktree: A fork's own checkout.
        worktree_git_dir: The repository git dir the worktree points into.
        fanout: A `run --parallel` coordinator's own record.
    """
    lineage = _parallel_lineage()
    # A fresh run carries the configured gate until `pin_gate` stamps the pair it resolved.
    verify_command, verify_origin = gate or (
        cfg.harness.verify_command,
        "configured" if cfg.harness.verify_command else "",
    )
    m = manifest.SessionManifest(
        agent6_version=__version__,
        session_id=session_id,
        # `fork` and `resume` act on session_mode(), never on this string.
        mode=mode,
        start_ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        # The display twin of the operator's words; the verbatim engine copy is in the snapshot.
        user_task=task_text.operator_task_text(user_task)[:4000],
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        git_control=cfg.git.control,
        models=manifest.ModelsBrief(
            # The role that drives this mode: a plan run's worker never ran.
            driver=_model_brief(cfg.models.resolve(kinds.session_kind(mode).role)),
            reviewer=_model_brief(cfg.models.resolve("reviewer")),
            driver_from_flag=driver_from_flag,
        ),
        harness=manifest.HarnessStamp(
            review_trigger=cfg.review.trigger,
            revise_prompt=cfg.prompt.revise_prompt,
            preset=effective_preset,
            preset_from_flag=preset_from_flag,
            verify_command=tuple(verify_command),
            verify_origin=verify_origin,
        ),
        policy=_policy_stamp(cfg, isolation),
        source_session_id=source_session_id,
        parent_session_id=parent_session_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
        worktree=worktree,
        worktree_git_dir=worktree_git_dir,
        parallel=lineage,
        fanout=fanout,
    )
    write_manifest(layout.manifest_path, m)


def _parallel_lineage() -> manifest.ParallelLineage | None:
    """Return the fan-out lineage the spawner stamped into this lane's environment, or None.

    `AGENT6_PARALLEL_LINEAGE=<coordinator>:<group>:<lane>`. Read at the manifest's write so the
    grouping survives a coordinator death.
    """
    raw = os.environ.get("AGENT6_PARALLEL_LINEAGE", "")
    coordinator, _, rest = raw.partition(":")
    group, sep, lane = rest.rpartition(":")
    if not sep or not coordinator or not group or not lane.isdigit():
        return None
    return manifest.ParallelLineage(group=group, lane=int(lane), coordinator=coordinator)


def stamp_parked(session_dir: pathlib.Path, *, task: str, reason: str) -> None:
    """Record that the run was submitted and never started; `unpark` replaces the stamp.

    Args:
        session_dir: The run.
        task: The verbatim task a resume starts fresh.
        reason: Why the run waits.
    """
    m = manifest.read_manifest(session_dir)
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(update={"parked_task": task, "parked_reason": reason, "run_branch": None}),
    )


def parked_stamp(session_dir: pathlib.Path) -> tuple[str, str] | None:
    """Return a parked submission's (task, reason), or None without a manifest or a park."""
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        return None
    return (m.parked_task, m.parked_reason) if m.parked_task else None


def unpark(session_dir: pathlib.Path, *, run_branch: str | None) -> None:
    """End the park at the execution's start and record the branch the start cut.

    A manifest carrying no park, or none readable, is left alone.
    """
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError:
        return
    if not m.parked_task:
        return
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(update={"parked_task": "", "parked_reason": "", "run_branch": run_branch}),
    )


def stamp_execution(session_dir: pathlib.Path, cfg: Config, mode: str, isolation: str) -> None:
    """Re-stamp what an execution owns: the models driving it and the policy it runs under.

    `agent6 exec` joins the recorded policy's jail and `sessions show` reads the recorded model.
    """
    m = manifest.read_manifest(session_dir)
    harness = m.harness
    if not harness.preset_from_flag:
        harness = harness.model_copy(update={"preset": cfg.preset})
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(
            update={
                "models": manifest.ModelsBrief(
                    driver=_model_brief(cfg.models.resolve(kinds.session_kind(mode).role)),
                    reviewer=_model_brief(cfg.models.resolve("reviewer")),
                    driver_from_flag=m.models.driver_from_flag,
                ),
                "harness": harness,
                "policy": _policy_stamp(cfg, isolation),
            }
        ),
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


def stamp_fork_task(session_dir: pathlib.Path, steer: str, *, source_dir: pathlib.Path) -> None:
    """Record the first steer that sends a fork elsewhere as the fork's own task.

    A fork starts with its source's task; later steers are follow-ups within the task, so the
    stamp fires only while the task is still the source's. A newer manifest, or a pruned
    source, leaves the task as it stands rather than failing the resume.
    """
    m = manifest.read_manifest(session_dir)
    if m.parent_session_id is None:
        return
    try:
        source_task = manifest.read_manifest(source_dir).user_task
    except manifest.ManifestError:
        return  # the source is gone; its task is whatever the fork already carries
    if m.user_task != source_task:
        return  # already sent somewhere; this steer is a follow-up
    with contextlib.suppress(manifest.ManifestError, OSError):
        _write_task(session_dir, m, steer)


def stamp_task(session_dir: pathlib.Path, steer: str) -> None:
    """Record the steer as the run's task; a newer manifest keeps its task rather than fail."""
    with contextlib.suppress(manifest.ManifestError, OSError):
        _write_task(session_dir, manifest.read_manifest(session_dir), steer)


def _write_task(session_dir: pathlib.Path, m: manifest.SessionManifest, steer: str) -> None:
    """Write the steer as the manifest's task, clipped like the first write."""
    write_manifest(
        session_dir / manifest.MANIFEST_NAME,
        m.model_copy(update={"user_task": task_text.operator_task_text(steer)[:4000]}),
    )


def stamp_verify_gate(session_dir: pathlib.Path, argv: Sequence[str], origin: str) -> None:
    """Pin the gate the run is judged by and where it came from.

    From here on the pair is the run's, so a mid-run edit to AGENTS.md cannot move the gate.
    """
    m = manifest.read_manifest(session_dir)
    harness = m.harness.model_copy(update={"verify_command": tuple(argv), "verify_origin": origin})
    write_manifest(session_dir / manifest.MANIFEST_NAME, m.model_copy(update={"harness": harness}))


def pin_gate(
    session_dir: pathlib.Path,
    argv: Sequence[str],
    origin: str,
    *,
    events: agent6_events.EventSink,
    reporter: app_reporter.Reporter,
) -> None:
    """Pin the execution's gate, and re-pin it when the loop adopts one mid-run.

    A failed stamp is reported, never raised or swallowed: the execution is still worth
    running, and the manifest is where every viewer and the next execution read the gate.

    Args:
        session_dir: The run.
        argv: The gate command.
        origin: Where the gate came from.
        events: The run's event sink, watched for an adopted gate.
        reporter: Where a failed stamp is reported.
    """

    def _stamp(gate: Sequence[str], why: str) -> None:
        try:
            stamp_verify_gate(session_dir, gate, why)
        except (manifest.ManifestError, OSError) as exc:
            reporter.note(f"could not record this run's verify gate: {exc}")

    _stamp(argv, origin)

    def _repin_adopted_gate(event: dict[str, Any]) -> None:
        """Re-pin on a `loop.verify_inferred` event that adopted a gate."""
        if event.get("type") == "loop.verify_inferred" and event.get("adopted_at") is not None:
            command = tuple(event.get("command", ()))
            _stamp(command, "adopted" if command else "unadopted")

    # EventSink swallows a listener's exceptions; _stamp reports for itself.
    events.subscribe(_repin_adopted_gate)
