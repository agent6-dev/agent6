# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 fork`: clone a run (rolled back to a checkpoint) into a NEW run.

A fork copies a source run's state, as of checkpoint turn N, into a fresh run
dir with a new id and the same repo, recording lineage (parent run + the turn).
The source run is never mutated -- this is Pi-style "sessions as trees" done as
clone-to-new-session, not in-place branching. The new run is a normal resumable
run that starts mid-conversation; by default `fork` immediately continues it
from turn N (reusing the resume path).

Phase 1 scope: fork from the latest checkpoint or a recorded `--at-turn N`, and
copy the curator DAG verbatim. Replaying the journal to reconstruct the DAG as
of an older `graph_version` is deferred; forking a past turn copies the source's
current DAG and says so.

The tree a fork starts from: a fork cuts its branch at the checkpoint's committed
HEAD, so its working tree is the repo exactly as of that commit. That is the whole
mental model -- a fork is the repo at a commit plus the conversation up to that
turn. The one thing to know: on a gated run (commits fire only on a green verify),
an edit made but not yet committed at the forked turn is absent from the fork's
tree even though the copied transcript mentions it. This is the same "head_sha
tracks committed history only" posture `resume` documents (resume differs only in
that it continues on the live working tree, which still holds the edit). The forked
run picks it back up by re-reading the files it needs and seeing their real on-disk
content. The committed-sha model is the design choice: a fork is a commit plus a
conversation, which is predictable and cheap, instead of snapshotting uncommitted
working-tree bytes into every checkpoint.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import GitError, create_branch_at
from agent6.graph.storage import append_jsonl, list_checkpoint_turns
from agent6.portable import atomic_write
from agent6.runs.id import RunIdError, new_friendly_id, resolve_run_id
from agent6.runs.layout import RunLayout
from agent6.ui.cli._common import _BudgetOverrides, _state_dir
from agent6.ui.cli._manifest import write_run_manifest as _write_run_manifest
from agent6.ui.cli.plan_watch import _most_recent_run_id
from agent6.ui.cli.run import _cmd_resume
from agent6.workflows._run_state import load_checkpoint

# Curator-owned DAG artifacts copied verbatim into the fork (Phase 1). Each is a
# top-level entry under the run dir; `graph/` is a directory.
_DAG_ARTIFACTS: tuple[str, ...] = ("graph", "graph.jsonl", "graph.dot", "cursor.json")


def _lineage_entry(*, child: str, parent: str, turn: int, sha: str, ts: str) -> dict[str, object]:
    """One per-repo lineage event. Pure: the caller passes the timestamp in."""
    return {"child": child, "parent": parent, "turn": turn, "sha": sha, "ts": ts}


def _copy_dag(src: RunLayout, dst: RunLayout) -> None:
    """Copy the curator DAG artifacts from *src* into *dst*, verbatim."""
    for name in _DAG_ARTIFACTS:
        src_path = src.run_dir / name
        if not src_path.exists():
            continue
        dst_path = dst.run_dir / name
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def _select_checkpoint_path(src: RunLayout, at_turn: int | None) -> Path | None:
    """Resolve which checkpoint of *src* to fork from, or None on error (printed).

    Returns the latest checkpoint by default, the ``--at-turn N`` one when given,
    or degrades to ``loop_state.json`` for a pre-checkpoint (old) run.
    """
    legacy = src.run_dir / "loop_state.json"
    source_id = src.run_id
    turns = list_checkpoint_turns(src)
    if at_turn is None:
        return _select_latest_checkpoint_path(src, turns, legacy)
    return _select_explicit_checkpoint_path(src, turns, legacy, at_turn, source_id)


def _select_latest_checkpoint_path(src: RunLayout, turns: list[int], legacy: Path) -> Path | None:
    if legacy.is_file():
        return legacy
    if turns:
        return src.checkpoint_path(turns[-1])
    print(
        f"ERROR: {src.run_id} has no checkpoints and no loop_state.json; nothing to fork.",
        file=sys.stderr,
    )
    return None


def _select_explicit_checkpoint_path(
    src: RunLayout,
    turns: list[int],
    legacy: Path,
    at_turn: int,
    source_id: str,
) -> Path | None:
    if at_turn in turns:
        return src.checkpoint_path(at_turn)
    if legacy.is_file():
        legacy_turn = _snapshot_turn(legacy)
        if legacy_turn == at_turn:
            return legacy
        if not turns:
            print(
                f"NOTE: {source_id} predates the checkpoint store; --at-turn is unavailable. "
                "Forking from its latest snapshot (loop_state.json).",
                file=sys.stderr,
            )
            return legacy
    avail = ", ".join(str(t) for t in turns)
    print(
        f"ERROR: no checkpoint at turn {at_turn} for {source_id}. Available turns: {avail}",
        file=sys.stderr,
    )
    return None


def _snapshot_turn(path: Path) -> int | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("next_iteration")
    if not isinstance(value, str | int):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cmd_fork(  # noqa: PLR0911
    config_path: Path | None,
    source_run_id: str,
    *,
    at_turn: int | None = None,
    new_run_id: str = "",
    no_run: bool = False,
    tui: bool = False,
    budget_overrides: _BudgetOverrides | None = None,
) -> int:
    """Create a new run cloned from *source_run_id* at checkpoint *at_turn*.

    Default: fork from the latest checkpoint and immediately continue the new run
    from that turn (resume-like). ``--no-run`` just creates the fork dir.
    """
    cwd = Path.cwd()
    state_dir = _state_dir(cwd)
    runs_dir = state_dir / "runs"
    if not source_run_id:
        # "fork my last run" -- omitting the id forks the most recent run.
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}; nothing to fork.", file=sys.stderr)
            return 2
        source_run_id = latest
        print(f"[agent6] forking most recent run: {source_run_id}", file=sys.stderr)
    try:
        source_id = resolve_run_id(runs_dir, source_run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    src = RunLayout(state_dir=state_dir, run_id=source_id)
    if not src.run_dir.is_dir():
        print(f"ERROR: no such run dir: {src.run_dir}", file=sys.stderr)
        return 2

    checkpoint_path = _select_checkpoint_path(src, at_turn)
    if checkpoint_path is None:
        return 2

    try:
        checkpoint = load_checkpoint(checkpoint_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: failed to load checkpoint {checkpoint_path}: {exc}", file=sys.stderr)
        return 1

    # Read the source manifest to carry base_sha / base_branch forward.
    src_base_sha = ""
    src_base_branch = ""
    src_mode = "run"
    try:
        sm = json.loads(src.manifest_path.read_text(encoding="utf-8"))
        src_base_sha = str(sm.get("base_sha", "")) or ""
        src_base_branch = str(sm.get("base_branch", "")) or ""
        src_user_task = str(sm.get("user_task", "")) or ""
        src_mode = str(sm.get("mode", "run")) or "run"
    except (OSError, ValueError):
        src_user_task = ""

    forked_from_sha = checkpoint.head_sha
    if not forked_from_sha:
        print(
            "ERROR: the chosen checkpoint records no head_sha, so the fork branch "
            "cannot be cut. (A checkpoint from before per-turn sha capture.)",
            file=sys.stderr,
        )
        return 1

    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    child_id = new_run_id or new_friendly_id()
    rc = _materialize_fork(
        cwd=cwd,
        src=src,
        dst=RunLayout(state_dir=state_dir, run_id=child_id),
        checkpoint_path=checkpoint_path,
        forked_from_turn=checkpoint.turn,
        forked_from_sha=forked_from_sha,
        base_sha=src_base_sha,
        base_branch=src_base_branch,
        user_task=src_user_task,
        mode=src_mode,
        cfg=cfg,
    )
    if rc != 0:
        return rc

    if no_run:
        print(f"[agent6] fork created (not started): {child_id}", file=sys.stderr)
        print(f"  resume it with: agent6 resume {child_id}", file=sys.stderr)
        return 0

    # Continue the new run from turn N by reusing the resume path. The fork just
    # cloned the checkpoint (its head_sha) and cut agent6/<child> at that same
    # sha, so the resume head guard passes by construction; force stays off so a
    # real mismatch (a broken fork) still refuses.
    return _cmd_resume(
        config_path,
        child_id,
        force=False,
        tui=tui,
        budget_overrides=budget_overrides,
    )


def _materialize_fork(
    *,
    cwd: Path,
    src: RunLayout,
    dst: RunLayout,
    checkpoint_path: Path,
    forked_from_turn: int,
    forked_from_sha: str,
    base_sha: str,
    base_branch: str,
    user_task: str,
    mode: str,
    cfg: Config,
) -> int:
    """Write the fork's state on disk: clone the checkpoint + DAG, the manifest,
    the git branch, and the lineage record. Returns 0 on success, else an error
    code (after printing). The source run is never touched."""
    if dst.run_dir.exists():
        print(f"ERROR: target run dir already exists: {dst.run_dir}", file=sys.stderr)
        return 2
    dst.ensure()

    # Seed the new run's resume pointer + origin checkpoint from the chosen
    # checkpoint, then clone the curator DAG verbatim.
    blob = checkpoint_path.read_text(encoding="utf-8")
    atomic_write(dst.run_dir / "loop_state.json", blob)
    atomic_write(dst.checkpoint_path(0), blob)
    _copy_dag(src, dst)

    run_branch = f"agent6/{dst.run_id}"
    _write_run_manifest(
        dst,
        run_id=dst.run_id,
        user_task=user_task,
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        cfg=cfg,
        mode=mode,
        parent_run_id=src.run_id,
        forked_from_turn=forked_from_turn,
        forked_from_sha=forked_from_sha,
    )

    # Cut the fork's branch at the historical sha WITHOUT touching the operator's
    # checkout (additive `git branch <name> <sha>`).
    try:
        create_branch_at(cwd, run_branch, forked_from_sha)
    except GitError as exc:
        print(f"ERROR: could not cut fork branch {run_branch}: {exc}", file=sys.stderr)
        # The fork dir was just materialized; don't leave an orphan run dir +
        # manifest (and a lineage gap) when the branch couldn't be cut.
        shutil.rmtree(dst.run_dir, ignore_errors=True)
        return 1

    # Append the per-repo lineage event (ts minted here, passed into the pure helper).
    append_jsonl(
        src.state_dir / "lineage.jsonl",
        _lineage_entry(
            child=dst.run_id,
            parent=src.run_id,
            turn=forked_from_turn,
            sha=forked_from_sha,
            ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        ),
    )
    print(
        f"[agent6] forked {src.run_id}@turn {forked_from_turn} -> {dst.run_id} "
        f"(branch {run_branch} at {forked_from_sha[:12]})",
        file=sys.stderr,
    )
    return 0
