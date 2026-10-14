# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve subordinate workdirs and manage a lane's files from config to cleanup."""

from __future__ import annotations

import contextlib
import pathlib
import shlex
import shutil
from collections.abc import Callable

from agent6 import git_ops, paths
from agent6.config import Config, layer
from agent6.harness import subrun
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest


def subordinate_workdir_root(cfg: Config, origin: pathlib.Path, group: str) -> pathlib.Path:
    """Return the base dir for a group of subordinate working trees.

    Fan-out ids, `machine-<id>` and fork ids are all groups, so one prune sweep
    covers them; the dir is scoped by repo id, so another repo's clones never enter
    a sweep.

    Args:
        cfg: The run's config.
        origin: The repository.
        group: The group's name.

    Returns:
        `<workdir base>/<repo id>/<group>`.
    """
    return workdir_base(cfg, origin) / group


def workdir_base(cfg: Config, origin: pathlib.Path) -> pathlib.Path:
    """Return the per-repo dir every subordinate working tree of the origin sits in.

    Args:
        cfg: The run's config.
        origin: The repository.

    Returns:
        `[parallel].workdir` as an absolute path, or the cache's `parallel` dir, plus the
        repo id.
    """
    return paths.parallel_base(cfg.parallel.workdir) / paths.repo_id(origin)


def adopt_orphan_lane(
    origin: pathlib.Path,
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
) -> str | None:
    """Import an orphaned fan-out lane so an ordinary merge can land it.

    A coordinator death leaves a finished lane's branch only in its clone, with
    the origin still holding the live-view symlink.

    Args:
        origin: The repository.
        cfg: The run's config.
        layout: The lane's layout in the origin.
        manifest: The lane's manifest.

    Returns:
        The note to print, or None when the session is not an orphaned lane.

    Raises:
        SubrunError: The import failed; the symlink is restored.
    """
    if (
        not manifest.run_branch
        or manifest.parallel is None
        or git_ops.branch_exists(origin, manifest.run_branch)
        or not layout.session_dir.is_symlink()
    ):
        return None
    lineage = manifest.parallel
    clone = subordinate_workdir_root(cfg, origin, lineage.group) / f"lane-{lineage.lane}"
    if not (clone / ".git").exists() or not git_ops.branch_exists(clone, manifest.run_branch):
        return None
    real = layout.session_dir.resolve()
    layout.session_dir.unlink()
    try:
        subrun.import_run(origin, clone, manifest.run_branch, real, layout.state_dir)
    except subrun.SubrunError:
        layout.session_dir.symlink_to(real)
        raise
    return f"imported orphaned lane branch {manifest.run_branch} from {clone}"


def recorded_merged(origin: pathlib.Path, clone: pathlib.Path, tip: str) -> bool:
    """Tell whether a session's manifest records the tip as merged into a live branch.

    A squash-merged lane's commits are unreachable by design, so this is the second
    proof that its clone may go.

    Args:
        origin: The repository.
        clone: The lane's clone.
        tip: The commit to prove.

    Returns:
        Whether a merge stamp covers it.
    """
    state = paths.state_dir(origin)
    for branch in git_ops.list_run_branches(clone):
        layout = sessions_layout.session_layout(state, branch.removeprefix("agent6/"))
        if layout is None:
            continue
        try:
            merged = sessions_manifest.read_manifest(layout.session_dir).merged
        except sessions_manifest.ManifestError:
            continue
        if merged is not None and merged.tip == tip and git_ops.branch_exists(origin, merged.into):
            return True
    return False


def lane_keep_reason(clone: pathlib.Path) -> str:
    """Return why a live lane or an unconfirmed startup keeps the clone, else "".

    Args:
        clone: The lane's clone.

    Returns:
        The keep reason; a spawn timeout can leave a live child with no session record.
    """
    state = paths.state_dir(clone)
    dirs = [
        d
        for bucket in sessions_layout.HUB_BUCKETS
        for d in sessions_layout.bucket_dir(state, bucket).glob("*")
        if d.is_dir()
    ]
    if not dirs:
        return (
            "no lane session record, so a lane may still be starting; once none is, remove it"
            f" with: rm -r -- {shlex.quote(str(clone.parent))}"
        )
    return "live lane; stop it before pruning" if any(ipc.worker_is_alive(d) for d in dirs) else ""


def sweep_fanout_clones(
    origin: pathlib.Path,
    cfg: Config,
    *,
    note_keep: Callable[[pathlib.Path, str], None] = lambda _path, _why: None,
) -> tuple[int, int]:
    """Delete fan-out clone dirs whose every tip the origin already reaches.

    A clone holding a commit the origin lacks, or a live lane, keeps its whole
    group dir: it may be the only copy. A clone with no lane session record is kept because
    a spawn timeout can leave a live child before its first write. Only a dir holding
    `lane-*` clones is a group; a fork's worktree or an operator's dir is left alone.

    Args:
        origin: The repository.
        cfg: The run's config.
        note_keep: Receives each kept group path and its reason.

    Returns:
        `(swept, kept)` group counts.
    """
    base = workdir_base(cfg, origin)
    if not base.is_dir():
        return 0, 0
    swept = kept = 0
    for fanout in sorted(p for p in base.iterdir() if p.is_dir()):
        clones = [c for c in sorted(fanout.glob("lane-*")) if (c / ".git").is_dir()]
        if not clones:
            continue
        why = ""
        for clone in clones:
            try:
                tips = [git_ops.chain_tip(clone, br) for br in git_ops.list_run_branches(clone)]
                # A machine-state clone's work rides its chain ref, not a branch.
                tips += [sha for _ref, sha in git_ops.list_chain_refs(clone)]
                if not tips and (why := lane_keep_reason(clone)):
                    break
                # Reachability, not existence: a loose object is one `git gc` from gone.
                if any(
                    tip is not None
                    and not git_ops.commit_is_reachable(origin, tip)
                    and not recorded_merged(origin, clone, tip)
                    for tip in tips
                ):
                    why = "commits this repo cannot reach; merge or archive the lane first"
                    break
            except git_ops.GitError as exc:
                why = (
                    f"git could not check its commits: {exc}; review this clone before removing it"
                )
                break
        if not why:
            for clone in clones:
                shutil.rmtree(paths.jail_cache_home(clone), ignore_errors=True)
            shutil.rmtree(fanout, ignore_errors=True)
            swept += 1
        else:
            kept += 1
            note_keep(fanout, why)
    return swept, kept


def write_lane_config(
    cfg: Config, spec: subrun.LaneSpec, *, explicit_leaves: frozenset[str]
) -> pathlib.Path:
    """Write the origin's explicit settings and non-default lane values to `--config`.

    The clone's per-repo config is empty; unset defaults stay defaults in the lane.

    Args:
        cfg: The origin's effective config.
        spec: The lane.
        explicit_leaves: The dotted leaves the origin's config layers set.

    Returns:
        The config file's path.
    """
    lane_cfg = cfg.with_model_route("worker", spec.route) if spec.route else cfg
    # The import fetches the lane's branch, so branch_per_run stays on.
    lane_cfg = lane_cfg.model_copy(
        update={"git": lane_cfg.git.model_copy(update={"branch_per_run": True})}
    )
    config_path = spec.workdir.parent / f"lane-{spec.lane}-config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        layer.materialize(lane_cfg, explicit_leaves=explicit_leaves), encoding="utf-8"
    )
    return config_path


def cleanup(
    imported: list[subrun.LaneSpec], *, workdir_root: pathlib.Path, base: pathlib.Path, cfg: Config
) -> None:
    """Tear down the clone, state dir, jail home and config of each imported lane, best-effort.

    Every level from the group dir up to the per-repo base is removed once empty,
    never the dir above it.

    Args:
        imported: The lanes whose import landed.
        workdir_root: The group dir.
        base: The per-repo base dir.
        cfg: The run's config.
    """
    for spec in imported:
        shutil.rmtree(paths.state_dir(spec.workdir), ignore_errors=True)
        shutil.rmtree(paths.jail_cache_home(spec.workdir), ignore_errors=True)
        shutil.rmtree(spec.workdir, ignore_errors=True)
        (spec.workdir.parent / f"lane-{spec.lane}-config.toml").unlink(missing_ok=True)
    level = workdir_root
    while level.is_relative_to(base):
        with contextlib.suppress(OSError):
            level.rmdir()  # succeeds only when nothing was kept
        if level == base:
            break
        level = level.parent
