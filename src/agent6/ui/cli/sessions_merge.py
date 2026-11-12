# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions merge`: land a run's chain on its base."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sys
from collections.abc import Mapping

from agent6 import errors, terminal
from agent6.app import merge, resume, workdirs
from agent6.config import Config, ConfigError, layer
from agent6.harness import subrun
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.ui.cli import common, sessions_cmds
from agent6.vcs import commit_message
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing, scan_session_log, tail_events, worker_models

# The gate words that mean the gate did not certify the run's final tree.
_UNVERIFIED_WARNINGS = {
    "failed": "the verify gate was red on this run's final tree",
    "timed_out": "the verify gate timed out on this run's final tree",
    "unverified": "nothing verified this run's final tree",
}


@dataclasses.dataclass(frozen=True, slots=True)
class _MergePlan:
    """A validated, mutation-ready merge: everything `_cmd_merge` needs after every guard passed.

    Attributes:
        layout: The run.
        manifest: Its manifest.
        run_branch: The branch or chain ref holding the work.
        target: The branch to land on.
        base_sha: The commit the run was cut from.
        method: The merge method.
        identity: The committer identity.
        cfg: The effective config.
    """

    layout: sessions_layout.SessionLayout
    manifest: sessions_manifest.SessionManifest
    run_branch: str
    target: str
    base_sha: str
    method: str
    identity: git_read.CommitIdentity
    cfg: Config


def _not_mergeable_now(layout: sessions_layout.SessionLayout, session_id: str) -> str:
    """Return why the run's work is not for merging: a live worker, or an `/undo` took it back.

    The raw pid, not session_is_live: after session.stop the finalizer may still be at work.

    Args:
        layout: The run.
        session_id: The id the refusal names.

    Returns:
        The refusal, or "" when nothing blocks the merge.
    """
    if process.worker_is_alive(layout.session_dir):
        return (
            f"run {session_id!r} is still live; a merge now lands only the"
            " commits so far (its later ones need another merge) and moves your"
            " index while the worker still edits. Stop it first:\n"
            f"    agent6 stop {session_id}"
        )
    scan = scan_session_log(layout.logs_path)
    if scan.finished and scan.stop_reason == "undone":
        return listing.undone_refusal(session_id, scan.undone_to)
    return ""


def _plan_merge(  # noqa: PLR0911
    cwd: pathlib.Path,
    session_id: str,
    into: str | None,
    method: str | None,
    *,
    config_paths: tuple[pathlib.Path, ...],
) -> _MergePlan | int:
    """Resolve and validate everything a merge needs, without touching the repo.

    Args:
        cwd: The repo.
        session_id: The run, or "" for the newest.
        into: The target branch; None takes the branch the run was cut from.
        method: The merge method; None takes `git.merge_method`.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The plan, or the exit code of a printed refusal or an already-pruned merge.
        Every guard runs before `_cmd_merge` mutates anything.
    """
    res = sessions_cmds._resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    layout, manifest = res
    if refusal := _not_mergeable_now(layout, manifest.session_id):
        common.refuse(refusal)
        return 2
    ref = sessions_cmds._commits_ref(cwd, manifest)
    if (
        (stamp := resume.covering_stamp(cwd, manifest)) is not None
        and not layout.session_dir.is_symlink()
        and (not ref.head_ref or git_read.chain_tip(cwd, ref.head_ref) is None)
    ):
        already = "already " if stamp.commit else ""
        print(f"[agent6] {already}{stamp.landed()} (run refs were pruned).")
        return 0
    # execute_merge refuses a missing base_sha too; here it is a refusal before anything moves.
    unmergeable = (
        merge.NO_BASE_SHA
        if not manifest.base_sha
        else f"this session has no branch to merge ({ref.reason})."
        if ref.reason
        else ""
    )
    if unmergeable:
        common.refuse(unmergeable)
        return 2
    run_branch = ref.head_ref
    target = into or manifest.base_branch
    if not target or target == run_branch:
        common.error(
            "no base branch (the run started on a detached HEAD); pass --into <existing-branch>."
            if not target
            else f"target {target!r} is the run branch itself; pass --into <other-branch>."
        )
        return 2
    try:
        cfg = layer.load_effective(cwd, config_paths).config
    except ConfigError as exc:
        common.error(f"{exc}")
        return 2
    # An orphaned lane (its coordinator died before importing it) is adopted: fetched, then merged.
    try:
        adopted = workdirs.adopt_orphan_lane(cwd, cfg, layout, manifest)
    except subrun.SubrunError as exc:
        common.error(f"{exc}")
        return 2
    if adopted is not None:
        print(f"[agent6] {adopted}")
        layout = sessions_layout.SessionLayout(
            state_dir=layout.state_dir, session_id=layout.session_id
        )
    # chain_tip resolves both shapes of head_ref: a branch name and the hidden chain ref.
    if git_read.chain_tip(cwd, run_branch) is None:
        if sessions_cmds._committed_nothing(cwd, manifest.session_id):
            # Not a failure: there is no work to land.
            print("[agent6] nothing to merge: this run committed nothing.")
            return 0
        common.error(
            f"run ref {run_branch!r} no longer exists; its commits survive at"
            f" {git_read.chain_ref_for(manifest.session_id)}."
        )
        return 2
    if not git_read.branch_exists(cwd, target):
        common.error(f"target branch {target!r} does not exist; pass --into <existing-branch>.")
        return 2
    identity = git_read.CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        trailer=commit_message.render_commit_trailer(
            cfg.git.commit.trailer,
            models=worker_models(
                tail_events(layout.session_dir / sessions_layout.LOGS_NAME, follow=False)
            )
            or ((manifest.models.driver.model,) if manifest.models.driver else ()),
        ),
    )
    try:
        git_read.verify_git_identity(cwd, identity)  # refuse cleanly before mutating anything
    except errors.GitError as exc:
        common.error(f"{exc}")
        return 2
    return _MergePlan(
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=target,
        base_sha=manifest.base_sha,
        method=method or cfg.git.merge_method,
        identity=identity,
        cfg=cfg,
    )


def _manual_merge_cmd(cwd: pathlib.Path, plan: _MergePlan) -> str:
    """Return the by-hand merge for a plumbing conflict, from the checkout as it is.

    Git refuses to merge over modified tracked files, and a finished run leaves its work in
    the tree until merged, so a dirty tree is stashed first; a checkout on another branch
    moves to the target.

    Args:
        cwd: The repo.
        plan: The merge that conflicted.

    Returns:
        The git commands, joined by `&&`.
    """
    steps: list[str] = []
    with contextlib.suppress(errors.GitError):
        st = git_read.status(cwd)
        if st.modified_count:
            steps.append("git stash")
        if st.branch != plan.target:
            steps.append(f"git checkout {plan.target}")
    # A by-hand landing of a `tasks` merge is one squash: the task ranges need the chain.
    squash = " --squash" if plan.method in ("squash", "tasks") else ""
    steps.append(f"git merge{squash} {plan.run_branch}")
    return " && ".join(steps)


def _cmd_merge(
    *,
    session_id: str,
    method: str | None,
    into: str | None,
    message: str | None,
    config_paths: tuple[pathlib.Path, ...],
) -> int:
    """Land a run's work on a target branch with the chosen method.

    Ref plumbing only: the checkout, index and worktree are never the medium, so a worktree
    still carrying the run's work is no obstacle.

    Args:
        session_id: The run, or "" for the newest.
        method: The merge method; None takes `git.merge_method`.
        into: The target branch; None takes the branch the run was cut from.
        message: The merge commit message; None takes the default.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The exit code: 0 landed or nothing to land, 1 a conflict, 2 a refusal.
    """
    cwd = pathlib.Path.cwd()
    plan = _plan_merge(cwd, session_id, into, method, config_paths=config_paths)
    if isinstance(plan, int):
        return plan
    if not git_read.list_run_commits(cwd, plan.base_sha, plan.run_branch):
        # A success line here would be indistinguishable from a real merge.
        print(f"[agent6] nothing to merge: run branch {plan.run_branch} has no commits.")
        return 0
    # The merge lands whatever the gate said; the operator hears it first.
    if unverified := _UNVERIFIED_WARNINGS.get(scan_session_log(plan.layout.logs_path).gate_word()):
        print(f"[agent6] warning: {unverified}; merging it.", file=sys.stderr)
    outcome = merge.execute_merge(
        cwd,
        layout=plan.layout,
        manifest=plan.manifest,
        run_branch=plan.run_branch,
        target=plan.target,
        base_sha=plan.base_sha,
        method=plan.method,
        message=message,
        cfg=plan.cfg,
        identity=plan.identity,
        warn=lambda m: print(f"[agent6] {m}", file=sys.stderr),
    )
    if outcome.status == "error":
        common.error(f"{outcome.error}")
        return 1
    if outcome.status == "conflict":
        resume = (
            "Resolve them in the run's own worktree:\n"
            f"    agent6 resume {plan.layout.session_id} --merge {plan.target}\n"
            if plan.manifest.worktree is not None
            else ""
        )
        print(
            f"CONFLICT: merging {plan.run_branch} into {plan.target} hit conflicts in "
            f"{', '.join(outcome.conflicts)}. Your checkout is untouched (no partial merge).\n"
            f"{resume}Or by hand:\n"
            f"    {_manual_merge_cmd(cwd, plan)}",
            file=sys.stderr,
        )
        return 1
    note = (
        f"\n  (merge record could not be written: {outcome.stamp_error};"
        " `sessions prune` will call this branch unmerged)"
        if outcome.stamp_error
        else ""
    )
    if outcome.status == "noop":
        print(f"[agent6] {merge.noop_merge_line(plan.run_branch, plan.target, outcome)}.{note}")
        return 0
    print(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.method}) -> {outcome.merged_sha[:12]}{note}"
    )
    if kept := merge.left_behind_line(plan.target, outcome):
        print(f"[agent6] {kept}")
    if others := listing.same_path_runs(
        cwd, plan.layout.session_id, plan.base_sha, plan.run_branch
    ):
        print(f"[agent6] {_same_paths_line(others)}")
    return 0


def _same_paths_line(runs: Mapping[str, tuple[str, ...]]) -> str:
    """Return the line naming the unmerged runs that changed paths this merge landed.

    Args:
        runs: Each run's id with the paths both changed.

    Returns:
        Four runs at most, each with its first shared path, then the count of the rest.
    """
    named = [
        f"{run} ({terminal.printable_name(shared[0])}"
        + (f" and {len(shared) - 1} more)" if len(shared) > 1 else ")")
        for run, shared in list(runs.items())[:4]
    ]
    more = f", and {len(runs) - len(named)} more" if len(runs) > len(named) else ""
    return "unmerged runs that changed some of the same paths: " + ", ".join(named) + more
