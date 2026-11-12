# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Plan and land a run's work on a branch: the one merge every surface calls.

The CLI passes its reporter; the TUI and the web call `merge_session_dir` and show what it
returns. Nothing here prints by itself.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib

from agent6 import errors, kinds, terminal
from agent6.app import merge, resume, worktrees
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError, layer
from agent6.harness import child_repo
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing, scan_session_log, wire

# The gate words that mean the gate did not certify the run's final tree.
UNVERIFIED_WARNINGS = {
    "failed": "the verify gate was red on this run's final tree",
    "timed_out": "the verify gate timed out on this run's final tree",
    "unverified": "nothing verified this run's final tree",
}


@dataclasses.dataclass(frozen=True, slots=True)
class MergePlan:
    """A validated, mutation-ready merge: everything `run_merge` needs after every guard passed.

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


@dataclasses.dataclass(frozen=True, slots=True)
class CommitsRef:
    """Where a session's commits end, for `base_sha..head_ref`.

    Attributes:
        head_ref: The run branch, the hidden chain ref for a run with branch_per_run off,
            or "" when the session made no commits.
        reason: Why there is no ref; "" exactly when `head_ref` is one, so the branch verbs
            refuse on it while diff reads `head_ref`.
    """

    head_ref: str
    reason: str


def commits_ref(cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest) -> CommitsRef:
    """Return where the run's commits end.

    Args:
        cwd: The repo.
        manifest: The run's manifest.

    Returns:
        The run branch while it covers the chain, else the chain ref; else the manifest's
        branch name while no chain exists (the verbs read its absence themselves: pruned,
        never cut, or a child's branch still in its clone); else the reason the run has no
        commits.
    """
    if ref := wire.commits_ref(manifest, cwd):
        return CommitsRef(head_ref=ref, reason="")
    if (
        manifest.run_branch
        and git_read.chain_tip(cwd, git_read.chain_ref_for(manifest.session_id)) is None
    ):
        return CommitsRef(head_ref=manifest.run_branch, reason="")
    if manifest.parked_task:
        # A parked run never started, so `base..HEAD` is the run it was parked behind.
        return CommitsRef(
            head_ref="", reason="this run was parked before it started, so it made no commits"
        )
    kind = kinds.SESSION_KINDS.get(manifest.mode)
    if kind is not None and not kind.edits:
        article = "an" if manifest.mode[:1] in "aeiou" else "a"
        return CommitsRef(
            head_ref="",
            reason=f"{article} {manifest.mode} does not write to the repo, so it made no commits",
        )
    return CommitsRef(head_ref="", reason="this run recorded no commits")


def chain_refusal(
    manifest: sessions_manifest.SessionManifest, *, allow_model: bool = False
) -> str | None:
    """Return why a session has no agent6 chain for the branch verbs to read, or None.

    Args:
        manifest: The session's manifest.
        allow_model: Permit a read-only comparison of a model-controlled run.

    Returns:
        The refusal for a group, whose children hold the commits, or for a run whose model
        managed git itself.
    """
    return None if allow_model else sessions_manifest.model_git_refusal(manifest, "sessions")


def not_mergeable_now(layout: sessions_layout.SessionLayout, session_id: str) -> str:
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


def plan_merge(  # noqa: PLR0911
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    into: str | None,
    method: str | None,
    *,
    config_paths: tuple[pathlib.Path, ...],
    reporter: app_reporter.Reporter,
) -> MergePlan | int:
    """Validate everything a merge needs, without touching the repo.

    Args:
        cwd: The repo.
        layout: The run.
        manifest: Its manifest.
        into: The target branch; None takes the branch the run was cut from.
        method: The merge method; None takes `git.merge_method`.
        config_paths: The `--config` files in command-line order, if any.
        reporter: Receives every refusal and note.

    Returns:
        The plan, or the exit code of a reported refusal or an already-pruned merge.
        Every guard runs before `run_merge` mutates anything.
    """
    if refusal := chain_refusal(manifest) or not_mergeable_now(layout, manifest.session_id):
        reporter.refuse(refusal)
        return 2
    ref = commits_ref(cwd, manifest)
    if (stamp := resume.covering_stamp(cwd, manifest)) is not None and (
        not ref.head_ref or git_read.chain_tip(cwd, ref.head_ref) is None
    ):
        already = "already " if stamp.commit else ""
        reporter.out(f"[agent6] {already}{stamp.landed()} (run refs were pruned).")
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
        reporter.refuse(unmergeable)
        return 2
    run_branch = ref.head_ref
    target = into or manifest.base_branch
    if not target or target == run_branch:
        reporter.error(
            "no base branch (the run started on a detached HEAD); pass --into <existing-branch>."
            if not target
            else f"target {target!r} is the run branch itself; pass --into <other-branch>."
        )
        return 2
    try:
        cfg = layer.load_effective(cwd, config_paths).config
    except ConfigError as exc:
        reporter.error(f"{exc}")
        return 2
    # An orphaned child (its parent died before importing it) is adopted: fetched, then merged.
    try:
        adopted = worktrees.adopt_orphan_child(cwd, cfg, layout, manifest)
    except child_repo.ChildRepoError as exc:
        reporter.error(f"{exc}")
        return 2
    if adopted is not None:
        reporter.out(f"[agent6] {adopted}")
    # chain_tip resolves both shapes of head_ref: a branch name and the hidden chain ref.
    if git_read.chain_tip(cwd, run_branch) is None:
        if git_read.chain_tip(cwd, git_read.chain_ref_for(manifest.session_id)) is None:
            # Not a failure: there is no work to land.
            reporter.out("[agent6] nothing to merge: this run committed nothing.")
            return 0
        reporter.error(
            f"run ref {run_branch!r} no longer exists; its commits survive at"
            f" {git_read.chain_ref_for(manifest.session_id)}."
        )
        return 2
    if not git_read.branch_exists(cwd, target):
        reporter.error(f"target branch {target!r} does not exist; pass --into <existing-branch>.")
        return 2
    identity = merge.commit_identity(cfg, layout.session_dir, manifest)
    try:
        git_read.verify_git_identity(cwd, identity)  # refuse cleanly before mutating anything
    except errors.GitError as exc:
        reporter.error(f"{exc}")
        return 2
    return MergePlan(
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=target,
        base_sha=manifest.base_sha,
        method=method or cfg.git.merge_method,
        identity=identity,
        cfg=cfg,
    )


def manual_merge_cmd(cwd: pathlib.Path, plan: MergePlan) -> str:
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


def run_merge(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    *,
    method: str | None,
    into: str | None,
    message: str | None,
    config_paths: tuple[pathlib.Path, ...],
    reporter: app_reporter.Reporter,
) -> int:
    """Land a run's work on a target branch with the chosen method.

    Ref plumbing only: the checkout, index and worktree are never the medium, so a worktree
    still carrying the run's work is no obstacle.

    Args:
        cwd: The repo.
        layout: The run.
        manifest: Its manifest.
        method: The merge method; None takes `git.merge_method`.
        into: The target branch; None takes the branch the run was cut from.
        message: The merge commit message; None takes the default.
        config_paths: The `--config` files in command-line order, if any.
        reporter: Receives every line the merge says.

    Returns:
        The exit code: 0 landed or nothing to land, 1 a conflict, 2 a refusal.
    """
    plan = plan_merge(
        cwd, layout, manifest, into, method, config_paths=config_paths, reporter=reporter
    )
    if isinstance(plan, int):
        return plan
    if not git_read.list_run_commits(cwd, plan.base_sha, plan.run_branch):
        # A success line here would be indistinguishable from a real merge.
        reporter.out(f"[agent6] nothing to merge: run branch {plan.run_branch} has no commits.")
        return 0
    # The merge lands whatever the gate said; the operator hears it first.
    if unverified := UNVERIFIED_WARNINGS.get(scan_session_log(plan.layout.logs_path).gate_word()):
        reporter.note(f"warning: {unverified}; merging it.")
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
        warn=reporter.note,
    )
    if outcome.status == "error":
        reporter.error(f"{outcome.error}")
        return 1
    if outcome.status == "conflict":
        in_worktree = (
            "Resolve them in the run's own worktree:\n"
            f"    agent6 resume {plan.layout.session_id} --merge {plan.target}\n"
            if plan.manifest.worktree is not None
            else ""
        )
        reporter.err(
            f"CONFLICT: merging {plan.run_branch} into {plan.target} hit conflicts in "
            f"{', '.join(outcome.conflicts)}. Your checkout is untouched (no partial merge).\n"
            f"{in_worktree}Or by hand:\n"
            f"    {manual_merge_cmd(cwd, plan)}"
        )
        return 1
    note = (
        f"\n  (merge record could not be written: {outcome.stamp_error};"
        " `sessions prune` will call this branch unmerged)"
        if outcome.stamp_error
        else ""
    )
    if outcome.status == "noop":
        reporter.out(
            f"[agent6] {merge.noop_merge_line(plan.run_branch, plan.target, outcome)}.{note}"
        )
        return 0
    reporter.out(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.method}) -> {outcome.merged_sha[:12]}{note}"
    )
    if kept := merge.left_behind_line(plan.target, outcome):
        reporter.out(f"[agent6] {kept}")
    if same := same_paths_note(cwd, plan.layout.session_dir):
        reporter.out(f"[agent6] {same}")
    return 0


def merge_session_dir(
    cwd: pathlib.Path,
    session_dir: pathlib.Path,
    *,
    method: str | None,
    config_paths: tuple[pathlib.Path, ...],
) -> tuple[bool, str]:
    """Merge the session in a directory, for a front-end with no terminal of its own.

    Args:
        cwd: The repo.
        session_dir: The session's directory.
        method: The merge method; None takes `git.merge_method`.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        Whether the merge landed or had nothing to land, and every line it said, in order.
    """
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError as exc:
        reporter.error(f"could not read manifest: {exc}")
        return False, "\n".join(said)
    rc = run_merge(
        cwd,
        sessions_layout.layout_of(session_dir),
        manifest,
        method=method,
        into=None,
        message=None,
        config_paths=config_paths,
        reporter=reporter,
    )
    return rc == 0, "\n".join(said)


def same_paths_note(cwd: pathlib.Path, session_dir: pathlib.Path) -> str:
    """Return the line naming the unmerged runs that changed paths this run changed.

    A merge says it after landing, and a front-end's confirm before the click.

    Args:
        cwd: The repo.
        session_dir: The run's directory.

    Returns:
        Four runs at most, each with its first shared path, then the count of the rest;
        "" when no other unmerged run shares a path, or this run has no commits or no
        readable manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return ""
    ref = commits_ref(cwd, manifest)
    if not ref.head_ref or not manifest.base_sha:
        return ""
    runs = listing.same_path_runs(cwd, manifest.session_id, manifest.base_sha, ref.head_ref)
    if not runs:
        return ""
    named = [
        f"{run} ({terminal.printable_name(shared[0])}"
        + (f" and {len(shared) - 1} more)" if len(shared) > 1 else ")")
        for run, shared in list(runs.items())[:4]
    ]
    more = f", and {len(runs) - len(named)} more" if len(runs) > len(named) else ""
    return "unmerged runs that changed some of the same paths: " + ", ".join(named) + more
