# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run-branch merge engine shared by `sessions merge` and `git.auto_merge`.

`cli.sessions_cmds` validates + resolves a run, then calls `execute_merge`; the run
finalizer (`app.finalize.finalize_auto_merge`) calls it directly with the run
context it already holds. One place to mutate means both honor the same strategy
dispatch, clean tree on failure, checkout restore, and manifest record."""

from __future__ import annotations

import contextlib
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app.manifest import write_manifest
from agent6.config import Config
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    MergeResult,
    branch_exists,
    branch_tip_sha,
    condense_commit_message,
    create_branch,
    is_ancestor,
    list_run_commits,
    merge_branch,
    set_repo_hook_policy,
    squash_merge,
)
from agent6.sessions.layout import SessionLayout
from agent6.sessions.manifest import ManifestError, MergeStamp, SessionManifest, read_manifest


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Result of execute_merge. `status` is merged / noop / conflict / error; the
    other fields carry that status's detail.

    `noop` is an already-merged branch: git stages nothing and leaves the target
    where it was, so there is no merge sha to report or record."""

    status: Literal["merged", "noop", "conflict", "error"]
    merged_sha: str = ""
    conflicts: tuple[str, ...] = ()
    error: str = ""
    # Why the manifest stamp did not land, "" when it did. The merge happened
    # either way; without this, `prune` calls the branch unmerged.
    stamp_error: str = ""


def record_merge_in_manifest(
    layout: SessionLayout, *, merged_into: str, merged_sha: str, merged_tip: str = ""
) -> str:
    """Record a successful merge in the run manifest so later tooling can tell a
    merged run branch from an unmerged one. *merged_tip* is the run-branch tip
    that was merged: `sessions prune --delete-squashed` force-deletes only a branch
    still pointing there. Best-effort: a missing/corrupt manifest must not fail a
    merge that already happened.

    Returns "" when the stamp landed, else why it did not. Silence made `prune`
    call a branch agent6 had merged minutes earlier "NOT merged", and left
    `--delete-squashed` unable to clean it up ever."""
    try:
        m = read_manifest(layout.session_dir)
    except ManifestError as exc:
        return str(exc)
    stamped = m.model_copy(
        update={
            "merged": MergeStamp(
                into=merged_into,
                sha=merged_sha,
                tip=merged_tip,
                ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
            )
        }
    )
    # Also ManifestError: a manifest newer than this binary can rewrite is left
    # alone rather than downgraded, and the merge it records already happened.
    try:
        write_manifest(layout.manifest_path, stamped)
    except (OSError, ManifestError) as exc:
        return str(exc)
    return ""


def restore_checkout(cwd: Path, original: str, target: str) -> None:
    """Switch back to the user's original branch after a merge ran on *target*, so a
    merge does not silently leave them on a different branch. No-op if they were
    already on the target or on a detached HEAD."""
    if original and original not in (target, "HEAD") and branch_exists(cwd, original):
        with contextlib.suppress(GitError):
            create_branch(cwd, original)


def dispatch_merge(
    cwd: Path,
    strategy: str,
    run_branch: str,
    base_sha: str,
    manifest: SessionManifest,
    message: str | None,
    cfg: Config,
    identity: CommitIdentity,
) -> MergeResult:
    """Run the chosen strategy. squash condenses the per-step commit messages (and
    folds in the configured coauthor); merge/ff hand off to merge_branch."""
    if strategy != "squash":
        return merge_branch(
            cwd, run_branch, ff_only=(strategy == "ff"), message=message, identity=identity
        )
    rows = list_run_commits(cwd, base_sha, run_branch)
    default_msg, coauthors = condense_commit_message(
        rows, subject=manifest.user_task or "agent6 run"
    )
    if cfg.git.commit.coauthor and cfg.git.commit.coauthor.lower() not in {
        c.lower() for c in coauthors
    }:
        coauthors = (*coauthors, cfg.git.commit.coauthor)
    return squash_merge(
        cwd,
        run_branch,
        message or default_msg,
        identity=CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email),
        coauthors=coauthors,
    )


def execute_merge(  # noqa: PLR0911
    cwd: Path,
    *,
    layout: SessionLayout,
    manifest: SessionManifest,
    run_branch: str,
    target: str,
    base_sha: str,
    strategy: str,
    message: str | None,
    cfg: Config,
    identity: CommitIdentity,
    original: str,
) -> MergeOutcome:
    """Check out *target*, merge *run_branch* in with *strategy*, restore the
    *original* checkout, and record the merge. The caller validates first; this
    mutates. Leaves a clean tree on conflict or error."""
    set_repo_hook_policy(cfg.git.run_repo_hooks)
    if not branch_exists(cwd, target):
        # The merge target must already exist; never fabricate it (create_branch
        # would otherwise make it at HEAD). runs merge pre-checks this for a nicer
        # message; auto_merge relies on this guard if the base was deleted mid-run.
        return MergeOutcome("error", error=f"target branch {target!r} does not exist")
    if (
        strategy == "ff"
        and not is_ancestor(cwd, target, run_branch)
        and not is_ancestor(cwd, run_branch, target)
    ):
        # Pre-check what `git merge --ff-only` would refuse: without it the
        # raw `fatal: Not possible to fast-forward` plus git's rebase hints
        # spew at the operator with no agent6 reason (auto_merge with an ff
        # config would spew the same on a moved base). A run branch the target
        # already CONTAINS is not refused: --ff-only is a clean no-op there
        # ("Already up to date"), even when the target has moved past it.
        return MergeOutcome(
            "error",
            error=(
                f"{target!r} has moved since the run branch was cut, so a"
                " fast-forward is impossible; merge with --strategy merge or"
                " squash instead"
            ),
        )
    try:
        create_branch(cwd, target)  # checkout the (now-verified) target
    except GitError as exc:
        return MergeOutcome("error", error=f"could not check out target branch {target!r}: {exc}")
    # A run strands the checkout on its OWN branch (branch_per_run switches at
    # start and never switches back), so `sessions merge <id>` is often invoked from
    # agent6/<id> -- meaning `original` IS the branch being merged. Restoring to
    # it would leave the user on a now-merged (squash: unreachable) branch whose
    # tree no longer matches the target. Land on the target instead; that is
    # where the work now lives.
    land_on = target if original == run_branch else original
    # Where the target stood before we touched it. Every strategy that merges
    # something moves it (squash commits, merge commits, a fast-forward), so an
    # unmoved target means git had nothing to merge.
    target_tip_before = branch_tip_sha(cwd, target) or ""
    try:
        result = dispatch_merge(
            cwd, strategy, run_branch, base_sha, manifest, message, cfg, identity
        )
    except GitError as exc:
        restore_checkout(cwd, land_on, target)
        return MergeOutcome("error", error=f"merge failed: {exc}")
    restore_checkout(cwd, land_on, target)
    if result.conflicted:
        return MergeOutcome("conflict", conflicts=result.conflicts)
    if result.merged_sha and result.merged_sha == target_tip_before:
        # Nothing merged. Reporting the target's own tip as the merge sha
        # credits the run with whatever was committed there since, and stamping
        # it destroys the record of the merge that did happen.
        return MergeOutcome("noop", merged_sha=result.merged_sha)
    stamp_error = record_merge_in_manifest(
        layout,
        merged_into=target,
        merged_sha=result.merged_sha,
        merged_tip=branch_tip_sha(cwd, run_branch) or "",
    )
    return MergeOutcome("merged", merged_sha=result.merged_sha, stamp_error=stamp_error)
