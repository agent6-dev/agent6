# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs diff/commits/merge/prune` commands (the run-branch lifecycle)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.cli._common import _runs_dir, _state_dir
from agent6.cli._merge import execute_merge
from agent6.cli.plan_watch import _most_recent_run_id
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config_layer import load_effective
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    delete_branch_if_merged,
    is_ancestor,
    is_git_repo,
    list_run_branches,
    list_run_commits,
    verify_git_identity,
)
from agent6.git_ops import status as git_status
from agent6.graph.storage import RunLayout
from agent6.run_id import RunIdError, resolve_run_id


def _cmd_diff(*, run_id: str, stat: bool, paths: tuple[str, ...]) -> int:  # noqa: PLR0911
    """Print the git diff a run produced (manifest.base_sha -> branch HEAD).

    Resolves the run id (or unique prefix; empty string means most-recent),
    reads ``manifest.json`` for ``base_sha`` and ``run_branch``, then shells
    out to ``git diff`` with operator-controlled argv (no LLM input).
    """
    cwd = Path.cwd()
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return 2

    target_id = run_id
    if not target_id:
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}", file=sys.stderr)
            return 2
        target_id = latest
        print(f"[agent6] diffing most recent run: {target_id}", file=sys.stderr)
    else:
        try:
            target_id = resolve_run_id(runs_dir, target_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    layout = RunLayout(state_dir=_state_dir(cwd), run_id=target_id)
    if not layout.manifest_path.is_file():
        print(
            f"ERROR: run {target_id} has no manifest.json "
            "(predates manifest support, or was killed before setup)",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2

    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    if not base_sha:
        print("ERROR: manifest has no base_sha; nothing to diff against", file=sys.stderr)
        return 2

    head_ref = str(run_branch) if run_branch else "HEAD"
    argv: list[str] = ["git", "diff"]
    if stat:
        argv.append("--stat")
    argv.extend([f"{base_sha}..{head_ref}"])
    if paths:
        argv.append("--")
        argv.extend(paths)
    print(
        f"[agent6] {' '.join(argv)}  (base_branch={manifest.get('base_branch')!r})",
        file=sys.stderr,
    )
    proc = subprocess.run(argv, cwd=cwd, check=False)
    return proc.returncode


def _resolve_run_manifest(cwd: Path, run_id: str) -> tuple[RunLayout, dict[str, Any]] | int:
    """Resolve a run id (or '' for most-recent) to its (layout, manifest), or an exit
    code on error. Shared by `runs merge` and `runs commits`."""
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return 2
    target_id = run_id
    if not target_id:
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}", file=sys.stderr)
            return 2
        target_id = latest
        print(f"[agent6] using most recent run: {target_id}", file=sys.stderr)
    else:
        try:
            target_id = resolve_run_id(runs_dir, target_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    layout = RunLayout(state_dir=_state_dir(cwd), run_id=target_id)
    if not layout.manifest_path.is_file():
        print(f"ERROR: run {target_id} has no manifest.json", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2
    return layout, manifest


def _cmd_commits(*, run_id: str) -> int:
    """List the per-step commits on a run's branch (manifest.base_sha -> run branch)."""
    cwd = Path.cwd()
    res = _resolve_run_manifest(cwd, run_id)
    if isinstance(res, int):
        return res
    _layout, manifest = res
    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    if not run_branch or not base_sha:
        print(
            "ERROR: this run has no branch/base recorded (branch_per_run was off?).",
            file=sys.stderr,
        )
        return 2
    rows = list_run_commits(cwd, base_sha, str(run_branch))
    if not rows:
        print("[agent6] no commits on the run branch.")
        return 0
    for row in rows:
        print(f"{row.sha[:12]}  {row.subject}")
    print(f"\n[agent6] {len(rows)} commit(s) on {run_branch}", file=sys.stderr)
    return 0


@dataclass(frozen=True, slots=True)
class _MergePlan:
    """A validated, mutation-ready merge: everything `_cmd_merge` needs after every
    guard has passed. `_plan_merge` builds it without touching the repo."""

    layout: RunLayout
    manifest: dict[str, Any]
    run_branch: str
    target: str
    base_sha: str
    strategy: str
    identity: CommitIdentity
    cfg: Config
    original: str


def _plan_merge(  # noqa: PLR0911
    cwd: Path, run_id: str, into: str | None, strategy: str | None
) -> _MergePlan | int:
    """Resolve and validate everything a merge needs, or return an exit code. Pure:
    every guard fails before `_cmd_merge` mutates the repo."""
    res = _resolve_run_manifest(cwd, run_id)
    if isinstance(res, int):
        return res
    layout, manifest = res
    run_branch = manifest.get("run_branch")
    if not run_branch:
        print(
            "ERROR: this run has no branch to merge (branch_per_run was off, so the "
            "work already landed on your current branch).",
            file=sys.stderr,
        )
        return 2
    run_branch = str(run_branch)
    target = into or str(manifest.get("base_branch") or "")
    if not target:
        print(
            "ERROR: no target branch (manifest has no base_branch); pass --into <branch>.",
            file=sys.stderr,
        )
        return 2
    if target == run_branch:
        print(
            f"ERROR: target {target!r} is the run branch itself; pass --into <other-branch>.",
            file=sys.stderr,
        )
        return 2
    try:
        cfg = load_effective(cwd, None).config
        st = git_status(cwd)
    except (ConfigError, GitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not st.is_clean:
        print(
            "REFUSING: working tree is not clean; commit or stash your changes first.",
            file=sys.stderr,
        )
        return 2
    if not branch_exists(cwd, run_branch):
        print(f"ERROR: run branch {run_branch!r} no longer exists.", file=sys.stderr)
        return 2
    if not branch_exists(cwd, target):
        print(
            f"ERROR: target branch {target!r} does not exist; pass --into <existing-branch>.",
            file=sys.stderr,
        )
        return 2
    identity = CommitIdentity(
        name=cfg.git.commit.name, email=cfg.git.commit.email, coauthor=cfg.git.commit.coauthor
    )
    try:
        verify_git_identity(cwd, identity)  # refuse cleanly before mutating anything
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return _MergePlan(
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=target,
        base_sha=str(manifest.get("base_sha") or ""),
        strategy=strategy or cfg.git.merge_strategy,
        identity=identity,
        cfg=cfg,
        original=st.branch,
    )


def _cmd_merge(*, run_id: str, strategy: str | None, into: str | None, message: str | None) -> int:
    """Merge a run's branch into a target (default: the branch the run was cut
    from), with the chosen strategy (default: git.merge_strategy). Refuses a dirty
    worktree, leaves a clean tree on failure, restores your original checkout, and
    records the merge in the manifest."""
    cwd = Path.cwd()
    plan = _plan_merge(cwd, run_id, into, strategy)
    if isinstance(plan, int):
        return plan
    outcome = execute_merge(
        cwd,
        layout=plan.layout,
        manifest=plan.manifest,
        run_branch=plan.run_branch,
        target=plan.target,
        base_sha=plan.base_sha,
        strategy=plan.strategy,
        message=message,
        cfg=plan.cfg,
        identity=plan.identity,
        original=plan.original,
    )
    if outcome.status == "error":
        print(f"ERROR: {outcome.error}", file=sys.stderr)
        return 1
    if outcome.status == "conflict":
        print(
            f"CONFLICT: merging {plan.run_branch} into {plan.target} hit conflicts in "
            f"{', '.join(outcome.conflicts)}. The tree was left clean (no partial merge); "
            f"resolve it by hand if you want:\n"
            f"    git checkout {plan.target} && git merge {plan.run_branch}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.strategy}) -> {outcome.merged_sha[:12]}"
    )
    return 0


def _manifest_merged_into(state_dir: Path, branch: str) -> str:
    """The base branch the run owning *branch* (agent6/<run_id>) was merged into, or
    "" if there is no manifest or it was never recorded as merged."""
    run_id = branch.removeprefix("agent6/")
    try:
        manifest = json.loads(
            RunLayout(state_dir=state_dir, run_id=run_id).manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return ""
    return str(manifest.get("merged_into") or "") if manifest.get("merged_sha") else ""


def _cmd_prune() -> int:
    """Delete agent6/* run branches that `git branch -d` can safely remove
    (reachable-merged into HEAD, i.e. merge/ff strategies). Report squash-merged
    ones (remove by hand with git branch -D) and unmerged ones (review first);
    agent6 never force-deletes."""
    cwd = Path.cwd()
    if not is_git_repo(cwd):
        print("ERROR: not a git repository", file=sys.stderr)
        return 2
    branches = list_run_branches(cwd)
    if not branches:
        print("[agent6] no agent6/* run branches to prune.")
        return 0
    try:
        current = git_status(cwd).branch
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    state_dir = _state_dir(cwd)
    deleted = merged_kept = unmerged_kept = 0
    for br in branches:
        if br == current:
            print(f"[agent6] skipped {br} (checked out)", file=sys.stderr)
            continue
        if delete_branch_if_merged(cwd, br):
            deleted += 1
            print(f"[agent6] deleted {br} (merged)")
            continue
        merged_into = _manifest_merged_into(state_dir, br)
        if not merged_into:
            unmerged_kept += 1
            print(f"[agent6] kept {br} (NOT merged; review, then: git branch -D {br})")
            continue
        merged_kept += 1
        if branch_exists(cwd, merged_into) and is_ancestor(cwd, br, merged_into):
            # Reachable-merged into its base, so `git branch -d` only refused because
            # HEAD is not the base; deleting it cleanly needs to run from the base.
            print(
                f"[agent6] kept {br} (merged into {merged_into} but not reachable from "
                f"{current!r}; re-run prune on {merged_into}, or: git branch -D {br})"
            )
        else:
            print(
                f"[agent6] kept {br} (squash-merged into {merged_into}, unreachable; "
                f"remove with: git branch -D {br})"
            )
    print(
        f"\n[agent6] deleted {deleted}; kept {merged_kept + unmerged_kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged)",
    )
    return 0
