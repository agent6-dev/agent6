# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""End of a run: the composed end block, exit code, auto-merge / auto-stash
finalizers, and the operator notify hook."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from agent6.budget import BudgetTracker
from agent6.config import Config, NotifyConfig
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    create_branch,
    delete_branch_if_merged,
    restore_stash,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.runs.layout import RunLayout
from agent6.ui.cli._merge import execute_merge
from agent6.ui.viewmodel import status_word
from agent6.workflows.loop import RunResult

# Distinct exit code for a budget-exhausted run so automation can tell "raise
# the cap and `agent6 resume`" apart from a genuine failure. Documented in
# CONFIG.md ([budget]); a budget-stopped run is resumable from its snapshot.
_EXIT_BUDGET_EXHAUSTED = 3


def run_exit_code(result: RunResult) -> int:
    """Map a finished run to its process exit code (0 ok / 3 budget / 1 else)."""
    if result.completed:
        return 0
    if result.reason == "budget_exhausted":
        return _EXIT_BUDGET_EXHAUSTED
    return 1


def print_run_end(
    result: RunResult, *, layout: RunLayout, budget: BudgetTracker, console_stream: bool
) -> None:
    """One composed end-of-run block: outcome, summary, cost, and the next step.

    Replaces the old `result: completed=True reason=... iterations=...` repr line
    plus a re-print of the summary. When the live ConsoleView already rendered the
    `● done <summary>` terminator (console_stream), this omits the summary and
    just adds what the stream lacks: cost and the branch / next-step footer."""
    word, reason = status_word(finished=True, all_passed=result.completed, end_reason=result.reason)
    if not console_stream:
        # Headless: no ConsoleView ran, so this block is the only end output.
        headline = word if not reason else f"{word} · {reason.replace('_', ' ')}"
        print(f"\n{headline}")
        if result.summary:
            print(f"  {result.summary}")
    print()
    print(budget.format_summary())
    run_branch = ""
    with contextlib.suppress(OSError, ValueError):
        run_branch = json.loads(layout.manifest_path.read_text(encoding="utf-8")).get(
            "run_branch", ""
        )
    if result.completed and run_branch:
        print(f"\nchanges are on {run_branch}")
        print(f"  merge with:  agent6 runs merge {layout.run_id}")
        print(f"  inspect:     agent6 runs diff {layout.run_id}")
    elif not result.completed:
        print(f"\nresume with:  agent6 resume {layout.run_id}")


def finalize_auto_merge(cwd: Path, *, layout: RunLayout, cfg: Config) -> None:
    """After a successful run, merge the run branch into its base using
    git.merge_strategy (git.auto_merge). Reads the run context from the manifest, so
    run + resume share it. Ends on the base branch (the pre-run branch) with a clean
    tree. Non-fatal and best-effort: on conflict or error the run branch is left
    intact and the message says how to merge by hand. No-op when branch_per_run was
    off."""
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    run_branch = manifest.get("run_branch")
    base_branch = manifest.get("base_branch")
    if not run_branch or not base_branch:
        return  # branch_per_run was off: the work already landed on the base branch
    run_branch, base_branch = str(run_branch), str(base_branch)
    try:
        st = git_status(cwd)
    except GitError:
        return
    if not st.is_clean:
        print(
            f"[agent6] auto_merge skipped (worktree not clean); merge by hand:\n"
            f"    git checkout {base_branch} && git merge {run_branch}",
            file=sys.stderr,
        )
        return
    identity = CommitIdentity(
        name=cfg.git.commit.name, email=cfg.git.commit.email, coauthor=cfg.git.commit.coauthor
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        print(f"[agent6] auto_merge skipped: {exc}", file=sys.stderr)
        return
    outcome = execute_merge(
        cwd,
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=base_branch,
        base_sha=str(manifest.get("base_sha") or ""),
        strategy=cfg.git.merge_strategy,
        message=None,
        cfg=cfg,
        identity=identity,
        original="",  # stay on the base branch, where the work now lives
    )
    if outcome.status == "merged":
        print(
            f"[agent6] auto_merged {run_branch} into {base_branch} "
            f"({cfg.git.merge_strategy}) -> {outcome.merged_sha[:12]}",
            file=sys.stderr,
        )
        if cfg.git.auto_prune:
            if delete_branch_if_merged(cwd, run_branch):
                print(f"[agent6] auto_pruned {run_branch}", file=sys.stderr)
            else:
                print(
                    f"[agent6] auto_prune kept {run_branch} (squash-merged, unreachable; "
                    f"remove with: git branch -D {run_branch})",
                    file=sys.stderr,
                )
    elif outcome.status == "conflict":
        print(
            f"[agent6] auto_merge into {base_branch} hit conflicts "
            f"({', '.join(outcome.conflicts)}); left a clean tree on {base_branch} with the run "
            f"branch {run_branch} intact. Merge by hand:\n    git merge {run_branch}",
            file=sys.stderr,
        )
    else:
        print(f"[agent6] auto_merge failed: {outcome.error}", file=sys.stderr)


def finalize_auto_stash(
    cwd: Path, *, base_branch: str, run_branch: str | None, auto_pop: bool
) -> None:
    """Restore or report the pre-run auto-stash so the user's work is never left in a
    hidden stash. With auto_pop off, print how to pop it. With auto_pop on, pop it
    onto the base branch when that is safe (clean worktree, conflict-free apply);
    otherwise leave the stash with a message. Never reset --hard (refused)."""
    recover = f"git checkout {base_branch} && git stash pop" if run_branch else "git stash pop"
    if not auto_pop:
        print(
            f"[agent6] pre-run changes are stashed; restore them with: {recover}", file=sys.stderr
        )
        return
    try:
        st = git_status(cwd)
    except GitError:
        st = None
    if st is None or not st.is_clean:
        print(
            f"[agent6] pre-run changes left stashed (worktree not clean); restore with: {recover}",
            file=sys.stderr,
        )
        return
    if run_branch and st.branch == run_branch:
        if not branch_exists(cwd, base_branch):
            print(
                f"[agent6] base branch {base_branch} no longer exists; pre-run changes left "
                f"stashed (recover with: git stash pop)",
                file=sys.stderr,
            )
            return
        try:
            create_branch(cwd, base_branch)  # checks out the existing base branch
        except GitError as exc:
            print(
                f"[agent6] could not switch to {base_branch} to restore the stash ({exc}); "
                f"restore with: {recover}",
                file=sys.stderr,
            )
            return
    if restore_stash(cwd):
        print(f"[agent6] restored your pre-run changes onto {base_branch}", file=sys.stderr)
    else:
        print(
            "[agent6] restoring your pre-run changes hit a conflict; resolve the markers"
            " (your stash is preserved at stash@{0})",
            file=sys.stderr,
        )


def fire_notify_hook(
    notify: NotifyConfig,
    *,
    run_id: str,
    run_dir: Path,
    ok: bool,
    reason: str,
) -> None:
    """Run the operator-configured post-completion hook.

    The argv comes from `[notify].on_complete` in your config, operator-
    controlled, not LLM-controlled, so it does not go through the jail.
    Failures are logged to stderr and do not change the agent6 exit code.
    """
    if not notify.on_complete:
        return
    env = dict(os.environ)
    env["AGENT6_RUN_ID"] = run_id
    env["AGENT6_RUN_OK"] = "1" if ok else "0"
    env["AGENT6_RUN_REASON"] = reason
    env["AGENT6_RUN_DIR"] = str(run_dir)
    try:
        subprocess.run(
            list(notify.on_complete),
            env=env,
            timeout=notify.timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[agent6] notify.on_complete failed: {exc}", file=sys.stderr)
