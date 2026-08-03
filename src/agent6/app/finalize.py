# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""End of a run: the composed end block, exit code, auto-merge / auto-stash
finalizers, and the operator notify hook."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from agent6.app.baseline import gate_on_base
from agent6.app.merge import execute_merge
from agent6.budget import BudgetTracker
from agent6.config import Config, NotifyConfig
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    auto_stash_message,
    branch_exists,
    create_branch,
    delete_branch_if_merged,
    find_stash,
    restore_stash,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.types import IsolationLevel
from agent6.viewmodel import scan_run_log, summarize_run_dir
from agent6.viewmodel.format import format_cost
from agent6.workflows.loop import RunResult

# Distinct exit code for a budget-exhausted run so automation can tell "raise
# the cap and `agent6 resume`" apart from a genuine failure. Documented in
# CONFIG.md ([budget]); a budget-stopped run is resumable from its snapshot.
_EXIT_BUDGET_EXHAUSTED = 3
# The agent finished deliberately but the verify gate was red or stale. Its own
# code so a script can tell "the work is not green" from "the run broke" (1)
# without parsing the event log; `require_verify_to_finish` turns the same
# condition into a refusal to finish at all.
_EXIT_VERIFY_FAILED = 4


def run_exit_code(result: RunResult) -> int:
    """Map a finished run to its process exit code.

    0 finished (nothing to gate on, or the gate was green) / 3 budget /
    4 finished over a RED verify / 1 else."""
    if result.completed:
        return _EXIT_VERIFY_FAILED if result.verified == "failed" else 0
    if result.reason == "budget_exhausted":
        return _EXIT_BUDGET_EXHAUSTED
    return 1


def _sandbox_unreachable_tools(layout: RunLayout) -> list[str]:
    """Binaries the run flagged as host-present but jail-broken
    (loop.sandbox_tool_unreachable events), for the operator diagnostic."""
    out: list[str] = []
    try:
        for line in layout.logs_path.read_text(encoding="utf-8").splitlines():
            if '"loop.sandbox_tool_unreachable"' not in line:
                continue
            try:
                binary = json.loads(line).get("binary")
            except ValueError:
                continue
            if isinstance(binary, str) and binary and binary not in out:
                out.append(binary)
    except OSError:
        pass
    return out


def _print_baseline(
    result: RunResult, *, layout: RunLayout, cfg: Config, isolation: IsolationLevel
) -> None:
    """On a red gate, say whether it was red BEFORE this run.

    "your run failed" and "your change broke nothing new" are different facts,
    and the operator cannot tell them apart from a red exit alone -- least of
    all when the task WAS to change the tests. Only on red: green raises no
    question, and the answer would change nothing.
    """
    if result.verified != "failed":
        return
    base_sha = ""
    with contextlib.suppress(ManifestError):
        base_sha = read_manifest(layout.run_dir).base_sha
    baseline = gate_on_base(
        Path.cwd(),
        base_sha,
        argv=tuple(cfg.workflow.verify_command),
        isolation=isolation,
        timeout_s=cfg.workflow.verify_timeout_s,
    )
    print(f"\n{baseline.line()}")


def _print_stale_gate(result: RunResult) -> None:
    """Surface a proposed gate replacement, and say plainly that nothing moved.

    The worker may declare the configured gate stale instead of reverting
    correct work to satisfy it. Applying the proposal is the operator's call,
    so this prints the exact command rather than doing anything.
    """
    if not result.stale_gate:
        return
    print("\nthe worker says this run's verify gate no longer matches the task:")
    print(f"  it proposes: {result.stale_gate}")
    print("  nothing changed. To adopt it:")
    print(f"    agent6 config set workflow.verify_command {shlex.quote(result.stale_gate)}")


def print_run_end(
    result: RunResult,
    *,
    layout: RunLayout,
    budget: BudgetTracker,
    console_stream: bool,
    cfg: Config,
    isolation: IsolationLevel,
) -> None:
    """One composed end-of-run block: outcome, summary, cost, and the next step.

    When the live ConsoleView already rendered the `● done <summary>` terminator
    (console_stream), this omits the summary and just adds what the stream
    lacks: cost and the branch / next-step footer."""
    # Read the outcome from the SAME fold `agent6 runs` uses, not from
    # result.completed: completed means "the agent finished deliberately", which
    # is true for a finish_run even when verify never went green. status_word off
    # result.completed then prints "passed" while runs list reads the run.end
    # event's real all_passed and prints "finished" -- the exact disagreement
    # status_word exists to prevent. summarize_run_dir folds that event, so the
    # console headline and the listing can never diverge.
    summary = summarize_run_dir(layout.run_dir)
    word, reason = summary.status, summary.reason
    if not console_stream:
        # Headless: no ConsoleView ran, so this block is the only end output.
        headline = word if not reason else f"{word} · {reason.replace('_', ' ')}"
        print(f"\n{headline}")
        if result.summary:
            print(f"  {result.summary}")
    print()
    for binary in _sandbox_unreachable_tools(layout):
        print(
            f"WARNING: `{binary}` is installed on this machine but did not work"
            " inside agent6's sandbox."
        )
        print(
            "  Likely a per-user / version-manager install (rustup, pyenv, nvm, ...)"
            " whose config or toolchain the sandbox does not expose -- not an agent6"
            " bug. Fix options:"
        )
        print(f"    - make `{binary}` run from a clean shell (a system-wide install)")
        print("    - install it into a standard bin dir (~/.local/bin, /usr/local/bin)")
        print("    - grant its real directory via [sandbox].extra_read_paths")
        print("    - run with --dangerously-disable-sandbox")
    _print_baseline(result, layout=layout, cfg=cfg, isolation=isolation)
    _print_stale_gate(result)
    print(budget.format_summary())
    _print_run_total_across_legs(layout)
    run_branch = ""
    base_branch = ""
    merged_into = ""
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.run_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
        if manifest.merged is not None:
            merged_into = manifest.merged.into or base_branch
    if result.completed and run_branch and merged_into:
        # auto_merge already merged this branch into the base (and auto_prune may
        # have deleted it); don't tell the operator to merge it again.
        print(f"\nchanges merged into {merged_into}")
        print(f"  inspect:     agent6 runs diff {layout.run_id}")
    elif result.completed and run_branch:
        print(f"\nchanges are on {run_branch}")
        print(f"  merge with:  agent6 runs merge {layout.run_id}")
        print(f"  inspect:     agent6 runs diff {layout.run_id}")
        # The run left the checkout ON its branch (branch_per_run cuts it and
        # never switches back). Say so + how to leave, or the next run stacks on
        # it (see git.branch_from) and merge/prune defaults quietly shift.
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(Path.cwd()).branch
        if current == run_branch and base_branch and base_branch != run_branch:
            print(f"  you are on {run_branch}; return with: git switch {base_branch}")
    elif not result.completed:
        print(f"\nresume with:  agent6 resume {layout.run_id}")


def _print_run_total_across_legs(layout: RunLayout) -> None:
    """After the leg's token+cost banner: the run's true cumulative spend when
    resume legs precede this one. The tracker is per-leg (each resume starts a
    fresh budget), so its "TOTAL" line undersells a resumed run without this."""
    scan = scan_run_log(layout.run_dir / "logs.jsonl")
    if scan.legs > 1 and scan.cost_usd is not None:
        cost = format_cost(scan.cost_usd, partial=scan.usd_partial)
        print(f"  RUN TOTAL (all {scan.legs} legs): {cost}")


def print_interrupt_end(*, layout: RunLayout, budget: BudgetTracker) -> None:
    """After a Ctrl-C interrupt: the cost so far, the resume hint, and the
    branch-return hint. The interrupt cuts the run before ``print_run_end``, so
    without this the user saw only "run interrupted" -- no spend, no way to pick
    the (auto-committed, resumable) work back up, and no note they were left on
    the run branch. Mirrors the not-completed footer of ``print_run_end``."""
    print()
    print(budget.format_summary())
    _print_run_total_across_legs(layout)
    print(f"\nresume with:  agent6 resume {layout.run_id}")
    run_branch = ""
    base_branch = ""
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.run_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
    if run_branch:
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(Path.cwd()).branch
        if current == run_branch and base_branch and base_branch != run_branch:
            print(f"  you are on {run_branch}; return with: git switch {base_branch}")


def finalize_auto_merge(cwd: Path, *, layout: RunLayout, cfg: Config) -> None:
    """After a successful run, merge the run branch into its base using
    git.merge_strategy (git.auto_merge). Reads the run context from the manifest, so
    run + resume share it. Ends on the base branch (the pre-run branch) with a clean
    tree. Non-fatal and best-effort: on conflict or error the run branch is left
    intact and the message says how to merge by hand. No-op when branch_per_run was
    off."""
    try:
        manifest = read_manifest(layout.run_dir)
    except ManifestError:
        return
    run_branch = manifest.run_branch
    base_branch = manifest.base_branch
    if not run_branch or not base_branch:
        return  # branch_per_run was off: the work already landed on the base branch
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
        base_sha=manifest.base_sha,
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


def _stash_apply_cmd(sha: str, base_branch: str, *, needs_checkout: bool) -> str:
    """The manual-recovery command for a stash, worded once for every caller.

    Always apply-by-SHA: a positional ``pop 'stash@{N}'`` printed now but run
    later restores whatever sits at that position by then, which is how a
    bystander's stash got applied and the pre-run work stayed hidden."""
    apply = f"git stash apply {sha}"
    return f"git checkout {base_branch} && {apply}" if needs_checkout else apply


def stash_recovery_hint(cwd: Path, *, run_id: str, base_branch: str) -> str | None:
    """How to restore this run's pre-run auto-stash by hand, or None when the
    run pushed no stash. For callers that must tell the operator where their
    work went without restoring it (a detached run keeps the checkout on its
    run branch, so the stash has to wait)."""
    entry = find_stash(cwd, auto_stash_message(run_id))
    if entry is None:
        return None
    return _stash_apply_cmd(entry.sha, base_branch, needs_checkout=True)


def finalize_auto_stash(
    cwd: Path, *, base_branch: str, run_branch: str | None, auto_pop: bool, run_id: str
) -> None:
    """Restore or report the pre-run auto-stash so the user's work is never left in a
    hidden stash. With auto_pop off, print how to pop it. With auto_pop on, pop it
    onto the base branch when that is safe (clean worktree, conflict-free apply);
    otherwise leave the stash with a message. Never reset --hard (refused).

    The stash is found by the run-id message the run pushed it with, and
    restored by its immutable sha, never by position: a stash pushed DURING
    the run sat at stash@{0}, so a positional pop restored the wrong work and
    left the pre-run work hidden. The printed manual-recovery hint applies by
    sha too (``git stash apply <sha>``), which stays correct however the
    stash stack shifts later -- a positional ``pop 'stash@{N}'`` printed now
    but run after another stash push would restore the wrong one."""
    message = auto_stash_message(run_id)
    entry = find_stash(cwd, message)
    if entry is None:
        print(
            "[agent6] pre-run auto-stash not found (already restored?); nothing to pop",
            file=sys.stderr,
        )
        return
    # apply-by-sha is identity-stable; drop it yourself once you've confirmed.
    apply = f"git stash apply {entry.sha}"
    recover = _stash_apply_cmd(entry.sha, base_branch, needs_checkout=bool(run_branch))
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
                f"stashed (recover with: {apply})",
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
    try:
        restored = restore_stash(cwd, entry)
    except GitError as exc:
        # The apply itself landed; what failed is putting back a concurrent
        # stash the raced drop displaced. Say both -- finalization continues.
        print(
            f"[agent6] restored your pre-run changes onto {base_branch}, but {exc}",
            file=sys.stderr,
        )
        return
    if restored:
        print(f"[agent6] restored your pre-run changes onto {base_branch}", file=sys.stderr)
    else:
        print(
            "[agent6] restoring your pre-run changes hit a conflict; resolve the markers"
            f" (your stash is preserved; re-apply with: git stash apply {entry.sha})",
            file=sys.stderr,
        )


# What an operator notify hook inherits: enough to execute a script (PATH,
# HOME, locale, user identity) and to reach the desktop bus (notify-send needs
# DISPLAY/DBUS), plus the AGENT6_* facts the caller adds. Never the whole
# environment: it carries the provider API keys resolved via
# `[providers.*].api_key_env`, and a hook that logs or forwards its env
# (a shell wrapper, a webhook poster) would carry the key with it.
_HOOK_ENV_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


def hook_env(**agent6_vars: str) -> dict[str, str]:
    """The environment for an operator notify hook: the `_HOOK_ENV_KEEP` base
    plus the given ``AGENT6_*`` facts. The one owner for both hooks
    (`[notify].on_complete` here, `[machine.notify].on_event` in
    `app/machine/_preflight.py`), so their env-scope claims cannot drift."""
    env = {k: v for k in _HOOK_ENV_KEEP if (v := os.environ.get(k)) is not None}
    env.update(agent6_vars)
    return env


def fire_notify_hook(
    notify: NotifyConfig,
    *,
    run_id: str,
    run_dir: Path,
    ok: bool,
    reason: str,
    verified: str,
) -> None:
    """Run the operator-configured post-completion hook.

    The argv comes from `[notify].on_complete` in your config, operator-
    controlled, not LLM-controlled, so it does not go through the jail.
    Failures are logged to stderr and do not change the agent6 exit code.
    """
    if not notify.on_complete:
        return
    env = hook_env(
        AGENT6_RUN_ID=run_id,
        # OK = the agent stopped deliberately; VERIFIED = what the gate said
        # (passed / failed / not_applicable). A hook that wants "green" reads
        # the second: OK alone is true for a finish over a red verify.
        AGENT6_RUN_OK="1" if ok else "0",
        AGENT6_RUN_VERIFIED=verified,
        AGENT6_RUN_REASON=reason,
        AGENT6_RUN_DIR=str(run_dir),
    )
    try:
        subprocess.run(
            list(notify.on_complete),
            env=env,
            timeout=notify.timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[agent6] notify.on_complete failed: {exc}", file=sys.stderr)
