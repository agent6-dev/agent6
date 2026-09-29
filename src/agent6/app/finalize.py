# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end of a run.

The composed end block, the exit code, the auto-merge and auto-stash finalizers,
and the operator notify hook.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

from agent6.app.merge import execute_merge, left_behind_line, noop_merge_line
from agent6.app.reporter import Reporter
from agent6.budget import BudgetTracker
from agent6.child_env import curated_env
from agent6.commit_message import render_commit_trailer
from agent6.config import Config, NotifyConfig
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    auto_stash_message,
    branch_exists,
    chain_ref_for,
    chain_tip,
    create_branch,
    delete_branch_if_merged,
    find_stash,
    merge_stamp_holds,
    restore_stash,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.harness.loop import SessionResult
from agent6.sessions.layout import LOGS_NAME, SessionLayout, read_untracked_at_start
from agent6.sessions.manifest import ManifestError, SessionManifest, read_manifest
from agent6.verify_infer import line_to_argv
from agent6.viewmodel import scan_session_log, summarize_session_dir, tail_events, worker_models
from agent6.viewmodel.format import format_usd, status_label
from agent6.viewmodel.wire import commits_ref

# The run stopped at its budget; resumable from its snapshot once the cap is raised.
_EXIT_BUDGET_EXHAUSTED = 3
# The agent finished but the gate was red or unverified; the fan-out and a BLOCK verdict too.
EXIT_VERIFY_FAILED = 4
# The agent finished green but no commit landed and the edits sit uncommitted.
EXIT_NO_COMMIT_LANDED = 5


def session_exit_code(result: SessionResult, *, stranded: bool = False) -> int:
    """Map a finished run to its process exit code.

    An unverified finish exits 4 like a red one, so a worker cannot pass by never
    running the gate; a red gate outranks stranded edits.

    Args:
        result: The session's result.
        stranded: Whether the edits sit uncommitted with no commit landed.

    Returns:
        0 for a green or gateless finish, 3 budget, 4 not green, 5 stranded, 1 else.
    """
    if result.completed:
        if result.verified in ("failed", "unverified"):
            return EXIT_VERIFY_FAILED
        return EXIT_NO_COMMIT_LANDED if stranded else 0
    if result.reason == "budget_exhausted":
        return _EXIT_BUDGET_EXHAUSTED
    return 1


def stranded_edits(result: SessionResult, layout: SessionLayout, cwd: Path) -> bool:
    """Tell whether a completed run left its edits uncommitted with no commit landed.

    A run with no chain to commit to, or one configured never to commit, leaves the
    tree as it is by design. The exit code and the end banner read this one predicate.

    Args:
        result: The session's result.
        layout: The session's layout.
        cwd: The checkout.

    Returns:
        Whether the edits are stranded.
    """
    if not result.completed:
        return False
    try:
        manifest = read_manifest(layout.session_dir)
    except ManifestError:
        return False
    if manifest.mode != "run" or manifest.git_control == "model":
        return False
    if not manifest.policy.commit_per_step:
        return False
    merged = manifest.merged is not None and merge_stamp_holds(
        cwd, manifest.session_id, manifest.run_branch or "", manifest.merged.tip
    )
    if merged or commits_ref(manifest, cwd):
        return False
    dirty = False
    with contextlib.suppress(GitError):
        exclude = read_untracked_at_start(layout.session_dir)
        dirty = not git_status(cwd, exclude=exclude).is_clean
    return dirty


def auto_merge_eligible(result: SessionResult) -> bool:
    """Tell whether auto_merge may land the run.

    Args:
        result: The session's result.

    Returns:
        Whether the finish was green or gateless.
    """
    return result.completed and result.verified in ("passed", "not_applicable")


def _sandbox_unreachable_tools(layout: SessionLayout) -> list[str]:
    """List the binaries the run flagged as present on the host but unreachable in the jail.

    Args:
        layout: The session's layout.

    Returns:
        The binaries, in first-seen order.
    """
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


def _print_next_session(layout: SessionLayout, *, completed: bool, reporter: Reporter) -> None:
    """Print the next step after a plan or ask that produced something to act on.

    A plan's hints follow its plan.md, which only `finish_planning` writes; an ask's
    hint follows a completed session.

    Args:
        layout: The session's layout.
        completed: Whether the session finished deliberately.
        reporter: Receives the lines.
    """
    with contextlib.suppress(ManifestError):
        mode = read_manifest(layout.session_dir).mode
        session_id = layout.session_id
        if mode == "plan":
            # The plan is the deliverable, printed like an ask prints its answer.
            with contextlib.suppress(OSError):
                plan = (layout.session_dir / "plan.md").read_text(encoding="utf-8").rstrip()
                if plan:
                    reporter.out(f"\n{plan}")
                    reporter.out(f"\nedit:     agent6 plan edit {session_id}")
                    reporter.out(f'revise:   agent6 resume {session_id} --steer "<what to change>"')
                    reporter.out(f"execute:  agent6 run --from {session_id}")
        elif mode == "ask" and completed:
            reporter.out(f'\nnext:  agent6 run --from {session_id} "<what to do with it>"')


def _print_unknown_baseline(
    result: SessionResult, *, layout: SessionLayout, reporter: Reporter
) -> None:
    """Say when a red gate was never observed at the base, and name the check.

    A run whose first verify ran on an unmodified tree ends `gate_red_at_base`
    instead. Saying "unknown" beats a second gate run in the teardown.

    Args:
        result: The session's result.
        layout: The session's layout.
        reporter: Receives the lines.
    """
    if result.verified != "failed" or result.reason == "gate_red_at_base":
        return
    gate = ()
    base = ""
    with contextlib.suppress(ManifestError):
        m = read_manifest(layout.session_dir)
        gate, base = m.harness.verify_command, (m.forked_from_sha or m.base_sha)
    if not (gate and base):
        return
    reporter.out(
        f"\nthe gate is red, and nothing checked it before this run started ({base[:12]})."
    )
    # A worktree, not a stash: the run's work is committed, so a stash would exclude nothing.
    reporter.out("  to see whether this run caused it, check out the base commit somewhere else:")
    reporter.out(f"    git worktree add /tmp/agent6-base {base[:12]} \\")
    reporter.out(f"      && (cd /tmp/agent6-base && {shlex.join(gate)})")


def _print_unverified(result: SessionResult, *, layout: SessionLayout, reporter: Reporter) -> None:
    """Say what is missing after a gated end nothing observed.

    Args:
        result: The session's result.
        layout: The session's layout.
        reporter: Receives the lines.
    """
    if result.verified != "unverified":
        return
    reporter.out(
        "\nnothing verified the final tree: no verify ran this execution, or edits landed"
        " after the last green."
    )
    reporter.out(f'  resume and run the gate:  agent6 resume {layout.session_id} --steer "verify"')


def _print_stale_gate(result: SessionResult, *, reporter: Reporter) -> None:
    """Print the worker's proposed gate replacement, and that nothing moved.

    Applying the proposal is the operator's call. A proposal over a green gate is
    not shown: nothing found fault with that gate.

    Args:
        result: The session's result.
        reporter: Receives the lines.
    """
    if not result.stale_gate or result.verified not in ("failed", "unverified"):
        return
    reporter.out("\nthe worker says this run's verify gate no longer matches the task:")
    reporter.out(f"  it proposes: {result.stale_gate}")
    reporter.out("  nothing changed. To adopt it:")
    # `config set` takes argv as a JSON array; the inference's tokeniser wraps a pipeline.
    argv = json.dumps(list(line_to_argv(result.stale_gate) or ()))
    reporter.out(f"    agent6 config set harness.verify_command {shlex.quote(argv)}")


def print_session_end(
    result: SessionResult,
    *,
    layout: SessionLayout,
    cwd: Path,
    budget: BudgetTracker,
    console_stream: bool,
    reporter: Reporter,
) -> None:
    """Print the composed end-of-run block: outcome, summary, cost and the next step.

    Args:
        result: The session's result.
        layout: The session's layout.
        cwd: The checkout.
        budget: The execution's tracker.
        console_stream: Whether a live console view already rendered the done line, so
            only cost and the footer are added.
        reporter: Receives the lines.
    """
    # The outcome comes from the fold `agent6 sessions` reads, so the two cannot diverge.
    summary = summarize_session_dir(layout.session_dir)
    word, reason = summary.status, summary.reason
    if not console_stream:
        # Headless: this block is the only end output.
        headline = status_label(word, reason)
        reporter.out(f"\n{headline}")
        if result.summary:
            reporter.out(f"  {result.summary}")
    elif result.summary and result.reason not in ("finish_session", "finish_planning"):
        # The stream's done line carries a clean finish's summary only; a failure's reason
        # reaches the operator here alone.
        reporter.out(f"  {result.summary}")
    reporter.out("")
    if unreachable := _sandbox_unreachable_tools(layout):
        # One remedy for the set, not one per binary.
        names = ", ".join(f"`{b}`" for b in unreachable)
        verb = "is" if len(unreachable) == 1 else "are"
        reporter.out(
            f"WARNING: {names} {verb} installed on this machine but did not work"
            " inside agent6's sandbox."
        )
        reporter.out(
            "  Likely a per-user or version-manager install (rustup, pyenv, nvm)"
            " whose toolchain the sandbox does not expose. Fix options:"
        )
        reporter.out("    - run them from a clean shell (a system-wide install)")
        reporter.out("    - install them into a standard bin dir (~/.local/bin, /usr/local/bin)")
        reporter.out("    - grant their real directories via [sandbox].extra_read_paths")
        reporter.out("    - run with --dangerously-disable-sandbox")
    _print_next_session(layout, completed=result.completed, reporter=reporter)
    _print_unknown_baseline(result, layout=layout, reporter=reporter)
    _print_unverified(result, layout=layout, reporter=reporter)
    _print_stale_gate(result, reporter=reporter)
    reporter.cost(budget.format_summary())
    _print_run_total_across_executions(layout, reporter=reporter)
    _print_run_branch_footer(result, layout=layout, cwd=cwd, reporter=reporter)


def _print_run_branch_footer(
    result: SessionResult, *, layout: SessionLayout, cwd: Path, reporter: Reporter
) -> None:
    """Print where the run's changes are, every claim checked against git.

    Args:
        result: The session's result.
        layout: The session's layout.
        cwd: The checkout.
        reporter: Receives the lines.
    """
    run_branch = ""
    base_branch = ""
    merged_into = ""
    manifest: SessionManifest | None = None
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.session_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
        if manifest.merged is not None and merge_stamp_holds(
            cwd, manifest.session_id, run_branch, manifest.merged.tip
        ):
            merged_into = manifest.merged.into or base_branch
    if result.completed and manifest is not None and manifest.git_control == "model":
        # The model managed git: there is no agent6 branch to merge or diff.
        current = ""
        head = ""
        with contextlib.suppress(GitError):
            st = git_status(cwd)
            current = st.branch
            head = st.head_sha[:12]
        where = current or head or "the current checkout"
        reporter.out(
            f'\ngit was model-controlled ([git].control = "model"); its work is on {where}'
        )
        reporter.out("  inspect it with plain git (log/diff); sessions merge does not apply")
    elif result.completed and merged_into:
        # auto_merge landed it, and auto_prune may have deleted the branch.
        reporter.out(f"\nchanges merged into {merged_into}")
        reporter.out(f"  inspect:     agent6 sessions diff {layout.session_id}")
    elif result.completed and manifest is not None and (where := commits_ref(manifest, cwd)):
        # The run branch, or the chain ref a branchless run commits to.
        reporter.out(f"\nchanges are on {where}")
        reporter.out(f"  merge with:  agent6 sessions merge {layout.session_id}")
        reporter.out(f"  inspect:     agent6 sessions diff {layout.session_id}")
        # An operator who checked the run branch out should know how to leave it.
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(cwd).branch
        if run_branch and current == run_branch and base_branch and base_branch != run_branch:
            reporter.out(f"  you are on {run_branch}; return with: git switch {base_branch}")
    elif result.completed and manifest is not None and not manifest.policy.commit_per_step:
        reporter.out(
            "\nnothing was committed ([git].commit_per_step = false): the run's edits are in"
            " the working tree"
        )
    elif result.completed and (manifest is None or manifest.mode not in ("plan", "ask")):
        # A plan or an ask never commits: its deliverable is printed above.
        _print_no_commit_footer(
            result, layout=layout, cwd=cwd, run_branch=run_branch, reporter=reporter
        )
    elif not result.completed:
        reporter.out(f"\nresume with:  agent6 resume {layout.session_id}")


def _print_no_commit_footer(
    result: SessionResult,
    *,
    layout: SessionLayout,
    cwd: Path,
    run_branch: str,
    reporter: Reporter,
) -> None:
    """Print the footer for a run whose commit never landed.

    Stranded edits are a failure, a clean tree means the run recorded nothing, and a
    tree git cannot read gets an honest unknown.

    Args:
        result: The session's result.
        layout: The session's layout.
        cwd: The checkout.
        run_branch: The promised branch, or "".
        reporter: Receives the lines.
    """
    try:
        exclude = read_untracked_at_start(layout.session_dir)
        tree_clean: bool | None = git_status(cwd, exclude=exclude).is_clean
    except GitError as exc:
        tree_clean = None
        reporter.out(
            f"\ncould not check the working tree (git failed: {exc}); inspect it manually."
        )
    if tree_clean is not None and stranded_edits(result, layout, cwd):
        where = f" on {run_branch}, so the branch was never created" if run_branch else ""
        reporter.out(f"\nWARNING: the run finished with no commit{where}.")
        reporter.out(
            "  Edits are left uncommitted in the working tree (the commit failed; see the run log)."
        )
        reporter.out(f"  retry after fixing the cause:  agent6 resume {layout.session_id}")
    elif tree_clean is True:
        reporter.out("\nno changes were committed")


def _print_run_total_across_executions(layout: SessionLayout, *, reporter: Reporter) -> None:
    """Print the run's cumulative spend when earlier executions precede this one.

    Args:
        layout: The session's layout.
        reporter: Receives the line.
    """
    scan = scan_session_log(layout.session_dir / LOGS_NAME)
    if scan.executions > 1 and scan.cost_usd is not None:
        cost = format_usd(scan.cost_usd, partial=scan.usd_partial)
        reporter.cost(f"  RUN TOTAL (all {scan.executions} executions): {cost}")


def print_interrupt_end(
    *, layout: SessionLayout, cwd: Path, budget: BudgetTracker, reporter: Reporter
) -> None:
    """Print the cost so far and the resume and branch-return hints after an interrupt.

    Args:
        layout: The session's layout.
        cwd: The checkout.
        budget: The execution's tracker.
        reporter: Receives the lines.
    """
    reporter.out("")
    reporter.cost(budget.format_summary())
    _print_run_total_across_executions(layout, reporter=reporter)
    reporter.out(f"\nresume with:  agent6 resume {layout.session_id}")
    run_branch = ""
    base_branch = ""
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(layout.session_dir)
        run_branch = manifest.run_branch or ""
        base_branch = manifest.base_branch
    if run_branch:
        current = ""
        with contextlib.suppress(GitError):
            current = git_status(cwd).branch
        if current == run_branch and base_branch and base_branch != run_branch:
            reporter.out(f"  you are on {run_branch}; return with: git switch {base_branch}")


def finalize_auto_merge(
    cwd: Path,
    *,
    layout: SessionLayout,
    cfg: Config,
    reporter: Reporter,
    budget: BudgetTracker | None = None,
    events: EventSink | None = None,
) -> None:
    """Land the run branch on its base with `git.merge_strategy`, best-effort.

    Ref plumbing only: the checkout is never switched. On a conflict or error the
    run's refs stay intact and the note says how to merge by hand.

    Args:
        cwd: The checkout.
        layout: The session's layout.
        cfg: The run's config.
        reporter: Receives the notes.
        budget: The tracker a squash message's model call bills, when one runs.
        events: The sink the merge's events go to.
    """
    try:
        manifest = read_manifest(layout.session_dir)
    except ManifestError:
        return
    base_branch = manifest.base_branch
    # The visible branch, else the chain ref; an unborn ref has nothing to land.
    run_branch = manifest.run_branch or chain_ref_for(manifest.session_id)
    if not base_branch or chain_tip(cwd, run_branch) is None:
        return
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        trailer=render_commit_trailer(
            cfg.git.commit.trailer,
            models=worker_models(tail_events(layout.session_dir / LOGS_NAME, follow=False))
            or ((manifest.models.driver.model,) if manifest.models.driver else ()),
        ),
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        reporter.note(
            f"auto_merge skipped: {exc}",
        )
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
        budget=budget,
        events=events,
        warn=reporter.note,
    )
    if outcome.status == "merged":
        reporter.note(
            f"auto_merged {run_branch} into {base_branch} "
            f"({cfg.git.merge_strategy}) -> {outcome.merged_sha[:12]}"
        )
        if kept := left_behind_line(base_branch, outcome):
            reporter.note(kept)
    elif outcome.status == "noop":
        reporter.note(f"{noop_merge_line(run_branch, base_branch, outcome)}.")
    elif outcome.status == "conflict":
        reporter.note(
            f"auto_merge into {base_branch} hit conflicts "
            f"({', '.join(outcome.conflicts)}); nothing was moved and {run_branch} is "
            f"intact. Merge by hand:\n    git merge {run_branch}"
        )
    else:
        reporter.note(
            f"auto_merge failed: {outcome.error}",
        )
    if outcome.stamp_error:
        reporter.note(
            f"merge record could not be written: {outcome.stamp_error};"
            " `sessions prune` will call this branch unmerged"
        )
    # auto_prune is a branch verb: the chain ref stays as the run's record.
    landed = outcome.status == "merged" or outcome.recorded
    if landed and cfg.git.auto_prune and manifest.run_branch:
        if delete_branch_if_merged(cwd, run_branch):
            reporter.note(f"auto_pruned {run_branch}")
        else:
            reporter.note(
                f"auto_prune kept {run_branch} (squash-merged, unreachable; "
                f"remove with: git branch -D {run_branch})"
            )


def _stash_apply_cmd(cwd: Path, sha: str, base_branch: str) -> str:
    """Word the manual-recovery command for a stash, by sha since a position rots.

    Args:
        cwd: The checkout.
        sha: The stash commit.
        base_branch: The branch the stash belongs on.

    Returns:
        The command, prefixed with a checkout of the base only when the operator is
        elsewhere.
    """
    apply = f"git stash apply {sha}"
    current = ""
    with contextlib.suppress(GitError):
        current = git_status(cwd).branch
    return f"git checkout {base_branch} && {apply}" if current != base_branch else apply


def stash_recovery_hint(cwd: Path, *, session_id: str, base_branch: str) -> str | None:
    """Word how to restore the run's pre-run auto-stash by hand.

    Args:
        cwd: The checkout.
        session_id: The run whose stash it is.
        base_branch: The branch the stash belongs on.

    Returns:
        The command, or None when the run pushed no stash.
    """
    entry = find_stash(cwd, auto_stash_message(session_id))
    if entry is None:
        return None
    return _stash_apply_cmd(cwd, entry.sha, base_branch)


def finalize_auto_stash(
    cwd: Path,
    *,
    base_branch: str,
    run_branch: str | None,
    auto_pop: bool,
    session_id: str,
    exclude: Collection[str] = (),
    reporter: Reporter,
) -> None:
    """Restore the pre-run auto-stash when that is safe, else say how to.

    The stash is found by the run-id message it was pushed with and restored by its
    sha, never by position: a stash pushed during the run sits at stash@{0}.

    Args:
        cwd: The checkout.
        base_branch: The branch the stash belongs on.
        run_branch: The run's branch, or None.
        auto_pop: Whether to restore it; off prints the command instead.
        session_id: The run whose stash it is.
        exclude: The operator's untracked files, which do not make the tree unclean.
        reporter: Receives the notes.
    """
    message = auto_stash_message(session_id)
    entry = find_stash(cwd, message)
    if entry is None:
        reporter.note("pre-run auto-stash not found (already restored?); nothing to pop")
        return
    # apply by sha keeps the stash; the operator drops it once confirmed.
    apply = f"git stash apply {entry.sha}"
    recover = _stash_apply_cmd(cwd, entry.sha, base_branch)
    if not auto_pop:
        reporter.note(f"pre-run changes are stashed; restore them with: {recover}")
        return
    try:
        st = git_status(cwd, exclude=exclude)
    except GitError:
        st = None
    if st is None or not st.is_clean:
        reporter.note(f"pre-run changes left stashed (worktree not clean); restore with: {recover}")
        return
    if run_branch and st.branch == run_branch:
        if not branch_exists(cwd, base_branch):
            reporter.note(
                f"base branch {base_branch} no longer exists; pre-run changes left "
                f"stashed (recover with: {apply})"
            )
            return
        try:
            create_branch(cwd, base_branch)  # checks out the existing base branch
        except GitError as exc:
            reporter.note(
                f"could not switch to {base_branch} to restore the stash ({exc}); "
                f"restore with: {recover}"
            )
            return
    try:
        restored = restore_stash(cwd, entry)
    except GitError as exc:
        # The apply landed; putting back a concurrent stash the drop displaced failed.
        reporter.note(f"restored your pre-run changes onto {base_branch}, but {exc}")
        return
    if restored:
        reporter.note(
            f"restored your pre-run changes onto {base_branch}",
        )
    else:
        reporter.note(
            "restoring your pre-run changes hit a conflict; resolve the markers"
            f" (your stash is preserved; re-apply with: git stash apply {entry.sha})"
        )


def hook_env(**agent6_vars: str) -> dict[str, str]:
    """Build the environment for an operator notify hook.

    Args:
        **agent6_vars: The `AGENT6_*` facts to add.

    Returns:
        The curated base environment plus those facts.
    """
    return curated_env(extra=agent6_vars)


def run_notify_hook(
    argv: Sequence[str],
    env: dict[str, str],
    *,
    timeout_s: float,
    label: str,
    note: Callable[[str], None],
) -> None:
    """Run one operator notify hook on the host; a failure is noted, never fatal.

    The argv is operator-controlled, never model output, so it runs outside the
    jail. Its stdout is discarded: under `agent6 acp` the parent's stdout is the
    protocol stream.

    Args:
        argv: The hook command.
        env: Its environment.
        timeout_s: How long to wait for it.
        label: The config key named in a failure note.
        note: Receives the failure note.
    """
    try:
        res = subprocess.run(
            list(argv),
            stdout=subprocess.DEVNULL,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        note(f"{label} failed: {exc}")
        return
    if res.returncode != 0:
        note(f"{label} exited {res.returncode}")


def fire_notify_hook(
    notify: NotifyConfig,
    *,
    session_id: str,
    session_dir: Path,
    ok: bool,
    reason: str,
    verified: str,
    reporter: Reporter,
) -> None:
    """Run the `[notify].on_complete` hook, when one is configured.

    Args:
        notify: The notify config.
        session_id: The session's id.
        session_dir: The session's directory.
        ok: Whether the agent stopped deliberately.
        reason: The end reason.
        verified: What the gate said: passed, failed or not_applicable.
        reporter: Receives a failure note.
    """
    if not notify.on_complete:
        return
    env = hook_env(
        AGENT6_SESSION_ID=session_id,
        # A hook that wants "green" reads VERIFIED: OK is true over a red verify too.
        AGENT6_SESSION_OK="1" if ok else "0",
        AGENT6_SESSION_VERIFIED=verified,
        AGENT6_SESSION_REASON=reason,
        AGENT6_SESSION_DIR=str(session_dir),
    )
    run_notify_hook(
        notify.on_complete,
        env,
        timeout_s=notify.timeout_s,
        label="notify.on_complete",
        note=reporter.note,
    )
