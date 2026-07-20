# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs diff/commits/merge/prune/compare` commands (the run-branch
lifecycle)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.app.merge import execute_merge
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import load_effective
from agent6.git_ops import (
    DIFF_SHOW_SAFETY_FLAGS,
    CommitIdentity,
    GitError,
    branch_exists,
    branch_tip_sha,
    delete_branch_if_merged,
    diff_range,
    force_delete_squash_merged_branch,
    git_hardening_flags,
    is_ancestor,
    is_git_repo,
    list_run_branches,
    list_run_commits,
    verify_git_identity,
)
from agent6.git_ops import status as git_status
from agent6.runs.id import RunIdError, resolve_run_id
from agent6.runs.ipc import request_stop, worker_is_alive
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import ManifestError, RunManifest, read_manifest
from agent6.ui.cli._common import (
    _runs_dir,
    _state_dir,
    load_config_or_exit,
    resolve_or_newest_layout,
    sgr,
)
from agent6.ui.cli._compare import manifest_task, print_ranked_candidates, rank, verify_ok
from agent6.viewmodel import is_run_husk, is_winner, newest_run_dir, summarize_run_dir, task_snippet
from agent6.viewmodel.format import WINNER_GLYPH, format_cost, status_label
from agent6.workflows.judge import CandidateBrief

# ANSI styles for the shared status words (viewmodel.status_word), tty only:
# a listing where a provider_error death reads as plain text is how dead runs
# went unnoticed.
_STATUS_SGR = {
    "starting": "36",  # launching (pre-loop): in progress, lighter than running
    "running": "1;36",
    "waiting": "33",  # blocked on the operator (approval / question)
    "stale": "2",
    "passed": "32",
    "answered": "32",  # an ask that answered is terminal success
    "planned": "35",  # informational magenta (mauve on the TUI/web); not green, not red
    "stopped": "33",
    "failed": "1;31",
}


def _styled_status(status: str, reason: str, *, color: bool) -> tuple[str, str]:
    """(possibly-colored label, plain label) -- the plain form drives width math."""
    label = status_label(status, reason)
    sgr = _STATUS_SGR.get(status)
    if color and sgr:
        return f"\x1b[{sgr}m{label}\x1b[0m", label
    return label, label


def _cmd_list() -> int:
    """List this repo's runs (runs/ + asks/), newest first: updated (last-activity
    time), status (with the failure reason), mode, cost, id, task. The listing twin
    of the TUI/web hubs, built on the same shared summary."""
    import time  # noqa: PLC0415

    cwd = Path.cwd()
    dirs: list[Path] = []
    for sub in ("runs", "asks"):
        d = _state_dir(cwd) / sub
        if d.is_dir():
            dirs.extend(p for p in d.iterdir() if p.is_dir() and not is_run_husk(p))
    if not dirs:
        print('no runs yet. Start one with `agent6 run "<task>"`.')
        return 0
    winners = {d.name for d in dirs if is_winner(d)}  # fan-out compare winners
    summaries = sorted((summarize_run_dir(d) for d in dirs), key=lambda s: s.mtime, reverse=True)
    color = sys.stdout.isatty()
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for s in summaries:
        when = time.strftime("%m-%d %H:%M", time.localtime(s.mtime))
        styled, plain = _styled_status(s.status, s.reason, color=color)
        cost = "" if s.cost_usd <= 0 else format_cost(s.cost_usd)
        # A winner lane gets a ★ suffix on its id (folded into the width math, so
        # the columns stay aligned); a non-disruptive marker the hub/home mirror.
        run_id = f"{s.run_id} {WINNER_GLYPH}" if s.run_id in winners else s.run_id
        rows.append((when, styled, plain, s.mode, cost, run_id, task_snippet(s.task, max_chars=60)))
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    id_w = max(2, *(len(r[5]) for r in rows))
    print(
        f"{'updated':<11}  {'status':<{status_w}}  {'mode':<4}  {'cost':<8}  {'id':<{id_w}}  task"
    )
    for when, styled, plain, mode, cost, run_id, task in rows:
        pad = " " * (status_w - len(plain))
        print(f"{when:<11}  {styled}{pad}  {mode:<4}  {cost:<8}  {run_id:<{id_w}}  {task}")
    return 0


def _cmd_diff(*, run_id: str, stat: bool, paths: tuple[str, ...]) -> int:
    """Print the git diff a run produced (manifest.base_sha -> branch HEAD).

    Resolves the run id (or unique prefix; empty string means most-recent),
    reads ``manifest.json`` for ``base_sha`` and ``run_branch``, then shells
    out to ``git diff`` with operator-controlled argv (no LLM input). The call
    streams to the terminal, so it cannot go through git_ops._run; it carries
    the same host-RCE hardening (``git_hardening_flags``: a poisoned
    ``.git/config`` ``diff.external`` / ``diff.*.textconv`` / ``core.fsmonitor``
    / repo hook must not execute on the host) plus ``DIFF_SHOW_SAFETY_FLAGS``,
    which force the builtin diff renderer (git >= 2.53 executes even an EMPTY
    ``diff.external`` override, so the ``-c`` flags alone would kill the printed
    patch) and disable the per-file textconv driver the ``-c`` flags do not reach.
    """
    cwd = Path.cwd()
    res = _resolve_run_manifest(
        cwd,
        run_id,
        recent_note="diffing most recent run",
        missing_hint=" (predates manifest support, or was killed before setup)",
    )
    if isinstance(res, int):
        return res
    _layout, manifest = res

    base_sha = manifest.base_sha
    run_branch = manifest.run_branch
    if not base_sha:
        print("ERROR: manifest has no base_sha; nothing to diff against", file=sys.stderr)
        return 2
    if run_branch:
        pruned = _pruned_branch_note(cwd, manifest, run_branch)
        if pruned is not None:  # branch gone (pruned): say where the work went
            print(pruned)
            return 0

    head_ref = run_branch if run_branch else "HEAD"
    # The logical command; printed without the -c hardening overrides (the
    # same convention as git_ops error messages), executed with them.
    args: list[str] = ["diff", *DIFF_SHOW_SAFETY_FLAGS]
    if stat:
        args.append("--stat")
    args.extend([f"{base_sha}..{head_ref}"])
    if paths:
        args.append("--")
        args.extend(paths)
    print(
        f"[agent6] git {' '.join(args)}  (base_branch={manifest.base_branch!r})",
        file=sys.stderr,
    )
    # A zero-commit run would print the headers and then nothing; probe first
    # (`--quiet` = exit 0 when identical) and say so. Probe errors (rc > 1,
    # e.g. a missing sha) fall through so the real diff surfaces git's message.
    probe_args = ["diff", *DIFF_SHOW_SAFETY_FLAGS, "--quiet", f"{base_sha}..{head_ref}"]
    if paths:
        probe_args.extend(["--", *paths])
    probe = subprocess.run(
        ["git", *git_hardening_flags(), *probe_args], cwd=cwd, check=False, capture_output=True
    )
    if probe.returncode == 0:
        # No COMMITTED changes yet. A run commits only after a verify pass, so a
        # live run mid-work has its edits uncommitted on the worktree and this
        # reads as "the agent did nothing". If the run branch is the current
        # checkout and its worktree is dirty, say so instead of a bare silence.
        dirty = _dirty_worktree_note(cwd, run_branch)
        print(dirty if dirty else "(no changes)")
        return 0
    proc = subprocess.run(["git", *git_hardening_flags(), *args], cwd=cwd, check=False)
    return proc.returncode


def _dirty_worktree_note(cwd: Path, run_branch: object) -> str:
    """A note when the diffed run's branch is the current checkout and its
    worktree has uncommitted work (a run commits only after each verify pass),
    else "". Only speaks when the dirty files are unambiguously THIS run's:
    the current branch must equal run_branch. Best-effort; git errors -> "" ."""
    if not run_branch:
        return ""
    # Same host-RCE hardening as the diff/probe above: `git status` refreshes the
    # index and would fire a poisoned `.git/config` core.fsmonitor on the host.
    try:
        current = subprocess.run(
            ["git", *git_hardening_flags(), "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if current.returncode != 0 or current.stdout.strip() != str(run_branch):
            return ""
        status = subprocess.run(
            ["git", *git_hardening_flags(), "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    n = len([ln for ln in status.stdout.splitlines() if ln.strip()])
    if n == 0:
        return ""
    files = "file" if n == 1 else "files"
    return (
        f"(no committed changes yet; {n} {files} modified in the working tree: "
        "a run commits after each verify pass)"
    )


def _resolve_run_manifest(
    cwd: Path,
    run_id: str,
    *,
    recent_note: str = "using most recent run",
    missing_hint: str = "",
) -> tuple[RunLayout, RunManifest] | int:
    """Resolve a run id (or '' for most-recent) to its (layout, manifest), or an exit
    code on error. Shared by `runs diff`/`merge`/`commits`; the two note strings vary
    per caller."""
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return 2
    target_id = run_id
    if not target_id:
        latest = newest_run_dir([runs_dir])
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}", file=sys.stderr)
            return 2
        target_id = latest.name
        print(f"[agent6] {recent_note}: {target_id}", file=sys.stderr)
    else:
        try:
            target_id = resolve_run_id(runs_dir, target_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    layout = RunLayout(state_dir=_state_dir(cwd), run_id=target_id)
    if not layout.manifest_path.is_file():
        print(f"ERROR: run {target_id} has no manifest.json{missing_hint}", file=sys.stderr)
        return 2
    try:
        manifest = read_manifest(layout.run_dir)
    except ManifestError as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2
    return layout, manifest


def _cmd_stop(*, run_id: str) -> int:
    """Ask a running detached run to stop cleanly after its current step.

    Drops the same 'stop after this step' marker the TUI/web Stop button uses:
    the run finishes the in-flight step (its tool results and auto-commit land),
    then ends and is resumable. For a running run only; a finished one is a no-op
    with a note."""
    cwd = Path.cwd()
    try:
        layout = resolve_or_newest_layout(cwd, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if layout is None:
        print("ERROR: no runs to stop.", file=sys.stderr)
        return 2
    run_dir = layout.run_dir
    rid = run_dir.name
    if not worker_is_alive(run_dir):
        print(f"[agent6] {rid} is not running; nothing to stop.", file=sys.stderr)
        return 0
    request_stop(run_dir)
    print(f"[agent6] requested stop for {rid}; it ends after the current step.")
    print(f"  resume with:  agent6 resume {rid}")
    return 0


def _pruned_branch_note(cwd: Path, manifest: RunManifest, run_branch: str) -> str | None:
    """A friendly message when a run's branch no longer exists (it was pruned),
    or None if the branch is still present. Uses the manifest's recorded merge so
    diff/commits say where the work went instead of leaking a raw git fatal."""
    if branch_exists(cwd, run_branch):
        return None
    merged_into = manifest.merged.into if manifest.merged else ""
    merged_sha = manifest.merged.sha if manifest.merged else ""
    if merged_into:
        note = f"[agent6] run branch {run_branch} was pruned; squash-merged into {merged_into}"
        if merged_sha and set(merged_sha) != {"0"}:
            note += f" as {merged_sha[:12]}\n  see: git show {merged_sha[:12]}"
        return note
    return f"[agent6] run branch {run_branch} no longer exists (deleted, and no merge recorded)."


def _cmd_commits(*, run_id: str) -> int:
    """List the per-step commits on a run's branch (manifest.base_sha -> run branch)."""
    cwd = Path.cwd()
    res = _resolve_run_manifest(cwd, run_id)
    if isinstance(res, int):
        return res
    _layout, manifest = res
    base_sha = manifest.base_sha
    run_branch = manifest.run_branch
    if not run_branch or not base_sha:
        print(
            "ERROR: this run has no branch/base recorded (branch_per_run was off?).",
            file=sys.stderr,
        )
        return 2
    pruned = _pruned_branch_note(cwd, manifest, run_branch)
    if pruned is not None:
        print(pruned)
        return 0
    rows = list_run_commits(cwd, base_sha, run_branch)
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
    manifest: RunManifest
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
    # A live run keeps the shared checkout on its run branch and its tree is
    # clean for the whole duration of every provider call, so every guard below
    # passes mid-run -- and execute_merge would then switch the checkout to the
    # base branch under the worker, whose next auto-commit lands mid-run WIP
    # directly on it. Liveness is the gate, matching stop/resume/compact. The
    # run's own end-of-run finalize_auto_merge is unaffected (it calls
    # execute_merge directly, not this planner).
    if worker_is_alive(layout.run_dir):
        print(
            f"REFUSING: run {run_id!r} is still live; merging now would switch the"
            " shared checkout out from under the worker and its next auto-commit"
            " would land on the base branch. Stop it first:\n"
            f"    agent6 runs stop {run_id}",
            file=sys.stderr,
        )
        return 2
    run_branch = manifest.run_branch
    if not run_branch:
        print(
            "ERROR: this run has no branch to merge (branch_per_run was off, so the "
            "work already landed on your current branch).",
            file=sys.stderr,
        )
        return 2
    target = into or manifest.base_branch
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
        base_sha=manifest.base_sha,
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
    if plan.base_sha and not list_run_commits(cwd, plan.base_sha, plan.run_branch):
        # A success line here would be indistinguishable from a real merge.
        print(f"[agent6] nothing to merge: run branch {plan.run_branch} has no commits.")
        return 0
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
    "" if there is no (readable) manifest or it was never recorded as merged.

    Frozen semantics: `runs prune --delete-squashed` force-deletes a branch ONLY
    when this returns a base name (a manifest-confirmed merge with a recorded sha).
    An unreadable/corrupt/unmerged manifest returns "" -> the branch is KEPT, never
    force-deleted (fail-safe). The model's leniency preserves that: the legacy flat
    merged_into/merged_sha keys fold into `merged`, and any parse failure raises
    ManifestError -> ""."""
    run_id = branch.removeprefix("agent6/")
    try:
        manifest = read_manifest(RunLayout(state_dir=state_dir, run_id=run_id).run_dir)
    except ManifestError:
        return ""
    return manifest.merged.into if (manifest.merged and manifest.merged.sha) else ""


def _cmd_prune(*, delete_squashed: bool = False) -> int:
    """Delete agent6/* run branches that `git branch -d` can safely remove
    (reachable-merged into HEAD, i.e. merge/ff strategies). Report squash-merged
    ones and unmerged ones (review first).

    With ``--delete-squashed`` also force-delete branches the manifest confirms
    were squash-merged into an existing base -- their content is safe in that
    base commit, and each deletion prints the exact command to undelete it (the
    commit survives in the reflog until GC). Unmerged branches are never
    force-deleted."""
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
    deleted = squashed_deleted = merged_kept = unmerged_kept = 0
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
        reachable = branch_exists(cwd, merged_into) and is_ancestor(cwd, br, merged_into)
        if reachable:
            # Reachable-merged into its base, so `git branch -d` only refused because
            # HEAD is not the base; deleting it cleanly needs to run from the base.
            merged_kept += 1
            print(
                f"[agent6] kept {br} (merged into {merged_into} but not reachable from "
                f"{current!r}; re-run prune on {merged_into}, or: git branch -D {br})"
            )
            continue
        # Squash-merged into its base: content is in the base commit but the branch
        # is unreachable, so `git branch -d` refuses it.
        if delete_squashed and branch_exists(cwd, merged_into):
            sha = branch_tip_sha(cwd, br)
            if sha is not None and force_delete_squash_merged_branch(cwd, br):
                squashed_deleted += 1
                print(f"[agent6] deleted {br} (squash-merged into {merged_into})")
                # A faded undelete hint: the commit survives in the reflog until GC.
                print(sgr(f"          undelete: git branch {br} {sha[:12]}", "2"))
                continue
        merged_kept += 1
        print(
            f"[agent6] kept {br} (squash-merged into {merged_into}, unreachable; "
            f"remove with: runs prune --delete-squashed, or: git branch -D {br})"
        )
    kept = merged_kept + unmerged_kept
    total_deleted = deleted + squashed_deleted
    squashed_note = f" ({squashed_deleted} squash-merged)" if squashed_deleted else ""
    print(
        f"\n[agent6] deleted {total_deleted}{squashed_note}; kept {kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged)",
    )
    return 0


def _candidate_diff(cwd: Path, base_sha: str, run_branch: str) -> str:
    """The diff a run's branch introduced (base_sha..run_branch), read-only,
    without checking out the branch (unlike `_cmd_diff`, several candidates are
    compared in one call, and only one can be the current checkout). "" if the
    branch is gone -- never blocks the comparison."""
    if not base_sha or not run_branch or not branch_exists(cwd, run_branch):
        return ""
    return diff_range(cwd, base_sha, run_branch)


def _cmd_compare(*, run_ids: tuple[str, ...]) -> int:
    """Advisory ranked comparison across >=2 already-run candidates: the same
    ranked report `--parallel`'s auto-compare prints (judge via the reviewer
    model when configured, else the mechanical verify+cost ranking) -- for
    runs picked by hand, not necessarily from the same fan-out or even the
    same task (each candidate's own manifest `user_task` is its task).
    Read-only: no merges, no writes."""
    if len(run_ids) < 2:
        print(
            f"ERROR: runs compare needs at least 2 run ids (got {len(run_ids)}).",
            file=sys.stderr,
        )
        return 2
    cwd = Path.cwd()
    resolved: list[tuple[RunLayout, RunManifest]] = []
    seen: set[str] = set()
    for query in run_ids:
        res = _resolve_run_manifest(cwd, query)
        if isinstance(res, int):
            return res
        layout, manifest = res
        if layout.run_id in seen:
            print(f"ERROR: run {layout.run_id!r} was given more than once.", file=sys.stderr)
            return 2
        seen.add(layout.run_id)
        resolved.append((layout, manifest))
    eff = load_config_or_exit(cwd, None)
    if isinstance(eff, int):
        return eff
    cfg = eff.config

    candidates: list[CandidateBrief] = []
    for layout, manifest in resolved:
        base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""
        summary = summarize_run_dir(layout.run_dir)
        candidates.append(
            CandidateBrief(
                run_id=layout.run_id,
                task=manifest_task(layout.run_dir, fallback=layout.run_id),
                diff=_candidate_diff(cwd, base_sha, run_branch),
                verify_ok=verify_ok(summary.status),
                cost_usd=summary.cost_usd,
            )
        )

    reviewer = cfg.models.resolve("reviewer")
    # `runs compare` is advisory and stateless: it ranks + prints but never stamps
    # a manifest (only the fan-out's auto-compare does), so `ranked_by` is unused.
    outcome = rank(cfg, candidates, transcript_dir=_state_dir(cwd) / "compare")
    print(f"[agent6] comparing {len(candidates)} runs:")
    print_ranked_candidates(candidates, outcome)
    # A fresh judgment can contradict the fan-out's recorded verdict (the star in
    # listings comes from the auto-compare stamp, which this command never
    # rewrites); when re-judging one fan-out's own lanes, disclose the clash
    # rather than let the two surfaces silently disagree.
    groups = {manifest.parallel_id for _, manifest in resolved}
    if outcome.ranking and len(groups) == 1 and None not in groups:
        stamped = next(
            (
                layout.run_id
                for layout, manifest in resolved
                if manifest.compare is not None and manifest.compare.winner
            ),
            None,
        )
        if stamped is not None and stamped != outcome.ranking[0]:
            print(
                f"\nnote: the recorded fan-out verdict picked {stamped}"
                f" (the {WINNER_GLYPH} in listings); this fresh ranking is advisory"
                " and nothing was re-stamped."
            )
    if reviewer is None:
        print(
            "\n(no reviewer model configured; ranked mechanically: verify-pass first, then"
            " lower cost)"
        )
    return 0
