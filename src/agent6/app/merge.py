# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Land a run's branch, for `sessions merge` and `git.auto_merge`.

Landing is ref plumbing (`git_ops.plumb_merge`): no checkout and no clean-tree requirement,
so the worktree carrying the run's own work is never an obstacle. Both callers share the
strategy dispatch and the manifest record.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import pathlib
from collections.abc import Callable
from typing import Literal

from agent6 import budget as agent6_budget
from agent6 import commit_message, event_log, git_ops, secrets
from agent6.app import _setup, providers, stamps
from agent6.config import Config, ConfigError
from agent6.providers import ProviderError, TranscriptSink, call_for_text
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest


@dataclasses.dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Record what `execute_merge` did.

    A noop is a branch whose content the target already holds: the target stays where it was
    and the run is merged all the same. A noop over the tip already recorded leaves the record
    of the merge that did happen alone.

    Attributes:
        status: merged, noop, conflict or error.
        merged_sha: The target's tip after the merge; on a noop, its own tip.
        conflicts: The conflicting paths.
        error: Why the merge did not happen.
        stamp_error: Why the manifest stamp did not land; the merge happened either way.
        recorded: This call stamped a noop as merged, with NO_MERGE_COMMIT for the sha.
        left_behind: Paths the checkout kept its own version of (`left_behind_line`).
    """

    status: Literal["merged", "noop", "conflict", "error"]
    merged_sha: str = ""
    conflicts: tuple[str, ...] = ()
    error: str = ""
    stamp_error: str = ""
    recorded: bool = False
    left_behind: tuple[str, ...] = ()


def record_merge_in_manifest(
    layout: sessions_layout.SessionLayout,
    *,
    merged_into: str,
    merged_sha: str,
    merged_tip: str = "",
    into_tip: str = "",
) -> str:
    """Stamp the merge into the run manifest, so `prune` can tell a merged branch.

    A missing or corrupt manifest must not fail a merge that already happened.

    Args:
        layout: The run's directory layout.
        merged_into: The target branch.
        merged_sha: The merge commit, or NO_MERGE_COMMIT.
        merged_tip: The run-branch tip merged; `prune --delete-squashed` deletes only a branch
            still pointing there.
        into_tip: The target's tip a merge that added nothing saw.

    Returns:
        "" when the stamp landed, else why it did not.
    """
    try:
        m = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError as exc:
        return str(exc)
    stamped = m.model_copy(
        update={
            "merged": sessions_manifest.MergeStamp(
                into=merged_into,
                sha=merged_sha,
                tip=merged_tip,
                into_tip=into_tip,
                ts=_dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
            )
        }
    )
    # ManifestError: a manifest newer than this binary is left alone rather than downgraded.
    try:
        stamps.write_manifest(layout.manifest_path, stamped)
    except (OSError, sessions_manifest.ManifestError) as exc:
        return str(exc)
    return ""


def dispatch_merge(
    cwd: pathlib.Path,
    strategy: str,
    target: str,
    run_branch: str,
    base_sha: str,
    manifest: sessions_manifest.SessionManifest,
    message: str | None,
    cfg: Config,
    identity: git_ops.CommitIdentity,
    *,
    transcript_dir: pathlib.Path | None = None,
    budget: agent6_budget.BudgetTracker | None = None,
    events: event_log.EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
    merge_base: str | None = None,
) -> git_ops.MergeResult:
    """Run the strategy on the target through `plumb_merge`.

    A squash builds its message per `[git.commit.squash].message`; an operator message
    overrides any style.

    Args:
        cwd: The repository.
        strategy: ff, merge or squash.
        target: The branch to land on.
        run_branch: The run's branch or chain ref.
        base_sha: The commit the run started from.
        manifest: The run's manifest.
        message: The operator's message, else the style's.
        cfg: The resolved config.
        identity: The committer.
        transcript_dir: Where a model-written message's call is transcribed.
        budget: The run's tracker, when the merge runs inside one.
        events: The run's event sink, when the merge runs inside one.
        warn: Where a degraded message is reported.
        merge_base: The base to merge from, when git's own does not serve.

    Returns:
        The plumbing's result.
    """
    if strategy == "squash" and message is None:
        message = _squash_message(
            cwd,
            cfg,
            manifest,
            base_sha=base_sha,
            run_branch=run_branch,
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
            warn=warn,
        )
    if message is None and strategy == "merge":
        message = f"Merge {run_branch}"
    return git_ops.plumb_merge(
        cwd,
        target,
        run_branch,
        strategy=strategy,
        message=message,
        identity=identity,
        merge_base=merge_base,
    )


def landed_base(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    target: str,
    tip: str,
) -> str | None:
    """Return the merge base for a run the target already holds as a squash, else None.

    Git cannot relate a squash to the chain it came from, so its own base reads every earlier
    commit as new work. The base is the merged tip when the chain continues past it, else the
    point a fork left its ancestor's chain.

    Args:
        cwd: The repository.
        layout: The run's directory layout.
        manifest: The run's manifest; its stamp, or an ancestor's, names the merged tip.
        target: The branch to land on.
        tip: The run's chain tip.

    Returns:
        The base, or None where git's own serves.
    """
    node, fork_point = manifest, ""
    for _ in range(64):  # a lineage deeper than this is not a fork chain
        stamp = node.merged
        if stamp is not None and stamp.into == target and stamp.tip:
            if git_ops.is_ancestor(cwd, stamp.tip, target):
                break  # a real merge: git relates the two histories itself
            if git_ops.is_ancestor(cwd, stamp.tip, tip):
                return stamp.tip
            if fork_point and git_ops.is_ancestor(cwd, fork_point, stamp.tip):
                return fork_point
            break
        fork_point = node.forked_from_sha
        if not node.parent_session_id:
            break
        try:
            node = sessions_manifest.read_manifest(
                layout.session_dir.parent / node.parent_session_id
            )
        except (sessions_manifest.ManifestError, OSError):
            break
    return None


def _squash_message(
    cwd: pathlib.Path,
    cfg: Config,
    manifest: sessions_manifest.SessionManifest,
    *,
    base_sha: str,
    run_branch: str,
    transcript_dir: pathlib.Path | None,
    budget: agent6_budget.BudgetTracker | None,
    events: event_log.EventSink | None,
    warn: Callable[[str], None],
) -> str | None:
    """Return the squash message per `[git.commit.squash].message`; None lets git combine."""
    style = cfg.git.commit.squash.message
    rows = git_ops.list_run_commits(cwd, base_sha, run_branch)
    if style == "combine":
        # Git's own SQUASH_MSG shape; the plumbing merge never runs `merge --squash`.
        parts = ["Squashed commit of the following:\n"]
        parts += [
            f"commit {r.sha}\n\n    {r.message.rstrip().replace(chr(10), chr(10) + '    ')}"
            for r in rows
        ]
        return "\n".join(parts) if rows else None
    base_msg = commit_message.condense_commit_message(
        rows, subject=manifest.user_task or "agent6 run"
    )
    if style == "conventional":
        subject = commit_message.conventional_commit_subject(
            git_ops.range_name_status(cwd, base_sha, run_branch), summary=base_msg.splitlines()[0]
        )
        return "\n".join([subject, *base_msg.splitlines()[1:]])
    if style == "model":
        msg = _model_squash_message(
            cwd,
            cfg,
            rows,
            base_sha=base_sha,
            run_branch=run_branch,
            task=manifest.user_task or "agent6 run",
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
            warn=warn,
        )
        if msg:
            return msg
    return base_msg


def _model_squash_message(
    cwd: pathlib.Path,
    cfg: Config,
    rows: tuple[commit_message.CommitRow, ...],
    *,
    base_sha: str,
    run_branch: str,
    task: str,
    transcript_dir: pathlib.Path | None,
    budget: agent6_budget.BudgetTracker | None,
    events: event_log.EventSink | None,
    warn: Callable[[str], None] = lambda _m: None,
) -> str | None:
    """Write the squash message with one provider call from git facts only.

    Args:
        cwd: The repository.
        cfg: The resolved config.
        rows: The run's commits.
        base_sha: The commit the run started from.
        run_branch: The run's branch or chain ref.
        task: The run's task.
        transcript_dir: Where the call is transcribed; None skips the call.
        budget: The run's tracker when auto_merge runs inside a run, so the call spends the
            run's remainder; None takes a fresh per-invocation cap.
        events: The run's event sink; with it the call's spend reaches the log.
        warn: Where a failure is reported.

    Returns:
        The message, or None on any failure (the caller degrades to the agent6 style).
    """
    if transcript_dir is None:
        return None
    try:
        tracker = budget if budget is not None else _setup.budget_tracker(cfg)
        provider = providers.build_role_provider(
            cfg,
            "worker",
            transcript_sink=TranscriptSink(transcript_dir),
            budget=tracker,
        )
        if events is not None:
            rm = cfg.models.resolve("worker")
            provider = providers.InstrumentedProvider(
                inner=provider,
                role="squash",
                model=rm.model if rm is not None else "",
                provider_name=rm.provider if rm is not None else "",
                events=events,
                budget=tracker,
            )
        steps = "\n".join(f"- {r.subject}" for r in rows[:100])
        files = "\n".join(
            f"{s}\t{p}" for s, p in git_ops.range_name_status(cwd, base_sha, run_branch)[:200]
        )
        msg = call_for_text(
            provider,
            system=(
                "Write a git commit message for a squashed branch: one"
                " imperative subject line under 72 characters, a blank line,"
                " then a short body. Use only the facts given. Output the"
                " message text only."
            ),
            user=(
                f"Task: {task}\nPer-step subjects:\n{steps}\nChanged files (status\tpath):\n{files}"
            ),
            max_tokens=500,
        )
    except (
        agent6_budget.BudgetExceededError,
        ConfigError,
        git_ops.GitError,
        OSError,
        ProviderError,
        secrets.SecretsError,
    ) as exc:
        # auto_merge runs the draft in a finished run's teardown, which must not crash on it.
        warn(f"model squash message failed ({exc}); using the agent6 style")
        return None
    if not msg:
        warn("model squash message failed (the model returned nothing); using the agent6 style")
    return msg or None


NO_BASE_SHA = "the manifest records no base_sha; nothing to merge from"


def execute_merge(
    cwd: pathlib.Path,
    *,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    run_branch: str,
    target: str,
    base_sha: str,
    strategy: str,
    message: str | None,
    cfg: Config,
    identity: git_ops.CommitIdentity,
    budget: agent6_budget.BudgetTracker | None = None,
    events: event_log.EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
) -> MergeOutcome:
    """Land the run's branch on the target and record the merge.

    Ref plumbing only: the checkout is never switched and the worktree is never required
    clean. The caller validates first; this mutates.

    Args:
        cwd: The repository.
        layout: The run's directory layout.
        manifest: The run's manifest.
        run_branch: The run's branch or chain ref.
        target: The branch to land on.
        base_sha: The commit the run started from.
        strategy: ff, merge or squash.
        message: The operator's message, else the style's.
        cfg: The resolved config.
        identity: The committer.
        budget: The run's tracker, when the merge runs inside one.
        events: The run's event sink, when the merge runs inside one.
        warn: Where a degraded squash message is reported.

    Returns:
        What the merge did.
    """
    _setup.apply_git_ops_policy(cfg)
    # Without base_sha `git log ..<branch>` counts from HEAD: a wrong list with a clean exit.
    refusal = (
        NO_BASE_SHA
        if not base_sha
        else f"target branch {target!r} does not exist"
        if not git_ops.branch_exists(cwd, target)
        else None
    )
    if refusal is not None:
        return MergeOutcome("error", error=refusal)
    if (
        strategy == "ff"
        and not git_ops.is_ancestor(cwd, target, run_branch)
        and not git_ops.is_ancestor(cwd, run_branch, target)
    ):
        # A run the target already contains is not refused: that is a clean noop below.
        return MergeOutcome(
            "error",
            error=(
                f"{target!r} has moved since the run started, so a"
                " fast-forward is impossible; merge with --strategy merge or"
                " squash instead"
            ),
        )
    # Every strategy that merges something moves the target; unmoved means nothing to merge.
    target_tip_before = git_ops.branch_tip_sha(cwd, target) or ""
    try:
        merge_base = landed_base(
            cwd, layout, manifest, target, git_ops.chain_tip(cwd, run_branch) or ""
        )
        result = dispatch_merge(
            cwd,
            strategy,
            target,
            run_branch,
            base_sha,
            manifest,
            message,
            cfg,
            identity,
            transcript_dir=layout.session_dir / "transcripts",
            budget=budget,
            events=events,
            warn=warn,
            merge_base=merge_base,
        )
    except git_ops.GitError as exc:
        return MergeOutcome("error", error=f"merge failed: {exc}")
    if result.conflicted:
        return MergeOutcome("conflict", conflicts=result.conflicts)
    noop = bool(result.merged_sha) and result.merged_sha == target_tip_before
    merged_tip = git_ops.chain_tip(cwd, run_branch) or ""
    if noop and manifest.merged is not None and manifest.merged.tip == merged_tip:
        # Stamping the target's tip over the record would credit the run with later commits.
        return MergeOutcome("noop", merged_sha=result.merged_sha)
    # A merge that added nothing still records the run as merged up to this tip.
    stamp_error = record_merge_in_manifest(
        layout,
        merged_into=target,
        merged_sha=sessions_manifest.NO_MERGE_COMMIT if noop else result.merged_sha,
        merged_tip=merged_tip,
        into_tip=target_tip_before if noop else "",
    )
    return MergeOutcome(
        "noop" if noop else "merged",
        merged_sha=result.merged_sha,
        stamp_error=stamp_error,
        recorded=noop and not stamp_error,
        left_behind=result.left_behind,
    )


def left_behind_line(target: str, outcome: MergeOutcome) -> str:
    """Return the line for merged files the checkout keeps its own version of, or "".

    The plumbing brings the checkout forward only where a file still matches what the branch
    held, so an operator's edit is never overwritten; the tree they then test holds the older
    content.
    """
    if not outcome.left_behind:
        return ""
    named = ", ".join(outcome.left_behind[:4]) + (", ..." if len(outcome.left_behind) > 4 else "")
    return f"your checkout keeps its own {named}; {target} holds what was merged"


def noop_merge_line(run_branch: str, target: str, outcome: MergeOutcome) -> str:
    """Return the line for a merge that added nothing; a stamp error is the caller's note."""
    if outcome.recorded or outcome.stamp_error:
        recorded = f"; recorded as merged into {target}" if outcome.recorded else ""
        return f"nothing to add from {run_branch}: {target} already has its content{recorded}"
    return f"nothing left to merge from {run_branch} into {target}"
