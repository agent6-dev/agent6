# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions list/diff/commits/stop/dir/remove`: the run-branch read side.

`merge` is `sessions_merge`, `prune` is `sessions_prune`, `compare` is `sessions_compare`.
"""

from __future__ import annotations

import contextlib
import pathlib
import shutil
import subprocess
import sys

from agent6 import errors, portable
from agent6 import paths as agent6_paths
from agent6.app import fork_worktrees, merge_plan, resume
from agent6.sessions import id, lineage, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import common, steer_tty
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import policy as git_policy
from agent6.vcs.git import read as git_read
from agent6.viewmodel import format, newest_session_dir, session_dirs, summarize_session_dir


def _cmd_diff(*, session_id: str, stat: bool, paths: tuple[str, ...], paginate: bool = True) -> int:
    """Print the git diff a run produced, from its base sha to its branch head.

    Streams to the terminal, so it cannot go through git_ops; it carries the same host-RCE
    hardening (a poisoned `.git/config` `diff.external`, textconv, `core.fsmonitor` or hook
    must not execute on the host) plus `DIFF_SAFETY_FLAGS`, which force the builtin
    renderer (git 2.53 executes even an empty `diff.external` override) and disable the
    per-file textconv driver the `-c` flags do not reach.

    Args:
        session_id: The run, or "" for the newest.
        stat: Print `--stat` only.
        paths: The pathspecs.
        paginate: Let git page; off for the `run -i` REPL, whose prompt loop the pager
            would take over.

    Returns:
        git's exit code; 2 when the run cannot be resolved or has no branch.
    """
    cwd = pathlib.Path.cwd()
    res = _resolve_session_manifest(
        cwd,
        session_id,
        recent_note="diffing most recent run",
        missing_hint=" (predates manifest support, or was killed before setup)",
    )
    if isinstance(res, int):
        return res
    _layout, manifest = res

    ref = merge_plan.commits_ref(cwd, manifest)
    if not ref.head_ref:
        print(f"[agent6] {ref.reason}.")
        return 0
    if manifest.run_branch and ref.head_ref == manifest.run_branch:
        # A branch may stand in for a pruned one: say where the work went.
        pruned = _pruned_branch_note(cwd, manifest, manifest.run_branch)
        if pruned is not None:
            print(pruned)
            return 0
    base_sha = manifest.base_sha
    if not base_sha:
        common.error("manifest has no base_sha; nothing to diff against")
        return 2

    head_ref = ref.head_ref
    # Printed without the -c hardening overrides, as git_ops error messages are; executed with them.
    args: list[str] = ["diff", *git_policy.DIFF_SAFETY_FLAGS]
    if stat:
        args.append("--stat")
    args.extend([f"{base_sha}..{head_ref}"])
    if paths:
        args.append("--")
        args.extend(paths)
    base = f"base branch {manifest.base_branch}" if manifest.base_branch else "detached HEAD"
    print(f"[agent6] git {' '.join(args)}  ({base})", file=sys.stderr)
    # Probe first so a zero-commit run says so; a probe error falls through to git's message.
    probe_args = [
        "diff",
        *git_policy.DIFF_SAFETY_FLAGS,
        "--quiet",
        f"{base_sha}..{head_ref}",
    ]
    if paths:
        probe_args.extend(["--", *paths])
    probe = subprocess.run(
        ["git", *git_policy.git_hardening_flags(cwd), *probe_args],
        env=git_policy.git_environment(),
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0:
        # A live run mid-work has uncommitted edits on the worktree: say so, not a bare silence.
        dirty = _dirty_worktree_note(cwd, manifest.run_branch)
        print(dirty if dirty else "(no changes)")
        return 0
    pager = () if paginate else ("--no-pager",)
    proc = subprocess.run(
        ["git", *pager, *git_policy.git_hardening_flags(cwd), *args],
        cwd=cwd,
        check=False,
        env=git_policy.git_environment(),
    )
    if proc.returncode != 0 and (note := git_policy.unsupported_note()):
        print(note, file=sys.stderr)
    return proc.returncode


def _dirty_worktree_note(cwd: pathlib.Path, run_branch: object) -> str:
    """Return a note when the diffed run's branch is checked out with uncommitted work, else "".

    Only when the dirty files are unambiguously this run's: the current branch must equal
    the run branch.

    Args:
        cwd: The checkout git runs from.
        run_branch: The run's branch from its manifest; empty gives "".

    Returns:
        The note, counting the dirty files. Best effort; a git error reads as "".
    """
    if not run_branch:
        return ""
    try:
        status = git_read.status(cwd)
    except (errors.GitError, OSError):
        return ""
    n = status.modified_count + status.untracked_count
    if status.branch != str(run_branch) or n == 0:
        return ""
    files = "file" if n == 1 else "files"
    return (
        f"(no committed changes yet; {n} {files} modified in the working tree: "
        "a run commits after each verify pass)"
    )


def _resolve_session_manifest(
    cwd: pathlib.Path,
    session_id: str,
    *,
    recent_note: str = "using most recent run",
    missing_hint: str = "",
    allow_model: bool = False,
) -> tuple[sessions_layout.SessionLayout, sessions_manifest.SessionManifest] | int:
    """Resolve a run id, or "" for the newest, to its layout and manifest.

    Shared by `sessions diff`, `merge` and `commits`; the two note strings vary per caller.

    Args:
        cwd: The repo.
        session_id: The run.
        recent_note: What to print when the newest run was taken.
        missing_hint: What to add when nothing matches.
        allow_model: Permit read-only comparison of model-controlled runs.

    Returns:
        `(layout, manifest)`, or the exit code of a printed error.
    """
    runs_dir = common._runs_dir(cwd)
    if not session_id:
        # No id: the most recent run; a plan or an ask has no branch for these verbs.
        latest = newest_session_dir([runs_dir]) if runs_dir.is_dir() else None
        if latest is None:
            # Only branchless sessions: say so; a fresh state dir keeps the first-contact copy.
            common.print_nothing_yet(
                "runs" if session_dirs(agent6_paths.state_dir(cwd)) else "sessions"
            )
            return 2
        layout = sessions_layout.layout_of(latest)
        print(f"[agent6] {recent_note}: {layout.session_id}", file=sys.stderr)
    else:
        # Every bucket: a plan the operator named exists, it just has no branch to show.
        try:
            layout = common.resolve_session_layout(cwd, session_id)
        except id.SessionIdError as exc:
            common.error(f"{exc}")
            return 2
    target_id = layout.session_id
    if not layout.manifest_path.is_file():
        common.error(f"session {target_id} has no manifest.json{missing_hint}")
        return 2
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError as exc:
        common.error(f"could not read manifest: {exc}")
        return 2
    refusal = merge_plan.chain_refusal(manifest, allow_model=allow_model)
    if refusal is not None:
        common.refuse(f"{refusal}")
        return 2
    return layout, manifest


def _committed_nothing(cwd: pathlib.Path, session_id: str) -> bool:
    """Return whether a run left no commit anywhere.

    Args:
        cwd: The repo.
        session_id: The run's id.

    Returns:
        True when the chain ref it commits to was never created, so its branch was never
        cut either.
    """
    return git_read.chain_tip(cwd, git_read.chain_ref_for(session_id)) is None


def _pruned_branch_note(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest, run_branch: str
) -> str | None:
    """Return where the work went when a run's branch is absent, or None when it is there.

    Args:
        cwd: The repo.
        manifest: The run's manifest.
        run_branch: The manifest's run branch.

    Returns:
        The note, separating the ways to get here: a merged-then-pruned branch (the stamp
        covering every commit), a branch deleted past its stamp or with no merge recorded
        (the chain ref keeps the commits), and a run that committed nothing.
    """
    if git_read.branch_exists(cwd, run_branch):
        return None
    stamp = resume.covering_stamp(cwd, manifest)
    if stamp is not None:
        note = f"[agent6] run branch {run_branch} was pruned; {stamp.description()}"
        if stamp.commit:
            note += f"\n  see: git show {stamp.commit}"
        return note
    if _committed_nothing(cwd, manifest.session_id):
        return f"[agent6] this run committed nothing, so {run_branch} was never cut."
    chain = git_read.chain_ref_for(manifest.session_id)
    if manifest.merged is not None:
        return (
            f"[agent6] run branch {run_branch} is gone; its commits survive at {chain},"
            f" past the merge into {manifest.merged.into}."
        )
    return (
        f"[agent6] run branch {run_branch} is gone with no merge recorded; its commits"
        f" survive at {chain}."
    )


def _cmd_commits(*, session_id: str) -> int:
    """List the per-turn commits on a run's branch or chain ref.

    Args:
        session_id: The run, or "" for the newest.

    Returns:
        The exit code; 2 when the run cannot be resolved, or names no branch and has no chain ref.
    """
    cwd = pathlib.Path.cwd()
    res = _resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    _layout, manifest = res
    ref = merge_plan.commits_ref(cwd, manifest)
    if not ref.head_ref:
        common.error(f"this session has no branch to list commits from ({ref.reason}).")
        return 2
    base_sha = manifest.base_sha
    if not base_sha:
        common.error("manifest has no base_sha; nothing to list commits from")
        return 2
    head_ref = ref.head_ref
    # A branch may stand in for a pruned one: say where the work went.
    pruned = (
        _pruned_branch_note(cwd, manifest, manifest.run_branch)
        if manifest.run_branch and head_ref == manifest.run_branch
        else None
    )
    if pruned is not None:
        print(pruned)
        return 0
    rows = git_read.list_run_commits(cwd, base_sha, head_ref)
    if not rows:
        print(f"[agent6] no commits on {head_ref}.")
        return 0
    for row in rows:
        print(f"{row.sha[:12]}  {row.subject}")
    print(f"\n[agent6] {len(rows)} commit(s) on {head_ref}", file=sys.stderr)
    return 0


def _cmd_sessions_dir(session_id: str = "") -> int:
    """Print the per-repo state dir, or the named session's own directory.

    One bare line so it composes (`ls "$(agent6 sessions dir)"`). Sessions live under
    `sessions/<bucket>/`, one bucket per mode.

    Args:
        session_id: The session, or "" for the state dir.

    Returns:
        The exit code; 2 when the session cannot be resolved.
    """
    cwd = pathlib.Path.cwd()
    if not session_id:
        print(agent6_paths.state_dir(cwd))
        return 0
    try:
        layout = common.resolve_session_layout(cwd, session_id)
    except id.SessionIdError as exc:
        common.error(f"{exc}")
        return 2
    print(layout.session_dir)
    return 0


def _remove_asks(cwd: pathlib.Path, session_id: str, *, yes: bool) -> int:
    """Clear this directory's asks bucket.

    Args:
        cwd: The directory whose asks bucket is cleared.
        session_id: The run id typed beside `--asks`; any id is refused.
        yes: Delete without asking.

    Returns:
        The exit code; 1 when a deletion failed, never a success line over a surviving dir.
    """
    if session_id:
        common.error("--asks clears this directory's asks; drop the run id.")
        return 2
    state_dir = agent6_paths.state_dir(cwd)
    bucket = sessions_layout.bucket_dir(state_dir, "asks")
    sessions = tuple(bucket.iterdir()) if bucket.is_dir() else ()
    reviewers = lineage.reviewer_records(state_dir, {session.name for session in sessions})
    gone = len(sessions)
    count = f"{gone} ask{'' if gone == 1 else 's'}"
    if gone and not yes and not portable.has_controlling_tty():
        common.refuse(
            f"no controlling terminal to confirm deleting {count} saved from {cwd}; use --yes."
        )
        return 2
    if (
        gone
        and not yes
        and not steer_tty.delete_confirmed(
            f"sessions remove --asks will delete {count} saved from {cwd}", "Delete them?"
        )
    ):
        return 0
    try:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(bucket)
        for reviewer in reviewers:
            shutil.rmtree(reviewer)
    except OSError as exc:
        common.error(f"could not remove {exc.filename or bucket}: {exc}")
        return 1
    try:
        if git_read.is_git_repo(cwd):
            for session in (*sessions, *reviewers):
                for ref in git_read.session_refs(cwd, session.name):
                    git_ops.delete_ref(cwd, ref)
    except errors.GitError as exc:
        common.error(f"removed asks, but could not remove their refs: {exc.safe_message}")
        return 1
    print(f"removed {count} from {cwd}")
    return 0


def _remove_refusal(
    layout: sessions_layout.SessionLayout, worktree: pathlib.Path | None, tips: tuple[str, ...]
) -> str:
    """Return why this record cannot be deleted, or "".

    The record is the only thing that names a fork's worktree, so deleting one that still
    holds work no commit has would leave nothing to find it by.

    Args:
        layout: The session.
        worktree: The fork's own worktree; None when another session shares and keeps it.
        tips: The branch tips the worktree's commits must reach.

    Returns:
        The refusal when the session or a child of it is live, or its worktree holds work
        no commit has.
    """
    for live in (layout.session_dir, *lineage.descendants_of(layout.session_dir)):
        if process.worker_is_alive(live):
            return f"{live.name} is still live; stop it first (agent6 stop {layout.session_id})."
    if worktree is None or not (dirt := fork_worktrees.uncommitted_in_worktree(worktree, tips)):
        return ""
    return (
        f"{layout.session_id}'s worktree {dirt} ({worktree}); deleting the record"
        f" would leave it with nothing naming it. Keep that work, then:"
        f" git -C {worktree} status"
    )


def _remove_preview(cwd: pathlib.Path, layout: sessions_layout.SessionLayout) -> str:
    """Return the lines naming what `sessions remove` is about to delete.

    Args:
        cwd: The checkout git is read from.
        layout: The session.

    Returns:
        The id, task and state lines, its child count, then for unmerged commits whether a
        branch keeps them.
    """
    tips = git_read.run_ref_tips(cwd)
    summary = summarize_session_dir(layout.session_dir, branch_tips=tips, repo=cwd)
    task = summary.task_label.line(100).replace("\n", "\n" + " " * 11)
    lines = [
        f"sessions remove will delete {layout.session_id}",
        f"  task:    {task}",
        f"  state:   {format.status_label(summary.status, summary.detail)}",
    ]
    children = [
        child
        for child in lineage.descendants_of(layout.session_dir)
        if child.parent.name != sessions_layout.EVIDENCE_BUCKET
    ]
    if children:
        lines.append(f"  children:   {len(children)} go with it")
    if summary.retained_ref:
        lines.append(f"  commits: retained ref {summary.retained_ref} goes with the record")
    elif summary.unmerged:
        branch = git_read.run_branch_for(layout.session_id)
        kept = f"branch {branch} keeps them" if branch in tips else "they go with it"
        lines.append(f"  commits: not merged; {kept}")
    return "\n".join(lines)


def _cmd_sessions_remove(*, session_id: str, asks: bool, yes: bool) -> int:
    """Delete run history from the state dir, with the run's chain ref and a fork's worktree.

    The chain ref (`refs/agent6/<id>/head`) is the gc anchor: meaningless once the record is
    gone, and left behind it would pin the run's objects forever. A fork's worktree goes
    unless another session (an `/undo` fork of it) still names it. The run's visible branch
    and its commits are git's and are left alone; `sessions prune` is the branch verb.
    `--asks` clears the asks made in this directory; asks made elsewhere are untouched.

    On a controlling terminal it names what will go and asks first; `yes` skips the question.
    Without one, deleting the newest session or the asks requires `yes`; an explicit id
    deletes without asking.

    Args:
        session_id: The session, or "" for the newest.
        asks: Clear the asks bucket instead.
        yes: Delete without asking.

    Returns:
        The exit code; 0 also when the operator answered no, 1 on a partial delete, 2 on a
        refusal.
    """
    if asks:
        return _remove_asks(pathlib.Path.cwd(), session_id, yes=yes)
    cwd = fork_worktrees.sweep_checkout(pathlib.Path.cwd())
    try:
        # remove is the surface that deletes a husk, so it resolves one.
        layout = common.resolve_or_newest_layout(cwd, session_id, allow_husk=True)
    except id.SessionIdError as exc:
        common.error(f"{exc}")
        return 2
    if layout is None or (not session_id and not yes and not portable.has_controlling_tty()):
        if layout is None:
            common.print_nothing_yet()
        else:
            common.refuse(
                "no session id and no controlling terminal to confirm deleting "
                f"{layout.session_id}; pass the id "
                f"(`agent6 sessions remove {layout.session_id}`) or use --yes."
            )
        return 2
    worktree: pathlib.Path | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        worktree = sessions_manifest.read_manifest(layout.session_dir).worktree
    sharing = (
        [
            d.name
            for d, _m in fork_worktrees.worktree_owners(agent6_paths.state_dir(cwd)).get(
                worktree, []
            )
            if d != layout.session_dir
        ]
        if worktree is not None
        else []
    )
    chain_tip = git_read.chain_tip(cwd, git_read.chain_ref_for(layout.session_id)) or ""
    tips = (chain_tip,) if chain_tip else ()
    reason = _remove_refusal(layout, worktree if not sharing else None, tips)
    if not reason and not yes:
        if not steer_tty.delete_confirmed(_remove_preview(cwd, layout), "Delete it?"):
            return 0
        # A resume from another terminal while the question stood makes the session live again.
        reason = _remove_refusal(layout, worktree if not sharing else None, tips)
    if reason:
        common.refuse(f"{reason}")
        return 2
    return _remove_session(cwd, layout, worktree, sharing, tips)


def _remove_session(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    worktree: pathlib.Path | None,
    sharing: list[str],
    tips: tuple[str, ...],
) -> int:
    """Delete one session's record with its children's, then its worktree and owned refs.

    The listings fold a run's working children under it and show no reviewer, so both go with
    the run. A child's branch stays, as the run's does.

    Args:
        cwd: The checkout git runs from.
        layout: The session.
        worktree: The fork's worktree the manifest records, or None.
        sharing: The other sessions that still name that worktree.
        tips: The branch tips the worktree's commits must reach.

    Returns:
        The exit code; 1 on a partial delete.
    """
    children = lineage.descendants_of(layout.session_dir)
    reviewers = [c for c in children if c.parent.name == sessions_layout.EVIDENCE_BUCKET]
    workers = [c for c in children if c not in reviewers]
    try:
        shutil.rmtree(layout.session_dir)
        for child in children:
            shutil.rmtree(child)
    except OSError as exc:
        # A partial delete leaves a remnant: no success line, and the chain ref stays as its anchor.
        common.error(f"could not remove {layout.session_dir}: {exc}")
        return 1
    went: list[str] = []  # what went with the record
    if workers:
        went.append(f"its {len(workers)} {'child' if len(workers) == 1 else 'children'}")
    if reviewers:
        went.append(f"{len(reviewers)} reviewer record{'' if len(reviewers) == 1 else 's'}")
    stays = ""
    if worktree is not None and sharing:
        verb = "names" if len(sharing) == 1 else "name"
        stays = f"; its worktree stays: {', '.join(sharing)} still {verb} it"
    elif worktree is not None:
        gone, why = fork_worktrees.remove_fork_worktree(cwd, worktree, tips)
        if gone:
            went.append("its worktree")
        elif why:
            stays = f"; its worktree stays: it {why} ({worktree})"
    chain = git_read.chain_ref_for(layout.session_id)
    try:
        if (chain_head := git_read.chain_tip(cwd, chain)) is not None:
            branch = git_read.run_branch_for(layout.session_id)
            branch_kept = git_read.branch_exists(cwd, branch)
            git_ops.delete_ref(cwd, chain)
            # A chain ref has no reflog: the sha on the deleting line is the only way back.
            went.append(
                "its chain ref"
                + (
                    f" (branch {branch} kept; `sessions prune` reports it)"
                    if branch_kept
                    else f" (its commits are now loose; until git gc:"
                    f" git branch <name> {chain_head[:12]})"
                )
            )
        snapshots = git_read.session_refs(cwd, layout.session_id)
        for ref in (
            *snapshots,
            *(r for child in children for r in git_read.session_refs(cwd, child.name)),
        ):
            git_ops.delete_ref(cwd, ref)
        if snapshots:
            count = len(snapshots)
            went.append(f"{count} review snapshot{'' if count == 1 else 's'}")
    except errors.GitError:
        pass  # not a repo here, or git unreadable: state-dir removal stands
    what = [layout.session_id, *went]
    removed = f"{', '.join(what[:-1])} and {what[-1]}" if went else what[0]
    print(f"removed {removed}{stays}")
    common.note_removed_cwd(cwd)
    return 0
