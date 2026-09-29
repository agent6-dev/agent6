# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions list/diff/commits/stop/dir/rm`: the run-branch read side.

`merge` and `prune` are `sessions_merge`, `compare` is `sessions_compare`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import shutil
import subprocess
import sys

from agent6 import git_ops, kinds
from agent6 import paths as agent6_paths
from agent6.app import fork_worktrees, resume
from agent6.sessions import id, ipc
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import _common
from agent6.viewmodel import (
    format,
    is_winner,
    newest_session_dir,
    session_dirs,
    summarize_session_dir,
    task_snippet,
    wire,
)
from agent6.viewmodel import listing as viewmodel_listing


def _cmd_list(*, as_json: bool = False, lanes: bool = False) -> int:
    """List this repo's sessions, newest first.

    Columns: updated, status (the mode folded in when the word does not imply it, the
    failure reason, the unmerged mark), cost, id, task. A fan-out's lanes nest under its
    row: folded into a count, or listed indented with `lanes`; the JSON row always nests
    them. Every bucket, unlike the TUI and web hubs, which give `machine create` drafts
    their own card.

    Args:
        as_json: Print the rows as JSON.
        lanes: List each fan-out's lanes.

    Returns:
        The exit code, 0.
    """
    cwd = pathlib.Path.cwd()
    dirs = session_dirs(agent6_paths.state_dir(cwd), sessions_layout.SESSION_BUCKETS)
    if not dirs:
        print(
            "[]" if as_json else _common.nothing_yet()
        )  # the empty listing is output, not an error
        return 0
    winners = {d.name for d in dirs if is_winner(d)}  # fan-out compare winners
    tips = git_ops.run_ref_tips(cwd)
    listing = viewmodel_listing.nested_rows(
        summarize_session_dir(d, branch_tips=tips) for d in dirs
    )
    if as_json:
        print(
            json.dumps([viewmodel_listing.row_json(r, winners=winners) for r in listing], indent=2)
        )
        return 0
    color = sys.stdout.isatty()

    def cells(
        row: viewmodel_listing.ListingRow, id_cell: str
    ) -> tuple[str, str, str, str, str, str]:
        """Return a row's six cells with the id cell as given."""
        s = row.summary
        styled, plain = _common.styled_status(
            s.status,
            s.reason,
            color=color,
            label=format.listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged),
        )
        return format.format_when(row.mtime), styled, plain, s.cost_cell, id_cell, s.task

    rows: list[tuple[str, str, str, str, str, str]] = []

    def emit(row: viewmodel_listing.ListingRow, depth: int) -> None:
        """Append the row's cells, then its lanes' when listing them."""
        s = row.summary
        id_cell = format.winner_id(s.session_id, winner=s.session_id in winners)
        if depth:
            id_cell = format.lane_id_cell(id_cell, depth)
        elif row.lanes and not lanes:
            id_cell += f" ({format.lane_count(len(row.lanes))})"
        rows.append(cells(row, id_cell))
        if lanes:
            for lane in row.lanes:
                emit(lane, depth + 1)

    for row in listing:
        emit(row, 0)
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    id_w = max(2, *(len(r[4]) for r in rows))
    # The task column takes what a tty has left (floor 24); piped output keeps a fixed 60.
    fixed = 11 + 2 + status_w + 2 + 8 + 2 + id_w + 2
    task_w = max(24, shutil.get_terminal_size().columns - fixed) if color else 60
    print(f"{'updated':<11}  {'status':<{status_w}}  {'cost':<8}  {'id':<{id_w}}  task")
    for when, styled, plain, cost, session_id, task in rows:
        pad = " " * (status_w - len(plain))
        snip = task_snippet(task, max_chars=task_w)
        print(f"{when:<11}  {styled}{pad}  {cost:<8}  {session_id:<{id_w}}  {snip}")
    return 0


def _cmd_diff(*, session_id: str, stat: bool, paths: tuple[str, ...], paginate: bool = True) -> int:
    """Print the git diff a run produced, from its base sha to its branch head.

    Streams to the terminal, so it cannot go through git_ops; it carries the same host-RCE
    hardening (a poisoned `.git/config` `diff.external`, textconv, `core.fsmonitor` or hook
    must not execute on the host) plus `DIFF_SHOW_SAFETY_FLAGS`, which force the builtin
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

    ref = _commits_ref(cwd, manifest)
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
        _common.error("manifest has no base_sha; nothing to diff against")
        return 2

    head_ref = ref.head_ref
    # Printed without the -c hardening overrides, as git_ops error messages are; executed with them.
    args: list[str] = ["diff", *git_ops.DIFF_SHOW_SAFETY_FLAGS]
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
    # Probe first so a zero-commit run says so; a probe error falls through to git's message.
    probe_args = ["diff", *git_ops.DIFF_SHOW_SAFETY_FLAGS, "--quiet", f"{base_sha}..{head_ref}"]
    if paths:
        probe_args.extend(["--", *paths])
    probe = subprocess.run(
        ["git", *git_ops.git_hardening_flags(cwd), *probe_args],
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
        ["git", *pager, *git_ops.git_hardening_flags(cwd), *args], cwd=cwd, check=False
    )
    return proc.returncode


def _dirty_worktree_note(cwd: pathlib.Path, run_branch: object) -> str:
    """Return a note when the diffed run's branch is checked out with uncommitted work, else "".

    Only when the dirty files are unambiguously this run's: the current branch must equal
    the run branch. Best effort; a git error reads as "".
    """
    if not run_branch:
        return ""
    # Hardened like the diff: `git status` would fire a poisoned core.fsmonitor on the host.
    try:
        current = subprocess.run(
            ["git", *git_ops.git_hardening_flags(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if current.returncode != 0 or current.stdout.strip() != str(run_branch):
            return ""
        status = subprocess.run(
            ["git", *git_ops.git_hardening_flags(cwd), "status", "--porcelain"],
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


@dataclasses.dataclass(frozen=True, slots=True)
class _CommitsRef:
    """Where a session's commits end, for `base_sha..head_ref`.

    Attributes:
        head_ref: The run branch, the hidden chain ref for a run with branch_per_run off,
            or "" when the session made no commits.
        reason: Why there is no ref; "" exactly when `head_ref` is one, so the branch verbs
            refuse on it while diff reads `head_ref`.
    """

    head_ref: str
    reason: str


def _commits_ref(cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest) -> _CommitsRef:
    """Return where the run's commits end.

    The run branch while it covers the chain, else the chain ref; else the manifest's branch
    name while no chain exists (the verbs read its absence themselves: pruned, never cut, or
    a lane's branch still in its clone); else the reason the run has no commits.
    """
    if ref := wire.commits_ref(manifest, cwd):
        return _CommitsRef(head_ref=ref, reason="")
    if (
        manifest.run_branch
        and git_ops.chain_tip(cwd, git_ops.chain_ref_for(manifest.session_id)) is None
    ):
        return _CommitsRef(head_ref=manifest.run_branch, reason="")
    if manifest.parked_task:
        # A parked run never started, so `base..HEAD` is the run it was parked behind.
        return _CommitsRef(
            head_ref="", reason="this run was parked before it started, so it made no commits"
        )
    kind = kinds.SESSION_KINDS.get(manifest.mode)
    if kind is not None and not kind.edits:
        article = "an" if manifest.mode[:1] in "aeiou" else "a"
        return _CommitsRef(
            head_ref="",
            reason=f"{article} {manifest.mode} does not write to the repo, so it made no commits",
        )
    return _CommitsRef(head_ref="", reason="this run recorded no commits")


def _resolve_session_manifest(
    cwd: pathlib.Path,
    session_id: str,
    *,
    recent_note: str = "using most recent run",
    missing_hint: str = "",
) -> tuple[sessions_layout.SessionLayout, sessions_manifest.SessionManifest] | int:
    """Resolve a run id, or "" for the newest, to its layout and manifest.

    Shared by `sessions diff`, `merge` and `commits`; the two note strings vary per caller.

    Args:
        cwd: The repo.
        session_id: The run.
        recent_note: What to print when the newest run was taken.
        missing_hint: What to add when nothing matches.

    Returns:
        `(layout, manifest)`, or the exit code of a printed error.
    """
    runs_dir = _common._runs_dir(cwd)
    if not session_id:
        # No id: the most recent run; a plan or an ask has no branch for these verbs.
        latest = newest_session_dir([runs_dir]) if runs_dir.is_dir() else None
        if latest is None:
            # Only branchless sessions: say so; a fresh state dir keeps the first-contact copy.
            _common.print_nothing_yet(
                "runs" if session_dirs(agent6_paths.state_dir(cwd)) else "sessions"
            )
            return 2
        layout = sessions_layout.layout_of(latest)
        print(f"[agent6] {recent_note}: {layout.session_id}", file=sys.stderr)
    else:
        # Every bucket: a plan the operator named exists, it just has no branch to show.
        try:
            layout = _common.resolve_session_layout(cwd, session_id)
        except id.SessionIdError as exc:
            _common.error(f"{exc}")
            return 2
    target_id = layout.session_id
    if not layout.manifest_path.is_file():
        _common.error(f"session {target_id} has no manifest.json{missing_hint}")
        return 2
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError as exc:
        _common.error(f"could not read manifest: {exc}")
        return 2
    # A fan-out commits nothing: its lanes hold the work.
    refusal = (
        f"{target_id} is a fan-out; its lanes hold the commits"
        f" (`agent6 sessions show {target_id}` lists them)"
        if manifest.fanout is not None
        else sessions_manifest.model_git_refusal(manifest, "sessions")
    )
    if refusal is not None:
        _common.refuse(f"{refusal}")
        return 2
    return layout, manifest


def _committed_nothing(cwd: pathlib.Path, session_id: str) -> bool:
    """Return whether a run left no commit anywhere.

    The chain ref it commits to was never created, so its branch was never cut either.
    """
    return git_ops.chain_tip(cwd, git_ops.chain_ref_for(session_id)) is None


def _pruned_branch_note(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest, run_branch: str
) -> str | None:
    """Return where the work went when a run's branch is absent, or None when it is there.

    Separates the ways to get here: a merged-then-pruned branch (the stamp covering every
    commit), a branch deleted past its stamp or with no merge recorded (the chain ref keeps
    the commits), and a run that committed nothing.
    """
    if git_ops.branch_exists(cwd, run_branch):
        return None
    stamp = resume.covering_stamp(cwd, manifest)
    if stamp is not None:
        note = f"[agent6] run branch {run_branch} was pruned; {stamp.landed()}"
        if stamp.commit:
            note += f"\n  see: git show {stamp.commit}"
        return note
    if _committed_nothing(cwd, manifest.session_id):
        return f"[agent6] this run committed nothing, so {run_branch} was never cut."
    chain = git_ops.chain_ref_for(manifest.session_id)
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
    """List the per-step commits on a run's branch or chain ref.

    Returns:
        The exit code; 2 when the run cannot be resolved or has no commits.
    """
    cwd = pathlib.Path.cwd()
    res = _resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    _layout, manifest = res
    ref = _commits_ref(cwd, manifest)
    if not ref.head_ref:
        _common.error(f"this session has no branch to list commits from ({ref.reason}).")
        return 2
    base_sha = manifest.base_sha
    if not base_sha:
        _common.error("manifest has no base_sha; nothing to list commits from")
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
    rows = git_ops.list_run_commits(cwd, base_sha, head_ref)
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

    Returns:
        The exit code; 2 when the session cannot be resolved.
    """
    cwd = pathlib.Path.cwd()
    if not session_id:
        print(agent6_paths.state_dir(cwd))
        return 0
    try:
        layout = _common.resolve_session_layout(cwd, session_id)
    except id.SessionIdError as exc:
        _common.error(f"{exc}")
        return 2
    print(layout.session_dir)
    return 0


def _rm_asks(cwd: pathlib.Path, session_id: str) -> int:
    """Clear this directory's asks bucket.

    Returns:
        The exit code; 1 when a deletion failed, never a success line over a surviving dir.
    """
    if session_id:
        _common.error("--asks clears this directory's asks; drop the run id.")
        return 2
    bucket = sessions_layout.bucket_dir(agent6_paths.state_dir(cwd), "asks")
    gone = sum(1 for _ in bucket.iterdir()) if bucket.is_dir() else 0
    try:
        shutil.rmtree(bucket)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _common.error(f"could not remove {bucket}: {exc}")
        return 1
    print(f"removed {gone} ask{'' if gone == 1 else 's'} from {cwd}")
    return 0


def _rm_refusal(
    layout: sessions_layout.SessionLayout, worktree: pathlib.Path | None, tips: tuple[str, ...]
) -> str:
    """Return why this record cannot be deleted, or "".

    The record is the only thing that names a fork's worktree, so deleting one that still
    holds work no commit has would leave nothing to find it by.

    Args:
        layout: The session.
        worktree: The fork's own worktree; None when another session shares and keeps it.
        tips: The branch tips the worktree's commits must reach.
    """
    if ipc.worker_is_alive(layout.session_dir):
        return (
            f"{layout.session_id} is still live; stop it first (agent6 stop {layout.session_id})."
        )
    if worktree is None or not (dirt := fork_worktrees.uncommitted_in_worktree(worktree, tips)):
        return ""
    return (
        f"{layout.session_id}'s worktree {dirt} ({worktree}); deleting the record"
        f" would leave it with nothing naming it. Keep that work, then:"
        f" git -C {worktree} status"
    )


def _cmd_sessions_rm(*, session_id: str, asks: bool) -> int:
    """Delete run history from the state dir, with the run's chain ref and a fork's worktree.

    The chain ref (`refs/agent6/<id>/head`) is the gc anchor: meaningless once the record is
    gone, and left behind it would pin the run's objects forever. A fork's worktree goes
    unless another session (an `/undo` fork of it) still names it. The run's visible branch
    and its commits are git's and are left alone; `sessions prune` is the branch verb.
    `--asks` clears the asks made in this directory; asks made elsewhere are untouched.

    Args:
        session_id: The session, or "" for the newest.
        asks: Clear the asks bucket instead.

    Returns:
        The exit code; 1 on a partial delete, 2 on a refusal.
    """
    cwd = pathlib.Path.cwd()
    if asks:
        return _rm_asks(cwd, session_id)
    try:
        # rm is the surface that deletes a husk, so it resolves one.
        layout = _common.resolve_or_newest_layout(cwd, session_id, allow_husk=True)
    except id.SessionIdError as exc:
        _common.error(f"{exc}")
        return 2
    if layout is None:
        _common.print_nothing_yet()
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
    landed = git_ops.chain_tip(cwd, git_ops.chain_ref_for(layout.session_id)) or ""
    tips = (landed,) if landed else ()
    if reason := _rm_refusal(layout, worktree if not sharing else None, tips):
        _common.refuse(f"{reason}")
        return 2
    try:
        shutil.rmtree(layout.session_dir)
    except OSError as exc:
        # A partial delete leaves a remnant: no success line, and the chain ref stays as its anchor.
        _common.error(f"could not remove {layout.session_dir}: {exc}")
        return 1
    went: list[str] = []  # what went with the record
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
    chain = git_ops.chain_ref_for(layout.session_id)
    try:
        if (chain_head := git_ops.chain_tip(cwd, chain)) is not None:
            branch = git_ops.run_branch_for(layout.session_id)
            branch_kept = git_ops.branch_exists(cwd, branch)
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
    except git_ops.GitError:
        pass  # not a repo here, or git unreadable: state-dir removal stands
    what = [layout.session_id, *went]
    removed = f"{', '.join(what[:-1])} and {what[-1]}" if went else what[0]
    print(f"removed {removed}{stays}")
    return 0
