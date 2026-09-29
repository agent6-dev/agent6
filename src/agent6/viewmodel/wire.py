# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the one-object wire snapshots of a session and of a machine instance.

`agent6 attach --json` prints them and the web serves them from the same fold.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6 import budget, git_ops
from agent6.machine import MachineJournal, load_machine
from agent6.sessions import ipc, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format, machine_state, tail
from agent6.viewmodel import state as viewmodel_state


def existing_run_branch(
    manifest: sessions_manifest.SessionManifest, repo: pathlib.Path | None
) -> str:
    """Return the run's branch while it exists, else "".

    The manifest names the branch at run start and git creates it at the first
    commit, so a run that never committed has none to merge or prune.

    Args:
        manifest: The session's manifest.
        repo: The repository to ask; None lets the manifest's word stand.

    Returns:
        The branch name, or "" when the manifest names none or the repo lacks it.
    """
    name = manifest.run_branch or ""
    if not name or (repo is not None and not git_ops.branch_exists(repo, name)):
        return ""
    return name


def commits_ref(manifest: sessions_manifest.SessionManifest, repo: pathlib.Path) -> str:
    """Return the ref holding the run's commits, what merge, diff and the footer read.

    The chain is the record and the branch a view of it: an operator's commit on the
    run branch takes it off the chain, and the run's later commits land on the chain
    alone, so the branch would show a frozen prefix of the run.

    Args:
        manifest: The session's manifest.
        repo: The repository.

    Returns:
        The run branch while it exists and still covers the chain, else the chain ref
        while it has a tip, else "" (the run recorded nothing).
    """
    chain = git_ops.chain_ref_for(manifest.session_id)
    head = git_ops.chain_tip(repo, chain)
    branch = existing_run_branch(manifest, repo)
    if branch and (head is None or git_ops.is_ancestor(repo, head, branch)):
        return branch
    return chain if head is not None else ""


def manifest_branches(
    session_dir: pathlib.Path, *, repo: pathlib.Path | None = None
) -> dict[str, str]:
    """Return the run header's branch facts from the session's manifest.

    The event fold does not carry them, and an operator needs to see where a run's
    work lives and where Merge lands.

    Args:
        session_dir: The session's state dir.
        repo: The repository; with it, `run_branch` is claimed only while the branch
            exists and `merged_into` only while the merge stamp still describes it
            (a resumed run commits past its stamp).

    Returns:
        `run_branch`, `base_branch`, `merged_into`, `commits_ref` and `branch_line`
        (their one wording), each present when known; empty for a run with no manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return {}
    return branch_facts(manifest, repo)


def branch_facts(
    manifest: sessions_manifest.SessionManifest, repo: pathlib.Path | None
) -> dict[str, str]:
    """Return `manifest_branches` for a manifest already read.

    Args:
        manifest: The session's manifest.
        repo: The repository, or None to take the manifest's word.

    Returns:
        The branch facts, each present when known.
    """
    out: dict[str, str] = {}
    run_branch = existing_run_branch(manifest, repo)
    if run_branch:
        out["run_branch"] = run_branch
    if manifest.base_branch:
        out["base_branch"] = manifest.base_branch
    if repo is not None and (ref := commits_ref(manifest, repo)):
        out["commits_ref"] = ref
    stamp = manifest.merged
    merged_into = ""
    if stamp and stamp.into:
        holds = repo is None or git_ops.merge_stamp_holds(
            repo, manifest.session_id, manifest.run_branch or "", stamp.tip
        )
        merged_into = stamp.into if holds else ""
    if merged_into:
        out["merged_into"] = merged_into
    # The line names the manifest's branch when merged, pruned or not, else the existing one.
    named = (manifest.run_branch or "") if merged_into else run_branch
    line = format.format_branch(named, manifest.base_branch or "", merged_into)
    if line:
        out["branch_line"] = line
    return out


def manifest_header(
    session_dir: pathlib.Path, *, repo: pathlib.Path | None = None
) -> dict[str, Any]:
    """Return the session-header fields the event fold does not carry.

    Merged into every session snapshot, one-shot and streamed, so the header a page
    paints from cannot drift.

    Args:
        session_dir: The session's state dir.
        repo: The repository, or None to take the manifest's word on branches.

    Returns:
        The branch facts, `git_control`, `base_sha`, the fork lineage as `forked_from`,
        the `worktree` and the fan-out `compare` outcome with its `line`; empty for a
        run with no readable manifest.
    """
    try:
        m = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return {}
    header: dict[str, Any] = dict(branch_facts(m, repo))
    header["git_control"] = m.git_control
    header["base_sha"] = m.base_sha
    lineage = format.format_lineage(m.parent_session_id, m.forked_from_turn, m.forked_from_sha)
    if lineage:
        header["forked_from"] = lineage
    if m.worktree is not None:
        header["worktree"] = str(m.worktree)
    if m.compare is not None:
        header["compare"] = m.compare.model_dump(mode="json")
        line, _rationale = format.format_compare(m.compare) or ("", "")
        header["compare"]["line"] = line
    return header


class UnknownStepError(ValueError):
    """A step sha that is none of the run's commits."""


def session_snapshot(
    session_dir: pathlib.Path, *, repo: pathlib.Path | None = None, step: str = ""
) -> dict[str, Any]:
    """Fold a session's state into the wire dict.

    The dict carries the dir-aware status (parked, stale, waiting, not the fold's
    blanket "running"), the dir-backed identity fill and the manifest header. A
    session with no log yet (a parked submission, a `fork --no-run`) folds nothing
    and lets the dir supply the word.

    Args:
        session_dir: The session's state dir.
        repo: The repository, or None to take the manifest's word on branches.
        step: A commit sha of the run; folds only up to that commit and stamps `as_of`.

    Returns:
        The session snapshot.

    Raises:
        UnknownStepError: `step` is none of the run's commits.
    """
    events = tail.tail_events(session_dir / layout.LOGS_NAME, follow=False)
    as_of: dict[str, Any] | None = None
    if step:
        at = viewmodel_state.fold_until_commit(events, step)
        if at is None:
            raise UnknownStepError(f"no commit {step} in this run")
        state = at
        as_of = {"iteration": at.steps[-1].iteration, "sha": at.steps[-1].sha}
    else:
        state = viewmodel_state.fold_session(events)
    snap = viewmodel_state.session_state_as_dict(state, session_dir)
    snap["as_of"] = as_of
    snap.update(manifest_header(session_dir, repo=repo))
    return snap


def machine_snapshot(
    machine_dir: pathlib.Path, *, execution: machine_state.AgentExecution | None = None
) -> dict[str, Any]:
    """Fold a machine instance's state into the wire dict.

    Carries the instance's `spend`: a machine runs unattended against
    `[budget].max_usd`, so every surface that watches one reads the same figure.

    Args:
        machine_dir: The instance's dir.
        execution: The newest agent execution to fold in, when the caller has it.

    Returns:
        The machine snapshot.

    Raises:
        MachineError: The machine source cannot be loaded.
        JournalError: The journal is corrupt.
    """
    spec = load_machine(machine_dir / "machine.asm.toml")
    events = MachineJournal(machine_dir).read()
    ms = machine_state.fold_machine(spec, events)
    d = machine_state.machine_state_as_dict(ms, machine_dir, execution=execution)
    spend, in_flight = machine_state.machine_spend(
        events, machine_dir, alive=ipc.worker_is_alive(machine_dir)
    )
    d["spend"] = {
        "usd": spend.usd,
        "usd_partial": spend.partial,
        "text": budget.format_usd(spend.usd, partial=spend.partial),
        "input_tokens": spend.input_tokens,
        "output_tokens": spend.output_tokens,
        "in_flight_state": in_flight,
    }
    return d
