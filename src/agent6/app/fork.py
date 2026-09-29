# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 fork` lifecycle, and the `/undo` rewind built on the same clone.

A fork copies a source session's state as of a checkpoint into a fresh session
with a new id, recording the lineage; the source is never mutated. A run fork is
the repo at the checkpoint's committed HEAD in its own linked worktree (under
`[parallel].workdir`, recorded in the manifest, removed by `sessions prune` once
merged) plus the conversation up to that turn; an uncommitted edit at the forked
turn is absent from the tree. The DAG is rebuilt at the checkpoint's
`graph_version` by `graph.replay`, not copied. A plan or ask fork reads the
operator's checkout. `/undo` (`app/undo.py`) adds no worktree: the fork keeps the
undone session's checkout, put back to the checkpoint's tree.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import pathlib
import shutil
from collections.abc import Callable, Sequence
from typing import Any

from agent6 import git_ops, kinds, paths, portable
from agent6.app import _setup, fork_worktrees, manifest, parallel, resume
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError, layer
from agent6.graph import replay, storage
from agent6.harness import _snapshot
from agent6.sandbox import detect
from agent6.sessions import id, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import newest_session_dir

# The DAG artifacts copied verbatim when there is no version to rebuild at.
_DAG_ARTIFACTS: tuple[str, ...] = ("graph", "graph.jsonl", "cursor.json")


def resolve_source(
    state_dir: pathlib.Path, query: str, *, reporter: app_reporter.Reporter
) -> layout.SessionLayout | None:
    """Resolve the session to fork, across every resumable bucket.

    Args:
        state_dir: The repo's state directory.
        query: A session id or prefix; empty for the most recent session.
        reporter: Receives the error or the "most recent" note.

    Returns:
        The session's layout, or None after reporting why.
    """
    if not query:
        # The bucket set bare `resume` uses, so the two agree on "most recent".
        latest = newest_session_dir(resume.resumable_bucket_dirs(state_dir))
        if latest is None:
            reporter.err('nothing to fork yet. Start a session with `agent6 run "<task>"`.')
            return None
        query = latest.name
        reporter.note(f"forking most recent session: {query}")
    try:
        return id.resolve_session(state_dir, query)
    except id.SessionIdError as exc:
        reporter.error(str(exc))
        return None


def _copy_dag(src: layout.SessionLayout, dst: layout.SessionLayout, *, graph_version: int) -> None:
    """Write the destination's DAG as the source's stood at a graph version.

    The read holds the source curator's per-mutation lock, so a live source cannot
    tear it; a crashed source's lock releases with its process.

    Args:
        src: The source session.
        dst: The fork.
        graph_version: The version to rebuild at; 0 or less copies the DAG verbatim.
    """
    with storage.flock(src.lock_path):
        if graph_version <= 0:
            for name in _DAG_ARTIFACTS:
                src_path = src.session_dir / name
                if not src_path.exists():
                    continue
                dst_path = dst.session_dir / name
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True, symlinks=True)
                else:
                    shutil.copy2(src_path, dst_path)
            return
        nodes = storage.load_graph(src)
        journal = _read_journal(src)
        replayed = replay.graph_at_version(
            nodes, journal, graph_version, current_cursor=storage.read_cursor(src)
        )
    dst.ensure()
    for node in replayed.nodes.values():
        storage.write_node(dst, replayed.nodes, node)
    storage.write_cursor(dst, replayed.cursor)
    # The journal prefix, so the fork's curator numbers on from the version it holds.
    kept = replay.journal_prefix(journal, graph_version)
    portable.atomic_write(
        dst.journal_path, "".join(json.dumps(e, sort_keys=True) + "\n" for e in kept)
    )


def _read_journal(src: layout.SessionLayout) -> list[dict[str, Any]]:
    """Read the source's graph journal, skipping torn and non-object lines.

    Args:
        src: The source session.

    Returns:
        The journal entries in order; empty when the file is unreadable.
    """
    out: list[dict[str, Any]] = []
    try:
        text = src.journal_path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)  # pyright: ignore[reportUnknownArgumentType]
    return out


def _select_checkpoint_path(
    src: layout.SessionLayout,
    at_turn: int | None,
    *,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> pathlib.Path | None:
    """Resolve which snapshot of the source to fork from.

    Args:
        src: The source session.
        at_turn: A checkpoint turn from the per-turn store; None takes the rolling
            `loop_state.json`, the newest state.
        reporter: Receives the error.

    Returns:
        The snapshot path, or None after reporting why.
    """
    turns = storage.list_checkpoint_turns(src)
    if at_turn is None:
        rolling = src.session_dir / "loop_state.json"
        if rolling.is_file():
            return rolling
        if turns:
            return src.checkpoint_path(turns[-1])
        reporter.error(
            f"{src.session_id} has no checkpoints and no loop_state.json; nothing to fork."
        )
        return None
    if at_turn in turns:
        return src.checkpoint_path(at_turn)
    avail = ", ".join(str(t) for t in turns) or "none"
    reporter.error(
        f"no checkpoint at turn {at_turn} for {src.session_id}. Available turns: {avail}"
    )
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class Checkout:
    """A fork's own checkout, recorded so its jail never reads the worktree's `.git` pointer.

    Attributes:
        worktree: The linked worktree.
        git_dir: The repository git dir the worktree points into.
    """

    worktree: pathlib.Path
    git_dir: pathlib.Path


class _ForkRefusedError(Exception):
    """The fork was refused before anything was created; the reason is reported.

    Attributes:
        rc: The exit code.
    """

    def __init__(self, rc: int) -> None:
        super().__init__(rc)
        self.rc = rc


@dataclasses.dataclass(frozen=True, slots=True)
class _ForkPlan:
    """Everything a fork writes, resolved first so a refusal creates nothing.

    Attributes:
        src: The source session.
        dst: The fork's layout.
        checkpoint_path: The snapshot the fork seeds from.
        graph_version: The DAG version to rebuild at.
        forked_from_turn: The turn the fork continues from.
        forked_from_sha: The committed HEAD at that turn.
        base_sha: The source's base commit, carried forward.
        base_branch: The source's base branch, carried forward.
        user_task: The source's task, carried forward.
        mode: The source's mode, kept.
        preset: The child's preset as the continuation stamps it.
        preset_from_flag: Whether the source's preset came from a flag.
        driver_from_flag: Whether the source's driver came from `--model`.
        cfg: The child's config.
        gate: The source's pinned verify command and its origin, inherited since an
            inferred or adopted gate has no config to derive from.
    """

    src: layout.SessionLayout
    dst: layout.SessionLayout
    checkpoint_path: pathlib.Path
    graph_version: int
    forked_from_turn: int
    forked_from_sha: str
    base_sha: str
    base_branch: str
    user_task: str
    mode: kinds.ResumableMode
    preset: str
    preset_from_flag: bool
    driver_from_flag: bool
    cfg: Config
    gate: tuple[Sequence[str], str]


def _plan_fork(
    config_path: pathlib.Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    cwd: pathlib.Path,
    sandbox_overrides: _setup.SandboxOverrides | None = None,
    refuse_continuation: Callable[[Config, str], str | None] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> _ForkPlan:
    """Resolve a fork of a source session at a checkpoint.

    The child's config is built as its continuation builds it, so the manifest
    stamps the policy the fork runs under.

    Args:
        config_path: An explicit config file, or None for discovery.
        source_session_id: The source session's id; empty for the most recent.
        at_turn: The checkpoint turn; None takes the newest state.
        new_session_id: An explicit id for the fork; empty allocates one.
        cwd: The repository.
        sandbox_overrides: This invocation's sandbox flags.
        refuse_continuation: Given the child's config and mode, returns why the
            continuation would refuse, or None; a reason refuses before anything is
            created, so the id stays free.
        reporter: Receives each refusal's reason.

    Returns:
        The resolved plan.

    Raises:
        _ForkRefusedError: The fork was refused; the reason is reported.
    """
    state = paths.state_dir(cwd)
    src = resolve_source(state, source_session_id, reporter=reporter)
    if src is None:
        raise _ForkRefusedError(2)

    checkpoint_path = _select_checkpoint_path(src, at_turn, reporter=reporter)
    if checkpoint_path is None:
        raise _ForkRefusedError(2)

    try:
        checkpoint = _snapshot.load_session_snapshot(checkpoint_path)
    except (OSError, ValueError) as exc:
        reporter.error(f"failed to load checkpoint {checkpoint_path}: {exc}")
        raise _ForkRefusedError(1) from exc

    # A damaged manifest fails loud: the mode must never fall open to "run".
    try:
        sm = sessions_manifest.read_manifest(src.session_dir)
        src_mode = sm.session_mode()
    except sessions_manifest.ManifestError as exc:
        reporter.error(f"cannot read source run manifest {src.manifest_path}: {exc}")
        raise _ForkRefusedError(2) from exc
    refusal = sessions_manifest.model_git_refusal(sm, "fork")
    if refusal is not None:
        reporter.error(refusal)
        raise _ForkRefusedError(2)

    forked_from_sha = checkpoint.head_sha
    if not forked_from_sha:
        reporter.error(
            "the chosen checkpoint records no head_sha, so the fork branch cannot be cut."
        )
        raise _ForkRefusedError(1)

    try:
        # The child's stamp derives from the same preset-resolved config resume replays.
        cfg = layer.load_effective(cwd, config_path, preset=sm.harness.replay_preset).config
        recorded = sm.models.replay_driver
        route = (
            kinds.ModelRoute(recorded.provider, recorded.model) if recorded is not None else None
        )
        cfg = _setup.session_config(
            cfg.with_model_route(kinds.session_kind(src_mode).role, route) if route else cfg,
            src_mode,
            sandbox_overrides,
        )
    except ConfigError as exc:
        reporter.error(str(exc))
        raise _ForkRefusedError(2) from exc
    if refuse_continuation is not None:
        refusal = refuse_continuation(cfg, src_mode)
        if refusal is not None:
            reporter.refuse(refusal)
            raise _ForkRefusedError(2)

    if new_session_id:
        try:
            id.validate_explicit_session_id(new_session_id)
        except id.SessionIdError as exc:
            reporter.error(str(exc))
            raise _ForkRefusedError(2) from exc
        # Ids are unique across buckets, so any bucket holding it refuses.
        if (held := id.session_id_bucket(state, new_session_id)) is not None:
            reporter.error(
                f"--session-id {new_session_id!r} already names a session under {held}/;"
                " ids are unique across every bucket. Pick another id."
            )
            raise _ForkRefusedError(2)
    child_id = new_session_id or id.unused_session_id(state, kinds.session_bucket(src_mode))
    return _ForkPlan(
        src=src,
        # A fork keeps its source's mode, so its dir belongs in that mode's bucket.
        dst=layout.SessionLayout(
            state_dir=state, session_id=child_id, subdir=kinds.session_bucket(src_mode)
        ),
        checkpoint_path=checkpoint_path,
        graph_version=checkpoint.graph_version,
        forked_from_turn=checkpoint.next_iteration,
        forked_from_sha=forked_from_sha,
        base_sha=sm.base_sha,
        base_branch=sm.base_branch,
        user_task=sm.user_task,
        mode=src_mode,
        # A flag-selected preset replays its name; a config-selected one re-derives.
        preset=sm.harness.replay_preset or cfg.preset,
        preset_from_flag=sm.harness.preset_from_flag,
        driver_from_flag=sm.models.driver_from_flag,
        cfg=cfg,
        gate=(sm.harness.verify_command, sm.harness.verify_origin),
    )


def create_fork(
    config_path: pathlib.Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    cwd: pathlib.Path,
    sandbox_overrides: _setup.SandboxOverrides | None = None,
    refuse_continuation: Callable[[Config, str], str | None] | None = None,
    worktree: bool = True,
    checkout: Checkout | None = None,
    checkout_untracked: frozenset[str] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> tuple[str, int]:
    """Create a new session cloned from a source at a checkpoint, without starting it.

    Writes the cloned checkpoint and DAG, the manifest, the refs cut at the
    checkpoint's committed HEAD, the lineage record and, for a run fork with
    `worktree` set, a linked worktree detached at that sha. A plan or ask fork gets
    no worktree either way.

    Args:
        config_path: An explicit config file, or None for discovery.
        source_session_id: The source session's id; empty for the most recent.
        at_turn: The checkpoint turn; None takes the newest state.
        new_session_id: An explicit id for the fork; empty allocates one.
        cwd: The repository.
        sandbox_overrides: This invocation's sandbox flags.
        refuse_continuation: Why the continuation would refuse, see `_plan_fork`.
        worktree: Whether a run fork gets its own linked worktree.
        checkout: With `worktree` off, the checkout the child works in; None for the
            operator's.
        checkout_untracked: The operator's untracked files in that checkout, which
            the source's set need not describe.
        reporter: Receives the refusal or the "forked" note.

    Returns:
        `(child_id, 0)` on success, else `("", rc)` after reporting the reason.
    """
    try:
        plan = _plan_fork(
            config_path,
            source_session_id,
            at_turn=at_turn,
            new_session_id=new_session_id,
            cwd=cwd,
            sandbox_overrides=sandbox_overrides,
            refuse_continuation=refuse_continuation,
            reporter=reporter,
        )
    except _ForkRefusedError as refused:
        return "", refused.rc
    added = worktree and plan.mode == "run"
    if added and plan.cfg.git.control == "model":
        reporter.error(
            "a fork runs in a linked worktree whose .git is read-only in the jail; under"
            ' [git].control = "model" the model could not commit there. Set control ='
            ' "agent6" for the fork, or take the run back with /undo in its checkout.'
        )
        return "", 2
    if added:
        path = parallel.subordinate_workdir_root(plan.cfg, cwd, plan.dst.session_id)
        try:
            # The git dir a fork execution's jail grants from.
            checkout = Checkout(path, git_ops.git_common_dir(cwd))
            git_ops.add_worktree(cwd, path, plan.forked_from_sha)
        except git_ops.GitError as exc:
            reporter.error(f"could not add the fork's worktree at {path}: {exc}")
            return "", 1
    rc = _materialize_fork(
        plan,
        cwd=cwd,
        checkout=checkout,
        fresh_checkout=added,
        checkout_untracked=checkout_untracked,
        reporter=reporter,
    )
    if rc != 0:
        if added and checkout is not None:
            fork_worktrees.remove_fork_worktree(cwd, checkout.worktree, (plan.forked_from_sha,))
        return "", rc
    return plan.dst.session_id, 0


def _materialize_fork(
    plan: _ForkPlan,
    *,
    cwd: pathlib.Path,
    checkout: Checkout | None,
    fresh_checkout: bool,
    checkout_untracked: frozenset[str] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Write the fork's state on disk; the source is never touched.

    Args:
        plan: The resolved fork.
        cwd: The repository.
        checkout: The worktree the fork works in and its git dir; None for the
            operator's checkout.
        fresh_checkout: Whether the checkout is a fresh worktree of the forked sha.
        checkout_untracked: The operator's untracked files in an existing checkout.
        reporter: Receives the error or the "forked" note.

    Returns:
        0 on success, else the exit code after reporting the reason.
    """
    src, dst = plan.src, plan.dst
    if dst.session_dir.exists():
        reporter.error(f"target run dir already exists: {dst.session_dir}")
        return 2
    dst.ensure()

    # The resume pointer and origin checkpoint, then the DAG as of the checkpoint.
    blob = plan.checkpoint_path.read_text(encoding="utf-8")
    portable.atomic_write(dst.session_dir / "loop_state.json", blob)
    portable.atomic_write(dst.checkpoint_path(0), blob)
    _copy_dag(src, dst, graph_version=plan.graph_version)
    # The files the fork's commits leave out: observed in a fresh worktree, given for an
    # existing checkout (where the run's own work still reads untracked), else the source's.
    if fresh_checkout and checkout is not None:
        excluded = git_ops.untracked_paths(checkout.worktree)
    elif checkout_untracked is not None:
        excluded = checkout_untracked
    else:
        excluded = layout.read_untracked_at_start(src.session_dir)
    layout.write_untracked_at_start(dst.session_dir, excluded)

    run_branch = (
        git_ops.run_branch_for(dst.session_id)
        if plan.mode == "run" and plan.cfg.git.branch_per_run
        else None
    )
    manifest.write_session_manifest(
        dst,
        session_id=dst.session_id,
        user_task=plan.user_task,
        base_sha=plan.base_sha,
        base_branch=plan.base_branch,
        run_branch=run_branch,
        cfg=plan.cfg,
        mode=plan.mode,
        effective_preset=plan.preset,
        preset_from_flag=plan.preset_from_flag,
        driver_from_flag=plan.driver_from_flag,
        parent_session_id=src.session_id,
        forked_from_turn=plan.forked_from_turn,
        forked_from_sha=plan.forked_from_sha,
        gate=plan.gate,
        isolation=detect.resolve_isolation(plan.cfg.sandbox.isolation, _setup.detect_env()),
        worktree=checkout.worktree if checkout is not None else None,
        worktree_git_dir=checkout.git_dir if checkout is not None else None,
    )

    # Additive ref writes only, never a checkout; plan and ask sessions own no refs.
    try:
        # The branch can refuse (it may exist at another sha), so it goes first.
        if run_branch is not None:
            git_ops.create_branch_at(cwd, run_branch, plan.forked_from_sha)
        if plan.mode == "run":
            git_ops.set_ref(cwd, git_ops.chain_ref_for(dst.session_id), plan.forked_from_sha)
    except git_ops.GitError as exc:
        reporter.error(f"could not cut fork refs at {plan.forked_from_sha[:12]}: {exc}")
        # A fork exists only with its refs; a chain ref under this id predates the fork.
        shutil.rmtree(dst.session_dir, ignore_errors=True)
        return 1

    storage.append_jsonl(
        src.state_dir / "lineage.jsonl",
        {
            "child": dst.session_id,
            "parent": src.session_id,
            "turn": plan.forked_from_turn,
            "sha": plan.forked_from_sha,
            "ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        },
    )
    at = (
        f"(branch {run_branch} "
        if run_branch
        else f"({git_ops.chain_ref_for(dst.session_id)} "
        if plan.mode == "run"
        else "("
    )
    where = f" in {checkout.worktree}" if checkout is not None else ""
    reporter.note(
        f"forked {src.session_id}@turn {plan.forked_from_turn} -> {dst.session_id} "
        f"{at}at {plan.forked_from_sha[:12]}){where}"
    )
    return 0
