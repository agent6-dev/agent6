# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a fresh run and claim the checkout in which it will run."""

from __future__ import annotations

import dataclasses
import os
import pathlib

from agent6 import directive, errors, kinds, paths
from agent6.app import (
    fork_worktrees,
    preflight,
    stamps,
    worktree_setup,
    worktrees,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.parallel import launch as child_launch
from agent6.app.review import directive as review_directive
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import model as config_model
from agent6.sessions import id
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing


@dataclasses.dataclass(frozen=True, slots=True)
class RunStart:
    """What a fresh session's checks resolved, and the checkout it claims.

    Attributes:
        cfg: The run's config: clamped to the mode, and once a worktree is claimed, with the
            paths hidden in it.
        task: The task, composed for a `/review`.
        operator_task: The words before composition.
        mode: The session mode.
        repo: The repository the run was started in.
        cwd: The run's checkout: the repository, or once claimed the worktree added for it.
        worktree: Whether the run works in its own worktree.
        worktree_git_dir: The repository git dir a linked worktree points into: the
            repository's own until a worktree is claimed.
        review_world: The resolved comparison of a `/review` task; None for any other.
        tui_enabled: Whether the TUI owns the terminal.
        base_sha: The commit the run starts from; "" without a head.
        base_branch: The branch the run starts from.
        state: The state directory the execution runs with.
        layout: The session's directory layout.
    """

    cfg: config_model.Config
    task: str
    operator_task: str
    mode: kinds.LoopMode
    repo: pathlib.Path
    cwd: pathlib.Path
    worktree: bool
    worktree_git_dir: pathlib.Path | None
    review_world: review_directive.ReviewWorld | None
    tui_enabled: bool
    base_sha: str
    base_branch: str
    state: pathlib.Path
    layout: sessions_layout.SessionLayout

    @property
    def run_branch(self) -> str | None:
        """The branch the run's first chain commit advances; nothing is cut or checked out."""
        cfg = self.cfg
        if cfg.git.branch_per_run and self.mode == "run" and cfg.git.control != "model":
            return git_read.run_branch_for(self.layout.session_id)
        return None


def resolve_review(
    repo: pathlib.Path,
    task: str,
    cfg: config_model.Config,
    *,
    worktree: bool,
    reporter: app_reporter.Reporter,
) -> review_directive.ReviewWorld | None:
    """Resolve a `/review` task into its comparison.

    Args:
        repo: The repository.
        task: The task text.
        cfg: The run's config, for the hidden paths.
        worktree: Whether `--worktree` was asked for.
        reporter: Receives the refusal, or the line of a review with nothing to review.

    Returns:
        The comparison; None for any other task.

    Raises:
        preflight.SessionRefusedError: The directive is refused, or names no ref under
            `--worktree`; a comparison with nothing to review raises exit 0.
    """
    review_target = directive.parse_review(task)
    if review_target is None:
        return None
    try:
        review_world = review_directive.build(repo, review_target, hide=cfg.sandbox.hide_paths)
    except review_directive.ReviewDirectiveError as exc:
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    if review_world.empty:
        reporter.out(review_world.message)
        raise preflight.SessionRefusedError(0)
    if worktree and not review_target:
        reporter.refuse("/review needs this working tree; drop --worktree or name a ref")
        raise preflight.SessionRefusedError(2)
    return review_world


def start_base(
    repo: pathlib.Path,
    cfg: config_model.Config,
    mode: kinds.LoopMode,
    review_world: review_directive.ReviewWorld | None,
    *,
    worktree: bool,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
) -> tuple[str, str]:
    """Run the git preflight, and find where the run starts from.

    Args:
        repo: The repository.
        cfg: The run's config.
        mode: The session mode.
        review_world: The comparison of a `/review` task, whose head is the start.
        worktree: Whether the run works in its own worktree.
        frontend: The surface that confirms a start on another run's branch.
        reporter: Receives the refusal.

    Returns:
        The commit and the branch the run starts from.

    Raises:
        preflight.SessionRefusedError: The git preflight refused, or `--worktree` has no
            commit to start from.
    """
    git = preflight.git_preflight(
        repo,
        cfg,
        mode,
        confirm_run_on_run_branch=frontend.confirm_run_on_run_branch,
        reporter=reporter,
    )
    base_sha, base_branch = git.base_sha, git.base_branch
    if review_world is not None:
        base_sha = review_world.target.head_commit or review_world.target.base_commit or ""
        base_branch = review_world.head_branch
    if worktree and not base_sha:
        reporter.refuse(
            "--worktree starts from HEAD's commit, and this repository has none. Commit first"
            ' (`git add -A && git commit -m "initial commit"`), or run without --worktree.'
        )
        raise preflight.SessionRefusedError(2)
    return base_sha, base_branch


def session_slot(
    record: pathlib.Path,
    session_id: str,
    mode: kinds.LoopMode,
    *,
    reporter: app_reporter.Reporter,
) -> sessions_layout.SessionLayout:
    """Take the session's id: the explicit one, else an unused one.

    Args:
        record: The state directory the session's record goes under.
        session_id: An explicit id; empty allocates one.
        mode: The session mode, which names the bucket.
        reporter: Receives the refusal.

    Returns:
        The session's directory layout.

    Raises:
        preflight.SessionRefusedError: The explicit id names a session in another bucket,
            or one in this bucket that is not parked.
    """
    bucket = kinds.session_bucket(mode)
    # Same-bucket reuse is the resume or park flow below; another bucket's id is a collision.
    if session_id and (other := id.session_id_bucket(record, session_id)) not in (None, bucket):
        reporter.error(
            f"--session-id {session_id!r} already names a session under {other}/;"
            " ids are unique across every bucket. Pick another id."
        )
        raise preflight.SessionRefusedError(2)
    layout = sessions_layout.SessionLayout(
        state_dir=record,
        session_id=session_id or id.unused_session_id(record, bucket),
        subdir=bucket,
    )
    # An existing session under the id is a resume, not a fresh start; a parked run and an
    # ask are the reusable dirs.
    if not session_id or mode == "ask" or not layout.manifest_path.exists():
        return layout
    try:
        parked = sessions_manifest.read_manifest(layout.session_dir).parked_task
    except sessions_manifest.ManifestError:
        parked = ""
        resume = ""
    else:
        resume = (
            f'`agent6 resume {session_id} --steer "<what to do next>"` to give it new work'
            if listing.finished_needs_new_work(layout.session_dir)
            else f"`agent6 resume {session_id}` to continue it"
        )
    if parked:
        return layout
    next_step = (
        f" Use {resume}, or choose a different --session-id."
        if resume
        else " Choose a different --session-id."
    )
    reporter.error(f"run {session_id!r} already exists.{next_step}")
    raise preflight.SessionRefusedError(2)


def setup_and_gate(
    cfg: config_model.Config,
    repo: pathlib.Path,
    review_world: review_directive.ReviewWorld | None,
    *,
    base_sha: str,
    worktree: bool,
    opening: directive.Segment | tuple[directive.Segment, ...] | None,
    reporter: app_reporter.Reporter,
) -> config_model.Config:
    """Read the setup command a new worktree or a child's clone needs, and check its gate.

    Args:
        cfg: The run's config.
        repo: The repository.
        review_world: The comparison of a `/review` task, which carries its own gate.
        base_sha: The commit the run starts from.
        worktree: Whether the run works in its own worktree.
        opening: The tasks the run dispatches to children's clones.
        reporter: Receives the refusal.

    Returns:
        The config, with the setup command when it was unset.

    Raises:
        preflight.SessionRefusedError: The gate cannot run on tracked files alone.
    """
    if worktree or stamps.parallel_lineage() is not None:
        cfg = worktree_setup.read_if_unset(cfg, repo, reporter=reporter, at=base_sha)
    # A new worktree and a child's clone hold tracked files only: a gate neither can run refuses.
    if (worktree or opening) and (
        gate_refusal := worktree_setup.gate_refusal(
            cfg,
            repo,
            cfg.harness.verify.command
            if review_world is not None
            else worktree_setup.origin_gate(cfg, repo),
            worktree="a new worktree" if worktree else "a child's clone",
            at=base_sha,
        )
    ):
        reporter.refuse(gate_refusal)
        raise preflight.SessionRefusedError(2)
    return cfg


def check_run(
    cfg: config_model.Config,
    task: str,
    *,
    frontend: app_frontend.SessionFrontend,
    repo: pathlib.Path,
    operator_task: str,
    session_id: str,
    interactive: bool,
    tui: bool,
    mode: kinds.LoopMode,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    model: str | kinds.ModelRoute | None,
    worktree: bool,
    opening: directive.Segment | tuple[directive.Segment, ...] | None,
    reporter: app_reporter.Reporter,
) -> tuple[RunStart, kinds.IsolationLevel | None]:
    """Run every check a fresh session passes before anything is created.

    Args:
        cfg: The run's config, overrides applied.
        task: The task text.
        frontend: The surface's seam.
        repo: The repository directory.
        operator_task: The words before composition.
        session_id: An explicit id; empty allocates one.
        interactive: Whether the ask REPL drives the session.
        tui: Whether `--tui` was asked for.
        mode: The session mode.
        sandbox_overrides: The sandbox flags.
        model: The `--model` value that set the driving role.
        worktree: Whether `--worktree` was asked for.
        opening: The tasks the run dispatches in place of its first model call.
        reporter: Receives the refusal.

    Returns:
        What the checks resolved, and the isolation selected on the repository: None for a
        run in its own worktree, which selects there.

    Raises:
        preflight.SessionRefusedError: A check refused; an empty `/review` raises exit 0.
    """
    if (repo / ".git").is_file() and not preflight.require_git_repo(repo, reporter=reporter):
        raise preflight.SessionRefusedError(2)
    # A startup grant is recorded before model commands can rewrite the pointer under partial.
    worktree_git_dir = paths.linked_worktree_git_dir(repo)
    # A first prompt is a task, not a composer line.
    if (problem := directive.steer_problem(task)) is not None:
        reporter.error(problem)
        raise preflight.SessionRefusedError(2)
    if session_id:
        try:
            id.validate_explicit_session_id(session_id)
        except id.SessionIdError as exc:
            reporter.error(str(exc))
            raise preflight.SessionRefusedError(2) from exc
    # The clamp comes before anything reads a knob; the invocation's grant lands after it.
    cfg = session_setup.session_config(cfg, mode, sandbox_overrides)
    review_world = resolve_review(repo, task, cfg, worktree=worktree, reporter=reporter)
    if review_world is not None:
        worktree = worktree or review_world.needs_worktree
        task = review_world.message
    if worktree and (top := paths.checkout_root(repo)) != repo.resolve():
        reporter.refuse(
            f"--worktree starts at the repository's top level, and this is a subdirectory of"
            f" {top}. Run it from there, or run without --worktree."
        )
        raise preflight.SessionRefusedError(2)
    # Refuse an unanswerable run before anything is created, so its id stays free. A fresh run
    # has no dir yet, so the env is the only away answer.
    tui_enabled = session_prepare.headless_check(
        cfg,
        frontend,
        tui=tui,
        interactive=interactive,
        mode=mode,
        away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
        reporter=reporter,
    )
    # Before isolation: its budget preflight prices the model from the refreshed listing.
    if not preflight.route_preflight(
        cfg,
        kinds.session_kind(mode).role,
        reporter=reporter,
        model_flag=session_setup.route_text(model),
    ):
        raise preflight.SessionRefusedError(2)
    # Before the git preflight. A worktree run selects in `claim_checkout`, once the worktree
    # its commands run in exists.
    isolation = (
        None
        if worktree
        else session_prepare.hold_workspace_and_select_isolation(
            cfg, repo, frontend=frontend, reporter=reporter, worktree_git_dir=worktree_git_dir
        )
    )
    base_sha, base_branch = start_base(
        repo, cfg, mode, review_world, worktree=worktree, frontend=frontend, reporter=reporter
    )
    record, state = session_prepare.record_dirs(repo, child_launch.read(), reporter=reporter)
    layout = session_slot(record, session_id, mode, reporter=reporter)
    cfg = setup_and_gate(
        cfg,
        repo,
        review_world,
        base_sha=base_sha,
        worktree=worktree,
        opening=opening,
        reporter=reporter,
    )
    start = RunStart(
        cfg=cfg,
        task=task,
        operator_task=operator_task,
        mode=mode,
        repo=repo,
        cwd=repo,
        worktree=worktree,
        worktree_git_dir=worktree_git_dir,
        review_world=review_world,
        tui_enabled=tui_enabled,
        base_sha=base_sha,
        base_branch=base_branch,
        state=state,
        layout=layout,
    )
    return start, isolation


def add_worktree(
    start: RunStart, held: session_prepare.Held, *, reporter: app_reporter.Reporter
) -> RunStart:
    """Add the run's own worktree, at the commit the run starts from.

    Args:
        start: What the checks resolved.
        held: The teardown record, which takes the worktree.
        reporter: Receives the refusal.

    Returns:
        The start in its worktree, with the paths hidden there.

    Raises:
        preflight.SessionRefusedError: The worktree could not be placed or added.
    """
    try:
        path = worktrees.place(start.cfg, start.repo, start.layout.session_id)
        checkout = fork_worktrees.add_session_worktree(
            start.repo, path, start.base_sha, start.cfg.git.worktree.kind
        )
    except errors.OperatorError as exc:
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    except (errors.GitError, OSError) as exc:
        reporter.error(f"could not add the run's worktree: {exc}")
        raise preflight.SessionRefusedError(1) from exc
    # A start that ends before its manifest records the worktree drops it.
    held.added_worktree = session_prepare.AddedWorktree(
        start.repo, checkout.worktree, start.base_sha
    )
    return dataclasses.replace(
        start,
        cwd=checkout.worktree,
        cfg=start.cfg.with_hidden_in_checkout(start.repo, checkout.worktree),
        worktree_git_dir=checkout.git_dir,
    )


def isolation_in_worktree(
    start: RunStart, *, frontend: app_frontend.SessionFrontend, reporter: app_reporter.Reporter
) -> kinds.IsolationLevel:
    """Select the isolation on the run's own worktree.

    Args:
        start: The start in its worktree.
        frontend: The surface that confirms an unconfined autorun.
        reporter: Receives the refusal.

    Returns:
        The isolation level.

    Raises:
        preflight.SessionRefusedError: The level was refused, or the model's git could not
            commit in a linked worktree under it.
    """
    isolation = session_prepare.hold_workspace_and_select_isolation(
        start.cfg,
        start.cwd,
        frontend=frontend,
        reporter=reporter,
        worktree_git_dir=start.worktree_git_dir,
    )
    if (
        start.cfg.git.control == "model"
        and isolation != "none"
        and start.worktree_git_dir is not None
    ):
        # The repository git dir lies outside the worktree: read-only to jailed commands.
        reporter.refuse(
            "--worktree runs in a linked worktree whose .git is read-only to commands"
            f' under sandbox.isolation = {isolation!r}; under [git].control = "model"'
            ' the model could not commit there. Set [git].control = "agent6"'
            " (`agent6 config set --repo git.control agent6`), set git.worktree.kind ="
            ' "clone", or run without --worktree.'
        )
        raise preflight.SessionRefusedError(2)
    return isolation


def claim_checkout(
    start: RunStart,
    isolation: kinds.IsolationLevel | None,
    held: session_prepare.Held,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
) -> tuple[RunStart, kinds.IsolationLevel]:
    """Claim the run's checkout and its session: the worktree, its isolation, the writer lock.

    Args:
        start: What the checks resolved.
        isolation: The isolation selected on the repository; None for a worktree run.
        held: The teardown record, which takes the worktree and the lock.
        frontend: The surface that confirms an unconfined autorun.
        reporter: Receives the refusal and the worktree's note.

    Returns:
        The start in its checkout, and the isolation its commands run under.

    Raises:
        preflight.SessionRefusedError: The worktree, its isolation or the writer lock refused.
    """
    if start.worktree:
        start = add_worktree(start, held, reporter=reporter)
        isolation = isolation_in_worktree(start, frontend=frontend, reporter=reporter)
    assert isolation is not None  # selected before the git preflight, or on the worktree
    # One writer per session dir, taken before any shared state is touched.
    held.worker_lock_fd = session_prepare.take_single_writer(start.layout, reporter=reporter)
    if start.worktree:
        reporter.note(
            f"working in the worktree {start.cwd} (at {start.base_sha[:12]}); this checkout and"
            " its uncommitted changes stay as they are"
        )
    return start, isolation
