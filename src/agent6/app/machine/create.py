# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Draft a machine and its scripts from a task, for `agent6 machine create`.

Each attempt is validated (structure, bundle, lint, offline tests, dry run) and the first
valid draft is written. Authoring runs the confined agent subprocess a machine's `agent` state
uses, over a drafting workspace of its own; agent6 validates what is on disk between attempts.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable

from agent6 import errors, kinds, path_safety, paths, portable
from agent6.app import preflight, worktrees
from agent6.app import reporter as app_reporter
from agent6.app.machine import agent, bundle
from agent6.app.machine import draft as drafting
from agent6.app.session import runtime as session_runtime
from agent6.app.session import setup as session_setup
from agent6.config import layer, validation
from agent6.config import model as config_model
from agent6.machine import schema
from agent6.sessions import event_log, id, journal, layout, lock, process
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import execution_settings


def _write_scripts(base_dir: pathlib.Path, scripts: dict[str, str]) -> None:
    """Write the bundle's scripts, keyed by bundle-relative path.

    Each script is written beside its target and renamed over it, owner-only like a new machine
    file. Both steps walk descriptors from the bundle's root (`path_safety.open_contained`), so
    a symlink at `scripts` or at a directory below it refuses the write; a symlink at a script's
    own name is replaced.

    Args:
        base_dir: The bundle's root, the directory the machine file goes in.
        scripts: Each script's source, written ending in a newline.

    Raises:
        ContainmentError: A directory on a script's path is a symlink or a file.
    """
    for rel, content in scripts.items():
        target = pathlib.Path(rel)
        tmp = path_safety.contain(
            base_dir, target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        )
        try:
            # Opening the temp file creates the missing directories on the same walk.
            fd = path_safety.open_contained(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, create_parents=True
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                os.fchmod(fd, 0o600)
                fh.write(content if content.endswith("\n") else content + "\n")
                fh.flush()
                os.fsync(fd)
            dir_fd = path_safety.open_contained(
                path_safety.contain(base_dir, target.parent), os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.replace(tmp.rel_path.name, target.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except path_safety.ContainmentError as exc:
            raise path_safety.ContainmentError(
                f"{rel}: a directory on its path is a symlink or a file"
            ) from exc
        finally:
            # After the rename the temp name is gone; a failure before it leaves one to remove.
            with contextlib.suppress(OSError, path_safety.ContainmentError):
                path_safety.unlink_contained(tmp)


def _discard_workspace(workspace: pathlib.Path, reporter: app_reporter.Reporter) -> None:
    """Remove a drafting workspace whose bundle is published, with its per-repo state dir.

    The prune sweep knows child and machine clones and fork worktrees, not drafting workspaces.

    Args:
        workspace: The drafting workspace.
        reporter: Receives the note when a removal fails.
    """
    errors: list[str] = []
    for path in (paths.state_dir(workspace), workspace):
        # A workspace that never ran has no state dir; rmtree reports a missing path through onexc.
        if path.exists():
            shutil.rmtree(path, onexc=lambda _fn, target, exc: errors.append(f"{target}: {exc}"))
    # The per-repo base every child's worktree sits in goes only when empty.
    with contextlib.suppress(OSError):
        workspace.parent.rmdir()
    if errors:
        reporter.err(f"machine create: the drafting workspace stays ({errors[0]})")


def _pass_env_to_allow(spec: schema.MachineSpec, allowed: tuple[str, ...]) -> list[str]:
    """Return the variables the machine's tools name that `[machine].pass_env` lacks, sorted."""
    asked = {
        name
        for state in spec.states.values()
        if isinstance(state, schema.ToolState)
        for name in state.pass_env
    }
    return sorted(asked - set(allowed))


def new_draft_dir(state: pathlib.Path) -> pathlib.Path:
    """Return a new draft's directory: an unused session id under the machine bucket."""
    bucket = kinds.session_bucket("machine")
    return layout.bucket_dir(state, bucket) / id.unused_session_id(state, bucket)


def _open_draft(
    task: str,
    output: pathlib.Path | None,
    config_paths: tuple[pathlib.Path, ...],
    budget_overrides: session_setup.BudgetOverrides | None,
    reporter: app_reporter.Reporter,
    confirm_unconfined: Callable[[kinds.IsolationLevel, config_model.Config], bool],
) -> drafting.Draft:
    """Check the config and open the draft journal before preparing the workspace.

    Args:
        task: The operator's prompt.
        output: The requested machine file.
        config_paths: Explicit config layers.
        budget_overrides: Budget flags for the draft.
        reporter: The output channels.
        confirm_unconfined: Confirms an unconfined autorun with the operator.

    Returns:
        The opened draft.

    Raises:
        SessionRefusedError: A preflight check refused the draft.
    """
    cwd = pathlib.Path.cwd()
    try:
        eff = layer.load_effective(cwd, config_paths)
        cfg = eff.config
        if budget_overrides is not None:
            cfg = budget_overrides.apply(cfg)
        cfg.require_runnable("worker")
    except validation.ConfigError as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    missing = session_setup.check_provider_keys(cfg)
    if missing is not None:
        reporter.err(missing)
        raise preflight.SessionRefusedError(2)
    if not lock.hold_workspace(cwd):
        reporter.refuse(lock.WORKSPACE_PACKING.format(root=cwd))
        raise preflight.SessionRefusedError(2)
    isolation = session_runtime.select_isolation(
        cfg, cwd=cwd, confirm_unconfined=confirm_unconfined, reporter=reporter
    )
    scratch = new_draft_dir(paths.state_dir(cwd))
    paths.mkdir_for_real_user(scratch.parent)
    scratch.mkdir(mode=0o700)
    (scratch / "prompt.txt").write_text(task, encoding="utf-8")
    # The draft's watchable log: this process owns session.start, the attempt markers and
    # session.stop; each attempt's subprocess appends its own events to the same file.
    events_log = scratch / layout.LOGS_NAME
    execution_id = uuid.uuid4().hex
    events = event_log.EventSink(events_log)
    # A terminal draft has its own session.stop, which the status reads first: no clearing.
    process.emit_session_start(
        events,
        scratch,
        journal.SessionStart(
            execution_id=execution_id, user_task=task, task_source="operator", mode="machine"
        ),
        settings=execution_settings.observations(cfg, eff),
    )
    reporter.err(
        f"machine create: drafting as {scratch.name} (follow live: agent6 attach {scratch.name})"
    )
    # An empty repo of its own beside the children's worktrees: the jail masks the state dir.
    workspace = worktrees.child_worktree_root(cfg, cwd, scratch.name)
    return drafting.Draft(
        workspace, events, execution_id, reporter, cwd, output, cfg, isolation, config_paths
    )


def _prepare_runner(draft: drafting.Draft) -> drafting.Runner:
    """Prepare the confined authoring execution and journal any refusal.

    Args:
        draft: The session receiving the attempt logs.

    Returns:
        The authoring runner.

    Raises:
        SessionRefusedError: The workspace or commit identity could not be prepared.
    """
    workspace = draft.workspace
    cfg = draft.cfg
    try:
        workspace.mkdir(parents=True)
        git_ops.init_repo(workspace)
    except (OSError, errors.GitError) as exc:
        draft.reporter.error(f"could not prepare the drafting workspace {workspace}: {exc}")
        draft.stop("workspace_failed")
        raise preflight.SessionRefusedError(1) from exc
    # The complete effective config as the overlay: omitting values equal to built-in defaults
    # loses an explicit repo reset when the workspace reloads the operator's global layer.
    overlay = cfg.model_dump(mode="json")
    overlay["agent6"].pop("preset", None)  # the overlay layer forbids a preset
    # Resolved on the host: the confined execution cannot read ~/.gitconfig.
    try:
        name, email = git_read.verify_git_identity(
            workspace, git_read.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        )
    except errors.GitError as exc:
        draft.reporter.error(str(exc))
        draft.stop("no_git_identity")
        raise preflight.SessionRefusedError(2) from exc
    return agent.build_machine_agent_runner(
        # The authoring agent is no state: its requests carry the empty name.
        {"": agent.StateOverlay(overlay, draft.isolation)},
        workspace,
        draft.events.path.parent / "agent_transcripts",
        commit_identity=git_read.CommitIdentity(name=name, email=email),
    )


def publish_draft(
    draft: drafting.Draft,
    valid: drafting.ValidatedDraft,
) -> int:
    """Publish the validated scripts before their machine file and check the result.

    Args:
        draft: The create session.
        valid: The bundle that passed every check.

    Returns:
        Zero on success, one on failure or two on an output collision.
    """
    reporter = draft.reporter
    workspace = draft.workspace
    spec = valid.spec
    draft_text = valid.path.read_text(encoding="utf-8")
    payload = draft_text if draft_text.endswith("\n") else draft_text + "\n"
    target = draft.output if draft.output is not None else draft.cwd / f"{spec.machine}.asm.toml"
    existing = [
        p
        for p in (target, *(target.parent / rel for rel in sorted(valid.scripts)))
        if p.exists() or p.is_symlink()
    ]
    # The default path clobbers nothing, scripts included; `-o` replaces existing targets.
    if draft.output is None and existing:
        draft.stop("output_collision")
        reporter.err("REFUSING to overwrite existing file(s):")
        for clash in existing:
            reporter.err(f"  {clash}")
        reporter.err(f"The validated bundle is in {workspace}; it is also on stdout.")
        reporter.out(payload.removesuffix("\n"))
        return 2
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Scripts first, the machine file last: a death mid-publish leaves inert scripts.
        _write_scripts(target.parent, valid.scripts)
        portable.atomic_write(target, payload)
    except (OSError, path_safety.ContainmentError) as exc:
        draft.stop("write_failed")
        reporter.err(f"FAILED: could not write the bundle to {target.parent}: {exc}")
        reporter.err(f"The validated bundle is in {workspace}; it is also on stdout.")
        reporter.out(payload.removesuffix("\n"))
        return 1
    # A pre-existing symlink under scripts/ can make the published bundle differ: check it.
    out_problems = bundle.validate_bundle(spec, target)
    if out_problems:
        draft.stop("bundle_invalid")
        reporter.err(f"FAILED: the bundle written to {target.parent} does not validate:")
        for problem in out_problems:
            reporter.err(f"  - {problem}")
        return 1
    draft.stop("machine_created", verification="passed")
    scripts_note = f" + {len(valid.scripts)} script(s)" if valid.scripts else ""
    replaced_note = (
        f"; replaced {', '.join(repr(str(path)) for path in existing)}" if existing else ""
    )
    reporter.err(
        f"OK: wrote draft to {target} ({spec.machine}, {len(spec.states)} states)"
        f"{scripts_note}{replaced_note}."
    )
    if git_read.is_git_repo(target.parent):
        reporter.err(
            "Review and commit the bundle; `machine run` checks it for uncommitted changes"
            " when run from this repository."
        )
    if to_allow := _pass_env_to_allow(spec, draft.cfg.machine.pass_env):
        reporter.err(
            "The draft's tools read operator environment variables; `machine run` refuses"
            " until your config allows each:"
        )
        for name in to_allow:
            reporter.err(f"  agent6 config add machine.pass_env {name}")
    # A failure keeps the workspace: the draft in it is what the operator reads.
    _discard_workspace(workspace, reporter)
    return 0


def create_machine(
    task: str,
    reporter: app_reporter.Reporter,
    *,
    output: pathlib.Path | None,
    max_attempts: int,
    confirm_unconfined: Callable[[kinds.IsolationLevel, config_model.Config], bool],
    config_paths: tuple[pathlib.Path, ...] = (),
    budget_overrides: session_setup.BudgetOverrides | None = None,
) -> int:
    """Draft, validate and write a machine.

    Args:
        task: The natural-language task the machine should carry out.
        reporter: The two output channels.
        output: Where to write the `.asm.toml`; None writes `<machine>.asm.toml` in cwd and
            refuses to overwrite anything.
        max_attempts: How many authoring attempts to make.
        confirm_unconfined: Confirms an unconfined autorun with the operator.
        config_paths: The `--config` files the effective config is read under.
        budget_overrides: The `--max-*` flags, capping the attempts as they cap a run.

    Returns:
        The exit code: 0 on a written bundle, 1 on a failed draft, 2 on a refusal.
    """
    if max_attempts < 1:
        reporter.error("--max-attempts must be >= 1.")
        return 2
    try:
        draft = _open_draft(
            task, output, config_paths, budget_overrides, reporter, confirm_unconfined
        )
        runner = _prepare_runner(draft)
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    valid = drafting.try_draft(draft, runner, task, max_attempts)
    return publish_draft(draft, valid) if valid is not None else 1
