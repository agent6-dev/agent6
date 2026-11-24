"""Validate and prepare the checkout for a parent-owned child continuation."""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import tempfile

from agent6 import child_records, errors, paths
from agent6.app import child_launch, child_trees, child_wait, stamps, worktrees
from agent6.config import Config
from agent6.harness import child_repo, loop_snapshot
from agent6.sessions import child_dispatch, manifest, process
from agent6.sessions import layout as sessions_layout
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log


def authorized(
    launch: child_launch.ChildLaunch | None,
    record: manifest.SessionManifest,
    *,
    session_id: str,
    root: pathlib.Path,
) -> bool:
    """Return whether a private launch authorizes this child's continuation.

    Args:
        launch: The validated private launch, or None outside a child.
        record: The session's manifest.
        session_id: The resolved session id.
        root: The execution's checkout.

    Returns:
        True only for the same run child, clone, parent and group.
    """
    return bool(
        launch is not None
        and launch.continuation
        and launch.contract is None
        and launch.session_id == session_id
        and launch.worktree.resolve() == root.resolve()
        and record.mode == "run"
        and record.parallel is not None
        and record.parallel.parent == launch.parent
        and record.parallel.group == launch.group
        and record.parallel.number == launch.number
    )


def check_workspace(
    spec: child_records.ChildSpec,
    private: pathlib.Path,
    *,
    parent: str,
    record_state: pathlib.Path,
) -> child_launch.CloneReceipt:
    """Refuse a missing, live or foreign child before replacing its launch file.

    Args:
        spec: The existing child.
        private: The group's private launch directory.
        parent: The owning parent.
        record_state: The origin's state directory.

    Raises:
        LaunchError: The clone, ownership or saved conversation cannot support continuation.

    Returns:
        The receipt to carry forward without rebinding the checkout.
    """
    previous = child_launch.read_file(private / f"child-{spec.number}.json")
    if (
        previous.session_id != spec.session_id
        or previous.parent != parent
        or previous.group != private.name
        or previous.number != spec.number
        or previous.worktree != spec.worktree
        or previous.record_state != record_state
        or previous.contract is not None
    ):
        raise child_launch.LaunchError(
            "the retained launch does not belong to this child and parent"
        )
    directory = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
    if process.worker_is_alive(directory):
        raise child_launch.LaunchError(f"child {spec.session_id} still has a live worker")
    if not (directory / "loop_state.json").is_file():
        raise child_launch.LaunchError(f"child conversation snapshot is missing: {directory}")
    return previous.checked_clone(spec.worktree)


@dataclasses.dataclass(frozen=True, slots=True)
class Workspace:
    """The receipt the bootstrap must check, and the workspace its parent reports.

    Attributes:
        clone: The receipt for the prepared checkout.
        notice: The workspace as the parent's answer names it.
        fresh: Whether `prepare` made the clone for this continuation.
    """

    clone: child_launch.CloneReceipt
    notice: str
    fresh: bool

    def checked(self, root: pathlib.Path) -> child_launch.CloneReceipt:
        """Return the receipt for the prepared clone.

        Args:
            root: The checkout the worker will enter.

        Returns:
            The unchanged receipt.

        Raises:
            LaunchError: The prepared checkout was replaced before launch.
        """
        if child_launch.CloneReceipt.capture(root) != self.clone:
            raise child_launch.LaunchError("the continuation clone does not match its receipt")
        return self.clone


@dataclasses.dataclass(frozen=True, slots=True)
class End:
    """The evidence for one ended child. Only a killed worker can lack a measured tree."""

    record: manifest.SessionManifest
    tip: str
    end_tree: str
    index_tree: str


def inspect(  # noqa: PLR0912  # each refusal names the missing evidence
    spec: child_records.ChildSpec,
    continuation: child_records.ChildContinuation,
    *,
    origin: pathlib.Path,
    group: str,
    parent: str,
    record_state: pathlib.Path,
) -> End:
    """Validate the saved execution and its imported work without changing either.

    Args:
        spec: The child from the parent's dispatch.
        continuation: The instruction and whether the workspace needs restoration.
        origin: The parent repository, which holds the imported work.
        group: The dispatch group.
        parent: The parent session id.
        record_state: Where the parent and child keep their records.

    Returns:
        The saved evidence for continuation.

    Raises:
        LaunchError: The record, conversation, commits or trees cannot be restored.
    """
    directory = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
    try:
        record = manifest.read_manifest(directory)
    except manifest.ManifestError as exc:
        raise child_launch.LaunchError(f"child manifest missing or unreadable: {exc}") from exc
    if not (
        record.session_id == spec.session_id
        and record.mode == "run"
        and record.parallel
        and record.parallel.parent == parent
        and record.parallel.group == group
        and record.parallel.number == spec.number
    ):
        raise child_launch.LaunchError("the child record does not belong to this parent and group")
    if process.worker_is_alive(directory):
        raise child_launch.LaunchError(f"child {spec.session_id} still has a live worker")
    try:
        snapshot = loop_snapshot.load_session_snapshot(directory / "loop_state.json")
    except (ValueError, OSError) as exc:
        raise child_launch.LaunchError(
            f"child conversation snapshot missing or unreadable: {exc}"
        ) from exc
    logs = directory / sessions_layout.LOGS_NAME
    scan = session_log.scan_session_log(logs)
    if not scan.saw_start:
        raise child_launch.LaunchError(f"child journal missing or unreadable: {logs}")
    chain = git_read.chain_ref_for(spec.session_id)
    model = record.git_control == "model"
    ref = chain if model else git_read.run_branch_for(spec.session_id)
    tip = git_read.chain_tip(origin, ref)
    if not tip and (
        model or continuation.commits or snapshot.head_sha not in ("", record.base_sha)
    ):
        raise child_launch.LaunchError(f"child work ref or commit missing: {ref}")
    if not record.base_sha or not git_read.chain_tip(origin, record.base_sha):
        raise child_launch.LaunchError(f"child base commit missing: {record.base_sha}")
    head = tip or record.base_sha
    if (
        continuation.commits
        and git_read.commit_count(origin, record.base_sha, head) != continuation.commits
    ):
        raise child_launch.LaunchError(f"{ref} no longer holds the child's recorded commits")
    if snapshot.head_sha and not git_read.is_ancestor(origin, snapshot.head_sha, head):
        raise child_launch.LaunchError(
            f"{ref} does not hold the child's saved commit {snapshot.head_sha}"
        )
    if not continuation.fresh:
        return End(record, head, scan.end_tree, scan.index_tree)
    if not scan.end_tree and scan.stop_reason:
        raise child_launch.LaunchError("the child's end tree was not recorded")
    if model and scan.end_tree and not scan.index_tree:
        raise child_launch.LaunchError("the child's index tree was not recorded")
    for kind, tree in (("end", scan.end_tree), ("index", scan.index_tree)):
        if tree:
            child_trees.verify_pin(origin, spec.session_id, kind, tree)
    return End(record, head, scan.end_tree, scan.index_tree)


def recoverable(
    group: str,
    child: child_dispatch.ChildRecord,
    *,
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    parent: str,
) -> bool:
    """Check the evidence behind the resumed parent's list without opening a checkout.

    Args:
        group: The dispatch group.
        child: Its ended child.
        cfg: Supplies the group's worktree base.
        origin: The repository holding the retained Git objects.
        record_state: The directory holding the records.
        parent: The parent session id.

    Returns:
        Whether the work and conversation can be restored.
    """
    if child.merge is None or not sessions_layout.is_safe_session_id(child.session_id):
        return False
    root = worktrees.child_worktree_root(cfg, origin, group) / f"child-{child.number}"
    spec = child_records.ChildSpec(child.number, child.session_id, root, None)
    continuation = child_records.ChildContinuation(
        child.number, "", child.merge.status != "stopped", child.merge.commits
    )
    try:
        inspect(
            spec, continuation, origin=origin, group=group, parent=parent, record_state=record_state
        )
        if not continuation.fresh or root.exists():
            private = child_launch.launch_dir(paths.state_dir(origin), group)
            check_workspace(spec, private, parent=parent, record_state=record_state)
    except (child_launch.LaunchError, errors.GitError, OSError):
        return False
    return True


def prepare(
    spec: child_records.ChildSpec,
    continuation: child_records.ChildContinuation,
    *,
    origin: pathlib.Path,
    private: pathlib.Path,
    parent: str,
    record_state: pathlib.Path,
) -> Workspace:
    """Bind the kept clone or restore a new one before starting the worker.

    Args:
        spec: The child from the parent's dispatch.
        continuation: The continuation request.
        origin: The parent's repository.
        private: The group's private launch directory.
        parent: The parent session id.
        record_state: Where the parent and child keep their records.

    Returns:
        The receipt and the workspace notice for the parent.

    Raises:
        GitError: The saved trees could not be restored.
        LaunchError: The child's evidence or checkout does not match its dispatch.
        OSError: The checkout or record could not be written.
    """
    end = inspect(
        spec,
        continuation,
        origin=origin,
        group=private.name,
        parent=parent,
        record_state=record_state,
    )
    if not continuation.fresh:
        return Workspace(
            check_workspace(spec, private, parent=parent, record_state=record_state),
            "its kept clone",
            fresh=False,
        )
    note = (
        "a fresh clone restored to its end tree"
        if end.end_tree
        else (
            "a fresh clone at its last commit; no end tree was recorded;"
            " uncommitted files could not be restored"
        )
    )
    if spec.worktree.exists() or spec.worktree.is_symlink():
        # Keep an unmeasured checkout that holds the only copy, outside the group sweep.
        check_workspace(spec, private, parent=parent, record_state=record_state)
        archive = (
            pathlib.Path(tempfile.mkdtemp(prefix="kept-", dir=spec.worktree.parent.parent))
            / spec.session_id
        )
        spec.worktree.rename(archive)
        note += f"; earlier clone kept at {archive}"
    child_repo.clone_worktree(origin, spec.worktree)
    try:
        git_ops.checkout_detached(spec.worktree, end.tip)
        if end.record.git_control != "model" and end.tip != end.record.base_sha:
            tip = end.tip
            chain = git_read.chain_ref_for(spec.session_id)
            ref = git_read.run_branch_for(spec.session_id)
            git_ops.fetch_branch(spec.worktree, origin, f"{tip}:{chain}")
            git_ops.create_branch_at(spec.worktree, ref, tip)
        if end.end_tree:
            git_ops.restore_child_trees(spec.worktree, end.end_tree, end.index_tree)
        clone = child_launch.CloneReceipt.capture(spec.worktree)
        directory = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
        stamps.write_manifest(
            directory / manifest.MANIFEST_NAME,
            end.record.model_copy(update={"worktree_set_up": False}),
        )
    except (OSError, errors.GitError, child_launch.LaunchError):
        # No worker ran in this new clone. Every source object remains in the origin.
        shutil.rmtree(spec.worktree, ignore_errors=True)
        raise
    return Workspace(clone, note, fresh=True)
