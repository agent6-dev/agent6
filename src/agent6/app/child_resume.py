"""Validate the private authority and retained clone for a parent's child continuation."""

from __future__ import annotations

import pathlib

from agent6 import child_records
from agent6.app import child_launch, child_wait
from agent6.sessions import manifest, process


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
    child_dir = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
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
    clone = previous.checked_clone(spec.worktree)
    if not (child_dir / "loop_state.json").is_file():
        raise child_launch.LaunchError(f"child conversation snapshot is missing: {child_dir}")
    if process.worker_is_alive(child_dir):
        raise child_launch.LaunchError(f"child {spec.session_id} still has a live worker")
    return clone
