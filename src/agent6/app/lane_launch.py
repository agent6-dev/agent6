# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The facts a parent hands a lane, in one private file named by `AGENT6_LANE`.

The bootstrap (`agent6.ui.cli.lane_agent`) reads it to start the run; the lane's config load
and manifest write read it again, so the grouping survives a parent death. A run outside
a lane has no launch file. The task and pins travel here, never on the host argv, under the
state base every jail hides.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import stat
import sys

import pydantic

from agent6 import budget, kinds, lane_records
from agent6.config import Config, ConfigError
from agent6.sessions import ipc, manifest

ENV = "AGENT6_LANE"
# The fixed bootstrap: `-P` keeps the cwd off `sys.path`, `-E` ignores `PYTHONPATH` and its
# siblings, so nothing in the clone or the environment chooses what the lane imports.
BOOTSTRAP = "agent6.ui.cli.lane_agent"
_FILE_MODE = 0o600
_DIR_MODE = 0o700


class LaunchError(ConfigError):
    """A launch file or its directory is missing, malformed, foreign-owned or exposed."""


class LaneLaunch(pydantic.BaseModel, frozen=True, extra="forbid"):
    """What a lane needs to start, and what it knows about itself that no config layer holds.

    **Support: internal.** Written by the parent for one spawn, read by that lane.

    Attributes:
        worktree: The lane's clone, the run's cwd.
        record_state: The state dir of the repository that dispatched the lane, where the
            lane's record lives from its first event.
        session_id: The lane's session id.
        task: The task text, exactly as the lane's manifest records it.
        pins: The parent's pins, out of band of the task.
        max_usd: The parent's forwarded `--max-usd`; None keeps the config's.
        allowance: What the parent had left when the lane started, which sets every cap;
            None for the config's caps.
        auto_approve: Whether the parent ran with `--auto-approve`.
        away: The lane's away mode: `wait` parks a question for `agent6 attach`, `deny`
            answers it empty.
        config: The lane's config snapshot, beside this file; the lane's sole `--config`.
        notes: The snapshot's serialization loss notes, recorded in the lane's manifest.
        parent: The session every listing nests the lane under.
        group: The fan-out the lane belongs to.
        lane: The lane's number, from 1.
        gate_source: Where the verify gate in the snapshot came from; "" for none.
        operator_task: The task's words before composition; None when they are the task.
        limits: The turn cap and deadline the lane runs under; None for the config's own.
        contract: The bounded conversation a `reviewer` lane drives; None for a run lane.
    """

    worktree: pathlib.Path
    record_state: pathlib.Path
    session_id: str = pydantic.Field(min_length=1)
    task: str = pydantic.Field(min_length=1)
    pins: tuple[str, ...] = ()
    max_usd: float | None = None
    allowance: budget.BudgetAllowance | None = None
    auto_approve: bool = False
    away: ipc.AwayMode = "wait"
    config: pathlib.Path
    notes: tuple[str, ...] = ()
    parent: str = pydantic.Field(min_length=1)
    group: str = pydantic.Field(min_length=1)
    lane: int = pydantic.Field(ge=1)
    gate_source: manifest.GateSource = ""
    operator_task: str | None = None
    limits: lane_records.LaneLimits | None = None
    contract: kinds.ConversationContract | None = None


def is_lane() -> bool:
    """Return whether this process runs a lane, whose work the session that dispatched it joins."""
    return bool(os.environ.get("AGENT6_SUBRUN"))


def dispatch_authority(
    cfg: Config, *, mode: str, session_id: str, root: pathlib.Path
) -> lane_records.DispatchAuthority:
    """Resolve authority from the private launch, never from a tool or steer argument.

    Args:
        cfg: The effective configuration, including review authorization.
        mode: The execution mode.
        session_id: The execution's session id, bound to its launch.
        root: The execution's checkout, bound to its launch.

    Returns:
        Work authority only in top-level runs; configured panels also in top-level ask/plan.
    """
    launch = read()
    subordinate = launch is not None or is_lane()
    worker = mode == "run"
    if subordinate:
        worker = (
            worker
            and launch is not None
            and launch.contract is None
            and launch.session_id == session_id
            and launch.worktree.resolve() == root.resolve()
        )
    deadline = launch.limits.deadline_at if launch is not None and launch.limits else None
    return lane_records.DispatchAuthority(
        work=worker and not subordinate,
        panels=(worker or (not subordinate and mode in ("ask", "plan")))
        and cfg.review.when != "never",
        deadline_at=deadline,
    )


def apply_limits(cfg: Config, limits: lane_records.LaneLimits | None) -> Config:
    """Return the config with `harness.max_turns` clamped to the lane's cap.

    Args:
        cfg: The lane's loaded config.
        limits: The lane's limits; None, or one without a turn cap, leaves the config alone.

    Returns:
        The config, the same object when nothing changes.
    """
    if limits is None or limits.max_turns is None:
        return cfg
    turns = limits.clamp_turns(cfg.harness.max_turns)
    if turns == cfg.harness.max_turns:
        return cfg
    return cfg.model_copy(update={"harness": cfg.harness.model_copy(update={"max_turns": turns})})


def bootstrap_argv() -> list[str]:
    """Return the fixed host argv that starts a lane; the lane's facts travel in its file."""
    return [sys.executable, "-P", "-E", "-m", BOOTSTRAP]


def launch_dir(origin_state: pathlib.Path, group: str) -> pathlib.Path:
    """Return the private dir a group's launch files and config snapshots live in.

    Args:
        origin_state: The origin's state dir, a private dir every jail hides.
        group: The group id.

    Returns:
        `<origin_state>/launch/<group>`.
    """
    return origin_state / "launch" / group


def private_dir(path: pathlib.Path) -> None:
    """Create the dir `0700`, or accept one this user owns that nobody else can write.

    Args:
        path: The dir; missing parents are created with the process's umask.

    Raises:
        LaunchError: The path is a symlink, not a dir, foreign-owned or writable by others.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        path.mkdir(mode=_DIR_MODE)
    _check_private(path, directory=True, exposed=0o022)


def write_private(path: pathlib.Path, text: str) -> None:
    """Write a `0600` file under a private dir, never through a symlink at its path.

    The text lands in a sibling created exclusively and then takes the name, so a planted
    link at the path is replaced, not followed.

    Args:
        path: The file.
        text: Its content.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    pathlib.Path(tmp).replace(path)


def write(directory: pathlib.Path, launch: LaneLaunch) -> pathlib.Path:
    """Write the launch file `AGENT6_LANE` names, as `lane-<n>.json` under a private dir.

    Args:
        directory: The group's launch dir (`launch_dir`), created private when missing.
        launch: The lane's facts.

    Returns:
        The file's path.

    Raises:
        LaunchError: The dir cannot be private.
    """
    private_dir(directory)
    path = directory / f"lane-{launch.lane}.json"
    write_private(path, launch.model_dump_json(indent=2) + "\n")
    return path


def read() -> LaneLaunch | None:
    """Return the launch file `AGENT6_LANE` names, or None outside a lane.

    Returns:
        The lane's facts, or None when the variable is unset.

    Raises:
        LaunchError: The variable names a missing, malformed, foreign-owned or exposed file.
    """
    raw = os.environ.get(ENV)
    if not raw:
        return None
    return read_file(pathlib.Path(raw))


def read_file(path: pathlib.Path) -> LaneLaunch:
    """Validate and return a launch file: a regular `0600` file of this user's in a private dir.

    Args:
        path: The file.

    Returns:
        The lane's facts.

    Raises:
        LaunchError: The file is missing, a symlink, foreign-owned, readable by others or
            malformed, or its dir is writable by others.
    """
    text = read_private(path)
    try:
        return LaneLaunch.model_validate_json(text)
    except pydantic.ValidationError as exc:
        raise LaunchError(f"invalid lane launch {path}: {exc}") from exc


def read_private(path: pathlib.Path) -> str:
    """Return a private file's text after checking the file and its dir.

    Args:
        path: The file.

    Returns:
        The text.

    Raises:
        LaunchError: The file or its dir fails the private-file checks.
    """
    _check_private(path.parent, directory=True, exposed=0o022)
    _check_private(path, directory=False, exposed=0o077)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise LaunchError(f"invalid lane launch {path}: {exc.strerror or exc}") from exc
    try:
        # The open file is checked again: the path check and the read must see one inode.
        _check_stat(path, os.fstat(fd), directory=False, exposed=0o077)
    except LaunchError:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


def _check_private(path: pathlib.Path, *, directory: bool, exposed: int) -> None:
    """Check a path without following a symlink at it.

    Args:
        path: The path.
        directory: Whether a dir is expected, else a regular file.
        exposed: The mode bits that must be clear.

    Raises:
        LaunchError: The path is missing or fails `_check_stat`.
    """
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise LaunchError(f"invalid lane launch {path}: {exc.strerror or exc}") from exc
    _check_stat(path, st, directory=directory, exposed=exposed)


def _check_stat(path: pathlib.Path, st: os.stat_result, *, directory: bool, exposed: int) -> None:
    """Refuse a symlink, the wrong kind, a foreign owner or a mode with `exposed` bits set.

    Args:
        path: The path, for the message.
        st: Its stat result.
        directory: Whether a dir is expected, else a regular file.
        exposed: The mode bits that must be clear.

    Raises:
        LaunchError: One of those.
    """
    kind = "directory" if directory else "file"
    if stat.S_ISLNK(st.st_mode):
        raise LaunchError(f"invalid lane launch {path}: a symlink where a {kind} must be")
    if directory != stat.S_ISDIR(st.st_mode) or not (directory or stat.S_ISREG(st.st_mode)):
        raise LaunchError(f"invalid lane launch {path}: not a regular {kind}")
    if st.st_uid != os.geteuid():
        raise LaunchError(f"invalid lane launch {path}: owned by uid {st.st_uid}, not this user")
    if st.st_mode & exposed:
        raise LaunchError(
            f"invalid lane launch {path}: mode {stat.S_IMODE(st.st_mode):04o} lets others"
            f" {'write' if directory else 'read'} it"
        )
