# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the internal value types: frozen dataclasses agent6 constructs itself.

The pydantic models sit at the trust boundaries instead: `config.model`,
`tools.schema`, `machine.spec`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Literal

TernaryMode = Literal["deny", "ask", "allow"]
# `none` runs commands as plain subprocesses: a host with no confinement, or an operator opt-out.
IsolationLevel = Literal["full", "partial", "none"]
NetworkMode = Literal["host", "session", "none"]
RoleName = Literal["worker", "reviewer", "planner"]
# The modes `agent6 resume` accepts; `session_kind` answers the wider "is this a known mode".
ResumableMode = Literal["run", "plan", "ask"]
# What the REPL's after-commit hook tells the loop to do next; `exit` stops with no prompt.
AutoCommitDirective = Literal["continue", "stop", "undo", "exit"]


@dataclasses.dataclass(frozen=True, slots=True)
class SessionKind:
    """What a mode may do, in one record.

    The one owner, derived from the persisted mode string at read time and never
    written, so a change to what a mode may do reinterprets old sessions.

    Attributes:
        name: The mode string.
        role: The model role that drives it.
        edits: May mutate the workspace in-process and own a background command.
        runs_commands: May execute commands at all.
        clamps_commands: Forces approval even where config says "allow".
        resumable: `agent6 resume` can pick it up; a machine's states are driven by the
            machine agent instead.
    """

    name: str
    role: RoleName
    edits: bool
    runs_commands: bool
    clamps_commands: bool
    resumable: bool


SESSION_KINDS: dict[str, SessionKind] = {
    kind.name: kind
    for kind in (
        SessionKind(
            name="run",
            role="worker",
            edits=True,
            runs_commands=True,
            clamps_commands=False,
            resumable=True,
        ),
        # Operator-present like ask, so it clamps commands to ask.
        SessionKind(
            name="plan",
            role="planner",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
        ),
        # No edit or graph tools; a jailed command may write the workspace, so it is approval-gated.
        SessionKind(
            name="ask",
            role="worker",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
        ),
        # A read-only machine state: the deliverable is the finish payload.
        SessionKind(
            name="machine",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
        ),
        SessionKind(
            name="agent",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
        ),
    )
}


# The modes an operator starts and resumes; machine and agent executions are the machine agent's.
OPERATOR_MODES: tuple[str, ...] = tuple(k.name for k in SESSION_KINDS.values() if k.resumable)

# The roles whose output is the session talking; every other role is a side call on the journal.
DRIVING_ROLES: frozenset[str] = frozenset(k.role for k in SESSION_KINDS.values())


def is_side_role(role: str) -> bool:
    """Return whether a `role.*` event's answer is a side call's, not the session's own.

    An unnamed role is not a side call: older events carry none.

    Args:
        role: The event's `role` field.

    Returns:
        True for a named role that drives no mode.
    """
    return bool(role) and role not in DRIVING_ROLES


class UnknownSessionKindError(ValueError):
    """A mode string this agent6 does not know."""


def session_kind(name: str) -> SessionKind:
    """Return the record for a mode string.

    Args:
        name: The mode, such as `run`.

    Returns:
        What the mode may do and the role that drives it.

    Raises:
        UnknownSessionKindError: The mode is unknown; a damaged manifest must never
            escalate a read-only session to the write tools.
    """
    kind = SESSION_KINDS.get(name)
    if kind is None:
        raise UnknownSessionKindError(f"unknown session mode {name!r}")
    return kind


def session_bucket(name: str) -> str:
    """Return the bucket a mode's sessions get their directories in, derived and never stored.

    Args:
        name: The mode, such as `run`.

    Returns:
        The mode's name plus `s`, such as `runs`.

    Raises:
        UnknownSessionKindError: The mode is unknown, or is `agent`, whose executions
            live under their machine instance.
    """
    kind = session_kind(name)
    if kind.name == "agent":
        raise UnknownSessionKindError(
            "an agent execution lives under its machine instance, not sessions/"
        )
    return f"{kind.name}s"


@dataclasses.dataclass(frozen=True, slots=True)
class CommandResult:
    """The result of running a command, in or out of the jail.

    Attributes:
        argv: The command.
        returncode: Its exit code.
        stdout: Its stdout, decoded.
        stderr: Its stderr, decoded.
        duration_s: How long it ran.
        exec_failed: The binary could not be executed at all, as distinct from a
            non-zero exit: an operator command that cannot execute is surfaced loudly.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command exited 0."""
        return self.returncode == 0


@dataclasses.dataclass(frozen=True, slots=True)
class ChildSnapshot:
    """The agent's children when a command started.

    Attributes:
        seq: The command's start order in the session, so a stop can bound its sweep.
        pids: The children's pids.
    """

    seq: int
    pids: frozenset[int]


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundHandoff:
    """A command that outlived its check-in and is still running.

    Its own type: a return code invented for a running command would be a lie every
    caller has to ignore.

    Attributes:
        argv: The command.
        pid: Its pid.
        log: Where its output goes.
        stdout: What it printed before the hand-off.
        stderr: What it printed to stderr before the hand-off.
        duration_s: How long it had run at the hand-off.
        before: The agent's children when it started, its stop's baseline.
    """

    argv: tuple[str, ...]
    pid: int
    log: str
    stdout: str
    stderr: str
    duration_s: float
    before: ChildSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class JailPolicy:
    """What the jail allows one child invocation.

    Attributes:
        cwd: The child's working directory, the workspace.
        argv: The command.
        isolation: The isolation level.
        env: The child's environment.
        network: The network the child joins: the machine's, the run's own shared
            with its siblings, or an empty one of its own.
        extra_ro_paths: Trees bound read-only at their own paths.
        extra_rw_paths: Trees bound read-write at their own paths.
        grant_sources: For each of those two whose path is a symlink or passes through one,
            (the path, what it resolved to): the launcher binds the resolved tree at the
            path and puts the Landlock rule on it, following no link of its own.
        extra_device_paths: Device nodes bound into the jail's `/dev`; each must be a
            char or block device on the host or the launcher refuses.
        extra_protect_paths: Paths inside the workspace made read-only from the child's
            view, so a model-driven command cannot rewrite `.git`.
        tool_paths: Operator tools outside the system dirs, bound read-only and
            executable at their real paths.
        hide_paths: The operator's additions to the hidden set, masked last, after
            every bind; agent6's own private dirs are unioned in at serialization.
        timeout_s: The command's deadline.
        memory_limit_mib: The per-process data-segment cap (RLIMIT_DATA) in MiB, inherited
            by every descendant; 0 disables, since capping costs real builds more than it buys.
    """

    cwd: pathlib.Path
    argv: tuple[str, ...]
    isolation: IsolationLevel = "full"
    env: tuple[tuple[str, str], ...] = ()
    network: NetworkMode = "none"
    extra_ro_paths: tuple[pathlib.Path, ...] = ()
    extra_rw_paths: tuple[pathlib.Path, ...] = ()
    grant_sources: tuple[tuple[pathlib.Path, pathlib.Path], ...] = ()
    extra_device_paths: tuple[pathlib.Path, ...] = ()
    extra_protect_paths: tuple[pathlib.Path, ...] = ()
    tool_paths: tuple[pathlib.Path, ...] = ()
    hide_paths: tuple[pathlib.Path, ...] = ()
    timeout_s: float = 600.0
    memory_limit_mib: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRoute:
    """A provider and a model on it, the pair every model choice resolves to.

    Attributes:
        provider: The provider entry.
        model: The model id.
    """

    provider: str
    model: str

    @property
    def spec(self) -> str:
        """The pair as `provider/model`, the one spelling every surface accepts."""
        return f"{self.provider}/{self.model}"


@dataclasses.dataclass(frozen=True, slots=True)
class RepoSummary:
    """The compact view of a repository the prompt carries.

    Attributes:
        root: The repository root.
        branch: The checked-out branch; "" outside git.
        head_sha: HEAD's sha; "" outside git.
        file_count: How many tracked files.
        top_level: The top-level entries.
        agents_md: The AGENTS.md text.
        recent_log: The recent one-line log; "" outside git.
        repo_map: A directory map of `path/  (N files: a, b, ...)` rows, capped to a
            few KB; "" outside git.
        is_git: The root is a repository; `agent6 ask` runs anywhere, and the prompt
            names the situation instead of a fake repo header.
    """

    root: pathlib.Path
    branch: str
    head_sha: str
    file_count: int
    top_level: tuple[str, ...]
    agents_md: str
    recent_log: str
    repo_map: str = ""
    is_git: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One `agent6 check` summary row.

    Attributes:
        name: The check.
        status: PASS, FAIL, WARN or INFO, carried to the summary line unchanged.
        detail: The one-line finding.
    """

    name: str
    status: Literal["PASS", "FAIL", "WARN", "INFO"]
    detail: str


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxReport:
    """The result of one sandbox self-test.

    Attributes:
        name: The test's name.
        status: PASS, FAIL or SKIP when the probe does not apply.
        detail: What it found.
    """

    name: str
    status: Literal["PASS", "FAIL", "SKIP"]
    detail: str


def workspace_setup_fields(result: CommandResult) -> dict[str, object]:
    """Return the shared workspace.setup event body.

    Args:
        result: The setup command's result.

    Returns:
        The command, its exit code, duration and exec failure, and its stdout and stderr,
        each cut to its last 2000 characters.
    """
    return {
        "cmd": list(result.argv),
        "exit_code": result.returncode,
        "duration_s": result.duration_s,
        "exec_failed": result.exec_failed,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
