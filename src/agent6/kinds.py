# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the internal value types: frozen dataclasses agent6 constructs itself.

The pydantic models sit at the trust boundaries instead: `config.model`,
`tools.schema`, `machine.spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TernaryMode = Literal["no", "ask", "yes"]
# `none` runs commands as plain subprocesses: a host with no confinement, or an operator opt-out.
IsolationLevel = Literal["strict", "hardened", "none"]
NetworkMode = Literal["host", "session", "none"]
RoleName = Literal["worker", "reviewer", "planner"]
# The modes `agent6 resume` accepts; `session_kind` answers the wider "is this a known mode".
ResumableMode = Literal["run", "plan", "ask"]
# What the REPL's after-commit hook tells the loop to do next; `exit` stops with no prompt.
AutoCommitDirective = Literal["continue", "stop", "undo", "exit"]


@dataclass(frozen=True, slots=True)
class SessionKind:
    """What a mode may do, in one record.

    The one owner, derived from the persisted mode string at read time and never
    written, so a change to what a mode may do reinterprets old sessions.

    Attributes:
        name: The mode string.
        role: The model role that drives it.
        edits: May mutate the workspace in-process and own a background command.
        runs_commands: May execute commands at all.
        clamps_commands: Forces approval even where config says "yes".
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
    """
    return bool(role) and role not in DRIVING_ROLES


class UnknownSessionKindError(ValueError):
    """A mode string this agent6 does not know."""


def session_kind(name: str) -> SessionKind:
    """Return the record for a mode string.

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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ChildSnapshot:
    """The agent's children when a command started.

    Attributes:
        seq: The command's start order in the session, so a stop can bound its sweep.
        pids: The children's pids.
    """

    seq: int
    pids: frozenset[int]


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class JailPolicy:
    """What the jail allows one child invocation.

    Attributes:
        cwd: The child's working directory, the workspace.
        argv: The command.
        isolation: The isolation level.
        env: The child's environment.
        network: The network the child joins: the machine's, the run's own shared
            with its siblings, or an empty one of its own.
        extra_ro_paths: Trees bound read-only under `/ro`.
        extra_rw_paths: Trees bound read-write.
        extra_device_paths: Device nodes bound into the jail's `/dev`; each must be a
            char or block device on the host or the launcher refuses.
        extra_protect_paths: Paths inside the workspace made read-only from the child's
            view, so a model-driven command cannot rewrite `.git`.
        tool_paths: Operator tools outside the system dirs, bound read-only and
            executable at their real paths, since `/ro` remapping breaks symlinks.
        hide_paths: The operator's additions to the hidden set, masked last, after
            every bind; agent6's own private dirs are unioned in at serialization.
        timeout_s: The command's deadline.
        memory_limit_mb: The per-process RLIMIT_DATA cap, inherited by every
            descendant; 0 disables, since capping costs real builds more than it buys.
    """

    cwd: Path
    argv: tuple[str, ...]
    isolation: IsolationLevel = "strict"
    env: tuple[tuple[str, str], ...] = ()
    network: NetworkMode = "none"
    extra_ro_paths: tuple[Path, ...] = ()
    extra_rw_paths: tuple[Path, ...] = ()
    extra_device_paths: tuple[Path, ...] = ()
    extra_protect_paths: tuple[Path, ...] = ()
    tool_paths: tuple[Path, ...] = ()
    hide_paths: tuple[Path, ...] = ()
    timeout_s: float = 600.0
    memory_limit_mb: int = 0


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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

    root: Path
    branch: str
    head_sha: str
    file_count: int
    top_level: tuple[str, ...]
    agents_md: str
    recent_log: str
    repo_map: str = ""
    is_git: bool = True


@dataclass(frozen=True, slots=True)
class SandboxReport:
    """The result of one sandbox self-test.

    Attributes:
        name: The test's name.
        ok: Whether it passed.
        detail: What it found.
    """

    name: str
    ok: bool
    detail: str
