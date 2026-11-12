# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the internal value types: frozen dataclasses agent6 constructs itself.

The pydantic models sit at the trust boundaries instead: `config.model`,
`tools.schema`, `machine.schema`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import uuid
from typing import Any, Literal

from agent6 import review_records

# The gate's word at the end: `failed` is an observed red, `unverified` a tree no verdict covers.
Verification = Literal["passed", "failed", "unverified", "not_applicable"]

TernaryMode = Literal["deny", "ask", "allow"]
# `none` runs commands as plain subprocesses: a host with no confinement, or an operator opt-out.
IsolationLevel = Literal["full", "partial", "none"]
NetworkMode = Literal["host", "session", "none"]
RoleName = Literal["default", "helper", "planner", "reviewer", "worker"]
MODEL_ROLES: tuple[RoleName, ...] = ("default", "helper", "planner", "reviewer", "worker")
# The provider's reasoning setting; its meaning depends on the backend.
EffortLevel = Literal["off", "low", "medium", "high", "xhigh", "max"]
# The modes `agent6 resume` accepts; `session_kind` answers the wider "is this a known mode".
ResumableMode = Literal["run", "plan", "ask"]
# The modes `run_task` starts and the Harness drives; `machine` is driven by the machine agent.
LoopMode = Literal["run", "plan", "ask", "read_only", "reviewer"]
# Every session mode, the tool surface's vocabulary.
SessionMode = Literal["run", "plan", "ask", "machine", "read_only", "reviewer"]
# What the REPL's after-commit hook tells the loop to do next; `exit` stops with no prompt.
AutoCommitDirective = Literal["continue", "stop", "undo", "exit"]


class MediaAttachmentError(ValueError):
    """A media attachment has an invalid media type."""


@dataclasses.dataclass(frozen=True, slots=True)
class MediaAttachment:
    """Hold media bytes in memory and derive their identifying metadata.

    Attributes:
        data: The original bytes, excluded from diagnostic representations.
        media_type: One of the supported image or document media types.
        source_path: The path requested by the tool call.
    """

    data: bytes = dataclasses.field(repr=False)
    media_type: str
    source_path: str

    def __post_init__(self) -> None:
        if self.media_type not in {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "application/pdf",
        }:
            raise MediaAttachmentError(f"unsupported media type: {self.media_type!r}")

    @property
    def byte_size(self) -> int:
        """The number of original bytes."""
        return len(self.data)

    @property
    def sha256(self) -> str:
        """The digest of the original bytes."""
        return hashlib.sha256(self.data).hexdigest()

    def metadata(self) -> dict[str, str | int | bool]:
        """Return persistence-safe metadata with an explicit omission marker."""
        return {
            "source_path": self.source_path,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "payload_omitted": True,
        }

    @property
    def modality(self) -> Literal["image", "document"]:
        """The input modality described by this media type."""
        return "document" if self.media_type == "application/pdf" else "image"

    def summary(self) -> str:
        """Return the shared text label without payload bytes."""
        return f"{self.modality}: {self.source_path}, {(self.byte_size + 1023) // 1024} KiB"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionKind:
    """What a mode may do, in one record.

    The one owner, derived from the persisted mode string at read time and never
    written, so a change to what a mode may do reinterprets old sessions.

    Attributes:
        name: The mode string.
        role: The model role that drives it.
        edits: May mutate the workspace in-process.
        runs_commands: May execute commands, in the foreground or as background commands.
        clamps_commands: Forces approval even where config says "allow".
        resumable: `agent6 resume` can pick it up; a machine's states are driven by the
            machine agent instead.
        bucket: The directory under `sessions/` its sessions get, named after the mode.
    """

    name: str
    role: RoleName
    edits: bool
    runs_commands: bool
    clamps_commands: bool
    resumable: bool
    bucket: str


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
            bucket="runs",
        ),
        # Operator-present like ask, so it clamps commands to ask.
        SessionKind(
            name="plan",
            role="planner",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
            bucket="plans",
        ),
        # No edit or graph tools; a jailed command may write the workspace, so it is approval-gated.
        SessionKind(
            name="ask",
            role="worker",
            edits=False,
            runs_commands=True,
            clamps_commands=True,
            resumable=True,
            bucket="asks",
        ),
        # A read-only machine state: the deliverable is the finish payload.
        SessionKind(
            name="machine",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
            bucket="machines",
        ),
        # A machine `agent` state's reading loop; its record lives in the machine's instance.
        SessionKind(
            name="read_only",
            role="worker",
            edits=False,
            runs_commands=False,
            clamps_commands=False,
            resumable=False,
            bucket="machines",
        ),
        # A child whose task is a review: it investigates under its parent's policy and
        # answers with a verdict. It edits nothing and dispatches nothing.
        SessionKind(
            name="reviewer",
            role="worker",
            edits=False,
            runs_commands=True,
            clamps_commands=False,
            resumable=False,
            bucket="reviewers",
        ),
    )
}


# Only resumable modes start outside the machine supervisor.
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
        The kind's bucket, such as `runs`.

    Raises:
        UnknownSessionKindError: The mode is unknown.
    """
    return session_kind(name).bucket


def bucket_mode(bucket: str) -> str:
    """Return the mode whose sessions a bucket holds.

    Args:
        bucket: A directory name under `sessions/`, such as `runs`.

    Returns:
        The mode, such as `run`.

    Raises:
        UnknownSessionKindError: No mode has the bucket.
    """
    for kind in SESSION_KINDS.values():
        if kind.bucket == bucket:
            return kind.name
    raise UnknownSessionKindError(f"no session mode has the bucket {bucket!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ConversationContract:
    """Bound one conversation the Harness drives in place of a worker's.

    The session it drives is a `reviewer`. It investigates with the tools its parent hands
    down and passes a judgement up; it makes no edit and dispatches no child.

    Attributes:
        system: The system prompt, in place of the worker's assembled one.
        opening: The first user message, in place of the task block.
        max_turns: The turns allowed; a last turn that still asks for tools ends the
            conversation `max_turns` with its text kept and those tools not run.
        temperature: The sampling temperature; None keeps the worker's.
        review_binding: Parent-owned review provenance, or None for other conversations.
        allowed_tools: The tools the parent hands down: the ones its own model is offered
            now, within the `reviewer` mode's surface.
    """

    system: str
    opening: str
    max_turns: int
    temperature: float | None = None
    review_binding: review_records.ReviewBinding | None = None
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("a conversation contract allows at least one turn")


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
        timed_out: The runner ended the command at its deadline.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command exited 0."""
        return self.returncode == 0


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundHandoff:
    """A command that outlived its check-in and is still running.

    Its own type: a return code invented for a running command would be a lie every
    caller has to ignore.

    Attributes:
        argv: The command.
        handle: Its launcher-instance-scoped job handle.
        stdout: What it printed before the hand-off.
        stderr: What it printed to stderr before the hand-off.
        duration_s: How long it had run at the hand-off.
    """

    argv: tuple[str, ...]
    handle: str
    stdout: str
    stderr: str
    duration_s: float


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
            by every descendant; -1 disables, since capping costs real builds more than it buys.
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
    memory_limit_mib: int = -1


ControlValue = str | int | tuple[str, ...]
EnforcementOwner = Literal["agent6", "container", "host file tools", "unsupported"]
ProbeState = Literal["not probed", "resolved", "prepared"]


@dataclasses.dataclass(frozen=True, slots=True)
class EffectiveControl:
    """One configured sandbox leaf and the enforcement actually selected for it.

    Attributes:
        leaf: The sandbox configuration field.
        configured: Its configured value before target resolution.
        effective: Its resolved value for this target.
        behaviour: The control's effective behaviour.
        owner: What enforces the control, or unsupported when it is not enforced.
        probe: How far target preparation confirmed the control.
        reason: The refusal or warning reason, if any.
    """

    leaf: str
    configured: ControlValue
    effective: ControlValue
    behaviour: str
    owner: EnforcementOwner
    probe: ProbeState = "resolved"
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    """The command environment, independent of the transport that carries it."""

    variables: tuple[tuple[str, str], ...]
    inherit: bool = False


EnvironmentLifecycle = Literal["left_running", "stopped", "unknown", "not_applicable"]


@dataclasses.dataclass(frozen=True, slots=True)
class EnvironmentOutcome:
    """What became of a prepared environment when its execution ended.

    Attributes:
        outcome: The observed lifecycle state; `not_applicable` means agent6 manages no lifecycle.
        reason: The observation's explanation, never an authorization to stop anything.
    """

    outcome: EnvironmentLifecycle
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """Immutable execution facts; None means unobserved, never an inferred identity.

    Local `user` is the decimal UID inherited by commands as their real and effective UID.

    Lifecycle ownership refers to an external lease, not mutable job or cleanup state.
    """

    transport: str
    host_root: pathlib.Path
    command_root: pathlib.Path
    user: str | None
    home: str | None
    environment: ExecutionEnvironment
    controls: tuple[EffectiveControl, ...] = ()
    runtime_id: str | None = None
    container_id: str | None = None
    launcher_instance: str | None = None
    protocol_version: int | None = None
    launcher_digest: str | None = None
    lifecycle_ref: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionEnvironmentPolicy:
    """Persist environment policy and variable names, never their potentially secret values.

    Attributes:
        inherit: Whether commands inherit the target environment before applying variables.
        variable_names: Names explicitly supplied to commands, sorted; values are omitted.
    """

    inherit: bool
    variable_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionTargetStamp:
    """The persistence-safe projection of an ExecutionTarget, with unknown facts preserved.

    Attributes:
        transport: The recorded command transport, distinct from local confinement.
        host_root: The host workspace used by file tools.
        command_root: The workspace used by commands.
        user: The selected command user, or None when unobserved.
        home: The command HOME, or None when unobserved.
        environment: Inheritance policy and variable names, without values.
        controls: Configured and effective controls with their enforcement owners.
        runtime_id: Observed runtime identity, or None when unknown.
        container_id: Observed container identity, or None when unknown.
        launcher_instance: Supervisor-reported instance identity, or None when unknown.
        protocol_version: Observed supervisor protocol version, or None without a handshake.
        launcher_digest: The launcher binary's `sha256:<hex>` where it was transferred, or None
            when unobserved.
        lifecycle_ref: Recorded lifecycle ownership reference, or None when unknown.
    """

    transport: str
    host_root: pathlib.Path
    command_root: pathlib.Path
    user: str | None
    home: str | None
    environment: ExecutionEnvironmentPolicy
    controls: tuple[EffectiveControl, ...] = ()
    runtime_id: str | None = None
    container_id: str | None = None
    launcher_instance: str | None = None
    protocol_version: int | None = None
    launcher_digest: str | None = None
    lifecycle_ref: str | None = None


def read_target_stamp(data: object) -> object:
    """Read a stamp record written before `launcher_version` became `launcher_digest`.

    Args:
        data: The record as persisted.

    Returns:
        The record under the current field names; any other input unchanged.
    """
    if isinstance(data, dict) and "launcher_version" in data:
        renamed = dict(data)
        version = renamed.pop("launcher_version")
        renamed.setdefault("launcher_digest", version)
        return renamed
    return data


@dataclasses.dataclass(frozen=True, slots=True)
class CallContext:
    """Identify one logical model call across its request attempts.

    Attributes:
        model_call_id: The call's id, shared by its events and transcripts.
        panel_id: The review panel's id; None outside a panel.
        seat_index: The seat's zero-based roster position; None outside a review seat.
        execution_id: The execution the call runs in; None for a call outside one.
    """

    model_call_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    panel_id: str | None = None
    seat_index: int | None = None
    execution_id: str | None = None

    def event_fields(self) -> dict[str, Any]:
        """Return the call identity, omitting absent seat and execution fields."""
        fields: dict[str, Any] = {"model_call_id": self.model_call_id}
        if self.panel_id is not None:
            fields["panel_id"] = self.panel_id
        if self.seat_index is not None:
            fields["seat_index"] = self.seat_index
        if self.execution_id is not None:
            fields["execution_id"] = self.execution_id
        return fields


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallOrigin:
    """The model call a tool call answers, and the provider's id for its tool-use block.

    Attributes:
        call: The model call whose response holds the block.
        provider_tool_call_id: The provider's tool-use id, the one the result is returned under.
        turn: The turn the call belongs to; None outside the turn loop.
    """

    call: CallContext
    provider_tool_call_id: str
    turn: int | None = None

    def event_fields(self) -> dict[str, Any]:
        """Return the origin's fields for a tool.call or tool.result row."""
        fields = {**self.call.event_fields(), "provider_tool_call_id": self.provider_tool_call_id}
        if self.turn is not None:
            fields["turn"] = self.turn
        return fields


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
class ModelSettings:
    """A route with the sampling settings that travel with it.

    Attributes:
        route: The provider and model.
        effort: The reasoning effort, None for the backend's default.
        temperature: The sampling temperature, None for the provider's default.
    """

    route: ModelRoute
    effort: EffortLevel | None = None
    temperature: float | None = None


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
        status: PASS, FAIL, WARN for a jail that runs degraded, or SKIP when the probe does
            not apply.
        detail: What it found.
    """

    name: str
    status: Literal["PASS", "FAIL", "WARN", "SKIP"]
    detail: str


def worktree_setup_fields(result: CommandResult) -> dict[str, Any]:
    """Return the shared worktree.setup event body.

    Args:
        result: The setup command's result.

    Returns:
        The command, its exit code, duration and exec failure, and its stdout and stderr,
        each cut to its last 2000 characters.
    """
    return {
        "command": list(result.argv),
        "exit_code": result.returncode,
        "duration_s": result.duration_s,
        "exec_failed": result.exec_failed,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
