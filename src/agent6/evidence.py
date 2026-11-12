"""Keep verification observations separate from claims about the final tree."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import pydantic

from agent6 import kinds


@dataclasses.dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Record one verification attempt and the content it observed.

    Attributes:
        verify_id: The verification attempt identifier, or null when unrecorded.
        execution_id: The execution that ran the attempt, or null when unrecorded.
        cmd: The verification command's argv.
        tree_before: The observed tree before the attempt, or null when unknown.
        tree_after: The observed tree after the attempt, or null when unknown.
        scoped: Whether verification ran over selected paths instead of the full gate.
        selected_paths: The paths selected for scoped verification.
        outcome: The attempt's observed outcome.
        exit_code: The process exit code, or null without an observed exit.
        duration_s: The elapsed attempt time in seconds.
        timeout_s: The configured timeout in seconds.
        stdout_tail: The retained standard-output tail.
        stderr_tail: The retained standard-error tail.
        stdout_truncated: Whether standard output was truncated, or null when unrecorded.
        stderr_truncated: Whether standard error was truncated, or null when unrecorded.
        baseline: Whether this was the pre-run baseline attempt.
        reason: Why the command did not run, separate from its output.
        target_id: The execution-target stamp, or null when unrecorded.
    """

    verify_id: str | None
    execution_id: str | None
    cmd: tuple[str, ...]
    tree_before: str | None
    tree_after: str | None
    scoped: bool
    selected_paths: tuple[str, ...]
    outcome: Literal["passed", "failed", "timed_out", "interrupted", "denied", "unavailable"]
    exit_code: int | None
    duration_s: float
    timeout_s: float
    stdout_tail: str
    stderr_tail: str
    stdout_truncated: bool | None
    stderr_truncated: bool | None
    baseline: bool = False
    reason: str = ""
    target_id: str | None = None

    @property
    def stable_tree(self) -> bool:
        """Whether both observations identify the same tree."""
        return bool(self.tree_before) and self.tree_before == self.tree_after

    def certifies(self, tree: str | None) -> bool:
        """Return whether this pass covers the supplied tree.

        Args:
            tree: The tree to inspect.

        Returns:
            Whether an unchanged, known tree passed.
        """
        return self.outcome == "passed" and self.stable_tree and self.tree_after == tree


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionTargetEvidence:
    """Bind observed target facts to an execution, without inventing launcher identity.

    Attributes:
        target_id: A host-issued evidence reference, not a launcher instance identifier.
        execution_id: The execution that selected the target, or None when unrecorded.
        target: Persistence-safe target facts; unknown observations remain None.
    """

    target_id: str
    execution_id: str | None = None
    target: Annotated[kinds.ExecutionTargetStamp, pydantic.BeforeValidator(kinds.read_target_stamp)]


@dataclasses.dataclass(frozen=True, slots=True)
class SetupEvidence:
    """Retain setup identity and output independently of later baseline attempts.

    Attributes:
        execution_id: The execution that ran setup, or None when unrecorded.
        target_id: The setup target reference, or None when unrecorded.
        cmd: The setup command's argv.
        exit_code: The observed exit code, or None when unknown.
        duration_s: Elapsed setup time in seconds.
        exec_failed: Whether the command could not be executed.
        stdout_tail: Retained standard output.
        stderr_tail: Retained standard error.
    """

    execution_id: str | None = None
    target_id: str | None = None
    cmd: tuple[str, ...] = ()
    exit_code: int | None = None
    duration_s: float = 0.0
    exec_failed: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class JobsObservation:
    """Record whether the command owner confirmed job settlement, not environment shutdown.

    Attributes:
        execution_id: The execution whose jobs were observed, or None when unrecorded.
        target_id: The observed target reference, or None when unrecorded.
        outcome: Whether the command owner confirmed settlement.
        reason: Why settlement is unknown; empty when confirmed.
    """

    execution_id: str | None = None
    target_id: str | None = None
    outcome: Literal["settled", "unknown"]
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentObservation:
    """Record an environment lifecycle observation without granting stop authority.

    Attributes:
        execution_id: The execution observing the lifecycle, or None when unrecorded.
        target_id: The observed target reference, or None when unrecorded.
        outcome: The observed lifecycle state; not_applicable means the host is not managed.
        reason: The observation's explanation, not an authorization to stop anything.
    """

    execution_id: str | None = None
    target_id: str | None = None
    outcome: Literal["left_running", "stopped", "unknown", "not_applicable"]
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AcquisitionObservation:
    """Record the container lease an execution acquired, as the lifecycle said it.

    Attributes:
        execution_id: The acquiring execution, or None when unrecorded.
        lifecycle_ref: The lease key, the target's `lifecycle_ref`.
        role: `owner` when this run's start made the container appear, else `borrower`.
        reason: Why the run holds that role.
        started: This run's start command made the container appear.
        start_s: How long the start took in seconds; None when none ran.
    """

    execution_id: str | None = None
    lifecycle_ref: str
    role: str
    reason: str = ""
    started: bool = False
    start_s: float | None = None


TARGET = pydantic.TypeAdapter(ExecutionTargetEvidence)
ACQUISITION = pydantic.TypeAdapter(AcquisitionObservation)
SETUP = pydantic.TypeAdapter(SetupEvidence)
JOBS = pydantic.TypeAdapter(JobsObservation)
ENVIRONMENT = pydantic.TypeAdapter(EnvironmentObservation)
_EVIDENCE = pydantic.TypeAdapter(VerificationEvidence)


class EvidenceError(ValueError):
    """Recorded execution evidence is malformed or contradicts an immutable stamp."""


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionTargets:
    """Fold immutable target history and its current execution binding.

    Attributes:
        recorded: Valid target stamps in observation order.
        current: The prepared or started execution's target, None when unproved.
        error: The latest target record's validation error, empty for a valid record.
    """

    recorded: tuple[ExecutionTargetEvidence, ...] = ()
    current: ExecutionTargetEvidence | None = None
    error: str = ""

    def observe(self, event: Mapping[str, Any]) -> ExecutionTargets:
        """Fold one record, clearing current evidence at an unmatched execution boundary.

        Args:
            event: A journal record, including preparation before session.start.

        Returns:
            The updated fold; invalid target records retain history but clear current evidence.
        """
        if event.get("type") == "execution.target":
            try:
                stamp = TARGET.validate_python(event)
            except (ValueError, TypeError) as exc:
                return dataclasses.replace(self, current=None, error=str(exc))
            previous = next((t for t in self.recorded if t.target_id == stamp.target_id), None)
            if previous is not None and previous != stamp:
                return dataclasses.replace(
                    self, current=None, error=f"Conflicting target stamp: {stamp.target_id}"
                )
            return ExecutionTargets(
                recorded=self.recorded if previous is not None else (*self.recorded, stamp),
                current=stamp if stamp.execution_id else None,
            )
        if (
            event.get("type") in {"session.start", "session.resume", "session.stop"}
            and self.current is not None
            and self.current.execution_id != event.get("execution_id")
        ):
            return dataclasses.replace(self, current=None)
        return self


@dataclasses.dataclass(slots=True)
class ExecutionObservations:
    """Fold target, setup and lifecycle records without inferring missing references."""

    targets: ExecutionTargets = dataclasses.field(default_factory=ExecutionTargets)
    setups: list[SetupEvidence] = dataclasses.field(default_factory=list)
    jobs: list[JobsObservation] = dataclasses.field(default_factory=list)
    environments: list[EnvironmentObservation] = dataclasses.field(default_factory=list)
    acquisitions: list[AcquisitionObservation] = dataclasses.field(default_factory=list)

    def observe(self, event: Mapping[str, Any]) -> None:
        """Validate one observation and retain its explicit identities.

        Args:
            event: A journal record; unrelated event types are ignored.

        Raises:
            EvidenceError: A target stamp is malformed or contradicts an earlier stamp.
            ValidationError: A setup or lifecycle observation is malformed.
        """
        self.targets = self.targets.observe(event)
        match event.get("type"):
            case "execution.target":
                if self.targets.error:
                    raise EvidenceError(self.targets.error)
            case "workspace.setup":
                self.setups.append(SETUP.validate_python(event))
            case "execution.jobs":
                self.jobs.append(JOBS.validate_python(event))
            case "execution.environment":
                self.environments.append(ENVIRONMENT.validate_python(event))
            case "environment.acquired":
                self.acquisitions.append(ACQUISITION.validate_python(event))
            case _:
                pass


def target_record(
    target_id: str, execution_id: str | None, target: kinds.ExecutionTarget
) -> ExecutionTargetEvidence:
    """Project a prepared target into persistence-safe evidence without environment values.

    Args:
        target_id: The host-issued evidence reference.
        execution_id: The execution that selected this target.
        target: The resolved command target.

    Returns:
        Immutable target evidence with the supplied execution and target references.
    """
    return ExecutionTargetEvidence(
        target_id=target_id,
        execution_id=execution_id,
        target=kinds.ExecutionTargetStamp(
            transport=target.transport,
            host_root=target.host_root,
            command_root=target.command_root,
            user=target.user,
            home=target.home,
            environment=kinds.ExecutionEnvironmentPolicy(
                inherit=target.environment.inherit,
                variable_names=tuple(sorted(name for name, _value in target.environment.variables)),
            ),
            controls=target.controls,
            runtime_id=target.runtime_id,
            container_id=target.container_id,
            launcher_instance=target.launcher_instance,
            protocol_version=target.protocol_version,
            launcher_digest=target.launcher_digest,
            lifecycle_ref=target.lifecycle_ref,
        ),
    )


def target_lines(target_id: str, target: kinds.ExecutionTargetStamp) -> tuple[str, ...]:
    """Describe recorded target facts without probing the environment.

    Args:
        target_id: The stamp's recorded reference.
        target: The persisted target facts.

    Returns:
        The target identity, mapping and effective-control lines.
    """
    lines = [
        f"Environment: {target.transport} (target {target_id})",
        f"Roots: host {target.host_root}; commands {target.command_root}",
        f"User: {target.user or 'unknown'}; HOME: {target.home or 'unknown'}",
        f"Launcher: {target.launcher_instance or 'unknown'}; "
        f"digest: {target.launcher_digest or 'unknown'}; "
        "protocol: "
        + str(target.protocol_version if target.protocol_version is not None else "unknown"),
        f"Environment policy: {'inherited' if target.environment.inherit else 'explicit'}; "
        f"lifecycle ownership: {target.lifecycle_ref or 'unknown'}",
    ]
    if target.runtime_id is not None or target.container_id is not None:
        lines.append(
            f"Runtime: {target.runtime_id or 'unknown'}; "
            f"container: {target.container_id or 'unknown'}"
        )
    for control in target.controls:
        if control.reason or control.owner != "agent6" or control.leaf == "sandbox.isolation":
            reason = f"; {control.reason}" if control.reason else ""
            lines.append(
                f"{control.leaf}: {control.behaviour} ({control.owner}, {control.probe}){reason}"
            )
    return tuple(lines)


def outcome(event: Mapping[str, Any]) -> str:
    """Return a recorded outcome or infer only what a historical exit code establishes.

    Args:
        event: A finished verification event.

    Returns:
        The attempt's outcome, never an in-flight state.
    """
    recorded = event.get("outcome")
    if isinstance(recorded, str) and recorded:
        return recorded
    code = event.get("exit_code")
    if event.get("timed_out") is True:
        return "timed_out"
    if isinstance(code, int) and code < 0:
        return "interrupted"
    return "passed" if code == 0 else "failed" if isinstance(code, int) else "unavailable"


def result_label(exit_code: object, outcome: str, reason: str = "", *, joiner: str = " ") -> str:
    """Name an exit code when present, otherwise the finished attempt's outcome.

    Args:
        exit_code: The command's exit code, if it ran to an observed exit.
        outcome: The recorded attempt's outcome.
        reason: The reason the command did not run, if recorded.
        joiner: What joins "exit" to the code: a space in prose, "=" on a key=value line.

    Returns:
        A result label that never prints a missing code.
    """
    label = f"exit{joiner}{exit_code}" if isinstance(exit_code, int) else outcome
    return f"{label}: {reason}" if reason else label


def from_event(event: Mapping[str, Any]) -> VerificationEvidence:
    """Validate a verification event, leaving historical capture completeness unknown.

    Args:
        event: A journal verify.end record.

    Returns:
        The validated evidence.

    Raises:
        ValidationError: A recorded field is malformed.
    """
    return _EVIDENCE.validate_python(
        {
            "verify_id": event.get("verify_id"),
            "execution_id": event.get("execution_id"),
            "target_id": event.get("target_id"),
            "cmd": event.get("cmd", ()),
            "tree_before": event.get("tree_before"),
            "tree_after": event.get("tree_after"),
            "scoped": event.get("scoped", False),
            "selected_paths": event.get("selected_paths", ()),
            "outcome": outcome(event),
            "exit_code": event.get("exit_code"),
            "duration_s": event.get("duration_s", 0.0),
            "timeout_s": event.get("timeout_s", 0.0),
            "stdout_tail": event.get("stdout_tail", ""),
            "stderr_tail": event.get("stderr_tail", ""),
            "stdout_truncated": event.get("stdout_truncated"),
            "stderr_truncated": event.get("stderr_truncated"),
            "reason": event.get("reason", ""),
        }
    )


def capture_label(truncated: bool | None) -> str:
    """Return the output cap's meaning without guessing historical completeness.

    Args:
        truncated: Whether capture lost output, None when unknown.

    Returns:
        A plain-language capture label.
    """
    if truncated is None:
        return "capture completeness unknown"
    return "truncated" if truncated else "complete capture"
