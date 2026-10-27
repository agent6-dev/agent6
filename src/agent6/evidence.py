"""Keep verification observations separate from claims about the final tree."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

import pydantic


@dataclasses.dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Record one verification attempt and the content it observed."""

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


_EVIDENCE = pydantic.TypeAdapter(VerificationEvidence)


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
    if code == 124:
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
