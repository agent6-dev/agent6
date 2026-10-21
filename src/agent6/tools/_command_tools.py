# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Jailed command execution, check-ins and background tool handlers."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import Any, Literal, Protocol

from agent6 import kinds
from agent6.sandbox import jail, jail_session
from agent6.tools import _result_format, background, results, schema
from agent6.tools import errors as tools_errors


class JailPolicyFactory(Protocol):
    """Resolve a command's policy with optional timeout and writable grants."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        extra_rw_paths: tuple[pathlib.Path, ...] = (),
    ) -> kinds.JailPolicy: ...


@dataclasses.dataclass(frozen=True, slots=True)
class JailAccess:
    """The dispatcher's lazy access to jail policy, session and network."""

    get_policy: JailPolicyFactory
    get_session: Callable[[], jail_session.JailSession | None]
    net_for: background.NetFor


def _roster(shells: background.BackgroundShells) -> tuple[str, ...]:
    """Return the background roster as lines."""
    return tuple(v.line() for v in shells.roster())


def start_detached(
    argv: tuple[str, ...],
    *,
    mode: Literal["run", "plan", "ask", "machine"],
    get_background: Callable[[], background.BackgroundShells],
    access: JailAccess,
) -> results.ExecResult:
    """Start a command detached: the same hand-back as a check-in, at zero seconds.

    Args:
        argv: The command.
        mode: The session's mode; only a session that edits owns a background command's
            lifetime, derived from the same tool set that withholds read_background elsewhere.
        get_background: Returns the run's background roster.
        access: The dispatcher's lazy jail accessors.

    Returns:
        The hand-back, with no output yet.

    Raises:
        ToolError: The mode cannot read a hand-back, or the command could not start.
    """
    if schema.ReadBackgroundInput.TOOL_NAME not in schema.mode_tools(mode).permitted:
        raise tools_errors.ToolError(
            f"background commands are not available in {mode} mode:"
            " nothing there could read or stop one before the run ends"
        )
    shells = get_background()
    try:
        view = shells.start(
            argv,
            lambda a, rw: access.get_policy(a, extra_rw_paths=rw),
            session=access.get_session(),
            net_for=access.net_for,
        )
    except background.BackgroundError as exc:
        raise tools_errors.ToolError(str(exc)) from exc
    return results.ExecResult(
        returncode=None,
        stdout="",
        stderr="",
        duration_s=0.0,
        exec_failed=False,
        background_id=view.id,
    )


def run_model_command(
    argv: tuple[str, ...],
    *,
    mode: Literal["run", "plan", "ask", "machine"],
    shells: background.BackgroundShells | None,
    checkin: float,
    interrupted: Callable[[], bool],
    emit: Callable[..., None],
    access: JailAccess,
) -> results.ExecResult:
    """Run a model command; check-ins hand it back and an immediate stop ends it.

    The check-in needs a jail session to own the running command, a background roster to
    hand it to, and a mode whose tools can read the hand-back; otherwise this is a bounded
    run with its timeout.

    Args:
        argv: The command.
        mode: The session's mode.
        shells: The run's background roster; None without a run directory.
        checkin: Seconds until a running command is handed back; 0 disables the hand-back.
        interrupted: Whether the operator asked for an immediate stop.
        emit: The run's event sink, for `command.backgrounded`.
        access: The dispatcher's lazy jail accessors.

    Returns:
        The outcome, or the hand-back with the output so far.

    Raises:
        ToolError: The jail is unavailable or the operator's stop could not end the command.
    """
    session = access.get_session()
    if (
        session is None
        or shells is None
        or checkin <= 0
        or schema.ReadBackgroundInput.TOOL_NAME not in schema.mode_tools(mode).permitted
    ):
        return run_argv_in_jail(
            argv,
            label="run_command",
            access=access,
        )
    try:
        policy = access.get_policy(argv)
        outcome = session.run(
            argv,
            env=policy.env,
            timeout_s=0.0,  # the check-in replaces the kill
            checkin_s=checkin,
            log_dir=str(shells.log_root),
            # An immediate stop asks for the hand-back now; a stop after the step waits.
            interrupted=interrupted,
        )
    except jail.JailUnavailableError as exc:
        raise tools_errors.ToolError(f"jail unavailable: {exc}") from exc
    if isinstance(outcome, kinds.CommandResult):
        return _result_format.exec_result(outcome)
    view = shells.adopt(outcome, session=session)
    stopped_by_operator = interrupted()
    if stopped_by_operator:
        view = shells.stop(view.id)
        if view.detail:
            raise tools_errors.ToolError(
                f"operator's stop could not end the command: {view.detail}"
            )
    else:
        emit("command.backgrounded", id=view.id, pid=outcome.pid, seconds=outcome.duration_s)
    return results.ExecResult(
        returncode=view.returncode if stopped_by_operator else None,
        stdout=_result_format.clip_tail(outcome.stdout),
        stderr=_result_format.clip_tail(outcome.stderr),
        duration_s=outcome.duration_s,
        exec_failed=False,
        background_id="" if stopped_by_operator else view.id,
        stopped_by_operator=stopped_by_operator,
    )


def read_background(
    raw: dict[str, Any],
    *,
    get_background: Callable[[], background.BackgroundShells],
    checkin: float,
    interrupted: Callable[[], bool],
) -> results.BackgroundResult:
    """Read a background command's output, or the roster when no id is given.

    Args:
        raw: The tool call's arguments.
        get_background: Returns the run's background roster.
        checkin: Seconds to wait for the command to end when the call gives no `wait_s`.
        interrupted: Cuts the wait short.

    Returns:
        The roster, with the output when an id was given.

    Raises:
        ToolError: No run directory was wired, or the id is unknown.
    """
    args = schema.ReadBackgroundInput.model_validate(raw)
    shells = get_background()
    if not args.id:
        return results.BackgroundResult(shells=_roster(shells))
    wait_s = checkin if args.wait_s is None else args.wait_s
    try:
        _view, output = shells.read(
            args.id,
            tail_lines=args.tail_lines,
            wait_s=wait_s,
            interrupted=interrupted,
        )
    except background.BackgroundError as exc:
        raise tools_errors.ToolError(str(exc)) from exc
    return results.BackgroundResult(shells=_roster(shells), output=output)


def stop_background(
    raw: dict[str, Any],
    *,
    get_background: Callable[[], background.BackgroundShells],
) -> results.BackgroundResult:
    """Stop a background command.

    Args:
        raw: The tool call's arguments.
        get_background: Returns the run's background roster.

    Returns:
        The roster after the stop.

    Raises:
        ToolError: No run directory was wired, or the id is unknown.
    """
    args = schema.StopBackgroundInput.model_validate(raw)
    shells = get_background()
    try:
        shells.stop(args.id)
    except background.BackgroundError as exc:
        raise tools_errors.ToolError(str(exc)) from exc
    return results.BackgroundResult(shells=_roster(shells))


def run_argv_raw(
    argv: tuple[str, ...],
    *,
    access: JailAccess,
    label: str,
    timeout_s: float | None = None,
) -> tuple[kinds.CommandResult, float]:
    """Run an argv in the jail without a check-in.

    Args:
        argv: The command.
        access: The dispatcher's lazy jail accessors.
        label: The tool's name, for the error.
        timeout_s: The wall-clock limit; the policy's default when None.

    Returns:
        The unclipped outcome (a caller parsing the metric score needs the real bytes) and
        the timeout the policy resolved.

    Raises:
        ToolError: The jail is unavailable.
    """
    try:
        policy = access.get_policy(argv, timeout_s=timeout_s)
        session = access.get_session()
        # The operator's gate needs a verdict, not a handle.
        outcome = (
            session.run(argv, env=policy.env, timeout_s=policy.timeout_s)
            if session is not None
            else jail.run_in_jail(policy, session_net=access.net_for(policy))
        )
    except jail.JailUnavailableError as exc:
        raise tools_errors.ToolError(f"{label}: jail unavailable: {exc}") from exc
    if isinstance(outcome, kinds.BackgroundHandoff):  # pragma: no cover - no check-in was asked for
        raise tools_errors.ToolError(
            f"{label}: the jail handed back a command that was never detachable"
        )
    return outcome, policy.timeout_s


def run_argv_in_jail(
    argv: tuple[str, ...],
    *,
    access: JailAccess,
    label: str,
    timeout_s: float | None = None,
) -> results.ExecResult:
    """Return the model's view of an argv run in the jail."""
    outcome, timeout = run_argv_raw(
        argv,
        label=label,
        timeout_s=timeout_s,
        access=access,
    )
    return _result_format.exec_result(outcome, timeout_s=timeout)
