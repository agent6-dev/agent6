# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Jailed command execution, check-ins and background tool handlers."""

from __future__ import annotations

import dataclasses
import tempfile
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent6 import kinds
from agent6.sandbox import jail, jail_session
from agent6.sessions import journal
from agent6.tools import (
    background,
    child_tools,
    policy,
    result_format,
    results,
    schema,
    tool_errors,
    waits,
)

if TYPE_CHECKING:
    from agent6.tools import dispatch


@dataclasses.dataclass(frozen=True, slots=True)
class JailAccess:
    """Resolved command policy, its prepared session and the resources the owner releases.

    The owner closes jobs before these resources. Grants also govern host file tools.
    """

    policy: kinds.JailPolicy
    session: jail_session.JailSession
    grants: policy.Grants
    network: jail.SessionNetwork | None = None
    temp: tempfile.TemporaryDirectory[str] | None = None
    owns_network: bool = False
    # Settles the environment's lifecycle after the session closed; None means agent6 manages none.
    lifecycle: Callable[[], kinds.EnvironmentOutcome] | None = None
    # Says what the environment looks like now, for a command that could not reach it.
    environment_state: Callable[[], str] | None = None

    def close(self) -> str:
        """Close the session before releasing its local storage and owned network.

        Returns:
            The session cleanup error, or an empty string.
        """
        try:
            return self.session.close()
        finally:
            if self.temp is not None:
                self.temp.cleanup()
            if self.owns_network and self.network is not None:
                self.network.close()

    def unavailable(self, exc: jail.JailUnavailableError) -> str:
        """Return a jail failure's words, with the environment's state when it is observed."""
        seen = self.environment_state() if self.environment_state is not None else ""
        return f"jail unavailable: {exc}" + (f"; {seen}" if seen else "")

    def settle_environment(self) -> kinds.EnvironmentOutcome:
        """Report the environment's lifecycle once the session is closed.

        Returns:
            The lifecycle's outcome; `not_applicable` for a local target, `unknown` for a
            prepared transport whose lifecycle nothing settles.
        """
        if self.lifecycle is not None:
            return self.lifecycle()
        target = self.session.target
        if target is None or target.transport == "local":
            return kinds.EnvironmentOutcome(
                "not_applicable", "Local host lifecycle is not managed by agent6."
            )
        return kinds.EnvironmentOutcome("unknown", "No lifecycle observation recorded.")


def start_detached(
    argv: tuple[str, ...],
    *,
    get_background: Callable[[], background.BackgroundShells],
    access: JailAccess,
) -> results.ExecResult:
    """Start a command detached: the same hand-back as a check-in, at zero seconds.

    Args:
        argv: The command.
        get_background: Returns the run's background roster.
        access: The resolved command policy and session.

    Returns:
        The hand-back, with no output yet.

    Raises:
        ToolError: The command could not start.
    """
    shells = get_background()
    try:
        view = shells.start(
            argv,
            lambda a, rw: dataclasses.replace(access.policy, argv=a),
            session=access.session,
            net_for=lambda _policy: access.network,
        )
    except background.BackgroundError as exc:
        raise tool_errors.ToolError(str(exc)) from exc
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
    shells: background.BackgroundShells | None,
    checkin: float,
    interrupted: Callable[[], bool],
    emit: Callable[..., None],
    access: JailAccess,
) -> results.ExecResult:
    """Run a model command; check-ins hand it back and an immediate stop ends it.

    The check-in needs a background roster to hand the running command to; otherwise this is
    a bounded run with its timeout.

    Args:
        argv: The command.
        shells: The run's background roster; None without a session directory.
        checkin: Seconds until a running command is handed back; 0 disables the hand-back.
        interrupted: Whether the operator asked for an immediate stop.
        emit: The run's event sink, for `command.backgrounded`.
        access: The resolved command policy and session.

    Returns:
        The outcome, or the hand-back with the output so far.

    Raises:
        ToolError: The jail is unavailable or the operator's stop could not end the command.
    """
    if shells is None or checkin <= 0:
        return run_argv_in_jail(
            argv,
            label="run_command",
            access=access,
        )
    try:
        policy = access.policy
        outcome = access.session.run(
            argv,
            env=policy.env,
            timeout_s=0.0,  # the check-in replaces the kill
            checkin_s=checkin,
            # An immediate stop asks for the hand-back now; a stop after the step waits.
            interrupted=interrupted,
        )
    except jail.JailUnavailableError as exc:
        raise tool_errors.ToolError(access.unavailable(exc)) from exc
    if isinstance(outcome, kinds.CommandResult):
        return result_format.exec_result(outcome)
    view = shells.adopt(outcome, session=access.session)
    stopped_by_operator = interrupted()
    if stopped_by_operator:
        view = shells.stop(view.id)
        if view.detail:
            raise tool_errors.ToolError(f"operator's stop could not end the command: {view.detail}")
    else:
        emit(
            journal.CommandBackgrounded(
                background_id=view.id, handle=outcome.handle, elapsed_s=outcome.duration_s
            )
        )
    return results.ExecResult(
        returncode=view.returncode if stopped_by_operator else None,
        stdout=result_format.clip_tail(outcome.stdout),
        stderr=result_format.clip_tail(outcome.stderr),
        duration_s=outcome.duration_s,
        exec_failed=False,
        background_id="" if stopped_by_operator else view.id,
        stopped_by_operator=stopped_by_operator,
    )


def read_background_command(
    args: schema.ReadBackgroundInput,
    *,
    shells: background.BackgroundShells,
    wait_s: float,
    interrupted: Callable[[], bool],
) -> results.BackgroundResult:
    """Read a background command's output, or the roster when no id is given.

    Args:
        args: The tool call's validated arguments.
        shells: The run's command roster.
        wait_s: The resolved maximum seconds to wait for the command to end.
        interrupted: Cuts the wait short.

    Returns:
        The roster, or the selected command's fields and output tail.

    Raises:
        ToolError: The command id is unknown.
    """
    if not args.id:
        return results.BackgroundRosterResult(shells=tuple(v.line() for v in shells.roster()))
    try:
        with waits.during(wait_s):
            view, output = shells.read(
                args.id,
                tail_lines=args.tail_lines,
                wait_s=wait_s,
                interrupted=interrupted,
            )
    except background.BackgroundError as exc:
        raise tool_errors.ToolError(str(exc)) from exc
    return results.BackgroundReadResult(view, output)


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
        access: The resolved command policy and session.
        label: The tool's name, for the error.
        timeout_s: The wall-clock limit; the policy's default when None.

    Returns:
        The unclipped outcome (a caller parsing the metric score needs the real bytes) and
        the timeout the policy resolved.

    Raises:
        ToolError: The jail is unavailable.
    """
    try:
        policy = dataclasses.replace(
            access.policy,
            argv=argv,
            timeout_s=access.policy.timeout_s if timeout_s is None else timeout_s,
        )
        # The operator's gate needs a verdict, not a handle.
        outcome = access.session.run(argv, env=policy.env, timeout_s=policy.timeout_s)
    except jail.JailUnavailableError as exc:
        raise tool_errors.ToolError(f"{label}: {access.unavailable(exc)}") from exc
    if isinstance(outcome, kinds.BackgroundHandoff):  # pragma: no cover - no check-in was asked for
        raise tool_errors.ToolError(
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
    return result_format.exec_result(outcome, timeout_s=timeout)


def run_command(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.ExecResult:
    """Return the outcome of a command the model chose, or its handle when detached."""
    args = schema.RunCommandInput.model_validate(raw)
    ctx.approve_command("run_command", args.argv)
    if args.background:
        return start_detached(
            args.argv,
            get_background=ctx.get_background,
            access=ctx.jail_access("run_command"),
        )
    with waits.during():
        return run_model_command(
            args.argv,
            shells=ctx.shells,
            checkin=ctx.config.harness.command_checkin_s,
            interrupted=ctx.operator_wants_out_now,
            emit=ctx.emit,
            access=ctx.jail_access("run_command"),
        )


def run_verify_command(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.ExecResult:
    """Return the gate's outcome, run at the model's asking."""
    schema.RunVerifyInput.model_validate(raw)
    return ctx.run_verify()


def run_metric_command(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.MetricResult:
    """Run the configured metric command in the jail and parse its score.

    Args:
        ctx: The session metric and command gates.
        raw: The tool call's arguments; the tool takes none.

    Returns:
        The outcome plus the score: the pattern's first capture group as a float, or None
        when it does not match or parse.

    Raises:
        ToolError: No metric is configured, or the jail is unavailable.
        ToolDeniedError: The operator did not approve.
        OperatorCommandUnexecutableError: The command cannot run in the jail.
    """
    schema.RunMetricInput.model_validate(raw)
    metric = ctx.metric
    if metric is None:
        raise tool_errors.ToolError(
            "no session metric; the operator can set one with --metric or /metric"
        )
    argv = metric.command
    ctx.approve_command("run_metric_command", argv)
    ctx.emit(journal.MetricStart(command=list(argv)))
    outcome, timeout_s = run_argv_raw(
        argv,
        label="metric_command",
        timeout_s=ctx.config.harness.verify.timeout_s,
        access=ctx.jail_access("metric_command"),
    )
    if outcome.exec_failed:
        raise tool_errors.OperatorCommandUnexecutableError(
            f"metric_command {list(argv)} could not be executed in the sandbox: "
            f"{outcome.stderr}. See run_verify_command's note: PATH is /usr/bin:/bin "
            "plus the standard bin dirs; install the tool into one of those on the "
            "host, use a path inside the workspace, or grant its real directory "
            "via sandbox.extra_read_paths."
        )
    # Scored from the unclipped outcome: the display clip's marker matches a loose pattern.
    score = result_format.parse_metric_score(outcome.stdout, outcome.stderr, pattern=metric.pattern)
    res = result_format.exec_result(outcome, timeout_s=timeout_s)
    ctx.emit(
        journal.MetricEnd(
            command=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
            score=score,
        )
    )
    return results.MetricResult.from_exec(res, score)


def read_background(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.BackgroundResult:
    """Return background output with the run's check-in, stop and steer policy.

    An `id` the run's children hold reads that child; any other is a command's.

    Args:
        ctx: The session's background work and wait policy.
        raw: The tool call's arguments.

    Returns:
        The rosters, or the selected job's fields and its output or transcript tail.
    """
    args = schema.ReadBackgroundInput.model_validate(raw)
    checkin = ctx.config.harness.command_checkin_s
    wait_s = checkin if args.wait_s is None else args.wait_s
    limited = checkin > 0 and wait_s >= checkin
    if limited:
        wait_s = checkin
    started = time.monotonic()
    roster = ctx.children.roster() if ctx.children is not None else ()
    if ctx.children is not None and child_tools.addressed(args, roster):
        result = child_tools.read(
            args,
            ctx.children,
            shells=() if args.id else tuple(view.line() for view in ctx.get_background().roster()),
            wait_s=wait_s,
            interrupted=ctx.background_wait_interrupted,
            state_dir=ctx.state_dir,
        )
    else:
        result = read_background_command(
            args,
            shells=ctx.get_background(),
            wait_s=wait_s,
            interrupted=ctx.background_wait_interrupted,
        )
        if isinstance(result, results.BackgroundRosterResult):
            lines = tuple(child.line() for child in roster)
            result = dataclasses.replace(result, children=lines)
    if (
        limited
        and result.running
        and time.monotonic() - started >= checkin
        and not ctx.background_wait_interrupted()
    ):
        note = f"Wait ended at harness.command_checkin_s = {checkin:g}s."
        result = dataclasses.replace(
            result, output=f"{result.output}\n\n{note}" if result.output else note
        )
    return result


def stop_background(
    ctx: dispatch.ToolContext, raw: dict[str, Any]
) -> results.BackgroundRosterResult:
    """Stop a command or a child.

    Args:
        ctx: The session's background work and command policy.
        raw: The tool call's arguments, naming the command or the child by its `id`.

    Returns:
        The row of the command or the child after the stop.

    Raises:
        ToolError: A command stop is refused, no session directory was wired or the id is unknown.
    """
    args = schema.StopBackgroundInput.model_validate(raw)
    if ctx.children is not None and any(c.id == args.id for c in ctx.children.roster()):
        ctx.children.stop(args.id)
        rows = tuple(c.line() for c in ctx.children.roster() if c.id == args.id)
        return results.BackgroundRosterResult(shells=(), children=rows)
    if refusal := ctx.policy_refusal(schema.RunCommandInput.TOOL_NAME):
        raise tool_errors.ToolError(f"command stop {refusal}")
    try:
        view = ctx.get_background().stop(args.id)
    except background.BackgroundError as exc:
        raise tool_errors.ToolError(str(exc)) from exc
    return results.BackgroundRosterResult(shells=(view.line(),))
