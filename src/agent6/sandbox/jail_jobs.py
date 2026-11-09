# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Persist supervisor-owned job outcomes on the host."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib

from agent6 import kinds, paths
from agent6.sandbox import jail, jail_session


class SessionJob:
    """Persist only outcomes received from the owning supervisor.

    Losing the transport without a completion response leaves the outcome unknown.
    """

    def __init__(
        self,
        session: jail_session.JailSession,
        handle: str,
        outcome_dir: pathlib.Path,
        *,
        owns_session: bool = False,
    ) -> None:
        self._session = session
        self._handle = handle
        self._outcome_dir = outcome_dir
        self._owns_session = owns_session
        self._final: jail_session.BackgroundStatus | None = None
        self._stopped = False

    def status(self) -> jail_session.BackgroundStatus:
        """Return the supervisor's status, retaining and persisting terminal answers."""
        if self._final is not None:
            return self._final
        try:
            status = self._session.status_background(self._handle)
        except jail.JailUnavailableError as exc:
            status = jail_session.BackgroundStatus(running=False, returncode=None, error=str(exc))
        if not status.running:
            self._settle(status)
        return status

    def stop(self) -> str:
        """Stop the job's descendants, preserving an already observed outcome.

        Returns:
            An empty string after confirmed cleanup, otherwise the reason it is unknown.
        """
        if self._stopped:
            return ""
        try:
            stopped = self._session.stop_background(self._handle)
        except jail.JailUnavailableError as exc:
            self._settle(
                jail_session.BackgroundStatus(
                    running=False,
                    returncode=self._final.returncode if self._final is not None else None,
                    error=str(exc),
                )
            )
            return str(exc)
        else:
            code = stopped.returncode
            if self._final is not None and self._final.returncode is not None:
                code = self._final.returncode
            error = (
                jail.survivors_message(stopped.survivors) if stopped.survivors else stopped.error
            )
            self._settle(jail_session.BackgroundStatus(running=False, returncode=code, error=error))
            self._stopped = not error
            return error
        finally:
            if self._owns_session:
                self._session.close()

    def _settle(self, status: jail_session.BackgroundStatus) -> None:
        self._final = status
        with contextlib.suppress(OSError):
            (self._outcome_dir / "result.json").write_text(
                json.dumps(dataclasses.asdict(status)), encoding="utf-8"
            )


def start_in_jail(
    policy: kinds.JailPolicy,
    *,
    outcome_dir: pathlib.Path,
    session_net: jail.SessionNetwork | None = None,
) -> SessionJob:
    """Start a background job in its own command session at any isolation level.

    Args:
        policy: The confinement and command.
        outcome_dir: Where observed outcomes are persisted on the host.
        session_net: The run's session network, when one exists.

    Returns:
        The job, whose stop also closes its private session.

    Raises:
        jail.JailUnavailableError: The session could not start the command.
    """
    paths.mkdir_for_real_user(outcome_dir)
    session = jail_session.JailSession.open(policy, session_net=session_net)
    try:
        handle = session.start_background(policy.argv, env=policy.env)
    except BaseException:
        session.close()
        raise
    return SessionJob(session, handle, outcome_dir, owns_session=True)
