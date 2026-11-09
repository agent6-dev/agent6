# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Persist supervisor-owned output and job outcomes on the host."""

from __future__ import annotations

import dataclasses
import json
import os
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
        paths.mkdir_for_real_user(outcome_dir)
        self._session = session
        self._handle = handle
        self._outcome_dir = outcome_dir
        self._owns_session = owns_session
        self._final: jail_session.BackgroundStatus | None = None
        self._stopped = False
        self._copied = 0
        self._lost = ""
        self._capture_error = ""

    def status(self) -> jail_session.BackgroundStatus:
        """Return the supervisor's status, retaining and persisting terminal answers."""
        if self._final is not None:
            return self._settle(self._final)
        try:
            status = self._session.status_background(self._handle)
        except jail.JailUnavailableError as exc:
            self._lost = self._lost or str(exc)
            status = jail_session.BackgroundStatus(running=False, returncode=None, error=self._lost)
        self._capture_error = status.capture_error or self._capture_error
        if not status.running:
            return self._settle(status)
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
            self._lost = self._lost or str(exc)
            self._settle(
                jail_session.BackgroundStatus(
                    running=False,
                    returncode=self._final.returncode if self._final is not None else None,
                    error=self._lost,
                )
            )
            return self._lost
        else:
            code = stopped.returncode
            if self._final is not None and self._final.returncode is not None:
                code = self._final.returncode
            error = (
                jail.survivors_message(stopped.survivors) if stopped.survivors else stopped.error
            )
            self._capture_error = stopped.capture_error or self._capture_error
            self._settle(jail_session.BackgroundStatus(running=False, returncode=code, error=error))
            self._stopped = not error
            return error

    def copy_output(self, log_fd: int) -> str:
        """Copy one size snapshot, advancing only bytes successfully written on the host.

        Args:
            log_fd: The host log descriptor opened before the command started.

        Returns:
            Why output is incomplete, or an empty string.

        Raises:
            OSError: Host persistence failed; the next call retries from the copied offset.
        """
        if self._lost:
            return f"output incomplete: {self._output_error()}"
        end: int | None = None
        try:
            while end is None or self._copied < end:
                limit = (
                    jail_session.READ_LIMIT
                    if end is None
                    else min(jail_session.READ_LIMIT, end - self._copied)
                )
                read = self._session.read_background(self._handle, offset=self._copied, limit=limit)
                if end is None:
                    end = read.size
                self._capture_error = read.capture_error or self._capture_error
                data = read.data
                while data:
                    written = os.pwrite(log_fd, data, self._copied)
                    if written <= 0:
                        raise OSError("background log write made no progress")
                    self._copied += written
                    data = data[written:]
        except jail.JailUnavailableError as exc:
            self._lost = self._lost or str(exc)
            self.status()
        if self._final is not None:
            self._settle(self._final)
        detail = self._output_error()
        return f"output incomplete: {detail}" if detail else ""

    def close(self) -> str:
        """Close a private session after the caller has copied its final capture.

        Returns:
            An empty string after confirmed cleanup, otherwise its failure reason.
        """
        return self._session.close() if self._owns_session else ""

    def _output_error(self) -> str:
        """Return capture and transport errors without appending cached detail."""
        return "; ".join(filter(None, (self._capture_error, self._lost)))

    def _settle(self, status: jail_session.BackgroundStatus) -> jail_session.BackgroundStatus:
        self._final = dataclasses.replace(status, capture_error=self._output_error())
        record = self._outcome_dir / "result.json"
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(dataclasses.asdict(self._final)), encoding="utf-8")
        temporary.replace(record)
        return self._final


def start_in_jail(
    policy: kinds.JailPolicy,
    *,
    outcome_dir: pathlib.Path,
    session_net: jail.SessionNetwork | None = None,
    session: jail_session.JailSession | None = None,
) -> SessionJob:
    """Start a background job through a prepared or private local command session.

    Args:
        policy: The confinement and command.
        outcome_dir: Where observed outcomes are persisted on the host.
        session_net: The run's session network, when one exists.
        session: The prepared session; None explicitly opens a private local session.

    Returns:
        The job; stop it and copy its output before closing its private session.

    Raises:
        jail.JailUnavailableError: The session could not start the command.
    """
    paths.mkdir_for_real_user(outcome_dir)
    owns_session = session is None
    if session is None:
        session = jail_session.JailSession.open(policy, session_net=session_net)
    try:
        handle = session.start_background(policy.argv, env=policy.env)
    except BaseException:
        if owns_session:
            session.close()
        raise
    return SessionJob(session, handle, outcome_dir, owns_session=owns_session)
