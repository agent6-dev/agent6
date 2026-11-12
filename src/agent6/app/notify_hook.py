# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the operator's `[notify].on_event` hook on the host, once per notification.

The argv is operator-controlled and never carries model output, so it runs outside the
jail; `hook_env` gives it the curated base environment plus the `AGENT6_*` facts.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from agent6 import child_env
from agent6.config import NotifyConfig
from agent6.viewmodel import format


def hook_env(**agent6_vars: str) -> dict[str, str]:
    """Build the environment for an operator notify hook.

    Args:
        **agent6_vars: The `AGENT6_*` facts to add.

    Returns:
        The curated base environment plus those facts.
    """
    return child_env.curated_env(extra=agent6_vars)


def run_notify_hook(
    argv: Sequence[str],
    env: dict[str, str],
    *,
    timeout_s: float,
    label: str,
    note: Callable[[str], None],
) -> None:
    """Run one operator notify hook on the host; a failure is noted, never fatal.

    The argv is operator-controlled, never model output, so it runs outside the
    jail. Its stdout is discarded: under `agent6 acp` the parent's stdout is the
    protocol stream.

    Args:
        argv: The hook command.
        env: Its environment.
        timeout_s: How long to wait for it.
        label: The config key named in a failure note.
        note: Receives the failure note.
    """
    try:
        res = subprocess.run(
            list(argv),
            stdout=subprocess.DEVNULL,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        note(f"{label} failed: {exc}")
        return
    if res.returncode != 0:
        note(f"{label} exited {res.returncode}")


def fire(
    notify: NotifyConfig,
    event: str,
    *,
    message: str,
    level: str,
    note: Callable[[str], None],
    **subject: str,
) -> None:
    """Run `[notify].on_event` for one notification, when one is configured.

    Args:
        notify: The notify config.
        event: `session.waiting`, `session.stop`, `machine.notify` or `machine.end`.
        message: The prompt's first line, the end reason, or the machine's notify message.
        level: `info` for a wait or a machine notify (`warn`, `error` as the machine says);
            `ok` or `failed` for an end.
        note: Receives a failure note.
        **subject: The `AGENT6_SESSION_*` or `AGENT6_MACHINE_*` facts.
    """
    if not notify.on_event:
        return
    env = hook_env(
        AGENT6_NOTIFY_EVENT=event,
        AGENT6_NOTIFY_MESSAGE=message,
        AGENT6_NOTIFY_LEVEL=level,
        **subject,
    )
    run_notify_hook(
        notify.on_event, env, timeout_s=notify.timeout_s, label="notify.on_event", note=note
    )


def attention(
    notify: NotifyConfig,
    *,
    session_id: str,
    session_dir: pathlib.Path,
    note: Callable[[str], None],
) -> Callable[[dict[str, Any]], None]:
    """Return a journal listener that fires `session.waiting` as a prompt opens.

    Args:
        notify: The notify config.
        session_id: The session's id.
        session_dir: The session's directory.
        note: Receives a failure note.

    Returns:
        The listener for `EventSink.subscribe`; every row that is not a prompt is ignored.
    """

    def listen(event: dict[str, Any]) -> None:
        asked = format.attention_line(event)
        if asked is None:
            return
        fire(
            notify,
            "session.waiting",
            message=asked,
            level="info",
            note=note,
            AGENT6_SESSION_ID=session_id,
            AGENT6_SESSION_DIR=str(session_dir),
        )

    return listen


def session_end(
    notify: NotifyConfig,
    *,
    session_id: str,
    session_dir: pathlib.Path,
    ok: bool,
    reason: str,
    verification: str,
    note: Callable[[str], None],
) -> None:
    """Fire `session.stop` for a run, resume, plan or ask that stopped.

    Args:
        notify: The notify config.
        session_id: The session's id.
        session_dir: The session's directory.
        ok: Whether the agent stopped deliberately.
        reason: The end reason.
        verification: What the gate said: passed, failed, unverified or not_applicable.
        note: Receives a failure note.
    """
    fire(
        notify,
        "session.stop",
        message=reason,
        level="ok" if ok else "failed",
        note=note,
        AGENT6_SESSION_ID=session_id,
        # A hook that wants "green" reads VERIFIED: OK is true over a red verify too.
        AGENT6_SESSION_OK="1" if ok else "0",
        AGENT6_SESSION_VERIFIED=verification,
        AGENT6_SESSION_REASON=reason,
        AGENT6_SESSION_DIR=str(session_dir),
    )
