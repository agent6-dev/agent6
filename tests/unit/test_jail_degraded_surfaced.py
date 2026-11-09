# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import json
import pathlib

import pytest

from agent6.app import session_runtime
from agent6.config import Config
from agent6.sandbox import jail, jail_session
from agent6.sessions import event_log
from agent6.tools import dispatch


@pytest.mark.parametrize("warning", ["", "[agent6-jail] warning: fresh /proc mount failed (EPERM)"])
def test_prepared_session_surfaces_startup_stderr_once(
    tmp_path: pathlib.Path, warning: str
) -> None:
    """A prepared session emits its startup diagnostic on injection, not per command."""
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "none", "network": "host", "run_commands": "allow"}}
    )
    access = session_runtime.prepare_local_execution(cfg, tmp_path, "none")
    assert access.session is not None
    # Only the diagnostic is injected; commands use the real supervisor.
    access.session.startup_stderr = warning
    log = tmp_path / "events.jsonl"
    d = dispatch.ToolDispatcher(
        root=tmp_path, config=cfg, command_access=access, events=event_log.EventSink(log)
    )
    try:
        initial = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        assert [event["detail"] for event in initial if event["type"] == "jail.degraded"] == (
            [warning] if warning else []
        )
        for message in ("first", "second"):
            result = d.dispatch("run_command", {"argv": ["/bin/echo", message]}).to_wire()
            assert result["returncode"] == 0
            assert result["stdout"] == f"{message}\n"
    finally:
        d.close()
        d.close()
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    degraded = [event for event in events if event["type"] == "jail.degraded"]
    assert [event["detail"] for event in degraded] == ([warning] if warning else [])


def test_failed_preparation_never_invokes_one_shot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preparation error propagates before any dispatcher or fallback can run."""

    def refuse(*args: object, **kwargs: object) -> jail_session.JailSession:
        raise jail.JailUnavailableError("preparation refused")

    def one_shot(*args: object, **kwargs: object) -> None:
        pytest.fail("failed preparation invoked a one-shot runner")

    monkeypatch.setattr(jail_session.JailSession, "open", refuse)
    monkeypatch.setattr(jail, "run_in_jail", one_shot)
    cfg = Config.model_validate({"sandbox": {"isolation": "none", "network": "host"}})
    with pytest.raises(jail.JailUnavailableError, match="preparation refused"):
        session_runtime.prepare_local_execution(cfg, tmp_path, "none")
