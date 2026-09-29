# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run whose jail came up degraded says so once, at session open, not per command.

`JailSession.open()` stores the launcher's setup stderr at its ready handshake (a refused /proc
mount under rootless podman, a skipped grant).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import events as agent6_events
from agent6.config import Config
from agent6.tools import dispatch


class _StubSession:
    def __init__(self, startup_stderr: str) -> None:
        self.startup_stderr = startup_stderr

    def close(self) -> None:  # pragma: no cover - dispatcher teardown
        pass


def _patch_open(monkeypatch: pytest.MonkeyPatch, stub: _StubSession) -> None:
    def fake_open(cls: object, policy: object, *, session_net: object = None) -> _StubSession:
        return stub

    monkeypatch.setattr("agent6.sandbox.jail.JailSession.open", classmethod(fake_open))


def _events(path: pathlib.Path, kind: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        e
        for line in path.read_text(encoding="utf-8").splitlines()
        if (e := json.loads(line)).get("type") == kind
    ]


def _dispatcher(
    tmp_path: pathlib.Path, events: agent6_events.EventSink, stub: _StubSession
) -> dispatch.ToolDispatcher:
    # network = "host" needs no session netns; isolation must be strict for a session to open.
    (tmp_path / "s").mkdir(exist_ok=True)
    return dispatch.ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"network": "host"}}),
        isolation="strict",
        events=events,
        session_dir=tmp_path / "s",
        use_jail_session=True,
    )


def test_a_degraded_session_emits_jail_degraded_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = "[agent6-jail] warning: fresh /proc mount failed (EPERM)"
    stub = _StubSession(warning)
    _patch_open(monkeypatch, stub)
    log = tmp_path / "e.jsonl"
    d = _dispatcher(tmp_path, agent6_events.EventSink(log), stub)
    try:
        assert d._run_session() is stub  # pyright: ignore[reportPrivateUsage]
        d._run_session()  # already open -> no second emit  # pyright: ignore[reportPrivateUsage]
    finally:
        d.close()
    degraded = _events(log, "jail.degraded")
    assert len(degraded) == 1, degraded
    assert warning in str(degraded[0].get("detail"))


def test_a_clean_session_emits_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubSession("")
    _patch_open(monkeypatch, stub)
    log = tmp_path / "e.jsonl"
    d = _dispatcher(tmp_path, agent6_events.EventSink(log), stub)
    try:
        d._run_session()  # pyright: ignore[reportPrivateUsage]
    finally:
        d.close()
    assert _events(log, "jail.degraded") == []


def test_concurrent_callers_open_exactly_one_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lazy session open is locked: two threads reaching it together open one launcher.

    A dropped `JailSession` leaks its launcher, its namespaces and its network holder; the sleep
    widens the window the race needs.
    """
    import threading
    import time

    opened: list[_StubSession] = []

    def slow_open(cls: object, policy: object, *, session_net: object = None) -> _StubSession:
        time.sleep(0.05)  # the window a check-then-set leaves open
        stub = _StubSession("")
        opened.append(stub)
        return stub

    monkeypatch.setattr("agent6.sandbox.jail.JailSession.open", classmethod(slow_open))
    d = _dispatcher(tmp_path, agent6_events.EventSink(tmp_path / "e.jsonl"), _StubSession(""))
    seen: list[object] = []
    try:
        threads = [
            threading.Thread(
                target=lambda: seen.append(d._run_session())  # pyright: ignore[reportPrivateUsage]
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        d.close()

    assert len(opened) == 1, f"opened {len(opened)} sessions; {len(opened) - 1} leaked"
    assert seen == [opened[0]] * 4
