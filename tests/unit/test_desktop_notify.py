# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the best-effort desktop-notification helper."""

from __future__ import annotations

from typing import Any

import pytest

from agent6.frontend import notify


def test_desktop_notify_noop_when_notify_send_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", lambda _name: None)
    called: list[Any] = []
    monkeypatch.setattr(notify.subprocess, "Popen", lambda *a, **k: called.append(a))
    assert notify.desktop_notify("t", "b") is False
    assert called == []


def test_desktop_notify_fires_fixed_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", lambda _name: "/usr/bin/notify-send")
    seen: list[list[str]] = []

    def fake_popen(argv: list[str], **_kw: Any) -> Any:
        seen.append(argv)
        return object()

    monkeypatch.setattr(notify.subprocess, "Popen", fake_popen)
    assert notify.desktop_notify("agent6: m", "attention") is True
    # Fixed argv: exe + two positional data args, never a shell string.
    assert seen == [["/usr/bin/notify-send", "agent6: m", "attention"]]


def test_desktop_notify_swallows_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", lambda _name: "/usr/bin/notify-send")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no display")

    monkeypatch.setattr(notify.subprocess, "Popen", boom)
    assert notify.desktop_notify("t", "b") is False
