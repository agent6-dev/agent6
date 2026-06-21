# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point agent6's per-repo state base + global config at throwaway dirs.

    Run state + the per-repo config live out of the workspace under the state
    base (``AGENT6_STATE_HOME``). Isolating that base keeps tests off the real
    ``~/.local/state``; isolating the global config dir (``AGENT6_CONFIG_HOME``,
    pointed at an empty dir) is what makes ``AGENT6_STATE_HOME`` authoritative,
    since a global ``[agent6].state_dir`` would otherwise override it in
    ``state_base()``. A test may still override ``AGENT6_STATE_HOME`` itself (its
    body runs after this fixture); tests that need a global config set
    ``AGENT6_CONFIG_HOME`` to their own dir.
    """
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path_factory.mktemp("agent6-state")))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path_factory.mktemp("agent6-config")))
