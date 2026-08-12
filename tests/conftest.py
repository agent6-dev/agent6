# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_git(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Pin a suite-owned git identity; blank the system/global git config.

    Tests commit in throwaway repos and CLONES (a clone does not inherit the
    origin's repo-local user.name/email). The developer's ~/.gitconfig silently
    supplied the identity locally while a bare CI runner has none, so the suite
    was green here and red in CI. One suite-owned global config makes the two
    environments identical. A test that needs a MISSING identity overrides
    GIT_CONFIG_GLOBAL itself (see test_verify_git_identity_missing_raises).
    """
    cfg = tmp_path_factory.mktemp("git-identity") / "gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


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
    ``state_base()``. The cache and data homes are isolated for the same reason
    as the git config above: the developer's model-price cache made USD
    assertions pass locally while a bare CI runner has none, and the
    developer's installed skills would index into any run a test starts. A
    test that needs a price or a skill seeds its own dir. A test may still
    override any of these itself (its body runs after this fixture).
    """
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path_factory.mktemp("agent6-state")))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path_factory.mktemp("agent6-config")))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path_factory.mktemp("agent6-cache")))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path_factory.mktemp("agent6-data")))
