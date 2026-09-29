# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
import tree_sitter_language_pack
from tree_sitter_language_pack import PackConfig

from tests.jail_env import require_userns_jail


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Make the `needs_namespaces` marker real.

    Skip, with the host's actual reason, where the userns jail cannot run, instead of failing
    with JailUnavailableError; the marker alone registers a policy nothing enforces.
    """
    if item.get_closest_marker("needs_namespaces") is not None:
        require_userns_jail()


@pytest.fixture(autouse=True)
def _hermetic_git(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Pin a suite-owned git identity and blank the system and global git config.

    Tests commit in throwaway repos and clones (a clone does not inherit the origin's
    repo-local user.name and email), and a developer's ~/.gitconfig would supply an identity
    a bare runner lacks. A test that needs a missing identity overrides GIT_CONFIG_GLOBAL
    itself (see test_verify_git_identity_missing_raises).
    """
    cfg = tmp_path_factory.mktemp("git-identity") / "gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


# Resolved once, with the operator's own environment: the grammars already downloaded.
_GRAMMAR_CACHE = tree_sitter_language_pack.cache_dir()


@pytest.fixture(autouse=True)
def _isolate_state(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point agent6's per-repo state base and global config at throwaway dirs.

    Run state and the per-repo config live under ``$XDG_STATE_HOME/agent6``; isolating that
    base keeps tests off the real ``~/.local/state``, and an empty ``$XDG_CONFIG_HOME/agent6``
    keeps operator config and secrets out. The cache and data homes are isolated too: a
    developer's model-price cache or installed skills would otherwise reach a test. A test
    that needs a price or a skill seeds its own dir, and may override any of these itself.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("agent6-state")))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("agent6-config")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("agent6-cache")))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("agent6-data")))
    # The tree-sitter language pack keeps its grammars under the same XDG cache; keep them.
    tree_sitter_language_pack.configure(PackConfig(cache_dir=_GRAMMAR_CACHE))
