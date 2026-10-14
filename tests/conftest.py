# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared pytest fixtures."""

from __future__ import annotations

import contextlib
import fcntl
import os
import pathlib
import shutil
import tempfile
import termios

import pytest
import tree_sitter_language_pack

from agent6.sandbox import tool_paths
from tests.jail_env import require_userns_jail

# Resolved once, with the operator's own environment: the grammars already downloaded.
_GRAMMAR_CACHE = tree_sitter_language_pack.cache_dir()

_XDG_HOMES = ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME")
_SESSION_HOMES = pytest.StashKey[pathlib.Path]()


def pytest_configure(config: pytest.Config) -> None:
    """Run the whole session on throwaway XDG homes, under the per-test ones.

    A thread that outlives its test runs after the per-test environment is restored; it
    lands here, never in the operator's config or state.
    """
    base = pathlib.Path(tempfile.mkdtemp(prefix="agent6-test-session-"))
    config.stash[_SESSION_HOMES] = base
    for name in _XDG_HOMES:
        home = base / name.lower()
        home.mkdir()
        os.environ[name] = str(home)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Detach the session from its controlling terminal, as CI and the gate unit have none.

    A test that reaches a `/dev/tty` prompt (`sessions rm` without `--yes`) then gets the
    headless path instead of waiting on the developer's keyboard. A pty test forks its own
    session leader, which takes the pty as its terminal unaffected.
    """
    with contextlib.suppress(OSError):
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        try:
            fcntl.ioctl(fd, termios.TIOCNOTTY)
        finally:
            os.close(fd)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the session's throwaway homes."""
    shutil.rmtree(config.stash.get(_SESSION_HOMES, pathlib.Path(os.devnull)), ignore_errors=True)


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


@pytest.fixture
def jail_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> pathlib.Path:
    """Pin the jail PATH to one directory holding every runner verify inference names.

    Inference passes over a command whose runner the jail PATH lacks, so a test asserting an
    inferred gate pins the PATH here; unlinking a runner makes it missing.
    """
    bindir = tmp_path_factory.mktemp("jail-bin")
    for name in ("cargo", "go", "make", "node", "npm", "pytest", "python3", "sh"):
        runner = bindir / name
        runner.write_text("#!/bin/sh\n", encoding="utf-8")
        runner.chmod(0o755)
    monkeypatch.setattr(tool_paths, "jail_search_path", lambda: str(bindir))
    return bindir


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
    tree_sitter_language_pack.configure(
        tree_sitter_language_pack.PackConfig(cache_dir=_GRAMMAR_CACHE)
    )
