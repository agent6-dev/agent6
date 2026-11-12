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
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import tree_sitter_language_pack
from xdist import workermanage

from agent6 import kinds
from agent6.app.session import runtime as session_runtime
from agent6.config import Config
from agent6.sandbox import tool_paths
from agent6.sessions import lock
from agent6.tools import dispatch
from tests.browser_env import require_browser
from tests.collector_env import require_collector
from tests.jail_env import require_userns_jail

# Resolved once, with the operator's own environment: the grammars already downloaded.
_GRAMMAR_CACHE = tree_sitter_language_pack.cache_dir()

_XDG_HOMES = ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME")
_SESSION_HOMES = pytest.StashKey[pathlib.Path]()
_SESSION_GRAMMAR_CACHE = pytest.StashKey[str]()


def pytest_configure(config: pytest.Config) -> None:
    """Run the whole session on throwaway XDG homes, under the per-test ones.

    A thread that outlives its test runs after the per-test environment is restored; it
    lands here, never in the operator's config or state.
    """
    config.stash[_SESSION_GRAMMAR_CACHE] = getattr(config, "workerinput", {}).get(
        "grammar_cache", _GRAMMAR_CACHE
    )
    base = pathlib.Path(tempfile.mkdtemp(prefix="agent6-test-session-"))
    config.stash[_SESSION_HOMES] = base
    for name in _XDG_HOMES:
        home = base / name.lower()
        home.mkdir()
        os.environ[name] = str(home)


def pytest_configure_node(node: workermanage.WorkerController) -> None:
    """Pass the original grammar cache to workers before their XDG homes are isolated."""
    node.workerinput["grammar_cache"] = node.config.stash[_SESSION_GRAMMAR_CACHE]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Detach the session from its controlling terminal, as CI and the gate unit have none.

    A test that reaches a `/dev/tty` prompt (`sessions remove` without `--yes`) then gets the
    headless path instead of waiting on the developer's keyboard. A pty test forks its own
    session leader, which takes the pty as its terminal unaffected.
    """
    detach_controlling_tty()


def detach_controlling_tty() -> None:
    """Drop the controlling terminal, unless this process leads the session that holds it.

    `TIOCNOTTY` from a session leader hangs up the terminal's foreground group, the leader
    included, so a pytest started as its own session leader on a pty keeps its terminal.
    """
    if os.getsid(0) == os.getpid():
        return
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
    """Skip capability-marked tests with the environment's actual reason."""
    if item.get_closest_marker("needs_namespaces") is not None:
        require_userns_jail()
    if item.get_closest_marker("browser") is not None:
        require_browser()
    if marker := item.get_closest_marker("collector"):
        require_collector(marker.kwargs.get("binary", "otelcol-contrib"))
    needs_dev_tty = item.get_closest_marker("needs_dev_tty") is not None
    if needs_dev_tty or item.get_closest_marker("needs_pty") is not None:
        try:
            master, slave = os.openpty()
        except OSError as exc:
            pytest.skip(f"PTY unavailable: {exc}")
        else:
            os.close(master)
            os.close(slave)
    if needs_dev_tty and not pathlib.Path("/dev/tty").exists():
        pytest.skip("no /dev/tty node here: a prompt on the controlling terminal cannot run")
    if marker := item.get_closest_marker("needs_grammar"):
        language = marker.kwargs["language"]
        cache = item.config.stash[_SESSION_GRAMMAR_CACHE]
        tree_sitter_language_pack.configure(tree_sitter_language_pack.PackConfig(cache_dir=cache))
        if language not in tree_sitter_language_pack.downloaded_languages():
            pytest.skip(f"tree-sitter grammar {language!r} is not cached at {cache}")


@pytest.fixture(autouse=True)
def _hermetic_git(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Pin a suite-owned Git identity and blank the system and global git config.

    The tests' own git calls read the fixture's global config; agent6's git passes only the
    author and committer variables through, so the identity rides those too. HOME stays the
    operator's: rustup and the home rows of `check sandbox` read it.
    """
    cfg = tmp_path_factory.mktemp("git-identity") / "gitconfig"
    cfg.write_text("[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@t")


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


@pytest.fixture
def linked_worktrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every session worktree a linked one, whatever `[git.worktree].kind` says.

    For the tests that pin the linked kind's own facts (the recorded git dir, `git worktree
    list`, refs written by hand in the worktree): a module asks for it with `pytestmark`.
    """
    from agent6.app import fork_worktrees

    make = fork_worktrees.add_session_worktree

    def linked(
        repo: pathlib.Path, path: pathlib.Path, sha: str, _kind: str
    ) -> fork_worktrees.Checkout:
        return make(repo, path, sha, "linked")

    monkeypatch.setattr(fork_worktrees, "add_session_worktree", linked)


@pytest.fixture(autouse=True)
def _release_workspace_leases() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Drop the workspace leases a test's run, exec or machine held; a real process exits."""
    yield
    lock.release_workspaces()


@pytest.fixture(autouse=True)
def _isolate_state(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
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
        tree_sitter_language_pack.PackConfig(cache_dir=request.config.stash[_SESSION_GRAMMAR_CACHE])
    )


@pytest.fixture
def host_tmp_path() -> Iterator[pathlib.Path]:
    """Provide private storage under the host's /tmp, independent of basetemp and TMPDIR."""
    with tempfile.TemporaryDirectory(prefix="agent6-host-tmp-", dir="/tmp") as directory:
        yield pathlib.Path(directory).resolve()


@pytest.fixture
def short_repo() -> Iterator[pathlib.Path]:
    """Provide a repository path short enough for a reversible repository ID."""
    with tempfile.TemporaryDirectory(prefix="repo-") as directory:
        repo = pathlib.Path(directory).resolve()
        assert len(str(repo).encode()) < 100
        yield repo


@pytest.fixture
def build_dispatcher() -> Iterator[Callable[..., dispatch.ToolDispatcher]]:
    """Build dispatchers over a prepared local command session; teardown closes them all.

    `root`, `config` and `isolation` (default `none`) prepare the session; every other keyword
    argument goes to `ToolDispatcher`.
    """
    built: list[dispatch.ToolDispatcher] = []

    def build(
        *,
        root: pathlib.Path,
        config: Config,
        isolation: kinds.IsolationLevel = "none",
        **kwargs: Any,
    ) -> dispatch.ToolDispatcher:
        access = session_runtime.prepare_local_execution(config, root, isolation)
        try:
            dispatcher = dispatch.ToolDispatcher(
                root=root, config=config, isolation=isolation, command_access=access, **kwargs
            )
        except BaseException:
            access.close()
            raise
        built.append(dispatcher)
        return dispatcher

    yield build
    for dispatcher in built:
        dispatcher.close()
