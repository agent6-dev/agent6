# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Capability markers test what their callers need, without network access."""

from __future__ import annotations

import contextlib
import errno
import os
import pathlib
from unittest import mock

import pytest
import tree_sitter_language_pack

from tests import conftest


def test_xdist_workers_receive_the_original_grammar_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker must not look for grammars in the controller's throwaway XDG cache."""
    cache = str(tmp_path / "grammars")
    monkeypatch.setattr(conftest, "_GRAMMAR_CACHE", cache)
    monkeypatch.setattr(conftest, "_XDG_HOMES", ())
    node = mock.Mock(config=mock.Mock(stash=pytest.Stash()), workerinput={})
    del node.config.workerinput
    try:
        conftest.pytest_configure(node.config)
        conftest.pytest_configure_node(node)
        assert node.workerinput["grammar_cache"] == cache
    finally:
        conftest.pytest_unconfigure(node.config)


def test_a_worker_keeps_the_controllers_grammar_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache passed by xdist takes precedence over the worker's inherited XDG path."""
    cache = str(tmp_path / "grammars")
    monkeypatch.setattr(conftest, "_GRAMMAR_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr(conftest, "_XDG_HOMES", ())
    config = mock.Mock(stash=pytest.Stash(), workerinput={"grammar_cache": cache})
    try:
        conftest.pytest_configure(config)
        assert config.stash[conftest._SESSION_GRAMMAR_CACHE] == cache
    finally:
        conftest.pytest_unconfigure(config)


def test_the_pty_marker_reports_an_allocation_failure(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test needing a PTY skips with the allocation error, not a namespace verdict."""

    def unavailable() -> tuple[int, int]:
        raise FileNotFoundError(errno.ENOENT, "no PTY device")

    monkeypatch.setattr(os, "openpty", unavailable)
    request.node.add_marker(pytest.mark.needs_pty)
    assert isinstance(request.node, pytest.Item)
    with pytest.raises(pytest.skip.Exception, match=r"PTY unavailable: .*no PTY device"):
        conftest.pytest_runtest_setup(request.node)


def test_the_pty_marker_closes_both_probe_descriptors(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful PTY probe leaves no descriptors open."""
    descriptors = os.pipe()
    monkeypatch.setattr(os, "openpty", lambda: descriptors)
    request.node.add_marker(pytest.mark.needs_pty)
    assert isinstance(request.node, pytest.Item)
    try:
        conftest.pytest_runtest_setup(request.node)
        for fd in descriptors:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(fd)
    finally:
        for fd in descriptors:
            with contextlib.suppress(OSError):
                os.close(fd)


@pytest.mark.parametrize("cached", [False, True])
def test_the_grammar_marker_checks_the_requested_language_without_downloading(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    """An absent grammar skips; a cached one reaches the test, even without a network."""

    def no_download(language: str) -> None:
        pytest.fail(f"the capability check tried to load {language}")

    monkeypatch.setattr(tree_sitter_language_pack, "get_parser", no_download)
    monkeypatch.setattr(
        tree_sitter_language_pack,
        "downloaded_languages",
        lambda: ["python"] if cached else ["rust"],
    )
    request.node.add_marker(pytest.mark.needs_grammar(language="python"))
    assert isinstance(request.node, pytest.Item)
    if cached:
        conftest.pytest_runtest_setup(request.node)
    else:
        with pytest.raises(
            pytest.skip.Exception, match="tree-sitter grammar 'python' is not cached"
        ):
            conftest.pytest_runtest_setup(request.node)
