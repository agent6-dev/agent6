# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Collector-marked tests skip when their external binary is missing."""

from __future__ import annotations

import pathlib

import pytest

from tests import collector_env, conftest


@pytest.mark.parametrize("binary", ("otelcol-contrib", "promtool"))
def test_the_collector_marker_skips_with_the_missing_binarys_name(
    binary: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered marker alone does not skip a test whose binary is absent."""
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AGENT6_OTELCOL", raising=False)
    request.node.add_marker(pytest.mark.collector(binary=binary))
    assert isinstance(request.node, pytest.Item)
    with pytest.raises(pytest.skip.Exception, match=f"{binary} is not"):
        conftest.pytest_runtest_setup(request.node)


@pytest.mark.parametrize("source", ("PATH", "AGENT6_OTELCOL"))
def test_the_binary_path_survives_a_subprocess_working_directory_change(
    source: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative tool locations resolve before the collector starts in its temporary directory."""
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    binary = tools / "otelcol-contrib"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", "tools" if source == "PATH" else "")
    monkeypatch.delenv("AGENT6_OTELCOL", raising=False)
    if source == "AGENT6_OTELCOL":
        monkeypatch.setenv(source, "tools/otelcol-contrib")
    assert collector_env.require_collector() == str(binary)


def test_a_non_executable_collector_override_is_missing(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file at AGENT6_OTELCOL must be executable to satisfy the collector marker."""
    binary = tmp_path / "otelcol-contrib"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AGENT6_OTELCOL", str(binary))
    monkeypatch.setenv("PATH", "")
    request.node.add_marker(pytest.mark.collector)
    assert isinstance(request.node, pytest.Item)
    with pytest.raises(pytest.skip.Exception, match="AGENT6_OTELCOL"):
        conftest.pytest_runtest_setup(request.node)
