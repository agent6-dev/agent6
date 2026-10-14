# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The test session itself runs on throwaway XDG homes, under the per-test ones."""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

_HOMES = ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME")


@pytest.fixture(scope="module")
def between_tests() -> dict[str, str]:
    """The environment as it stands outside any test: a module fixture sets up first."""
    return {name: os.environ.get(name, "") for name in _HOMES}


def test_outside_a_test_the_environment_names_no_operator_home(
    between_tests: dict[str, str],
) -> None:
    """A thread that outlives its test must not reach the operator's config or state.

    The per-test environment is restored when a test ends, so a straggler ran
    `config set` against the real `~/.config/agent6/config.toml`.
    """
    tmp = pathlib.Path(tempfile.gettempdir()).resolve()
    for name, value in between_tests.items():
        assert value, f"{name} is unset between tests"
        assert pathlib.Path(value).resolve().is_relative_to(tmp), f"{name} = {value}"
