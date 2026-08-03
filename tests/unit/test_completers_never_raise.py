# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""No completer raises into the operator's shell.

argcomplete runs these on Tab, inside the shell, with nowhere to show an error:
an exception there is a traceback dumped over the command line. Several guarded
themselves ad hoc and several did not, which is the same gap in as many places.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent6.ui.cli import completers

_COMPLETERS = [
    (name, fn)
    for name, fn in vars(completers).items()
    if name.startswith("_complete_") and inspect.isfunction(fn)
]


def test_there_are_completers_to_check() -> None:
    assert len(_COMPLETERS) >= 10, [n for n, _ in _COMPLETERS]


@pytest.mark.parametrize(("name", "fn"), _COMPLETERS, ids=[n for n, _ in _COMPLETERS])
def test_an_unresolvable_state_dir_does_not_reach_the_shell(
    name: str, fn: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realistic failure: the config does not parse, so resolving the state
    dir raises. Forced directly -- pointing cwd at a bad config passed without
    ever reaching the raising path, which proved nothing.
    """
    from agent6.config import ConfigError
    from agent6.ui.cli import _common

    def _boom(_root: Path) -> Path:
        raise ConfigError("config is not valid TOML")

    monkeypatch.setattr(completers, "_state_dir", _boom, raising=False)
    monkeypatch.setattr(_common, "_state_dir", _boom)

    result = fn("", parsed_args=None)  # pyright: ignore[reportCallIssue, reportGeneralTypeIssues]
    assert isinstance(result, list), f"{name} returned {result!r}"
