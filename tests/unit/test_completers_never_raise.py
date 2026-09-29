# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""No completer raises into the operator's shell.

argcomplete runs them on Tab with nowhere to show an error; one decorator guards them all.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from agent6 import paths
from agent6.ui.cli import completers

_COMPLETERS = [
    (name, fn)
    for name, fn in vars(completers).items()
    if name.startswith("_complete_") and inspect.isfunction(fn)
]


def test_there_are_completers_to_check() -> None:
    assert len(_COMPLETERS) >= 10, [n for n, _ in _COMPLETERS]


# The completers that consult the state dir on a bare prefix; the decorator test carries the rest.
_STATE_DIR_CONSUMERS = [
    "_complete_session_ids",
    "_complete_session_ports",
    "_complete_resumable_ids",
    "_complete_plan_session_ids",
    "_complete_machine_ids",
    "_complete_watch_targets",
    "_complete_machine_files",
]


@pytest.mark.parametrize("name", _STATE_DIR_CONSUMERS)
def test_an_unresolvable_state_dir_does_not_reach_the_shell(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that does not parse, so the state dir cannot resolve, reaches no traceback.

    Forced directly: a bad config under cwd never reaches the raising path.
    """
    from agent6.config import ConfigError

    calls: list[pathlib.Path] = []

    def _boom(root: pathlib.Path) -> pathlib.Path:
        calls.append(root)
        raise ConfigError("config is not valid TOML")

    monkeypatch.setattr(paths, "state_dir", _boom)
    monkeypatch.setattr(paths, "state_dir", _boom)

    fn = getattr(completers, name)
    result = fn("", parsed_args=None)
    assert isinstance(result, list), f"{name} returned {result!r}"
    assert calls, f"{name} never consulted the state dir; drop it from _STATE_DIR_CONSUMERS"


def test_any_completer_bug_yields_no_suggestions_not_a_traceback() -> None:
    """Any completer bug yields no suggestions, never a traceback.

    The guard catches every exception.
    """

    @completers._never_raises
    def boom(prefix: str, **_kw: object) -> list[str]:
        raise KeyError("bug")

    assert boom("") == []
