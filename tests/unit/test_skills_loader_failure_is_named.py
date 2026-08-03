# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A skills LOADER failure is not reported as "that skill does not exist".

`skills enable/disable` check the name against what discovery found. Swallowing
a discovery failure to an empty tuple turned "I could not read your skills" into
"unknown skill 'x'; installed: (none)" -- which sends the operator looking for a
skill that is missing, when one is broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli import skills_cmds


def _break_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_dirs: object) -> tuple[tuple[object, ...], tuple[str, ...]]:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(skills_cmds, "discover_skills", _raise)


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_it_says_the_skills_could_not_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _break_discovery(monkeypatch)

    if verb == "enable":
        rc = skills_cmds._cmd_skills_enable(  # pyright: ignore[reportPrivateUsage]
            "thinking-hard", always=False, repo=False
        )
    else:
        rc = skills_cmds._cmd_skills_disable(  # pyright: ignore[reportPrivateUsage]
            "thinking-hard", repo=False
        )

    err = capsys.readouterr().err
    assert rc == 2
    assert "Permission denied" in err, err
    assert "installed: (none)" not in err, err


def test_completion_still_degrades_to_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """argcomplete has no way to show an error and must never raise into the
    shell: an empty list is the right answer there."""
    _break_discovery(monkeypatch)
    assert skills_cmds.resolved_skill_names_for_completion(tmp_path) == []
