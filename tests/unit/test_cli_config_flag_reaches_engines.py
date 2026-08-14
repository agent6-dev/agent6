# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The top-level --config reaches every command's config load.

`agent6 --config F sessions merge <id>` loaded the two standard layers and
silently ignored F: the squash style stayed default and the model drafter
never fired (caught live). The planner, compare, exec, and machine create now
thread the explicit path into load_effective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.ui.cli import sessions_cmds


def test_merge_planner_passes_the_explicit_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path | None] = []

    def fake_load(cwd: Path, explicit: Path | None) -> Any:
        seen.append(explicit)
        raise sessions_cmds.ConfigError("stop here")

    monkeypatch.setattr(sessions_cmds, "load_effective", fake_load)

    # A resolvable path must flow through unchanged when the planner DOES
    # reach the load; drive it far enough by stubbing resolution to succeed.
    class _Layout:
        session_dir = tmp_path / "sess"

    layout = _Layout()

    def _dead(d: Path) -> bool:
        return False

    monkeypatch.setattr(sessions_cmds, "worker_is_alive", _dead)

    class _Manifest:
        base_branch = "main"
        base_sha = "0" * 40
        run_branch = "agent6/x"

    def _resolved(cwd: Path, sid: str) -> Any:
        return (layout, _Manifest())

    def _exists(cwd: Path, b: str) -> bool:
        return True

    monkeypatch.setattr(sessions_cmds, "_resolve_session_manifest", _resolved)
    monkeypatch.setattr(sessions_cmds, "branch_exists", _exists)
    explicit = tmp_path / "special.toml"
    rc = sessions_cmds._plan_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, "sid", None, None, config_path=explicit
    )
    assert rc == 2  # the stubbed ConfigError surfaced as the exit path
    assert seen == [explicit]
