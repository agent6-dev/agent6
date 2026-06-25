# SPDX-License-Identifier: Apache-2.0
"""Regression tests for clear_pending_answers (cli/ui bridge bugs #7, #22).

#7: a leftover `steer.request` marker from a prior session must be dropped at
    run/resume START, else the resumed run stalls on a phantom steer prompt.
#22: `tui.pid` must only be cleared when NO live TUI owns it, so a concurrently
    live `agent6 watch` watcher keeps bridging approval/question modals.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6.ui.approval import (
    clear_pending_answers,
    request_steer,
    steer_request_pending,
    tui_is_live,
    write_tui_pid,
)


def test_clear_pending_drops_leftover_steer_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request_steer(run_dir)
    assert steer_request_pending(run_dir)

    clear_pending_answers(run_dir)

    # The phantom steer marker must be gone so resume doesn't stall.
    assert not steer_request_pending(run_dir)


def test_clear_pending_preserves_live_tui_pid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Our own pid is a live process => a live foreign watcher.
    write_tui_pid(run_dir, os.getpid())
    assert tui_is_live(run_dir)

    clear_pending_answers(run_dir)

    # A live watcher's pid must survive so its modals stay wired up.
    assert tui_is_live(run_dir)
    assert (run_dir / "tui.pid").exists()


def test_clear_pending_drops_stale_tui_pid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A pid that is (essentially certainly) not a live process.
    dead_pid = _find_dead_pid()
    write_tui_pid(run_dir, dead_pid)
    assert not tui_is_live(run_dir)

    clear_pending_answers(run_dir)

    # A stale (hard-killed) pid must be cleared so the poll doesn't block.
    assert not (run_dir / "tui.pid").exists()


def _find_dead_pid() -> int:
    for candidate in range(2_000_000, 2_000_100):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    # Fallback: very unlikely to be reached.
    return 2_000_000
