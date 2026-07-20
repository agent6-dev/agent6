# SPDX-License-Identifier: Apache-2.0
"""Regression tests for clear_pending_answers (cli/ui bridge bugs #7, #22).

#7: a leftover `steer.request` marker from a prior session must be dropped at
    run/resume START, else the resumed run stalls on a phantom steer prompt.
#22: `frontend.pid` must only be cleared when NO live TUI owns it, so a concurrently
    live `agent6 attach` watcher keeps bridging approval/question modals.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6.runs.ipc import (
    clear_pending_answers,
    frontend_is_live,
    register_frontend,
    request_steer,
    steer_request_pending,
    unregister_frontend,
)


def test_clear_pending_drops_leftover_steer_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request_steer(run_dir)
    assert steer_request_pending(run_dir)

    clear_pending_answers(run_dir)

    # The phantom steer marker must be gone so resume doesn't stall.
    assert not steer_request_pending(run_dir)


def test_clear_pending_preserves_live_frontend_claims(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Our own pid is a live process => a live foreign watcher.
    register_frontend(run_dir, os.getpid())
    assert frontend_is_live(run_dir)

    clear_pending_answers(run_dir)

    # A live watcher's claim must survive so its modals stay wired up.
    assert frontend_is_live(run_dir)


def test_dead_frontend_claims_are_pruned_by_the_liveness_probe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dead_pid = _find_dead_pid()
    register_frontend(run_dir, dead_pid)
    # A hard-killed front-end's claim reads not-live and is pruned in passing,
    # so the answer-poll never blocks on it and the dir stays tidy.
    assert not frontend_is_live(run_dir)
    assert not (run_dir / "frontends" / str(dead_pid)).exists()


def test_concurrent_frontends_do_not_deregister_each_other(tmp_path: Path) -> None:
    """The single-slot frontend.pid let one front-end's exit strand another
    (attach claims -> web clobbers -> web releases -> attach deregistered, its
    answers never read). One claim file per front-end kills the class: any
    number watch concurrently and each removes only its own claim."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    attach_pid = os.getpid()
    web_pid = _find_dead_pid()  # stands in for a second front-end's pid slot
    register_frontend(run_dir, attach_pid)
    register_frontend(run_dir, web_pid)
    unregister_frontend(run_dir, web_pid)  # the browser closes
    assert frontend_is_live(run_dir)  # the attach watcher keeps bridging
    unregister_frontend(run_dir, attach_pid)
    assert not frontend_is_live(run_dir)
    # Unregistering an absent claim is a no-op.
    unregister_frontend(run_dir, attach_pid)


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
