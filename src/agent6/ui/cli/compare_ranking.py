# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLI's adapter over the ranking core in `agent6.app.compare`.

It supplies the two pieces the headless core cannot: the `judging...` spinner and the
spawner that starts each judge as a reviewer child of the command.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import threading
import uuid
from collections.abc import Generator

from agent6 import child_records, paths
from agent6.app import compare
from agent6.app.parallel import children as app_children
from agent6.config import layer_data
from agent6.ui.cli import console_render, terminal_guard
from agent6.ui.cli import parallel as cli_parallel
from agent6.viewmodel import format

__all__ = ["rank"]


@contextlib.contextmanager
def _judging_status() -> Generator[None]:
    """Show progress around the judge call, which is otherwise silent for about a minute.

    A terminal gets the run stream's spinner and cadence; a non-tty gets one plain
    line, so no animation frames reach a log file.

    Yields:
        Nothing; the spinner runs for the block.
    """
    if not sys.stdout.isatty():
        print("judging...")
        yield
        return
    stop = threading.Event()

    def spin() -> None:
        i = 0
        while True:
            terminal_guard.raw_stream(sys.stdout).write("\r\x1b[2K")
            sys.stdout.write(f"{format.spinner_frame(i)} judging...")
            sys.stdout.flush()
            i += 1
            if stop.wait(console_render._HEARTBEAT_TICK_S):
                return

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        terminal_guard.raw_stream(sys.stdout).write("\r\x1b[2K")
        sys.stdout.flush()


def rank(
    effective: layer_data.EffectiveConfig,
    candidates: list[child_records.CandidateBrief],
    *,
    cwd: pathlib.Path,
) -> compare.RankOutcome:
    """Rank the candidates best first through the core, with the CLI's two pieces bound.

    The judges are reviewer children of no run: they are filed under an id the command
    mints, `compare-<hex>`, in the repository's reviewers bucket, and a judge that needs an
    approval gets none.

    Args:
        effective: The command's config, with the leaves its layers set.
        candidates: The runs' briefs.
        cwd: The repository.

    Returns:
        The core's outcome.
    """
    cfg = effective.config
    spawner = app_children.build_child_spawner(
        cfg,
        cwd,
        paths.state_dir(cwd),
        explicit_leaves=effective.explicit_leaves,
        parent_session_id=f"compare-{uuid.uuid4().hex[:8]}",
        runtime=cli_parallel.child_runtime(),
        child_away="deny",
    )
    return compare.rank(cfg, candidates, spawner=spawner, judging_status=_judging_status)
