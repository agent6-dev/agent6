# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLI's adapter over the ranking core in `agent6.app.compare`.

It supplies the two pieces the headless core cannot: the `judging...` spinner
and the reviewer-provider builder; the fan-out binds the same two through its
`LaneRuntime`.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import threading
from collections.abc import Generator

from agent6 import budget as agent6_budget
from agent6.app import compare, providers
from agent6.config import Config
from agent6.harness import judge
from agent6.providers import Provider, TranscriptSink
from agent6.ui.cli import _console_view, _terminal_guard
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
            _terminal_guard.raw_stream(sys.stdout).write("\r\x1b[2K")
            sys.stdout.write(f"{format.spinner_frame(i)} judging...")
            sys.stdout.flush()
            i += 1
            if stop.wait(_console_view._HEARTBEAT_TICK_S):
                return

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        _terminal_guard.raw_stream(sys.stdout).write("\r\x1b[2K")
        sys.stdout.flush()


def _reviewer_provider(
    cfg: Config, sink: TranscriptSink, budget: agent6_budget.BudgetTracker
) -> Provider:
    """Return the configured `reviewer` provider for the judge call."""
    return providers.build_role_provider(cfg, "reviewer", transcript_sink=sink, budget=budget)


def rank(
    cfg: Config, candidates: list[judge.CandidateBrief], *, transcript_dir: pathlib.Path
) -> compare.RankOutcome:
    """Rank the candidates best first through the core, with the CLI's two pieces bound.

    Args:
        cfg: The run's config.
        candidates: The lanes' briefs.
        transcript_dir: Where the judge's transcript goes.

    Returns:
        The core's outcome.
    """
    return compare.rank(
        cfg,
        candidates,
        transcript_dir=transcript_dir,
        build_provider=_reviewer_provider,
        judging_status=_judging_status,
    )
