# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem layout of one run's state directory.

A leaf: pure path arithmetic over the resolved run-state base, imported by
the graph storage/curator stack, the CLI, and the MCP server without pulling
in any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Filesystem layout for one `agent6 run`.

    ``state_dir`` is the resolved run-state base
    (``$XDG_STATE_HOME/agent6/<repo-id>`` by default, or wherever
    ``[agent6].state_dir`` points). See ``agent6.paths.state_dir``.
    """

    state_dir: Path
    run_id: str
    # Top-level bucket under state_dir. "runs" for `agent6 run`/`plan`; "asks"
    # for `agent6 ask` so read-only Q&A sessions stay separate from real runs.
    subdir: str = "runs"

    @property
    def run_dir(self) -> Path:
        return self.state_dir / self.subdir / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def graph_dir(self) -> Path:
        return self.run_dir / "graph"

    @property
    def journal_path(self) -> Path:
        return self.run_dir / "graph.jsonl"

    @property
    def dot_path(self) -> Path:
        return self.run_dir / "graph.dot"

    @property
    def cursor_path(self) -> Path:
        return self.run_dir / "cursor.json"

    @property
    def lock_path(self) -> Path:
        return self.run_dir / ".lock"

    @property
    def checkpoints_dir(self) -> Path:
        """Append-only per-turn resume checkpoints (``<NNNN>.json``).

        Each holds the same RunSnapshot bytes as ``loop_state.json`` for that
        turn (workspace ``head_sha`` + curator ``graph_version`` included), so
        ``agent6 fork`` can roll a run back to turn N. ``loop_state.json`` stays
        the "latest" pointer for plain ``resume``.
        """
        return self.run_dir / "checkpoints"

    @property
    def transcripts_dir(self) -> Path:
        return self.run_dir / "transcripts"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "logs.jsonl"

    @property
    def user_inputs_path(self) -> Path:
        """JSONL audit log of every interactive prompt + the operator's answer.

        Separate from logs.jsonl so the human-decision trail stays readable
        without grepping through machine telemetry.
        """
        return self.run_dir / "user_inputs.jsonl"

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def checkpoint_path(self, turn: int) -> Path:
        """Path of the checkpoint for ``turn`` (zero-padded to 4 digits)."""
        return self.checkpoints_dir / f"{turn:04d}.json"
