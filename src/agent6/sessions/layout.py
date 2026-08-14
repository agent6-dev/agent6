# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem layout of one session's state directory.

A leaf: pure path arithmetic over the resolved state base, imported by
the graph storage/curator stack, the CLI, and the MCP server without pulling
in any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def is_safe_session_id(session_id: str) -> bool:
    """True iff *session_id* is a single path component (no separator, not `.`/`..`).

    `SessionLayout.session_dir` is `<state>/sessions/<bucket>/<id>`: an unchecked id
    with `/` or `..` traverses out of the runs dir, and pathlib treats an
    absolute right operand as a replacement, so an externally supplied id must
    be validated at every trust boundary before it reaches a layout."""
    return (
        bool(session_id)
        and "/" not in session_id
        and "\\" not in session_id
        and session_id not in {".", ".."}
    )


# The event journal's filename. Named once, and used everywhere: a reader that
# hardcodes the wrong one silently finds nothing, which is indistinguishable
# from an empty session.
LOGS_NAME = "logs.jsonl"


@dataclass(frozen=True, slots=True)
class SessionLayout:
    """Filesystem layout for one `agent6 run`.

    `state_dir` is the resolved run-state base
    (`$XDG_STATE_HOME/agent6/<repo-id>` by default, or wherever
    `[agent6].state_dir` points). See `agent6.paths.state_dir`.
    """

    state_dir: Path
    session_id: str
    # Top-level bucket under state_dir. "runs" for `agent6 run`/`plan`; "asks"
    # for `agent6 ask` so read-only Q&A sessions stay separate from real runs.
    subdir: str = "runs"

    @property
    def session_dir(self) -> Path:
        return bucket_dir(self.state_dir, self.subdir) / self.session_id

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "manifest.json"

    @property
    def graph_dir(self) -> Path:
        return self.session_dir / "graph"

    @property
    def journal_path(self) -> Path:
        return self.session_dir / "graph.jsonl"

    @property
    def dot_path(self) -> Path:
        return self.session_dir / "graph.dot"

    @property
    def cursor_path(self) -> Path:
        return self.session_dir / "cursor.json"

    @property
    def lock_path(self) -> Path:
        return self.session_dir / ".lock"

    @property
    def checkpoints_dir(self) -> Path:
        """Append-only per-turn resume checkpoints (`<NNNN>.json`).

        Each holds the same SessionSnapshot bytes as `loop_state.json` for that
        turn (workspace `head_sha` + curator `graph_version` included), so
        `agent6 fork` can roll a run back to turn N. `loop_state.json` stays
        the "latest" pointer for plain `resume`.
        """
        return self.session_dir / "checkpoints"

    @property
    def transcripts_dir(self) -> Path:
        return self.session_dir / "transcripts"

    @property
    def logs_path(self) -> Path:
        return self.session_dir / LOGS_NAME

    def ensure(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def checkpoint_path(self, turn: int) -> Path:
        """Path of the checkpoint for `turn` (zero-padded to 4 digits)."""
        return self.checkpoints_dir / f"{turn:04d}.json"


# Every session directory lives under this one root, so the state dir's own
# `machines/` stays the live machine INSTANCES and a machine-authoring session
# can still be named for its mode (`sessions/machines/`).
SESSIONS_ROOT = "sessions"
# One bucket per session mode, named after it (`types.session_bucket`; a test
# pins the two together). Defined beside SessionLayout.subdir because it is a
# fact about the on-disk layout, not about any one front-end: both the CLI's id
# resolution and the resume lifecycle need it.
SESSION_BUCKETS: tuple[str, ...] = ("runs", "plans", "asks", "machines")
# What a hub lists as an ordinary session. Machine authoring is excluded: every
# hub gives it its own card, keyed by the machine being authored. (An `agent`
# state has no bucket at all; its sessions live inside the machine instance.)
HUB_BUCKETS: tuple[str, ...] = ("runs", "plans", "asks")


def bucket_dir(state_dir: Path, bucket: str) -> Path:
    """The directory holding sessions of one bucket: the one owner of that
    arithmetic, so a layout change lands in one place."""
    return state_dir / SESSIONS_ROOT / bucket


def layout_of(session_dir: Path) -> SessionLayout:
    """The layout of an ALREADY-RESOLVED session directory.

    Rebuilding one from the directory's NAME loses the bucket and defaults to
    runs/, which for a plan or an ask silently retargets a path that does not
    exist -- and the callers that do it sit inside a `suppress`, so it goes
    unnoticed.
    """
    return SessionLayout(
        state_dir=session_dir.parent.parent.parent,
        session_id=session_dir.name,
        subdir=session_dir.parent.name,
    )


def session_matches(state_dir: Path, session_id: str) -> list[SessionLayout]:
    """Every session *session_id* names or prefixes, across all buckets.

    Exact ids win outright: a full id that also prefixes a longer one is not
    ambiguous. A caller reports the list; only a single match is actionable.
    """
    if not session_id:
        return []
    exact: list[SessionLayout] = []
    prefix: list[SessionLayout] = []
    for subdir in SESSION_BUCKETS:
        bucket = bucket_dir(state_dir, subdir)
        if not bucket.is_dir():
            continue
        for entry in sorted(bucket.iterdir()):
            if not entry.is_dir():
                continue
            layout = SessionLayout(state_dir=state_dir, session_id=entry.name, subdir=subdir)
            if entry.name == session_id:
                exact.append(layout)
            elif entry.name.startswith(session_id):
                prefix.append(layout)
    return exact or prefix


def session_layout(state_dir: Path, session_id: str) -> SessionLayout | None:
    """The layout for *session_id* in whichever bucket holds it, or None.

    One id-to-layout resolution, so a command that accepts a session id reaches
    an ask the same way it reaches a run. An ambiguous prefix resolves to
    nothing rather than a guess.
    """
    matches = session_matches(state_dir, session_id)
    return matches[0] if len(matches) == 1 else None
