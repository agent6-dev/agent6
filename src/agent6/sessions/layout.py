# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The filesystem layout of one session's state directory.

A leaf: path arithmetic over the resolved state base.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Collection, Sequence

from agent6 import paths as agent6_paths
from agent6 import portable
from agent6.sessions import manifest


def is_safe_session_id(session_id: str) -> bool:
    """Return whether an id is a single path component.

    An unchecked id with `/` or `..` traverses out of the sessions dir, and pathlib treats
    an absolute right operand as a replacement, so an external id is validated at every
    trust boundary before it reaches a layout.

    Args:
        session_id: The id.

    Returns:
        True when it is non-empty, has no separator, and is not `.` or `..`.
    """
    return (
        bool(session_id)
        and "/" not in session_id
        and "\\" not in session_id
        and session_id not in {".", ".."}
    )


# A reader hardcoding the wrong name finds nothing, indistinguishable from an empty session.
LOGS_NAME = "logs.jsonl"
# Repo-relative, NUL-separated; every chain commit and dirty check leaves these files out.
UNTRACKED_AT_START_NAME = "untracked-at-start"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionLayout:
    """The paths of one session's state.

    Attributes:
        state_dir: The repository's state dir, `agent6.paths.state_dir`.
        session_id: The session id.
        subdir: The bucket under the sessions root, one per mode.
    """

    state_dir: pathlib.Path
    session_id: str
    subdir: str = "runs"

    @property
    def session_dir(self) -> pathlib.Path:
        """The session directory."""
        return bucket_dir(self.state_dir, self.subdir) / self.session_id

    @property
    def manifest_path(self) -> pathlib.Path:
        """The manifest file."""
        return self.session_dir / manifest.MANIFEST_NAME

    @property
    def graph_dir(self) -> pathlib.Path:
        """The task graph's node store."""
        return self.session_dir / "graph"

    @property
    def journal_path(self) -> pathlib.Path:
        """The task graph's journal."""
        return self.session_dir / "graph.jsonl"

    @property
    def cursor_path(self) -> pathlib.Path:
        """The task graph's cursor."""
        return self.session_dir / "cursor.json"

    @property
    def lock_path(self) -> pathlib.Path:
        """The task graph's lock file."""
        return self.session_dir / ".lock"

    @property
    def checkpoints_dir(self) -> pathlib.Path:
        """The per-turn checkpoints, `<NNNN>.json`, each the snapshot bytes of that turn."""
        return self.session_dir / "checkpoints"

    @property
    def transcripts_dir(self) -> pathlib.Path:
        """The provider transcripts."""
        return self.session_dir / "transcripts"

    @property
    def logs_path(self) -> pathlib.Path:
        """The event journal."""
        return self.session_dir / LOGS_NAME

    def ensure(self) -> None:
        """Create the session dir and its subdirs, 0700 and owned by the real operator.

        Under sudo the handover is now, not at teardown: a killed run must not leave a
        root-owned base.
        """
        agent6_paths.mkdir_for_real_user(self.graph_dir)
        agent6_paths.mkdir_for_real_user(self.transcripts_dir)
        agent6_paths.mkdir_for_real_user(self.checkpoints_dir)

    def checkpoint_path(self, turn: int) -> pathlib.Path:
        """Return the checkpoint file of a turn.

        Args:
            turn: The turn index.

        Returns:
            `<checkpoints>/<NNNN>.json`.
        """
        return self.checkpoints_dir / f"{turn:04d}.json"


def read_untracked_at_start(session_dir: pathlib.Path) -> frozenset[str]:
    """Read the run's `untracked-at-start` set.

    Args:
        session_dir: The session directory.

    Returns:
        The repo-relative paths; empty when the run recorded none.
    """
    try:
        raw = (session_dir / UNTRACKED_AT_START_NAME).read_bytes()
    except FileNotFoundError:
        return frozenset()
    return frozenset(p.decode("utf-8", "surrogateescape") for p in raw.split(b"\0") if p)


def write_untracked_at_start(session_dir: pathlib.Path, paths: Collection[str]) -> None:
    """Write the run's `untracked-at-start` set.

    Args:
        session_dir: The session directory.
        paths: The repo-relative paths.
    """
    portable.atomic_write(
        session_dir / UNTRACKED_AT_START_NAME,
        b"\0".join(p.encode("utf-8", "surrogateescape") for p in sorted(paths)),
    )


# Under one root, so the state dir's own `machines/` stays the live machine instances.
SESSIONS_ROOT = "sessions"
# One bucket per session mode, named after it; a test pins it to `kinds.session_bucket`.
SESSION_BUCKETS: tuple[str, ...] = ("runs", "plans", "asks", "machines")
# A hub gives machine authoring its own card; an `agent` state's sessions live in the instance.
HUB_BUCKETS: tuple[str, ...] = ("runs", "plans", "asks")


def session_has_record(session_dir: pathlib.Path) -> bool:
    """Return whether a session dir holds a record: a manifest or a journal.

    A dir with neither was refused before it started, or orphaned.

    Args:
        session_dir: The session directory.

    Returns:
        True when either file exists.
    """
    return (session_dir / manifest.MANIFEST_NAME).exists() or (session_dir / LOGS_NAME).exists()


def machines_root(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the directory of machine instances.

    A `machine create` draft is a session and lives under the `machines` bucket instead.

    Args:
        state_dir: The repo's state dir.

    Returns:
        `<state>/machines`.
    """
    return state_dir / "machines"


def bucket_dir(state_dir: pathlib.Path, bucket: str) -> pathlib.Path:
    """Return the directory holding one bucket's sessions.

    Args:
        state_dir: The repo's state dir.
        bucket: The bucket name.

    Returns:
        `<state>/sessions/<bucket>`.
    """
    return state_dir / SESSIONS_ROOT / bucket


def layout_of(session_dir: pathlib.Path) -> SessionLayout:
    """Return the layout of a resolved session directory.

    Rebuilding one from the directory's name alone loses the bucket and defaults to `runs`,
    which retargets a plan or an ask at a path that does not exist.

    Args:
        session_dir: The session directory.

    Returns:
        The layout.
    """
    return SessionLayout(
        state_dir=session_dir.parent.parent.parent,
        session_id=session_dir.name,
        subdir=session_dir.parent.name,
    )


def session_matches(
    state_dir: pathlib.Path, session_id: str, *, buckets: Sequence[str] = SESSION_BUCKETS
) -> list[SessionLayout]:
    """Return every session an id names or prefixes.

    An exact id wins outright: a full id that also prefixes a longer one is not ambiguous.

    Args:
        state_dir: The repo's state dir.
        session_id: The id or prefix.
        buckets: The buckets to search.

    Returns:
        The exact matches, else the prefix matches; empty for an empty id.
    """
    if not session_id:
        return []
    exact: list[SessionLayout] = []
    prefix: list[SessionLayout] = []
    for subdir in buckets:
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


def session_layout(state_dir: pathlib.Path, session_id: str) -> SessionLayout | None:
    """Return the layout an id names in whichever bucket holds it, or None.

    Args:
        state_dir: The repo's state dir.
        session_id: The id or prefix.

    Returns:
        The one match; None when there is none or the prefix is ambiguous.
    """
    matches = session_matches(state_dir, session_id)
    return matches[0] if len(matches) == 1 else None
