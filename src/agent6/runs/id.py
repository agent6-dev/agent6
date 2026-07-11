# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Friendly run identifiers and prefix resolution.

Run IDs have the shape ``<adjective>-<noun>-<suffix>`` where ``suffix``
is six Crockford base32 characters: four derived from a fresh ULID's
timestamp tail followed by two random. Example: ``sunny-otter-K4Q7B2``.

The leading 4 chars of the suffix encode the low 20 bits of the
current millisecond timestamp and are lexicographically sortable, so
directory listings under the per-repo run-state dir are mostly chronological
within the same ``<adjective>-<noun>`` pair (the timestamp rolls over
roughly every 17 minutes, which is fine for the typical dev session
listing). The trailing 2 chars supply 10 bits of entropy to keep IDs
unique even when several are minted in the same millisecond. The
format is otherwise purely cosmetic; nothing parses run IDs except the
prefix resolver here. Treat them as opaque strings everywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6._data.words import ADJECTIVES, NOUNS
from agent6.graph.ulid import new_ulid

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class RunIdError(Exception):
    """Raised when a user-supplied run id cannot be resolved. ``ambiguous`` is
    True when the query matched more than one run (vs no match), so a caller can
    surface the disambiguation instead of treating it as 'not found'."""

    def __init__(self, message: str, *, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous


def validate_explicit_run_id(run_id: str) -> str:
    """Return *run_id* if it is a safe single path component, else raise.

    A run id becomes a directory name under the state dir (``state_dir/<subdir>/
    <run_id>``). An operator ``--run-id`` with a separator, ``..``, or an
    absolute path would place run state outside the state dir; reject those so an
    explicit id can only ever name a run, never a traversal. Generated ids are
    slug-safe by construction and skip this."""
    if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
        raise RunIdError(
            f"invalid --run-id {run_id!r}: must be a single name with no '/', '\\', or '..'"
        )
    return run_id


def new_friendly_id() -> str:
    """Return a new ``<adj>-<noun>-<suffix>`` run id."""

    rand = os.urandom(6)
    adj = ADJECTIVES[(rand[0] << 8 | rand[1]) % len(ADJECTIVES)]
    noun = NOUNS[(rand[2] << 8 | rand[3]) % len(NOUNS)]
    # 4 timestamp-derived chars (low 20 bits of ms timestamp = 1ms
    # resolution, wraps every ~17 min) followed by 2 random chars for
    # in-millisecond uniqueness.
    ts_part = new_ulid()[6:10]
    rnd_part = _CROCKFORD[rand[4] % 32] + _CROCKFORD[rand[5] % 32]
    return f"{adj}-{noun}-{ts_part}{rnd_part}"


def list_run_ids(runs_dir: Path) -> list[str]:
    """Return run-id directory names under ``runs_dir`` (unsorted)."""

    if not runs_dir.is_dir():
        return []
    return [p.name for p in runs_dir.iterdir() if p.is_dir()]


def resolve_run_id(runs_dir: Path, query: str) -> str:
    """Resolve ``query`` to an exact run-id under ``runs_dir``.

    Accepts an exact match or an unambiguous prefix. Raises
    ``RunIdError`` if no match or more than one match is found.
    """

    if not query:
        raise RunIdError("empty run id")
    ids = list_run_ids(runs_dir)
    if query in ids:
        return query
    matches = [rid for rid in ids if rid.startswith(query)]
    if not matches:
        raise RunIdError(f"no run matches {query!r} under {runs_dir}")
    if len(matches) > 1:
        preview = ", ".join(sorted(matches)[:5])
        raise RunIdError(
            f"run id {query!r} is ambiguous ({len(matches)} matches): {preview}",
            ambiguous=True,
        )
    return matches[0]
