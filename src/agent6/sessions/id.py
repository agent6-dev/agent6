# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Friendly session ids and prefix resolution.

An id is `<adjective>-<noun>-<suffix>`, the suffix six Crockford base32 characters: four
from a fresh ULID's timestamp tail (the low 20 bits of the millisecond, wrapping every 17
minutes) and two random. Nothing parses an id except the prefix resolver here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from agent6._data.words import ADJECTIVES, NOUNS
from agent6.git_ops import valid_branch_name
from agent6.graph.ulid import new_ulid
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    SessionLayout,
    bucket_dir,
    is_safe_session_id,
    session_matches,
)

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class SessionIdError(Exception):
    """A user-supplied session id cannot be resolved.

    Attributes:
        no_match: Nothing matched the query. A caller with somewhere else to look (`attach`
            tries machine names next) falls through on this alone; every other refusal (an
            ambiguous prefix, a husk) reaches the operator.
    """

    def __init__(self, message: str, *, no_match: bool = False) -> None:
        """Record the message and whether nothing matched.

        Args:
            message: The refusal.
            no_match: Nothing matched the query.
        """
        super().__init__(message)
        self.no_match = no_match


def validate_explicit_session_id(session_id: str) -> str:
    """Refuse an explicit session id that cannot name both a directory and a git ref.

    Checked before any state is created: a separator or `..` would place state outside the
    state dir, and a name git's ref grammar rejects makes every commit's `update-ref` fail.
    Generated ids are safe by construction and skip this.

    Args:
        session_id: The `--session-id` value.

    Returns:
        The id unchanged.

    Raises:
        SessionIdError: The id is not a single path component or not a valid branch name.
    """
    if not is_safe_session_id(session_id):
        raise SessionIdError(
            f"invalid --session-id {session_id!r}: must be a single name with no '/', '\\', or '..'"
        )
    if not valid_branch_name(session_id):
        raise SessionIdError(
            f"invalid --session-id {session_id!r}: must be usable as a git branch name "
            "(no spaces or any of ~^:?*[\\, no '..' or '@{', "
            "no leading '-'/'.', no trailing '.' or '.lock')"
        )
    return session_id


def friendly_token() -> str:
    """Mint a fresh `<adj>-<noun>-<suffix>` token.

    A session directory is named through `unused_session_id`, which also checks the
    buckets; this is for any other readable unique string (an ACP connection, a fan-out).

    Returns:
        The token.
    """
    rand = os.urandom(6)
    adj = ADJECTIVES[(rand[0] << 8 | rand[1]) % len(ADJECTIVES)]
    noun = NOUNS[(rand[2] << 8 | rand[3]) % len(NOUNS)]
    # Four timestamp chars, then two random ones for in-millisecond uniqueness.
    ts_part = new_ulid()[6:10]
    rnd_part = _CROCKFORD[rand[4] % 32] + _CROCKFORD[rand[5] % 32]
    return f"{adj}-{noun}-{ts_part}{rnd_part}"


def session_id_bucket(state_dir: Path, session_id: str) -> str | None:
    """Return the bucket whose directory already holds the id, or None.

    Ids are one namespace across every bucket: every surface addresses a session by bare
    id, so an id living in two buckets is ambiguous everywhere.

    Args:
        state_dir: The repo's state dir.
        session_id: The id.

    Returns:
        The bucket name, or None when no bucket holds it.
    """
    for bucket in SESSION_BUCKETS:
        if (bucket_dir(state_dir, bucket) / session_id).exists():
            return bucket
    return None


def unused_session_id(state_dir: Path, bucket: str) -> str:
    """Mint an id whose directory exists in no session bucket.

    Two ids minted in the same millisecond collide about once in 30 million.

    Args:
        state_dir: The repo's state dir.
        bucket: The bucket the id is destined for, named in the failure.

    Returns:
        The id.

    Raises:
        RuntimeError: Eight mints in a row collided.
    """
    for _ in range(8):
        candidate = friendly_token()
        if session_id_bucket(state_dir, candidate) is None:
            return candidate
    raise RuntimeError(f"could not mint an unused session id under {bucket_dir(state_dir, bucket)}")


def resolve_session(
    state_dir: Path, query: str, *, buckets: Sequence[str] = SESSION_BUCKETS
) -> SessionLayout:
    """Resolve an id or a unique prefix to one session.

    Args:
        state_dir: The repo's state dir.
        query: The id or prefix.
        buckets: The buckets to search.

    Returns:
        The one matching session.

    Raises:
        SessionIdError: The query is empty or ambiguous, or, with `no_match` set, matches
            nothing; an ambiguous prefix never reads as "no such session".
    """
    if not query:
        raise SessionIdError("empty run id")
    matches = session_matches(state_dir, query, buckets=buckets)
    if len(matches) > 1:
        preview = ", ".join(f"{m.subdir}/{m.session_id}" for m in matches[:5])
        raise SessionIdError(f"run id {query!r} is ambiguous ({len(matches)} matches): {preview}")
    if not matches:
        raise SessionIdError(
            f"no session matches {query!r} (looked under {state_dir})", no_match=True
        )
    return matches[0]
