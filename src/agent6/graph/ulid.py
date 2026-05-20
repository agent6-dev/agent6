# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tiny ULID generator — Crockford-base32 26-char sortable IDs.

We use ULIDs (rather than uuid4) because they are time-sortable in lexicographic
order, which makes the on-disk graph trivially diff-friendly: nodes created
earlier sort earlier in `ls` output. Implementing this here avoids a runtime
dependency on `python-ulid`.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """Return a fresh 26-character Crockford-base32 ULID.

    Format: 48-bit ms timestamp (10 chars) || 80-bit random (16 chars).
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")  # 80 bits
    value = (ts_ms << 80) | rand
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def is_ulid(s: str) -> bool:
    """Cheap structural validity check (length + alphabet)."""
    if len(s) != 26:
        return False
    return all(c in _CROCKFORD for c in s)
