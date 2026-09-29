# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A ULID generator: 26-character Crockford base32 ids that sort by creation time.

Ids minted in the same millisecond stay monotonic by incrementing the previous random part,
the standard ULID rule. In-tree, so no runtime dependency on `python-ulid`.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time

# TaskNode's id validator enforces the alphabet where an id becomes a path component.
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


@dataclasses.dataclass
class _Monotonic:
    """The last id minted, under a lock.

    Attributes:
        lock: Serializes minting.
        last_ms: The last id's millisecond; -1 before the first.
        last_rand: The last id's random part.
    """

    lock: threading.Lock
    last_ms: int = -1
    last_rand: int = 0


_state = _Monotonic(lock=threading.Lock())


def new_ulid() -> str:
    """Mint a fresh ULID.

    Returns:
        A 48-bit millisecond timestamp and 80 random bits as 26 Crockford base32 characters,
        strictly increasing within the process, across same-millisecond calls and small
        clock steps backward.
    """
    with _state.lock:
        now_ms = int(time.time() * 1000) & ((1 << 48) - 1)
        if now_ms <= _state.last_ms:
            # The same millisecond, or the clock stepped back: bump the random part.
            _state.last_rand += 1
            if _state.last_rand >= 1 << 80:
                # Only a start value at the top of the range can overflow: borrow the next ms.
                _state.last_ms += 1
                _state.last_rand = 0
        else:
            _state.last_ms = now_ms
            _state.last_rand = int.from_bytes(os.urandom(10), "big")
        value = (_state.last_ms << 80) | _state.last_rand
    chars: list[str] = []
    for _ in range(26):
        chars.append(CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))
