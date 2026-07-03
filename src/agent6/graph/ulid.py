# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tiny ULID generator, Crockford-base32 26-char sortable IDs.

We use ULIDs (rather than uuid4) because they are time-sortable in lexicographic
order, which makes the on-disk graph trivially diff-friendly: nodes created
earlier sort earlier in `ls` output, and `_first_ready_subtask` relies on id
sort as creation order. Ids created in the same millisecond are made monotonic
by incrementing the previous random part (the standard ULID monotonicity rule);
without that, same-ms ids sort randomly. Implementing this here avoids a
runtime dependency on `python-ulid`.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


@dataclass
class _Monotonic:
    lock: threading.Lock
    last_ms: int = -1
    last_rand: int = 0


_state = _Monotonic(lock=threading.Lock())


def new_ulid() -> str:
    """Return a fresh 26-character Crockford-base32 ULID.

    Format: 48-bit ms timestamp (10 chars) || 80-bit random (16 chars).
    Strictly increasing within a process, including across same-millisecond
    calls and small clock steps backward.
    """
    with _state.lock:
        now_ms = int(time.time() * 1000) & ((1 << 48) - 1)
        if now_ms <= _state.last_ms:
            # Same millisecond (or the clock stepped back): bump the previous
            # random part so the new id still sorts after it.
            _state.last_rand += 1
            if _state.last_rand >= 1 << 80:
                # Counting cannot reach 2^80 in one ms; only a start value at
                # the very top of the range can overflow. Borrow the next
                # millisecond instead of failing.
                _state.last_ms += 1
                _state.last_rand = 0
        else:
            _state.last_ms = now_ms
            _state.last_rand = int.from_bytes(os.urandom(10), "big")
        value = (_state.last_ms << 80) | _state.last_rand
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))
