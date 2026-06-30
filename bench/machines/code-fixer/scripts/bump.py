# SPDX-License-Identifier: Apache-2.0
"""Increment the attempt counter for the code-fixer machine.

Reads the current count from argv[1] and prints ``{"attempts": <count+1>}`` as a
single JSON line. A non-integer argument is a hard error (non-zero exit), which
the machine routes to its give-up terminal.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: bump.py <current-attempts>", file=sys.stderr)
        return 2
    try:
        current = int(sys.argv[1])
    except ValueError:
        print(f"not an integer: {sys.argv[1]!r}", file=sys.stderr)
        return 2
    print(json.dumps({"attempts": current + 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
