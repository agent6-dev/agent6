# SPDX-License-Identifier: Apache-2.0
"""Move one processed inbox item into a bucket directory.

argv: ``process.py <item-name> <bucket>``. Moves ``inbox/<item>`` to
``processed/<bucket>/<item>``, draining the inbox one item per cycle.
Idempotent: a missing source (already moved on a retry) is a no-op success, so
the at-least-once executor never double-fails.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: process.py <item-name> <bucket>", file=sys.stderr)
        return 2
    item, bucket = sys.argv[1], sys.argv[2]
    src = Path("inbox") / item
    dest_dir = Path("processed") / bucket
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = False
    if src.is_file():
        shutil.move(str(src), str(dest_dir / item))
        moved = True
    print(json.dumps({"item": item, "bucket": bucket, "moved": moved}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
