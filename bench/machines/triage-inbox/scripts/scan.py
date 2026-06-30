# SPDX-License-Identifier: Apache-2.0
"""Scan the ``inbox/`` directory for the triage-inbox machine.

Prints one JSON line matching the `scan_result` schema: the sorted list of
pending ``*.txt`` item names, how many remain, and the name + text of the head
item (empty strings when the inbox is drained). Read-only: it never moves files,
so it is safe to re-run on replay.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    inbox = Path("inbox")
    items = sorted(p.name for p in inbox.glob("*.txt")) if inbox.is_dir() else []
    head = items[0] if items else ""
    head_text = (inbox / head).read_text(encoding="utf-8").strip() if head else ""
    print(
        json.dumps(
            {"pending": items, "remaining": len(items), "head": head, "head_text": head_text}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
