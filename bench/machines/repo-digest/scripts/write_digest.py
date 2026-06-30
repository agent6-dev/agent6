# SPDX-License-Identifier: Apache-2.0
"""Render the agent's digest to the machine's persistent data dir.

argv: ``write_digest.py <headline> [highlight ...]`` -- the highlights arrive as
separate argv elements (the machine splices the agent's ``list[str]`` field).
Writes ``$AGENT6_MACHINE_DATA_DIR/digest.md`` and prints ``{"path": ...,
"highlights": int}`` so the machine can whole-capture the result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: write_digest.py <headline> [highlight ...]", file=sys.stderr)
        return 2
    headline = sys.argv[1]
    highlights = sys.argv[2:]
    data = os.environ.get("AGENT6_MACHINE_DATA_DIR")
    if not data:
        print("AGENT6_MACHINE_DATA_DIR is not set", file=sys.stderr)
        return 2
    out = Path(data) / "digest.md"
    body = "\n".join(f"- {h}" for h in highlights) or "- (no highlights)"
    out.write_text(f"# {headline}\n\n{body}\n", encoding="utf-8")
    print(json.dumps({"path": str(out), "highlights": len(highlights)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
