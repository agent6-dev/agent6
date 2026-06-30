# SPDX-License-Identifier: Apache-2.0
"""Collect the last N commits of the repo at the cwd as typed JSON.

Prints one JSON line ``{"count": int, "range": str, "log": str}`` matching the
machine's `gitlog` schema. An empty or non-git directory yields count 0 (the
machine routes that to its no-op terminal), never a crash.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    n = sys.argv[1] if len(sys.argv) > 1 else "20"
    res = subprocess.run(
        ["git", "log", f"-n{n}", "--pretty=format:%h %s (%an)", "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()] if res.returncode == 0 else []
    rng = ""
    if lines:
        rng = f"{lines[-1].split(' ', 1)[0]}..{lines[0].split(' ', 1)[0]}"
    print(json.dumps({"count": len(lines), "range": rng, "log": "\n".join(lines)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
