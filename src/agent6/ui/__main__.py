# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`python -m agent6.ui --watch <run-dir>` entrypoint for the TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent6.ui")
    parser.add_argument("--watch", required=True, help="Run directory to tail (.agent6/runs/<id>)")
    args = parser.parse_args(argv)

    run_dir = Path(args.watch).expanduser().resolve()
    if not run_dir.exists():
        print(f"agent6 ui: run dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    try:
        from agent6.ui.tui import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"agent6 ui: {e}", file=sys.stderr)
        return 3
    return run_tui(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
