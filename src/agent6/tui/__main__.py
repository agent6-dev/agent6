# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`python -m agent6.tui --watch <run-dir>` entrypoint for the TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent6.tui")
    parser.add_argument("--watch", required=True, help="Run directory to tail (<run-dir>)")
    parser.add_argument(
        "--exit-on-end",
        action="store_true",
        help="Close the dashboard when the run ends (used by `agent6 run`'s auto-spawn).",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.watch).expanduser().resolve()
    if not run_dir.exists():
        print(f"agent6 tui: run dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    try:
        from agent6.tui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"agent6 tui: {e}", file=sys.stderr)
        return 3
    return run_tui(run_dir, exit_on_end=args.exit_on_end)


if __name__ == "__main__":
    raise SystemExit(main())
