# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `python -m agent6.ui.tui --watch <run-dir>` entry point."""

from __future__ import annotations

import argparse
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    """Open the TUI on a run directory.

    Args:
        argv: The command line; None reads the process's own.

    Returns:
        The exit code: 2 for a missing run dir, 3 when textual is not installed.
    """
    parser = argparse.ArgumentParser(prog="python -m agent6.ui.tui")
    parser.add_argument("--watch", required=True, help="Run directory to tail (<run-dir>)")
    parser.add_argument(
        "--exit-on-end",
        action="store_true",
        help="Close the dashboard when the run ends (used by `agent6 run`'s auto-spawn).",
    )
    args = parser.parse_args(argv)

    session_dir = pathlib.Path(args.watch).expanduser().resolve()
    if not session_dir.exists():
        print(f"agent6 tui: run dir does not exist: {session_dir}", file=sys.stderr)
        return 2

    try:
        from agent6.ui.tui import app  # noqa: PLC0415  # textual is optional  # noqa: PLC0415  # textual is optional
    except ImportError as e:
        print(f"agent6 tui: {e}", file=sys.stderr)
        return 3
    app.run_tui(session_dir, exit_on_end=args.exit_on_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
