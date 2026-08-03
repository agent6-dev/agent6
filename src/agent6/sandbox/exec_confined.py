# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Apply Landlock to this process, then become the given command.

    python -m agent6.sandbox.exec_confined --read A --read B --write C -- argv...

For a long-lived child agent6 spawns but does not drive: an MCP server the
operator configured. The jail launcher is the wrong tool there -- it captures
stdio and owns the process to completion, while an MCP server needs a live
bidirectional pipe for the whole session.

Landlock is restrict-self-then-exec: the domain is irrevocable and inherited
across `execve`, so confining ourselves and then becoming the server confines
the server and everything it spawns. `preexec_fn` would be the obvious place
for this and is unsafe in a process with threads, which the MCP client has.

Nothing here is reachable by the model: the paths come from the operator's
config, the argv from the same, and this module is only ever spawned by
agent6 itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent6.sandbox.landlock import LandlockNotSupportedError, apply_agent_landlock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent6.sandbox.exec_confined",
        description="Landlock this process, then exec the command after `--`.",
    )
    parser.add_argument("--read", action="append", default=[], metavar="PATH")
    parser.add_argument("--write", action="append", default=[], metavar="PATH")
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail rather than exec unconfined when the kernel has no Landlock.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- ARGV")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after `--`")

    try:
        apply_agent_landlock(
            read_paths=tuple(Path(p).expanduser() for p in args.read),
            write_paths=tuple(Path(p).expanduser() for p in args.write),
        )
    except LandlockNotSupportedError as exc:
        if args.require:
            print(f"agent6: refusing to run unconfined: {exc}", file=sys.stderr)
            return 2
        # Degrade, never silently: the caller asked for confinement this kernel
        # cannot give, and the server is still worth running.
        print(f"agent6: WARNING: running this server unconfined: {exc}", file=sys.stderr)

    try:
        # The operator's argv, from their own config; no shell, and no LLM
        # input reaches here.
        os.execvp(command[0], command)  # noqa: S606
    except OSError as exc:
        print(f"agent6: could not exec {command[0]}: {exc}", file=sys.stderr)
        return 127
    return 0  # pragma: no cover -- execvp does not return


if __name__ == "__main__":  # pragma: no cover -- the module entry point
    raise SystemExit(main())
