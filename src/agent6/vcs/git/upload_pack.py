# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the sending side of a local fetch under the jail of the workspace fetched from.

A `.git` a jailed command can write names what `git upload-pack` reads: an alternate, a
`gitdir:` file, a `commondir`, a symlinked ref. `git fetch --upload-pack=<command>` starts
the sending side with that command. `command` returns one that runs this module, which
starts `git upload-pack` through `spawn_in_jail` on the fetch's own stdin and stdout.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shlex
import sys

import pydantic

from agent6 import kinds
from agent6.sandbox import jail

_RECORD = pydantic.TypeAdapter(kinds.JailPolicy)


def command(sender: kinds.JailPolicy, repository: pathlib.Path, record: pathlib.Path) -> str:
    """Record the sending side's policy and return the `--upload-pack` value that runs it.

    Args:
        sender: The policy the commands of the repository's workspace run under.
        repository: The repository fetched from.
        record: Where the policy is written, in a directory no jail reaches.

    Returns:
        The command line git runs through `sh -c`, with the repository's path appended. The
        shell hands the module its own parent, the fetch, as `$PPID`.
    """
    served = dataclasses.replace(
        sender,
        # The jail mounts a workspace at its resolved path.
        argv=("git", "upload-pack", str(repository.resolve())),
        # upload-pack reaches no network, and only the run itself joins its session network.
        network="none" if sender.isolation == "full" else sender.network,
    )
    record.write_bytes(_RECORD.dump_json(served))
    argv = [sys.executable, "-P", "-E", "-m", "agent6.vcs.git.upload_pack", str(record)]
    # `exec`: the fetch is then this module's parent, whose end `main` follows.
    return "exec " + shlex.join(argv) + ' "$PPID"'


def main(argv: list[str]) -> int:
    """Run the recorded `git upload-pack` on this process's stdin and stdout.

    git appends the repository's path, which the record already names.

    Args:
        argv: This module's argv: the record's path, then the pid of the fetch that started
            the shell.

    Returns:
        upload-pack's exit code; 1 when the jail cannot start.
    """
    # A stopped fetch takes the sending side with it, so nothing holds its caller's pipes.
    # The shell named the fetch: one that died since is no longer this process's parent.
    jail.die_with_parent(int(argv[2]))()
    policy = _RECORD.validate_json(pathlib.Path(argv[1]).read_bytes())
    # Without the fetch's version upload-pack speaks version 0, which refuses a tree's id.
    if protocol := os.environ.get("GIT_PROTOCOL"):
        policy = dataclasses.replace(policy, env=(*policy.env, ("GIT_PROTOCOL", protocol)))
    try:
        child = jail.spawn_in_jail(policy, stdin=0, stdout=1, stderr=2)
    except jail.JailUnavailableError as exc:
        print(f"agent6: the sending side has no jail to run in: {exc}", file=sys.stderr)
        return 1
    try:
        code = child.popen.wait()
    finally:
        child.close()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
