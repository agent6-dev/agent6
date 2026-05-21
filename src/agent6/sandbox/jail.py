# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Python-side launcher for the `agent6-jail` Rust binary.

Serializes a JailPolicy to JSON on stdin and reads child stdout/stderr/return code
from the launcher's output. If the launcher is not available, falls back to a
plain (un-sandboxed) subprocess invocation only when the policy explicitly
opts in via `cwd-only-mode` — otherwise raises JailUnavailableError. This keeps
"silently weaker" failure modes out of the system.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from agent6.types import CommandResult, JailPolicy


class JailUnavailableError(RuntimeError):
    """`agent6-jail` could not be located or refused to set up the namespace."""


# Override for tests; checked first.
_ENV_VAR = "AGENT6_JAIL_BIN"


def _locate_jail_binary() -> Path | None:
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override)
        return p if p.is_file() else None
    # Bundled inside the installed package (the wheel ships the binary
    # under agent6/sandbox/_bin/agent6-jail; see hatch_build.py).
    bundled = Path(__file__).resolve().parent / "_bin" / "agent6-jail"
    if bundled.is_file():
        return bundled
    # Look in PATH
    found = shutil.which("agent6-jail")
    if found:
        return Path(found)
    # Look beside the repo (un-bundled dev checkout fallback)
    candidates = [
        Path(__file__).resolve().parents[3] / "jail" / "target" / "release" / "agent6-jail",
        Path(__file__).resolve().parents[3] / "jail" / "target" / "debug" / "agent6-jail",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _policy_to_json(policy: JailPolicy) -> str:
    return json.dumps(
        {
            "profile": policy.profile,
            "cwd": str(policy.cwd),
            "argv": list(policy.argv),
            "env": [list(pair) for pair in policy.env],
            "allow_network": policy.allow_network,
            "extra_ro_paths": [str(p) for p in policy.extra_ro_paths],
            "extra_rw_paths": [str(p) for p in policy.extra_rw_paths],
            "extra_protect_paths": [str(p) for p in policy.extra_protect_paths],
            "timeout_s": policy.timeout_s,
        }
    )


def run_in_jail(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` inside the sandbox.

    Raises JailUnavailableError if the launcher binary is missing or setup fails.
    """
    binary = _locate_jail_binary()
    if binary is None:
        raise JailUnavailableError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    spec = _policy_to_json(policy)
    start = time.monotonic()
    proc = subprocess.run(
        [str(binary)],
        input=spec,
        capture_output=True,
        text=True,
        check=False,
        timeout=policy.timeout_s + 5.0,
    )
    duration = time.monotonic() - start
    # The launcher prints a single JSON line on stdout describing the child's result,
    # then exits 0 itself. Anything else means setup failed.
    if proc.returncode != 0:
        raise JailUnavailableError(
            f"agent6-jail launcher exited {proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        result_json = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise JailUnavailableError(
            f"agent6-jail produced unparseable output: {proc.stdout!r}"
        ) from exc
    return CommandResult(
        argv=tuple(policy.argv),
        returncode=int(result_json["returncode"]),
        stdout=str(result_json.get("stdout", "")),
        stderr=str(result_json.get("stderr", "")),
        duration_s=duration,
    )
