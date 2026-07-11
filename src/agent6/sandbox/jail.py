# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Python-side launcher for the `agent6-jail` Rust binary.

Serializes a JailPolicy to JSON on stdin and reads child stdout/stderr/return code
from the launcher's output. If the launcher is not available, falls back to a
plain (un-sandboxed) subprocess invocation only when the policy explicitly
opts in via `cwd-only-mode`, otherwise raises JailUnavailableError. This keeps
"silently weaker" failure modes out of the system.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from agent6.types import CommandResult, JailPolicy


class JailUnavailableError(Exception):
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
    # Look beside the repo (un-bundled dev checkout fallback). The crate
    # lives at src/agent6/jail; this file is src/agent6/sandbox/jail.py, so
    # the crate is one level up from the sandbox package.
    pkg_root = Path(__file__).resolve().parents[1]
    candidates = [
        pkg_root / "jail" / "target" / "release" / "agent6-jail",
        pkg_root / "jail" / "target" / "debug" / "agent6-jail",
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
            "tool_paths": [str(p) for p in policy.tool_paths],
            "timeout_s": policy.timeout_s,
        }
    )


def _run_unsandboxed(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` as a plain subprocess (no confinement).

    Used only for the `none` profile on non-Linux hosts. Inherits the parent
    environment (so `PATH` etc. resolve normally) overlaid with `policy.env`;
    runs in `policy.cwd`. The sandbox-only knobs (network, ro/rw/protect paths)
    have no effect here, there is no kernel mechanism to enforce them.
    """
    env = {**os.environ, **{k: v for k, v in policy.env}}
    start = time.monotonic()
    # Unsandboxed escape hatch (non-Linux only); see run_in_jail docstring.
    try:
        proc = subprocess.run(
            list(policy.argv),
            cwd=str(policy.cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=policy.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        # Match the jailed profiles' contract: a timeout is an rc=124 result, not
        # a raised exception the caller would have to special-case. (text=True
        # makes the partial output str at runtime; the stub still types it bytes,
        # so coerce defensively.)
        def _text(v: object) -> str:
            if isinstance(v, bytes):
                return v.decode(errors="replace")
            return v if isinstance(v, str) else ""

        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    duration = time.monotonic() - start
    return CommandResult(
        argv=tuple(policy.argv),
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=duration,
    )


@functools.lru_cache(maxsize=1)
def strict_namespaces_work() -> bool:
    """Return True iff the jail binary can actually set up a `strict` namespace.

    The cheap ``unshare -U -r true`` probe in ``detect.probe_userns_supported``
    under-reports on an AppArmor-restricted host (Ubuntu 24.04+ with
    ``kernel.apparmor_restrict_unprivileged_userns=1``) where a profile grants
    the *agent6-jail* binary userns but not ``/usr/bin/unshare``. This runs the
    real jail binary with a trivial `strict` policy to get the authoritative
    answer. Cached for the process lifetime; the kernel/profile state does not
    change mid-run. Returns False if the jail binary is missing.
    """
    if not Path("/usr/bin/true").exists():
        return False
    probe_cwd = Path(tempfile.gettempdir())
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=probe_cwd,
                argv=("/usr/bin/true",),
                profile="strict",
                allow_network=False,
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        return False
    return res.returncode == 0


def run_in_jail(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` inside the sandbox.

    Raises JailUnavailableError if the launcher binary is missing or setup fails.

    The `none` profile is the unsandboxed path used on non-Linux hosts (see
    `agent6.detect.select_profile`): the command runs as a plain subprocess
    with no kernel confinement. This is never reached on Linux and never from
    config, `select_profile` only returns `none` when `profile = "auto"` on a
    platform without the Linux sandbox, and the CLI prints a prominent warning
    before any such run.

    Security review note: this is the single place where an
    LLM-influenced argv runs without the jail. It exists solely so agent6 is
    usable on platforms (macOS) where the Landlock/seccomp/namespace sandbox
    does not exist. All real-isolation profiles still go through the Rust
    launcher; nothing here weakens the Linux boundary.
    """
    if policy.profile == "none":
        return _run_unsandboxed(policy)
    binary = _locate_jail_binary()
    if binary is None:
        raise JailUnavailableError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path src/agent6/jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    spec = _policy_to_json(policy)
    start = time.monotonic()
    # Launch the launcher in its own session (group leader) so that, if it ever
    # hangs — e.g. a backgrounded grandchild holds the stdout pipe open past the
    # timeout — we can kill its whole process group and reap any orphaned
    # pidns-init/grandchild, not just the launcher itself. Use Popen (not
    # subprocess.run) so we keep the pid to target os.killpg.
    launcher = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = launcher.communicate(input=spec, timeout=policy.timeout_s + 5.0)
    except subprocess.TimeoutExpired as exc:
        # Kill the whole group, then drain whatever output was produced. Mirror
        # _run_unsandboxed: surface a timeout as the documented rc=124 result, not
        # a raised exception the caller would have to special-case.
        try:
            os.killpg(os.getpgid(launcher.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            launcher.kill()
        try:
            out, err = launcher.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            out, err = "", ""

        def _text(v: object) -> str:
            if isinstance(v, bytes):
                return v.decode(errors="replace")
            return v if isinstance(v, str) else ""

        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_text(out) or _text(exc.stdout),
            stderr=_text(err) or _text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    proc = subprocess.CompletedProcess(
        args=[str(binary)],
        returncode=launcher.returncode,
        stdout=out,
        stderr=err,
    )
    duration = time.monotonic() - start
    # The launcher prints a single JSON line on stdout describing the child's result,
    # then exits 0 itself. Anything else means setup failed, with one exception:
    # a child that could not be EXECUTED at all (bad path, missing interpreter)
    # also surfaces as a launcher error, but the jail itself worked fine. Report
    # that as an ordinary failed command (shell-style 127) so the model fixes
    # its argv instead of concluding the sandbox is broken.
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "child execution failed" in stderr:
            return CommandResult(
                argv=tuple(policy.argv),
                returncode=127,
                stdout="",
                stderr=f"{policy.argv[0]}: command not found or not executable ({stderr})",
                duration_s=duration,
                exec_failed=True,
            )
        raise JailUnavailableError(f"agent6-jail launcher exited {proc.returncode}: {stderr}")
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
