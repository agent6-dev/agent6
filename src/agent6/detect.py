# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Environment + kernel detection.

Pure stdlib, no agent6 imports. Read-only.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.types import SandboxProfile


@dataclass(frozen=True, slots=True)
class KernelInfo:
    """Parsed Linux kernel version."""

    raw: str
    major: int
    minor: int

    @property
    def supports_landlock_tcp(self) -> bool:
        """Landlock ABI v4 (TCP rules) requires Linux >= 6.7."""
        return (self.major, self.minor) >= (6, 7)

    @property
    def supports_landlock_fs(self) -> bool:
        """Landlock FS rules require Linux >= 5.13 (ABI v1)."""
        return (self.major, self.minor) >= (5, 13)


@dataclass(frozen=True, slots=True)
class Environment:
    """Detected execution environment."""

    in_container: bool
    container_signals: tuple[str, ...]
    kernel: KernelInfo
    userns_supported: bool
    sandbox_available: bool

    @property
    def detected_profile(self) -> SandboxProfile:
        """The strongest jail profile this environment can actually run.

        On non-Linux hosts (macOS) the kernel sandbox does not exist at all,
        so the only profile is `none` (unsandboxed): child commands run as
        plain subprocesses with no confinement. Callers are expected to warn
        loudly when this is selected.

        On Linux, `strict` requires `CLONE_NEWUSER` (and friends) to succeed;
        on hosts where userns is blocked (default-seccomp Docker,
        AppArmor-restricted Ubuntu, locked-down kiosks) we fall back to
        `hardened`, which keeps Landlock + seccomp + capset + rlimits +
        NO_NEW_PRIVS but skips namespaces. `hardened` is still real
        kernel-enforced isolation.
        """
        if not self.sandbox_available:
            return "none"
        return "strict" if self.userns_supported else "hardened"


_KERNEL_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def _parse_kernel(raw: str) -> KernelInfo:
    match = _KERNEL_VERSION_RE.match(raw)
    if match is None:
        return KernelInfo(raw=raw, major=0, minor=0)
    return KernelInfo(raw=raw, major=int(match.group(1)), minor=int(match.group(2)))


def read_kernel() -> KernelInfo:
    """Read the running kernel version from `/proc/sys/kernel/osrelease`."""
    try:
        raw = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").strip()
    except OSError:
        return KernelInfo(raw="unknown", major=0, minor=0)
    return _parse_kernel(raw)


def detect_container_signals() -> tuple[str, ...]:
    """Return the names of all container indicators present (empty = bare host)."""
    signals: list[str] = []
    if Path("/.dockerenv").exists():
        signals.append("/.dockerenv")
    if os.environ.get("REMOTE_CONTAINERS") == "true":
        signals.append("REMOTE_CONTAINERS")
    if os.environ.get("CODESPACES") == "true":
        signals.append("CODESPACES")
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        cgroup = ""
    if any(token in cgroup for token in ("docker", "containerd", "kubepods", "podman")):
        signals.append("cgroup")
    return tuple(signals)


@functools.lru_cache(maxsize=1)
def probe_userns_supported() -> bool:
    """Return True iff this process can create an unprivileged user namespace.

    Uses `unshare -U -r true` as a side-effect-free probe: if the call
    succeeds, the kernel + container policy allow `CLONE_NEWUSER`, which is a
    prerequisite for the `strict` jail profile. Cached for the process
    lifetime; this never changes mid-run.
    """
    unshare = "/usr/bin/unshare"
    if not Path(unshare).is_file():
        return False
    try:
        result = subprocess.run(  # fixed argv, no shell, no LLM input
            [unshare, "-U", "-r", "/usr/bin/true"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def sandbox_available() -> bool:
    """Return True iff the Linux kernel sandbox can be used on this host.

    The sandbox (jail launcher + Landlock + seccomp + namespaces + egress
    broker) is Linux-only. On every other platform there is no confinement
    mechanism, so we run unsandboxed (`profile = none`) and refuse any config
    that explicitly asked for isolation.
    """
    return sys.platform.startswith("linux")


def detect() -> Environment:
    """Detect kernel + container indicators + userns capability."""
    signals = detect_container_signals()
    return Environment(
        in_container=bool(signals),
        container_signals=signals,
        kernel=read_kernel(),
        userns_supported=probe_userns_supported(),
        sandbox_available=sandbox_available(),
    )


def select_profile(requested: str, env: Environment) -> SandboxProfile:
    """Resolve `[sandbox] profile` ("auto"|"strict"|"hardened") against the host.

    Raises `RuntimeError` if the user asked for a profile the kernel + container
    cannot provide. This is the "no silent downgrade" rule: we never give the
    user less isolation than they configured.
    """
    if not env.sandbox_available:
        # Non-Linux host: there is no kernel sandbox. `auto` resolves to the
        # unsandboxed `none` profile (callers warn); an explicit request for
        # real isolation is refused rather than silently downgraded.
        if requested == "auto":
            return "none"
        raise RuntimeError(
            f"sandbox.profile = {requested!r} requires the Linux kernel sandbox "
            f"(Landlock + seccomp + namespaces), which is not available on "
            f"{sys.platform!r}. Set profile = 'auto' to run unsandboxed on this "
            f"platform, or run agent6 on Linux for kernel-enforced isolation."
        )
    if requested == "auto":
        return env.detected_profile
    if requested == "strict":
        if not env.userns_supported:
            raise RuntimeError(
                "sandbox.profile = 'strict' requires unprivileged user namespaces "
                "(`unshare -U -r true`) to succeed, but this host blocks them. "
                "Set profile = 'hardened' (or 'auto') to run without namespaces "
                "while keeping Landlock + seccomp + capset + rlimits."
            )
        return "strict"
    if requested == "hardened":
        return "hardened"
    raise RuntimeError(f"unknown sandbox.profile: {requested!r}")
