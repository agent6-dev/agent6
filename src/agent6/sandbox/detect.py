# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Detect the host's kernel, container and confinement capabilities.

A read-only leaf: imports only `agent6.kinds` and the sibling `landlock` probe.
Probes shell out with fixed argv only.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import platform
import re
import subprocess
import sys

from agent6 import kinds
from agent6.sandbox import landlock


@dataclasses.dataclass(frozen=True, slots=True)
class KernelInfo:
    """Parsed Linux kernel version."""

    raw: str
    major: int
    minor: int


@dataclasses.dataclass(frozen=True, slots=True)
class Environment:
    """The detected execution environment.

    Attributes:
        in_container: A container indicator is present.
        container_signals: The names of the indicators present.
        kernel: The running kernel's version.
        userns_supported: This process can create an unprivileged user namespace.
        landlock_abi: The probed Landlock ABI version, 0 without Landlock; the syscall
            probe, not a kernel-version guess, since a kernel can ship with the LSM
            compiled out or disabled via `lsm=`.
        seccomp_arch_supported: The jail's seccomp filter exists for this CPU; mirrors
            the arch set in `jail/src/main.rs` `apply_seccomp`, which fails closed, and
            both strict and hardened promise that filter.
        sandbox_available: The host is Linux.
    """

    in_container: bool
    container_signals: tuple[str, ...]
    kernel: KernelInfo
    userns_supported: bool
    landlock_abi: int
    seccomp_arch_supported: bool
    sandbox_available: bool

    @property
    def detected_isolation(self) -> kinds.IsolationLevel:
        """The strongest jail isolation this environment can run.

        `strict` needs user namespaces; without them `hardened` keeps Landlock,
        seccomp and NO_NEW_PRIVS, and since Landlock is its only filesystem boundary
        it needs the Landlock probe to succeed. A host with neither, or no Linux
        kernel, resolves to `none`: unsandboxed and loudly warned by the caller,
        never a hardened label that confines nothing.
        """
        if not self.sandbox_available or not self.seccomp_arch_supported:
            return "none"
        if self.userns_supported:
            return "strict"
        return "hardened" if self.landlock_abi >= 1 else "none"


_KERNEL_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def _parse_kernel(raw: str) -> KernelInfo:
    """Return the major and minor version parsed from an osrelease string."""
    match = _KERNEL_VERSION_RE.match(raw)
    if match is None:
        return KernelInfo(raw=raw, major=0, minor=0)
    return KernelInfo(raw=raw, major=int(match.group(1)), minor=int(match.group(2)))


def read_kernel() -> KernelInfo:
    """Return the running kernel version from `/proc/sys/kernel/osrelease`."""
    try:
        raw = pathlib.Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").strip()
    except OSError:
        return KernelInfo(raw="unknown", major=0, minor=0)
    return _parse_kernel(raw)


def detect_container_signals() -> tuple[str, ...]:
    """Return the names of the container indicators present; empty on a bare host."""
    signals: list[str] = []
    if pathlib.Path("/.dockerenv").exists():
        signals.append("/.dockerenv")
    # Rootless podman often lacks a "podman" token in /proc/1/cgroup; this file is its marker.
    if pathlib.Path("/run/.containerenv").exists():
        signals.append("/run/.containerenv")
    if os.environ.get("REMOTE_CONTAINERS") == "true":
        signals.append("REMOTE_CONTAINERS")
    if os.environ.get("CODESPACES") == "true":
        signals.append("CODESPACES")
    try:
        cgroup = pathlib.Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        cgroup = ""
    if any(token in cgroup for token in ("docker", "containerd", "kubepods", "podman")):
        signals.append("cgroup")
    return tuple(signals)


def sandbox_disabled_by_env() -> bool:
    """Return whether `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` is set.

    The env form of `--dangerously-disable-sandbox`, read by `resolve_isolation`.
    A `machine run` supervisor resolves once and passes `none` to each agent
    subprocess in its request, which does not re-resolve. The LLM cannot set the
    launcher's environment.
    """
    return os.environ.get("AGENT6_DANGEROUSLY_DISABLE_SANDBOX") == "1"


@functools.lru_cache(maxsize=1)
def probe_userns_supported() -> bool:
    """Return whether this process can create an unprivileged user namespace.

    `unshare -U -r true` is the side-effect-free probe; `strict` needs it to succeed.
    Cached for the process lifetime.
    """
    unshare = "/usr/bin/unshare"
    if not pathlib.Path(unshare).is_file():
        return False
    try:
        result = subprocess.run(  # fixed argv, no LLM input
            [unshare, "-U", "-r", "/usr/bin/true"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def probe_landlock_abi() -> int:
    """Return the kernel's Landlock ABI version, 0 when unavailable.

    Fails closed: a probe error reads as no Landlock, so resolution refuses
    `hardened` and takes `auto` to the warned `none`. Cached for the process lifetime.
    """
    if not sandbox_available():
        return 0
    try:
        return max(0, landlock.landlock_abi())
    except landlock.LandlockError:
        return 0


def apparmor_userns_restricted() -> bool:
    """Return whether AppArmor restricts unprivileged user namespaces.

    Ubuntu 23.10+ ships `kernel.apparmor_restrict_unprivileged_userns=1`, so
    `strict` can be unavailable with `unprivileged_userns_clone = 1`; the fix is
    `agent6 system apparmor install` or the sysctl at 0. The proc file is absent
    on non-AppArmor kernels.
    """
    try:
        raw = pathlib.Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    return raw.strip() == "1"


def sandbox_available() -> bool:
    """Return whether the host is Linux, the only platform with the kernel sandbox."""
    return sys.platform.startswith("linux")


def _read_max_userns() -> str | None:
    """Return `user.max_user_namespaces`, or None when unreadable."""
    try:
        return (
            pathlib.Path("/proc/sys/user/max_user_namespaces").read_text(encoding="utf-8").strip()
        )
    except OSError:
        return None


def _userns_block_cause(env: Environment) -> str:
    """Return the mechanism blocking unprivileged user namespaces on this host."""
    if apparmor_userns_restricted():
        return (
            "unprivileged user namespaces are blocked by AppArmor "
            "(kernel.apparmor_restrict_unprivileged_userns=1, the Ubuntu 23.10+ default); "
            "`agent6 system apparmor install` grants userns to just the jail binary"
        )
    if _read_max_userns() == "0":
        return (
            "unprivileged user namespaces are disabled (user.max_user_namespaces = 0); "
            "raise it for strict"
        )
    if env.in_container:
        return (
            "unprivileged user namespaces are blocked in this container "
            "(the runtime's seccomp/AppArmor policy)"
        )
    return "unprivileged user namespaces are blocked on this host (`unshare -U -r true` fails)"


def degrade_reason(env: Environment) -> str | None:
    """Return why `auto` resolves below `strict` here, or None at full strength.

    Every surface reporting an auto-selected level below strict prints this, so a
    degraded level never appears without its cause.
    """
    if not env.sandbox_available:
        return f"there is no Linux kernel sandbox on {sys.platform!r}"
    if not env.seccomp_arch_supported:
        return (
            f"the jail's seccomp filter does not exist for {platform.machine()!r} "
            "(filters exist for x86_64 and aarch64)"
        )
    if env.userns_supported:
        return None
    cause = _userns_block_cause(env)
    if env.landlock_abi < 1:
        return (
            f"{cause}; this kernel offers no Landlock either "
            "(needs Linux >= 5.13 with the Landlock LSM enabled)"
        )
    return cause


def environment() -> Environment:
    """Return the host's kernel, container indicators and confinement capabilities."""
    signals = detect_container_signals()
    return Environment(
        in_container=bool(signals),
        container_signals=signals,
        kernel=read_kernel(),
        userns_supported=probe_userns_supported(),
        landlock_abi=probe_landlock_abi(),
        seccomp_arch_supported=platform.machine() in ("x86_64", "aarch64"),
        sandbox_available=sandbox_available(),
    )


class IsolationUnavailableError(Exception):
    """The host cannot provide the requested `[sandbox] isolation`.

    A distinct type, so the refusal sites catch exactly this and no unrelated fault.
    """


def resolve_isolation(requested: str, env: Environment) -> kinds.IsolationLevel:
    """Resolve `[sandbox] isolation` against the host.

    `auto` degrades to the strongest level the host offers; an explicit level the
    host cannot honor is refused, never downgraded.

    Args:
        requested: "auto", "strict", "hardened" or "none".
        env: The detected environment.

    Returns:
        The level to run.

    Raises:
        IsolationUnavailableError: The host cannot provide the requested level, or the
            level is unknown.
    """
    if sandbox_disabled_by_env():
        requested = "none"
    if not env.sandbox_available:
        if requested in ("auto", "none"):
            return "none"
        raise IsolationUnavailableError(
            f"sandbox.isolation = {requested!r} requires the Linux kernel sandbox "
            f"(Landlock + seccomp + namespaces), which is not available on "
            f"{sys.platform!r}. Set isolation = 'auto' to run unsandboxed on this "
            f"platform, or run agent6 on Linux for kernel-enforced isolation."
        )
    if requested == "auto":
        # `none` is never silent: callers warn, and with auto-approved run_command a confirm
        # gate fires too.
        return env.detected_isolation
    if requested == "none":
        # Self-authorizing: the config key, the flag and the env var are all operator-only.
        return "none"
    if requested in ("strict", "hardened") and not env.seccomp_arch_supported:
        raise IsolationUnavailableError(
            f"sandbox.isolation = {requested!r} requires the jail's seccomp filter, "
            f"which does not exist for {platform.machine()!r} (filters exist for "
            "x86_64 and aarch64). Set isolation = 'auto' to run unsandboxed on "
            "this machine, or 'none' to opt out explicitly."
        )
    if requested == "strict":
        if not env.userns_supported:
            raise IsolationUnavailableError(
                "sandbox.isolation = 'strict' requires unprivileged user namespaces "
                "(`unshare -U -r true`) to succeed, but this host blocks them. "
                "Set isolation = 'hardened' (or 'auto') to run without namespaces "
                "while keeping Landlock + seccomp + NO_NEW_PRIVS."
            )
        return "strict"
    if requested == "hardened":
        if env.landlock_abi < 1:
            raise IsolationUnavailableError(
                "sandbox.isolation = 'hardened' requires Landlock (Linux >= 5.13 "
                "with the Landlock LSM enabled), which this kernel does not provide. "
                "Hardened without it applies no filesystem confinement at all. Set "
                "isolation = 'auto', or 'none' to run unsandboxed."
            )
        if env.landlock_abi < 3:
            raise IsolationUnavailableError(
                "sandbox.isolation = 'hardened' requires Landlock ABI 3 (Linux 6.2) "
                "to confine file truncation, but this kernel's Landlock is ABI "
                f"{env.landlock_abi}: a jailed command could truncate "
                "(truncate/ftruncate) files outside its write grants. Set isolation "
                "= 'auto' to run with a warning, or upgrade the kernel."
            )
        return "hardened"
    raise IsolationUnavailableError(f"unknown sandbox.isolation: {requested!r}")
