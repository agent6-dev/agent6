# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 system`: host / OS-level setup that needs privileges.

Operator-driven admin actions, not LLM-driven: every command here shells out
with FIXED argv (run directly when already root, else prefixed with ``sudo``),
never with model-supplied input. Distinct from ``agent6 init`` (per-repo setup).
The first member is ``apparmor`` (install the bundled AppArmor profile that
lets the strict sandbox use unprivileged user namespaces on Ubuntu 24.04+); the
namespace is shaped to grow (e.g. a future ``system service`` for a systemd unit).

Security review note: this writes ``/etc/apparmor.d/agent6-jail`` and runs
``apparmor_parser`` via sudo. The profile content is a fixed in-repo constant
(below) and the argv is fixed; nothing here is influenced by LLM output. The
profile grants ``userns`` to the agent6-jail launcher binary ONLY (it is
``flags=(unconfined)`` because the launcher does its own sandboxing) and adds no
other capability.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

_APPARMOR_PROFILE_PATH = "/etc/apparmor.d/agent6-jail"

# The bundled AppArmor profile. Shipped as a constant so a pip/uv/pipx install
# carries it (no repo checkout needed). The attachment glob matches the bundled
# launcher wherever the wheel lands (.../agent6/sandbox/_bin/agent6-jail); a
# custom AGENT6_JAIL_BIN elsewhere needs its path added (warned at install).
_APPARMOR_PROFILE = """\
# AppArmor profile for the agent6-jail sandbox launcher (managed by
# `agent6 system apparmor`). It lifts the unprivileged user-namespace
# restriction for the launcher binary ONLY, so the strict sandbox profile works
# on kernels with kernel.apparmor_restrict_unprivileged_userns=1 (Ubuntu 24.04+).
# The launcher does its own sandboxing (userns/pivot_root/Landlock/seccomp/
# capset(0)/NO_NEW_PRIVS), so this adds no AppArmor confinement on top --
# flags=(unconfined). Without it, agent6 falls back to the hardened profile.
abi <abi/4.0>,
include <tunables/global>

profile agent6-jail /**/agent6/sandbox/_bin/agent6-jail flags=(unconfined) {
  userns,

  include if exists <local/agent6-jail>
}
"""


def _host_lsm() -> str:
    try:
        return Path("/sys/kernel/security/lsm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _apparmor_present() -> bool:
    """True when this host actually uses AppArmor (so the profile is meaningful)."""
    return shutil.which("apparmor_parser") is not None and "apparmor" in _host_lsm()


def _run_priv(argv: list[str], *, what: str) -> bool:
    """Run a fixed-argv privileged command: directly if already root, else via
    sudo. Returns True on success. The argv is operator/agent6-fixed, never LLM
    input (so a direct subprocess is within the security model)."""
    full = argv if os.geteuid() == 0 else ["sudo", *argv]
    print(f"[agent6] {' '.join(full)}", file=sys.stderr)
    try:
        rc = subprocess.run(full, check=False).returncode
    except OSError as exc:
        print(f"ERROR: could not {what}: {exc}", file=sys.stderr)
        return False
    if rc != 0:
        print(f"ERROR: {what} failed (exit {rc}).", file=sys.stderr)
    return rc == 0


def _cmd_system_apparmor(action: Literal["install", "remove", "status"]) -> int:
    """Install / remove / report the agent6-jail AppArmor profile."""
    installed = Path(_APPARMOR_PROFILE_PATH).is_file()

    if action == "status":
        print(f"AppArmor profile: {'installed' if installed else 'not installed'}")
        print(f"  path: {_APPARMOR_PROFILE_PATH}")
        print(f"  host LSM: {_host_lsm() or 'unknown'}")
        if not _apparmor_present():
            print("  NOTE: this host does not use AppArmor; the profile is a no-op here.")
        print("  Verify the effective sandbox profile with `agent6 check sandbox`.")
        return 0

    if not _apparmor_present():
        print(
            "This host does not use AppArmor (LSM: "
            f"{_host_lsm() or 'unknown'}), so the agent6-jail AppArmor profile is not"
            " applicable. On Ubuntu 24.04+ it lets the strict sandbox use user"
            " namespaces; other distros (e.g. Fedora/SELinux) allow them already.",
            file=sys.stderr,
        )
        return 1

    if action == "remove":
        if not installed:
            print(f"Nothing to remove: {_APPARMOR_PROFILE_PATH} is not present.")
            return 0
        # Unload from the kernel first (best-effort: the profile may be on disk
        # but not loaded, in which case -R exits non-zero harmlessly), then
        # delete the file. Success is "the file is gone", not the -R exit.
        _run_priv(["apparmor_parser", "-R", _APPARMOR_PROFILE_PATH], what="unload the profile")
        _run_priv(["rm", "-f", _APPARMOR_PROFILE_PATH], what="delete the profile")
        if Path(_APPARMOR_PROFILE_PATH).is_file():
            print(
                f"ERROR: {_APPARMOR_PROFILE_PATH} is still present after removal.",
                file=sys.stderr,
            )
            return 1
        print("Removed the agent6-jail AppArmor profile. The sandbox falls back to hardened.")
        return 0

    # install
    from agent6.sandbox.jail import _locate_jail_binary  # noqa: PLC0415 - avoid import cycle

    jail_bin = _locate_jail_binary()
    if jail_bin is not None and "/agent6/sandbox/_bin/agent6-jail" not in str(jail_bin):
        print(
            f"NOTE: your jail binary is at {jail_bin}, which the bundled profile's glob"
            " (/**/agent6/sandbox/_bin/agent6-jail) may not match. If `agent6 check"
            " sandbox` still reports hardened, add that path to the profile header.",
            file=sys.stderr,
        )
    with tempfile.NamedTemporaryFile("w", suffix=".apparmor", delete=False) as fh:
        fh.write(_APPARMOR_PROFILE)
        tmp = fh.name
    try:
        ok = _run_priv(["cp", tmp, _APPARMOR_PROFILE_PATH], what="install the profile")
        if ok:
            ok = _run_priv(
                ["apparmor_parser", "-r", _APPARMOR_PROFILE_PATH], what="load the profile"
            )
    finally:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
    if ok:
        print(
            f"Installed {_APPARMOR_PROFILE_PATH}; `agent6 check sandbox` now reports strict"
            " (per-command jailing). Heads-up: strict's default provider-egress also needs"
            " THIS process to create a user namespace, which this surgical profile does not"
            " grant -- so a default `agent6 run` falls back to hardened (with a note). For"
            " full strict, set kernel.apparmor_restrict_unprivileged_userns=0 (host-wide), or"
            " use sandbox.agent_network='open'."
        )
    return 0 if ok else 1
