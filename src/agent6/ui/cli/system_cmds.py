# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 system`: host-level setup that needs privileges.

Operator-driven: every command shells out with fixed argv, directly when already root,
else through `sudo`, never with model-supplied input. `apparmor` installs the bundled
profile that lets the strict sandbox use unprivileged user namespaces on Ubuntu 24.04+.

Security review note: `apparmor` writes `/etc/apparmor.d/agent6-jail` and runs
`apparmor_parser` via sudo. The profile is a fixed constant and the argv is fixed. The
profile grants `userns` to the agent6-jail launcher binary only, `flags=(unconfined)`
because the launcher does its own sandboxing, and adds no other capability.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from agent6.ui.cli._common import error, warn

_APPARMOR_PROFILE_PATH = "/etc/apparmor.d/agent6-jail"

# A constant so every install carries it; the glob matches the bundled launcher wherever the
# wheel lands, and a custom AGENT6_JAIL_BIN elsewhere needs its path added (warned at install).
_APPARMOR_PROFILE = """\
# AppArmor profile for the agent6-jail sandbox launcher (managed by
# `agent6 system apparmor`). It lifts the unprivileged user-namespace
# restriction for the launcher binary ONLY, so the strict sandbox isolation works
# on kernels with kernel.apparmor_restrict_unprivileged_userns=1 (Ubuntu 24.04+).
# The launcher does its own sandboxing (userns/pivot_root/Landlock/seccomp/
# NO_NEW_PRIVS), so this adds no AppArmor confinement on top --
# flags=(unconfined). Without it, agent6 falls back to the hardened isolation.
abi <abi/4.0>,
include <tunables/global>

profile agent6-jail /**/agent6/sandbox/_bin/agent6-jail flags=(unconfined) {
  userns,

  include if exists <local/agent6-jail>
}
"""


def _host_lsm() -> str:
    """Return the kernel's active LSM list, or "" when unreadable."""
    try:
        return Path("/sys/kernel/security/lsm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _apparmor_present() -> bool:
    """Return whether this host uses AppArmor, so the profile is meaningful."""
    return "apparmor" in _host_lsm()


def _run_priv(argv: list[str], *, what: str, required: bool = True) -> bool:
    """Run a fixed-argv privileged command, directly as root or through sudo.

    The argv is agent6's own, never LLM input, so a direct subprocess is within the
    security model.

    Args:
        argv: The command.
        what: What it does, for the error text.
        required: Report a failure as an error.

    Returns:
        Whether the command succeeded.
    """
    full = argv if os.geteuid() == 0 else ["sudo", *argv]
    print(f"[agent6] {' '.join(full)}", file=sys.stderr)
    try:
        rc = subprocess.run(full, check=False).returncode
    except OSError as exc:
        if required:
            error(f"could not {what}: {exc}")
        return False
    if rc != 0 and required:
        error(f"{what} failed (exit {rc}).")
    return rc == 0


def _discard_failed_install() -> None:
    """Remove the partial file a failed copy left, so `status` does not report a profile."""
    _run_priv(["rm", "-f", _APPARMOR_PROFILE_PATH], what="remove the failed install")
    if Path(_APPARMOR_PROFILE_PATH).is_file():
        warn(
            f"{_APPARMOR_PROFILE_PATH} was left on disk after the failed install;"
            " remove it with `agent6 system apparmor remove`."
        )
    else:
        print(
            f"Removed {_APPARMOR_PROFILE_PATH} again because the install did not complete.",
            file=sys.stderr,
        )


def _cmd_system_apparmor(action: Literal["install", "remove", "status"]) -> int:
    """Install, remove or report the agent6-jail AppArmor profile.

    Returns:
        The exit code; 1 when the host lacks AppArmor or a step failed.
    """
    installed = Path(_APPARMOR_PROFILE_PATH).is_file()

    if action == "status":
        print(f"AppArmor profile: {'installed' if installed else 'not installed'}")
        print(f"  path: {_APPARMOR_PROFILE_PATH}")
        print(f"  host LSM: {_host_lsm() or 'unknown'}")
        if not _apparmor_present():
            print("  NOTE: this host does not use AppArmor; the profile is a no-op here.")
        print("  Verify the effective sandbox isolation with `agent6 check sandbox`.")
        return 0

    if action == "remove":
        if not installed:
            print(f"Nothing to remove: {_APPARMOR_PROFILE_PATH} is not present.")
            return 0
        # Unload first, best effort (-R fails harmlessly when not loaded); success is the file gone.
        if _apparmor_present():
            _run_priv(
                ["apparmor_parser", "-R", _APPARMOR_PROFILE_PATH],
                what="unload the profile",
                required=False,
            )
        _run_priv(["rm", "-f", _APPARMOR_PROFILE_PATH], what="delete the profile")
        if Path(_APPARMOR_PROFILE_PATH).is_file():
            error(f"{_APPARMOR_PROFILE_PATH} is still present after removal.")
            return 1
        print("Removed the agent6-jail AppArmor profile. The sandbox falls back to hardened.")
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

    # install
    from agent6.sandbox.jail import locate_jail_binary  # noqa: PLC0415  # an import cycle

    jail_bin = locate_jail_binary()
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
    # Loaded from the temp file first, so a profile the parser refuses never reaches the path.
    try:
        ok = _run_priv(["apparmor_parser", "-r", tmp], what="load the profile") and _run_priv(
            ["cp", tmp, _APPARMOR_PROFILE_PATH], what="install the profile"
        )
    finally:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
    if not ok and Path(_APPARMOR_PROFILE_PATH).is_file() and not installed:
        _discard_failed_install()
    if ok:
        print(
            f"Installed {_APPARMOR_PROFILE_PATH}. The profile grants the launcher the"
            " user namespaces this kernel withholds; run `agent6 check sandbox` to see"
            " whether strict is now available."
        )
    return 0 if ok else 1
