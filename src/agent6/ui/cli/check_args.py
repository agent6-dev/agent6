"""Build the `check` and `system` parsers."""

from __future__ import annotations

import argparse

from agent6.ui.cli import common


def add_check_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `check`, the read-only pre-flight checks."""
    check_p = common._sub(
        sub,
        "check",
        help=(
            "Check this host and repo before a run: the sandbox, the config and its"
            " provider keys, the file and network boundaries, the MCP servers, the verify"
            " command. Reads only; changes nothing in the repository."
        ),
    )
    check_p.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "sandbox", "config", "mcp", "verify"),
        help="One section, or all (the default) with a PASS/FAIL summary at the end.",
    )
    common._add_config_flag(check_p)


def add_system_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `system <component> <action>`, the privileged host setup that uses sudo."""
    system_p = common._sub(
        sub,
        "system",
        help="Host/OS setup that needs privileges, and uses sudo to get them."
        " Installs or removes the AppArmor profile that sandbox.isolation = 'full' needs.",
    )
    system_sub = system_p.add_subparsers(
        dest="system_command", required=True, metavar="<subcommand>"
    )
    apparmor_p = common._sub(
        system_sub,
        "apparmor",
        help="Install or remove the AppArmor profile that sandbox.isolation = 'full' needs"
        " on Ubuntu 24.04+.",
    )
    apparmor_p.add_argument(
        "action",
        choices=("install", "remove", "status"),
        help="install the profile and reload AppArmor, remove it, or report its state.",
    )
