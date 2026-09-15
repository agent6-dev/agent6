# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `mcp` and its subcommands."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import REPO_FLAG_HELP, _sub
from agent6.ui.cli.completers import _complete_mcp_servers


def _add_mcp_server_parsers(mcp_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`connect`, `remove` and `list`, added to the `mcp` group beside `serve`."""
    connect = _sub(
        mcp_sub,
        "connect",
        help=(
            "Add an MCP server. agent6 starts it and shows its tools before saving it. If"
            " sandboxing is unavailable, a server started by a command is saved without testing"
            " it and gets a warning."
        ),
    )
    connect.add_argument(
        "name",
        help=(
            "Server name. Use letters, numbers, hyphens, or underscores, but never two"
            " underscores in a row. Its tools appear as mcp__NAME__TOOL."
        ),
    )
    connect.add_argument(
        # Not named "command": a positional's name is its dest, and the root
        # parser's subcommand verb already owns args.command.
        "server_command",
        nargs="*",
        metavar="COMMAND",
        help=(
            "Command and arguments used to start the server. Put `--` before the command so its"
            " options reach the server. Give either a command or a URL, not both."
        ),
    )
    connect.add_argument(
        "--url",
        default="",
        metavar="URL",
        help=(
            "URL of an MCP server that is already running. It must start with http:// or"
            " https://. Give either a URL or a command, not both."
        ),
    )
    connect.add_argument(
        "--token-env",
        dest="token_env",
        default="",
        metavar="VAR",
        help=(
            "Name of the environment variable that holds the bearer token for a URL server."
            " agent6 saves the variable name, not the token."
        ),
    )
    connect.add_argument(
        "--pass-env",
        dest="pass_env",
        action="append",
        default=[],
        metavar="VAR",
        help=(
            "Name of an environment variable to pass to a server started by a command. Repeat"
            " this option for each variable. Other variables come from agent6's limited base"
            " environment, which excludes model-provider API keys."
        ),
    )
    connect.add_argument(
        "--repo",
        dest="to_repo",
        action="store_true",
        help=REPO_FLAG_HELP,
    )

    remove = _sub(
        mcp_sub,
        "remove",
        help="Remove an MCP server from a config file. Default: global config file.",
    )
    remove_name = remove.add_argument("name", help="Server name shown by `agent6 mcp list`.")
    remove_name.completer = _complete_mcp_servers  # type: ignore[attr-defined]
    remove.add_argument(
        "--repo",
        dest="to_repo",
        action="store_true",
        help="Remove from the per-repo config instead of the global config.",
    )

    _sub(
        mcp_sub,
        "list",
        help=(
            "Show configured MCP servers and how agent6 reaches them. This does not start or test"
            " the servers."
        ),
    )
