# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `mcp` parser's server subcommands."""

from __future__ import annotations

import argparse

from agent6.ui.cli import _common, completers


def _add_mcp_server_parsers(mcp_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `connect`, `remove` and `list` to the `mcp` group beside `serve`."""
    connect = _common._sub(
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
        # A positional's name is its dest, and the root parser's verb already owns args.command.
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
        help=_common.REPO_FLAG_HELP,
    )

    remove = _common._sub(
        mcp_sub,
        "remove",
        help="Remove an MCP server from a config file. Default: global config file.",
    )
    remove_name = remove.add_argument("name", help="Server name shown by `agent6 mcp list`.")
    remove_name.completer = completers._complete_mcp_servers  # type: ignore[attr-defined]
    remove.add_argument(
        "--repo",
        dest="to_repo",
        action="store_true",
        help="Remove from the per-repo config instead of the global config.",
    )

    _common._sub(
        mcp_sub,
        "list",
        help=(
            "Show configured MCP servers and how agent6 reaches them. This does not start or test"
            " the servers."
        ),
    )
