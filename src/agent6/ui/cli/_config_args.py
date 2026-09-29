# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `config`, `connect` and `model` parsers."""

from __future__ import annotations

import argparse
import functools
import pathlib
from typing import get_args

from agent6.config import EffortLevel
from agent6.ui.cli import _common, completers


def _add_machine_file(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add `--machine-file FILE`, redirecting the verb to a machine file's [config] overlay."""
    arg = parser.add_argument(
        "--machine-file",
        dest="machine_file",
        type=pathlib.Path,
        default=None,
        metavar="FILE",
        help=help_text,
    )
    arg.completer = completers._complete_machine_files  # type: ignore[attr-defined]


def _add_config_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `config` and its subcommands over the layered config."""
    config_p = _common._sub(
        sub,
        "config",
        help=(
            "Show or change agent6 settings. With no subcommand, show every setting and where"
            " its value came from."
        ),
    )
    config_sub = config_p.add_subparsers(
        dest="config_command", required=True, metavar="<subcommand>"
    )
    config_show = _common._sub(
        config_sub,
        "show",
        help=(
            "Show every setting and where its value came from. The source column says default,"
            " global, repo, preset, flag, or machine. An asterisk marks a value supplied by a"
            " config source instead of the built-in defaults."
        ),
    )
    show_keys = config_show.add_argument(
        "keys",
        nargs="*",
        metavar="KEY",
        help=(
            "Setting or section names to show in full, such as sandbox.network or sandbox."
            " Default: all settings."
        ),
    )
    show_keys.completer = functools.partial(  # type: ignore[attr-defined]
        completers._complete_config_keys, settable=False, sections=True
    )
    config_show.add_argument(
        "--json", action="store_true", dest="as_json", help="Print JSON instead of a table."
    )
    config_show.add_argument(
        "--descriptions",
        action="store_true",
        help="Show what each setting controls below its value.",
    )
    _add_machine_file(config_show, "Apply the [config] section in FILE when reading settings.")
    config_fill = _common._sub(
        config_sub,
        "fill",
        help=(
            "Write every built-in default and global setting to the global config file. Keep"
            " preset definitions and the selected preset name, but do not copy repository"
            " settings or the preset's changes."
        ),
    )
    config_fill.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing global config file. Default: refuse.",
    )
    _common._sub(
        config_sub,
        "path",
        help="Show the paths agent6 uses for config, secrets, state, skills, and cache.",
    )
    _common._sub(
        config_sub,
        "presets",
        help=(
            "Show the available presets and the settings each one changes. Also show which"
            " preset is selected and where each preset came from."
        ),
    )
    config_get = _common._sub(
        config_sub, "get", help="Show one setting's current value and where it came from."
    )
    config_get_key = config_get.add_argument("key", help="Setting name, such as sandbox.network.")
    # `get` reads effective leaves; `[presets.*]` are stripped before validation, so not offered.
    config_get_key.completer = functools.partial(  # type: ignore[attr-defined]
        completers._complete_config_keys, settable=False
    )
    _add_machine_file(config_get, "Apply the [config] section in FILE when reading the setting.")
    for verb, blurb in (
        ("set", "Save a setting. Default: global config file."),
        (
            "unset",
            "Remove a saved setting. agent6 then uses a value from another config source or its"
            " built-in default. Default: remove it from the global config file.",
        ),
        ("add", "Add a value to a list setting. Default: global config file."),
        ("remove", "Remove a value from a list setting. Default: global config file."),
    ):
        p = _common._sub(config_sub, verb, help=blurb)
        key_arg = p.add_argument("key", help="Setting name, such as sandbox.network.")
        key_arg.completer = completers._complete_config_keys  # type: ignore[attr-defined]
        if verb != "unset":
            action = "save" if verb == "set" else verb
            val_arg = p.add_argument(
                "value",
                help=(
                    f"Value to {action}. TOML booleans, numbers, quoted strings, and lists keep"
                    " their types. Other text is treated as a string."
                ),
            )
            val_arg.completer = completers._complete_config_values  # type: ignore[attr-defined]
        p.add_argument(
            "--repo",
            action="store_true",
            help=_common.REPO_FLAG_HELP,
        )
        _add_machine_file(
            p,
            (
                "Save the change in FILE's [config] section. Settings that could access the"
                " host, plus providers, sandbox settings, presets, and MCP servers, are refused."
            ),
        )

    config_fix = _common._sub(
        config_sub,
        "fix",
        help=(
            "Remove settings that prevent the config from loading. By default, check the global"
            " and repository config files and print each removal. A machine file limits changes"
            " to its [config] section."
        ),
    )
    _add_machine_file(
        config_fix,
        "Remove invalid settings only from FILE's [config] section. Report problems in other"
        " config files without changing them.",
    )


def _add_connect_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `connect`, which adds a provider and its API key."""
    connect_p = _common._sub(
        sub,
        "connect",
        help=(
            "Set up a model provider. agent6 asks for connection and sign-in details, then saves"
            " the provider settings. Credentials, when needed, go in the private global secrets"
            " file."
        ),
    )
    connect_provider = connect_p.add_argument(
        "provider",
        nargs="?",
        default="",
        help=("Provider to set up, such as anthropic or openrouter. Default: ask for a provider."),
    )
    connect_provider.completer = completers._complete_providers  # type: ignore[attr-defined]
    connect_p.add_argument(
        "--logout",
        action="store_true",
        help=(
            "Remove the provider's saved credentials instead. agent6 tries to revoke a ChatGPT"
            " sign-in. The provider settings stay in the config file."
        ),
    )
    connect_p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help=(
            "Do not test a newly saved API key against the provider. Use this for a local or"
            " offline provider that does not offer a model list."
        ),
    )
    connect_p.add_argument(
        "--repo",
        action="store_true",
        help=_common.REPO_FLAG_HELP,
    )


def _add_model_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `model`, which shows or assigns a role's model and effort."""
    model_p = _common._sub(
        sub,
        "model",
        help="Show or set the model and reasoning effort for planning, work, and review.",
    )
    # default=None: argparse validates a string default against choices. "all" sets every role.
    model_p.add_argument(
        "role",
        nargs="?",
        choices=("planner", "worker", "reviewer", "all"),
        default=None,
        metavar="role",
        help=(
            "Assignment to change: planner, worker, reviewer, or all. Choose all to update every"
            " assignment. Default: show the current assignments."
        ),
    )
    model_route = model_p.add_argument(
        "route",
        nargs="?",
        default="",
        metavar="[PROVIDER/]MODEL",
        help=(
            "Provider and model, written as PROVIDER/MODEL. MODEL alone keeps the assignment's"
            " current provider. On a terminal, give only a configured PROVIDER to choose from"
            " its models, or omit this value to choose both. Without an interactive terminal, a"
            " configured PROVIDER prints its known models."
        ),
    )
    # Role-gated, so the routes do not bleed into the first positional.
    model_route.completer = completers._complete_model_verb_values  # type: ignore[attr-defined]
    model_p.add_argument(
        "--effort",
        choices=get_args(EffortLevel),
        default="",
        help=(
            "Reasoning effort to save with this assignment. Default: none saved, so the"
            " provider's own default applies."
        ),
    )
    model_p.add_argument(
        "--repo",
        action="store_true",
        help=_common.REPO_FLAG_HELP,
    )
