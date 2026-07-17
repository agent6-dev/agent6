# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for `config`/`connect`/`model`: inspect and edit the
layered config (global + repo + defaults + machine-file overlay), add a
provider + API key, and assign models to roles."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.ui.cli._common import _sub
from agent6.ui.cli.completers import (
    _complete_config_keys,
    _complete_config_values,
    _complete_machine_files,
    _complete_model_provider,
    _complete_models,
    _complete_providers,
)


def _add_config_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    config_p = _sub(
        sub,
        "config",
        help="Inspect and materialize the layered config (global + repo + defaults).",
    )
    config_sub = config_p.add_subparsers(
        dest="config_command", required=True, metavar="<subcommand>"
    )
    config_show = _sub(
        config_sub,
        "show",
        help=(
            "Print every effective config value and where it came from"
            " (default / global / repo / flag). `*` marks values that override"
            " the built-in default."
        ),
    )
    config_show.add_argument(
        "key",
        nargs="?",
        default="",
        help="Show just this leaf (or a section prefix, e.g. 'sandbox') UNTRUNCATED.",
    )
    config_show.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON instead of a table."
    )
    config_fill = _sub(
        config_sub,
        "fill",
        help=(
            "Write the fully-resolved config (every effective value, explicit)"
            " to a file: the global config by default, or the repo config with"
            " --repo. Handy before tightening defaults or for an audit snapshot."
        ),
    )
    config_fill.add_argument(
        "--repo",
        action="store_true",
        help="Write the per-repo config instead of the global config.",
    )
    config_fill.add_argument(
        "--force", action="store_true", help="Overwrite the target file if it already exists."
    )
    _sub(
        config_sub, "path", help="Print the resolved global + repo config (and secrets) file paths."
    )
    config_get = _sub(
        config_sub, "get", help="Print a leaf's effective value and which layer set it."
    )
    config_get_key = config_get.add_argument(
        "key", help="Dotted leaf path, e.g. sandbox.agent_network."
    )
    config_get_key.completer = _complete_config_keys  # type: ignore[attr-defined]
    config_get_machine = config_get.add_argument(
        "--machine-file",
        dest="machine_file",
        type=Path,
        default=None,
        metavar="FILE",
        help="View the value with a machine file's [config] overlay applied.",
    )
    config_get_machine.completer = _complete_machine_files  # type: ignore[attr-defined]
    for verb, blurb in (
        ("set", "Set a leaf to a scalar value (global by default)."),
        ("unset", "Remove a leaf, reverting it to the next-lower layer / default."),
        ("add", "Append a value to a list field (e.g. sandbox.allow_urls)."),
        ("remove", "Remove a value from a list field."),
    ):
        p = _sub(config_sub, verb, help=blurb)
        key_arg = p.add_argument("key", help="Dotted leaf path, e.g. sandbox.agent_network.")
        key_arg.completer = _complete_config_keys  # type: ignore[attr-defined]
        if verb != "unset":
            val_arg = p.add_argument("value", help="Value (TOML-typed; bare text is a string).")
            val_arg.completer = _complete_config_values  # type: ignore[attr-defined]
        p.add_argument(
            "--repo",
            action="store_true",
            help="Write the per-repo config instead of the global config.",
        )
        machine_arg = p.add_argument(
            "--machine-file",
            dest="machine_file",
            type=Path,
            default=None,
            metavar="FILE",
            help="Edit a machine file's [config] overlay (providers.* forbidden).",
        )
        machine_arg.completer = _complete_machine_files  # type: ignore[attr-defined]

    config_fix = _sub(
        config_sub,
        "fix",
        help=(
            "Drop invalid config entries (unknown keys, stale values) from the global"
            " and repo config, printing each and whether it was global or repo. Repairs"
            " a machine file's [config] overlay instead with --machine-file."
        ),
    )
    config_fix_machine = config_fix.add_argument(
        "--machine-file",
        dest="machine_file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Repair a machine file's [config] overlay instead of the global/repo config.",
    )
    config_fix_machine.completer = _complete_machine_files  # type: ignore[attr-defined]


def _add_connect_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    connect_p = _sub(
        sub,
        "connect",
        help="Interactively add a provider + API key (stored in the global secrets file).",
    )
    connect_provider = connect_p.add_argument(
        "--provider",
        default="",
        help="Provider name to add/update (e.g. anthropic, openrouter). Prompted if omitted.",
    )
    connect_provider.completer = _complete_providers  # type: ignore[attr-defined]
    connect_p.add_argument(
        "--repo",
        action="store_true",
        help="Write the [providers.*] block to the per-repo config instead of the global config.",
    )
    connect_p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the post-save read-only key check (a GET to the provider's /models)."
        " Use for offline/local endpoints (Ollama, llama.cpp) that have no models listing.",
    )


def _add_model_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model_p = _sub(
        sub,
        "model",
        help="Show or set which model + thinking level each role uses (planner/worker/reviewer).",
    )
    # choices gives both argparse validation and argcomplete tab-completion for
    # free. default=None (not "") so the omitted case isn't checked against
    # choices, argparse validates choices against a string default otherwise.
    # metavar="role" shows `role` in usage (not the noisy `{planner,...}`); the
    # choices stay listed in the help text. "all" is a pseudo-role (no config
    # field of that name) that sets every role at once, see _cmd_model.
    model_p.add_argument(
        "role",
        nargs="?",
        choices=("planner", "worker", "reviewer", "all"),
        default=None,
        metavar="role",
        help=(
            "Role to set: planner, worker, reviewer, or all (sets every role at"
            " once). Omit to print the current assignments."
        ),
    )
    model_provider = model_p.add_argument(
        "provider",
        nargs="?",
        default="",
        help="Provider name for the role (prompted from connected providers if omitted).",
    )
    # Role-gated (not _complete_providers) so the provider list doesn't bleed
    # into the first positional (role), see _complete_model_provider.
    model_provider.completer = _complete_model_provider  # type: ignore[attr-defined]
    model_model = model_p.add_argument(
        "model",
        nargs="?",
        default="",
        help="Model identifier for the role (prompted from the provider's catalog if omitted).",
    )
    model_model.completer = _complete_models  # type: ignore[attr-defined]
    model_p.add_argument(
        "--thinking",
        choices=("off", "low", "medium", "high"),
        default="",
        help="Reasoning/thinking effort for the role.",
    )
    model_p.add_argument(
        "--repo",
        action="store_true",
        help="Write to the per-repo config instead of the global config.",
    )
