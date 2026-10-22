# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `review`, `check` and `system` parsers."""

from __future__ import annotations

import argparse
from typing import get_args

from agent6.config import ReviewScope
from agent6.ui.cli import _common, completers


def _reviewer_count(raw: str) -> int:
    """Return the `--reviewers` count.

    Args:
        raw: The flag's value as typed.

    Returns:
        The count; 0 runs the single freeform review.

    Raises:
        ArgumentTypeError: The count is negative.
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("reviewer count must be non-negative")
    return value


def _add_check_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `check`, the read-only pre-flight checks."""
    check_p = _common._sub(
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
    _common._add_config_flag(check_p)


def _add_system_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `system <component> <action>`, the privileged host setup that uses sudo."""
    system_p = _common._sub(
        sub,
        "system",
        help="Host/OS setup that needs privileges, and uses sudo to get them."
        " Installs or removes the AppArmor profile that sandbox.isolation = 'full' needs.",
    )
    system_sub = system_p.add_subparsers(
        dest="system_command", required=True, metavar="<subcommand>"
    )
    apparmor_p = _common._sub(
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


def _add_review_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `review`, the read-only diff review."""
    review_p = _common._sub(
        sub,
        "review",
        help="Read-only code review of a diff (working tree, branch-vs-base, or arbitrary range).",
    )
    review_p.add_argument(
        "--base",
        default="",
        help=(
            "Base ref (review from its merge base with --head), or an explicit range such as"
            " main..HEAD for an endpoint diff. Default: uncommitted changes (working tree vs HEAD)."
        ),
    )
    review_p.add_argument(
        "--head",
        default="HEAD",
        help=(
            "Head ref (default: HEAD). Requires --base. With"
            " [review].scope = 'repo' it must be the checked-out commit, clean,"
            " which the seats read."
        ),
    )
    review_p.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Restrict the diff to PATH (repeatable; forwarded to `git diff -- PATH...`).",
    )
    review_model = review_p.add_argument(
        "--model",
        default="",
        metavar="[PROVIDER/]MODEL",
        help=(
            "The reviewer for this review, over every config layer: provider/model, or a"
            " model id on the reviewer's provider. A seat pinned in [review].reviewers keeps"
            " its own model."
        ),
    )
    review_model.completer = completers._complete_model_routes  # type: ignore[attr-defined]
    review_p.add_argument(
        "--reviewers",
        type=_reviewer_count,
        default=0,
        metavar="N",
        help=(
            "Run an adversarial review panel of N grounded reviewers instead of one"
            " freeform review. Findings are grounded against the diff (only real,"
            " block-eligible problems gate). 0 (default) = the single freeform review."
            " With [review].reviewers configured, N only turns the panel on; the"
            " configured list supplies the roster. Exits 0 PASS, 1 INCONCLUSIVE (every seat"
            " abstained), 2 refused before any seat ran, 3 budget, 4 BLOCK: the"
            " codes `agent6 run` uses."
        ),
    )
    review_p.add_argument(
        "--scope",
        choices=get_args(ReviewScope),
        default=None,
        help=(
            "Panel reading scope: diff (one call over the diff) or repo (a read-only tool loop"
            " over the checkout). Overrides [review].scope; ignored without --reviewers."
        ),
    )
    review_p.add_argument(
        "--focus",
        default="",
        help=(
            "The panel's focus values, comma-separated and cycled over --reviewers"
            " ('security,correctness,tests'). Pin models with [review].reviewers tables."
            " Default: a built-in set. Ignored without --reviewers, and when [review].reviewers is"
            " configured."
        ),
    )
