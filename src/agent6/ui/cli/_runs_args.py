# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builder for `runs` and its subcommands: list this repo's runs, or
inspect one (show/diff/merge/compare/commits/stop/prune/transcript/graph)."""

from __future__ import annotations

import argparse

from agent6.ui.cli._common import _sub
from agent6.ui.cli.completers import _complete_run_ids


def _add_runs_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    runs_p = _sub(
        sub,
        "runs",
        help=(
            "List this repo's runs (`agent6 runs`, or `runs list`) or inspect one:"
            " show (liveness/progress), diff, compare, transcript, graph. The run id"
            " is a positional everywhere (exact or unambiguous prefix; omit for the"
            " most recent). To follow a run live, use `agent6 attach`."
        ),
    )
    # No subcommand = list: "show me my runs" is the obvious bare meaning.
    runs_sub = runs_p.add_subparsers(dest="runs_command", required=False, metavar="<subcommand>")

    _sub(
        runs_sub,
        "list",
        help="List runs newest-first by update time: updated, status, mode, cost, id, task.",
    )

    runs_show = _sub(
        runs_sub,
        "show",
        help="One-shot liveness + progress of a run, then exit (`agent6 attach` follows live).",
    )
    runs_show_id = runs_show.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (omit for the most recent run).",
    )
    runs_show_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_show.add_argument(
        "--json",
        action="store_true",
        help="Emit the status as a single JSON object (for scripts/monitoring).",
    )

    runs_diff = _sub(
        runs_sub,
        "diff",
        help="Print the git diff produced by a run (manifest.base_sha -> HEAD of run branch).",
    )
    runs_diff_id = runs_diff.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unique prefix). Omit to diff the most recent run.",
    )
    runs_diff_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_diff.add_argument(
        "--stat",
        action="store_true",
        help="Show --stat summary instead of the full patch.",
    )
    runs_diff.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Restrict the diff to these paths.",
    )

    runs_merge = _sub(
        runs_sub,
        "merge",
        help="Merge a run's branch into a target (default: the branch it was cut from).",
    )
    runs_merge_id = runs_merge.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unique prefix). Omit to merge the most recent run.",
    )
    runs_merge_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_merge.add_argument(
        "--strategy",
        choices=("squash", "merge", "ff"),
        default=None,
        help="Override git.merge_strategy for this merge.",
    )
    runs_merge.add_argument(
        "--into",
        default=None,
        metavar="BRANCH",
        help="Target branch to merge into (default: the run's base branch).",
    )
    runs_merge.add_argument(
        "--message",
        "-m",
        default=None,
        help="Commit message for squash or merge (default: a condensed run summary).",
    )

    runs_compare = _sub(
        runs_sub,
        "compare",
        help=(
            "Advisory ranked comparison across >=2 runs (verify+cost, judged by the"
            " reviewer model when configured): the report `--parallel`'s auto-compare"
            " prints. Never merges."
        ),
    )
    runs_compare_ids = runs_compare.add_argument(
        "run_ids",
        nargs="+",
        metavar="RUN_ID",
        help="2 or more run ids (or unique prefixes) to compare.",
    )
    runs_compare_ids.completer = _complete_run_ids  # type: ignore[attr-defined]

    runs_commits = _sub(
        runs_sub,
        "commits",
        help="List the per-step commits on a run's branch.",
    )
    runs_commits_id = runs_commits.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unique prefix). Omit for the most recent run.",
    )
    runs_commits_id.completer = _complete_run_ids  # type: ignore[attr-defined]

    runs_stop = _sub(
        runs_sub,
        "stop",
        help="Ask a running detached run to stop cleanly after its current step (resumable).",
    )
    runs_stop.add_argument(
        "run_id", nargs="?", default="", help="Run id or unique prefix; omit for the most recent."
    )

    runs_prune = _sub(
        runs_sub,
        "prune",
        help="Delete agent6/* run branches that are safely merged; report the rest.",
    )
    runs_prune.add_argument(
        "--delete-squashed",
        action="store_true",
        help=(
            "Also force-delete run branches confirmed squash-merged into their base"
            " (git branch -d refuses these; the content is safe in the base commit)."
            " Each deletion prints an undelete command."
        ),
    )

    runs_tr = _sub(
        runs_sub,
        "transcript",
        help="Render a run's full LLM conversation (the lossless transcripts) as Markdown.",
    )
    runs_tr_id = runs_tr.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unambiguous prefix). Defaults to the most recent run.",
    )
    runs_tr_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_tr.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the raw transcript array (the per-call request/response objects) instead.",
    )
    runs_tr.add_argument(
        "--no-thinking", action="store_true", help="Omit the model's reasoning/thinking blocks."
    )
    runs_tr.add_argument(
        "--tools",
        choices=("both", "calls", "none"),
        default="both",
        help="Show tool calls + results (both), calls only, or neither.",
    )
    runs_tr.add_argument(
        "--seq",
        default="",
        help="Restrict to a round-trip seq window, e.g. 3 or 3-7 (default: all).",
    )

    runs_graph = _sub(
        runs_sub,
        "graph",
        help="Render the persisted task graph for a run as a DFS tree.",
    )
    runs_graph_id = runs_graph.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unambiguous prefix). Defaults to the most recent run.",
    )
    runs_graph_id.completer = _complete_run_ids  # type: ignore[attr-defined]
