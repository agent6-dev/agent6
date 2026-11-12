# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `sessions` parser and its subcommands."""

from __future__ import annotations

import argparse

from agent6.ui.cli import common, completers, sessions_findings


def _add_sessions_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `sessions`; a bare `sessions` is `sessions list`."""
    sessions_p = common._sub(
        sub,
        "sessions",
        help=(
            "List sessions for this repository with `agent6 sessions` or `agent6 sessions"
            " list`. Other subcommands inspect or manage saved work. Most subcommands take a"
            " session id or unambiguous prefix and default to the newest matching session."
            " `sessions dir` without an id prints the history directory. Use `agent6 attach`"
            " to follow a live session."
        ),
    )
    sessions_sub = sessions_p.add_subparsers(
        dest="sessions_command", required=True, metavar="<subcommand>"
    )

    sessions_findings.add_parsers(sessions_sub)
    receipt = common._sub(
        sessions_sub,
        "receipt",
        help="Inspect recorded testing, review and reversal evidence without running tests.",
    )
    common._add_session_id(receipt, completers._complete_session_ids)
    receipt.add_argument(
        "--json", action="store_true", help="Print the structured evidence receipt."
    )

    sessions_list = common._sub(
        sessions_sub,
        "list",
        help=(
            "List sessions by most recent activity, with update time, status, usage, id, and task."
        ),
    )
    sessions_list.add_argument(
        "--json",
        dest="list_json",
        action="store_true",
        help=(
            "Print a JSON array. Each row has session_id, mode, task (full text), continued_task,"
            " task_line, status, detail,"
            " stop_reason, label, level, last_activity, group_last_activity, when,"
            " cost_usd, usd_partial, plan_observed_points,"
            " plan_cap, usage, model, model_from_flag, id_cell, unmerged, verification,"
            " winner, lane,"
            " coordinator, worktree, review_queue, queue_group, lane_groups, and nested lanes."
        ),
    )
    sessions_list.add_argument(
        "--sort",
        dest="list_sort",
        choices=("activity", "review"),
        default="activity",
        help=(
            "Order the rows by most recent activity (the default), or by the review queue:"
            " questions nobody answered, then open block findings, then ended runs the gate"
            " did not certify, then clean runs waiting for a merge, then the rest; a review"
            " column names each row's place."
        ),
    )
    sessions_list.add_argument(
        "--lanes",
        dest="list_lanes",
        action="store_true",
        help=(
            "List each fan-out's lanes under its row. Default: show only the count. JSON always"
            " includes the nested lanes."
        ),
    )

    sessions_show = common._sub(
        sessions_sub,
        "show",
        help="Print a session's current status and progress once. `agent6 attach` follows it live.",
    )
    common._add_session_id(sessions_show, completers._complete_session_ids)
    sessions_show.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print one JSON object with task (full text), continued_task, task_line (headline),"
            " cost_usd and the session's status and progress."
        ),
    )

    sessions_diff = common._sub(
        sessions_sub,
        "diff",
        help="Print the committed changes a session made.",
    )
    common._add_session_id(sessions_diff, completers._complete_session_ids)
    sessions_diff.add_argument(
        "--stat",
        action="store_true",
        help="Print git's changed-file summary instead of the full patch.",
    )
    sessions_diff.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        metavar="PATH",
        help="Only include PATH. Repeat for more paths.",
    )

    sessions_merge = common._sub(
        sessions_sub,
        "merge",
        help=(
            "Merge a session's committed work into a branch. Default target: the branch where"
            " the session started. A pruned run reports its saved merge."
        ),
    )
    common._add_session_id(sessions_merge, completers._complete_session_ids)
    sessions_merge.add_argument(
        "--method",
        choices=("tasks", "squash", "merge", "ff_only"),
        default=None,
        help="Use this method. Default: git.merge_method from the active config.",
    )
    sessions_merge.add_argument(
        "--into",
        default=None,
        metavar="BRANCH",
        help="Merge into BRANCH. Default: the branch where the session started.",
    )
    sessions_merge.add_argument(
        "--message",
        "-m",
        default=None,
        help=(
            "Use this commit message for a squash or merge. Default: a squash follows"
            " git.commit.squash.message; a merge uses `Merge <run branch>`."
        ),
    )

    sessions_review = common._sub(
        sessions_sub,
        "review",
        help=(
            "Print a read-only review of a finished session's record on the reviewer model:"
            " how it ended, what went wrong and why, what you corrected, candidate memory"
            " facts and AGENTS.md lines with their evidence, and the memory entries the record"
            " contradicts. Saved under the state dir's reviews/. Nothing is written to the repo"
            " or the memory."
        ),
    )
    common._add_session_id(sessions_review, completers._complete_session_ids)
    review_model = sessions_review.add_argument(
        "--model",
        default="",
        metavar="[PROVIDER/]MODEL",
        help=(
            "The reviewer for this review, over every config layer: provider/model, or a"
            " model id on the reviewer's provider."
        ),
    )
    review_model.completer = completers._complete_model_routes  # type: ignore[attr-defined]

    sessions_compare = common._sub(
        sessions_sub,
        "compare",
        help=(
            "Rank two or more finished sessions: the reviewer model judges their changes,"
            " and a missing or failed judge falls back to verification result and cost."
            " Session ids create a fresh report; one"
            " fan-out id prints its saved report. This is the report `run --parallel`"
            " prints when it ends. Nothing is merged."
        ),
    )
    sessions_compare.add_argument(
        "--rejudge",
        action="store_true",
        help=(
            "Rank again instead of printing a fan-out's saved result. With a reviewer"
            " model, this spends a new call. The result can differ from listings and is not"
            " saved."
        ),
    )
    sessions_compare_ids = sessions_compare.add_argument(
        "session_ids",
        nargs="+",
        metavar="SESSION_ID",
        help=(
            "Two or more session ids or unambiguous prefixes. Or give one fan-out id to"
            " compare its lanes."
        ),
    )
    sessions_compare_ids.completer = completers._complete_session_ids  # type: ignore[attr-defined]

    sessions_commits = common._sub(
        sessions_sub,
        "commits",
        help="List the commits a session made.",
    )
    common._add_session_id(sessions_commits, completers._complete_session_ids)

    sessions_dir = common._sub(
        sessions_sub,
        "dir",
        help=(
            "Print the session history directory for this repository. With an id, print that"
            " session's directory. Output is one path for use in scripts."
        ),
    )
    common._add_session_id(
        sessions_dir,
        completers._complete_session_ids,
        help_text=f"{common.SESSION_ID}. Default: this repository's session history directory.",
    )

    sessions_remove = common._sub(
        sessions_sub,
        "remove",
        help=(
            "Delete a session's saved history, internal commit reference, and worktree."
            " Its git branch, if present, remains. On a controlling terminal it shows the"
            " session and asks [y/N] first; --yes skips the question. With no controlling"
            " terminal, pass a session id or --yes to delete a session without asking."
        ),
    )
    common._add_session_id(sessions_remove, completers._complete_session_ids)
    sessions_remove.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Delete without asking.",
    )
    sessions_remove.add_argument(
        "--asks",
        action="store_true",
        help=(
            "Delete every ask saved in this repository (outside one, this directory) instead of"
            " one session. Cannot be used with a session id. Asks of other repositories remain."
        ),
    )

    _add_prune_parser(sessions_sub)
    sessions_tr = common._sub(
        sessions_sub,
        "transcript",
        help="Print a session's complete model conversation as Markdown.",
    )
    common._add_session_id(
        sessions_tr,
        completers._complete_session_ids,
        help_text=(
            "Session id or unambiguous prefix; omit for the newest session"
            " with a transcripts directory."
        ),
    )
    sessions_tr.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print the saved request and response objects as a JSON array instead.",
    )
    sessions_tr.add_argument(
        "--no-thinking", action="store_true", help="Omit the model's reasoning blocks."
    )
    sessions_tr.add_argument(
        "--tools",
        choices=("both", "calls", "none"),
        default="both",
        help="Tool details: calls and results (`both`, default), calls only, or none.",
    )
    sessions_tr.add_argument(
        "--seq",
        default="",
        help="Only include round trip N or inclusive range N-M, such as 3 or 3-7. Default: all.",
    )

    sessions_graph = common._sub(
        sessions_sub,
        "graph",
        help="Print a session's saved task graph as a tree.",
    )
    common._add_session_id(
        sessions_graph,
        completers._complete_session_ids,
        help_text=(
            "Session id or unambiguous prefix; omit for the newest session with a graph directory."
        ),
    )


def _add_prune_parser(sessions_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `sessions prune`: git cleanup by default, session history with `--history`."""
    sessions_prune = common._sub(
        sessions_sub,
        "prune",
        help=(
            "Clean up merged session git data. Delete safe-to-remove agent6/* branches,"
            " internal commit references whose recorded targets contain their work, parallel"
            " clones whose commits are present, and the worktrees of sessions that are merged"
            " or hold no work. Keep unstarted forks. Report kept clones with their paths and"
            " reasons. With --history --older-than DAYS, prune session history instead."
        ),
    )
    sessions_prune.add_argument(
        "--history",
        action="store_true",
        help=(
            "Prune session history, not git data: with --older-than DAYS, delete the"
            " transcripts, checkpoints and objects of finished sessions whose last execution"
            " ended more than DAYS ago. Keep recovery state, journals, receipts and git data."
            " Nothing is deleted automatically."
        ),
    )
    sessions_prune.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="The age, in days, --history prunes beyond; nonnegative.",
    )
    sessions_prune.add_argument(
        "--delete-squashed",
        action="store_true",
        help=(
            "Also force-delete branches and internal commit references whose session records"
            " prove a squash merge remains on the recorded target. git's safe deletion rejects"
            " these. Each deletion prints a recovery command."
        ),
    )
