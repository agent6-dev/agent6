"""Handle review commands in the ACP front end without model interpretation."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import uuid
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

from agent6 import review_records
from agent6.app import review_actions
from agent6.sessions import process
from agent6.ui.acp import rpc, session, updates
from agent6.viewmodel import listing, receipt, review, tail, transcript


class _CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise rpc.RpcError(rpc.INVALID_PARAMS, f"{self.prog}: {message}")


def _parse(command: str, text: str) -> argparse.Namespace:
    parser = _CommandParser(prog=command, add_help=False, allow_abbrev=False)
    if command in ("/receipt", "/review-queue"):
        return parser.parse_args(shlex.split(text)[1:])
    if command == "/findings":
        parser.add_argument("--all", action="store_true")
    else:
        parser.add_argument("finding_id")
        if command == "/fix":
            parser.add_argument("--resume", action="store_true")
        else:
            parser.add_argument("--reason", required=True)
            if command == "/resolve":
                parser.add_argument("--as", dest="disposition", required=True)
                parser.add_argument("--duplicate-of")
    return parser.parse_args(shlex.split(text)[1:])


def _findings_text(view: review.ReviewView, *, include_resolved: bool) -> str:
    lines: list[str] = []
    for finding in view.findings:
        if finding.status != "open" and not include_resolved:
            continue
        brief = review_actions.finding_brief(finding)
        body = (
            f"{brief.id} · {brief.status} · {brief.severity} · {brief.location}\n"
            f"{brief.title}\n{brief.detail}\nFix: {finding.fix_state}"
        )
        if brief.pre_existing:
            body += (
                f"\n  Pre-existing line: {brief.pre_existing}"
                f"\n  Blame: {', '.join(finding.line_origin.commits)}"
            )
        if brief.resolution:
            body += f"\n{brief.resolution}"
        lines.append(body)
    return "\n\n".join(lines) or "No recorded findings."


def _read_only_reply(
    command: str, sessions: session.Sessions, current: session.Session, args: argparse.Namespace
) -> str | None:
    """Return the text a read-only command answers with.

    Args:
        command: The slash command.
        sessions: The session manager.
        current: The current conversation.
        args: The command's parsed arguments.

    Returns:
        The reply, or None for a command that acts on a finding.

    Raises:
        RpcError: A per-run command arrived in a conversation with no recorded run.
    """
    if command == "/review-queue":
        return review_queue_text(current.cwd)
    if command not in {"/receipt", "/findings"}:
        return None
    if not current.session_id:
        raise rpc.RpcError(rpc.INVALID_PARAMS, "this conversation has no recorded run")
    saved = current.layout(sessions.state_dir_for(current.cwd))
    if command == "/receipt":
        return receipt.render(receipt.read_receipt(saved.session_dir, current.cwd))
    return _findings_text(review_actions.read_review(saved.session_dir), include_resolved=args.all)


def review_queue_text(cwd: pathlib.Path) -> str:
    """Return the repository's sessions in review-queue order as text.

    Args:
        cwd: The repository.

    Returns:
        One line per session under its group heading, a blocked, unverified or awaiting-merge
        row with the CLI command that acts on it; a note when nothing is recorded.
    """
    rows, _winners = listing.rows_for(cwd, sort="review")
    if not rows:
        return "No sessions recorded for this repository."
    lines: list[str] = []
    group = ""
    for row in rows:
        own = listing.row_group(row)
        if own != group:
            group = own
            lines.append(f"{group.replace('_', ' ').capitalize()}:")
        lines.extend(_queue_lines(row, depth=1))
    return "\n".join(lines)


def _queue_lines(row: listing.ListingRow, *, depth: int) -> list[str]:
    """Return a row's line and its lanes', indented, a pending row naming what to run."""
    s = row.summary
    entry = s.review_queue
    commands = {
        "needs_answer": f"agent6 answer {s.session_id}",
        "blocked": f"agent6 sessions findings {s.session_id}",
        "unverified": f"agent6 sessions receipt {s.session_id}",
        "awaiting_merge": f"agent6 sessions merge {s.session_id}",
    }
    words = " · ".join(p for p in (s.status, entry.reason) if p)
    command = commands.get(entry.group, "")
    line = f"{'  ' * depth}{s.session_id}: {words}" + (f" ({command})" if command else "")
    out = [line]
    for lane in row.lanes:
        out.extend(_queue_lines(lane, depth=depth + 1))
    return out


def _recorded_updates(
    logs: pathlib.Path, before: int | None, request_id: str, acp_id: str
) -> Iterator[dict[str, Any]]:
    fold = transcript.TranscriptFold()
    for event in tail.tail_events(logs, follow=False, start_at=before):
        if event.get("request_id") == request_id:
            for item in fold.feed(event):
                yield from updates.updates_for(item, acp_session_id=acp_id)


def handle_command(
    sessions: session.Sessions,
    current: session.Session,
    text: str,
    *,
    notify: Callable[[dict[str, Any]], None],
    finish: Callable[[str], None],
) -> bool:
    """Handle an advertised review command before the prompt reaches the worker.

    Args:
        sessions: The session manager.
        current: The current conversation.
        text: The prompt text.
        notify: The callback that sends session updates.
        finish: The callback that ends the prompt with a stop reason.

    Returns:
        True when this command was handled; False for an ordinary prompt.

    Raises:
        RpcError: The command is invalid or its action was refused.
    """
    command = text.split(maxsplit=1)[0]
    if command.removeprefix("/") not in {
        c["name"] for c in updates.REVIEW_COMMANDS
    } or not command.startswith("/"):
        return False
    try:
        args = _parse(command, text)
        body = _read_only_reply(command, sessions, current, args)
        if body is not None:
            notify(updates.message_update(current.acp_id, body))
            finish("end_turn")
            return True
        if not current.session_id:
            raise rpc.RpcError(rpc.INVALID_PARAMS, "this conversation has no recorded run")
        saved = current.layout(sessions.state_dir_for(current.cwd))
        view = review_actions.read_review(saved.session_dir)
        if command == "/fix":
            if current.is_running() or process.worker_is_alive(saved.session_dir):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS,
                    "Fix requires an idle session; wait for its prompt to finish",
                )
            if not args.resume:
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, "Resume to fix requires confirmation; use /fix ID --resume"
                )
            if not (saved.session_dir / "loop_state.json").is_file():
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, "this run has no resume point; Fix cannot start a new run"
                )
        resolution = (
            review_records.FindingResolution.model_validate(
                dict(
                    disposition=args.disposition,
                    reason=args.reason,
                    duplicate_of=args.duplicate_of,
                )
            )
            if command == "/resolve"
            else None
        )
        action = review_records.FindingAction.model_validate(
            dict(
                request_id=uuid.uuid4().hex,
                expected_version=view.version,
                action=command[1:],
                resolution=resolution,
                reason=args.reason if command == "/reopen" else None,
                resume=command == "/fix",
            )
        )
        before = tail.journal_size(saved.logs_path)
        live = current.is_running() or process.worker_is_alive(saved.session_dir)

        def resume(steer: str) -> str:
            sessions.start_execution(current, steer, finish=finish, require_resume=True)
            return ""

        result = review_actions.act(
            saved.session_dir,
            args.finding_id,
            action,
            source="acp",
            resume=resume,
            wait_s=0,
        )
        # A live execution's tail owns its notices; idle dispositions have no tail.
        if command != "/fix" and not live:
            for frame in _recorded_updates(
                saved.logs_path, before, action.request_id, current.acp_id
            ):
                notify(frame)
        elif command != "/fix":
            notify(
                updates.message_update(
                    current.acp_id, f"{result.message} (request {result.request_id})"
                )
            )
        if command != "/fix":
            finish("end_turn")
        return True
    except (review_actions.ReviewActionError, OSError, ValueError) as exc:
        raise rpc.RpcError(rpc.INVALID_PARAMS, str(exc)) from exc
