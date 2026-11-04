"""Render materialized session transcripts as Markdown or JSON."""

from __future__ import annotations

import json
import pathlib
import sys

from agent6 import paths
from agent6.sessions import id, objects, retention
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import common
from agent6.viewmodel import transcript_render


def parse_seq_window(spec: str) -> tuple[int, int] | None:
    """Parse a `--seq` window: "" for all, "5" for one call, "3-7" for a range.

    Args:
        spec: The `--seq` value.

    Returns:
        `(first, last)`, or None for all.

    Raises:
        ValueError: The spec is junk or reversed.
    """
    spec = spec.strip()
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            raise ValueError(f"reversed window {spec!r}")
        return lo, hi
    n = int(spec)
    return n, n


def transcript_layout(cwd: pathlib.Path, session_id: str) -> sessions_layout.SessionLayout | int:
    """Resolve the session whose transcripts to render.

    Args:
        cwd: The repo.
        session_id: The id, or "" for the newest session with transcripts or a prune receipt.

    Returns:
        The layout, or the exit code of a printed error.
    """
    if session_id:
        try:
            return common.resolve_session_layout(cwd, session_id)
        except id.SessionIdError as exc:
            common.error(f"{exc}")
            return 2
    found = common.newest_layout_holding(cwd, "transcripts", "pruned.json")
    if found is None:
        common.error(f"no sessions with transcripts under {paths.state_dir(cwd)}")
        return 2
    print(f"[agent6] transcript for most recent session: {found.session_id}", file=sys.stderr)
    return found


def cmd_transcript(
    session_id: str, *, as_json: bool, no_thinking: bool, tools: str, seq: str
) -> int:
    """Render a session's full LLM conversation from its lossless per-call transcripts.

    The transcripts (`<run>/transcripts/*.json`) are the complete record, needing no join
    with logs.jsonl: assistant text and thinking plus every tool call with its full I/O,
    and a marker for every attempt that did not complete.
    `agent6 attach` and `history search` are the terse event timeline.

    Args:
        session_id: The session, or "" for the newest with transcripts.
        as_json: Print materialized attempt records as JSON.
        no_thinking: Leave the thinking blocks out.
        tools: How much of each tool call to show.
        seq: The `--seq` call window.

    Returns:
        The exit code; 2 for a bad window or an unresolvable session.
    """
    layout = transcript_layout(pathlib.Path.cwd(), session_id)
    if isinstance(layout, int):
        return layout

    try:
        window = parse_seq_window(seq)
        pruned = retention.read_pruned(layout.session_dir)
        if pruned is not None:
            common.note(pruned.notice(layout.session_id))
        transcripts = transcript_render.load_transcripts(layout.transcripts_dir)
    except ValueError as exc:
        common.error(
            str(exc)
            if isinstance(exc, (objects.ObjectStoreError, retention.RetentionError))
            else f"--seq expects N or N-M with N <= M, got {seq!r}"
        )
        return 2
    if not transcripts:
        if pruned is None:
            common.error(f"session {layout.session_id} has no transcripts")
        return 2

    if as_json:
        if window is not None:
            lo, hi = window
            transcripts = [
                t for t in transcripts if lo <= transcript_render.transcript_seq(t) <= hi
            ]
        print(json.dumps(transcripts, indent=2, ensure_ascii=False))
        return 0

    # Fold the full set (the per-seq walk needs every call), then window the messages.
    messages = transcript_render.fold_conversation(transcripts)
    if not messages and not any(transcript_render.drives_conversation(t) for t in transcripts):
        # A review-only dir: every round-trip is a side-call seat, so the fold prints nothing.
        print(
            f"session {layout.session_id} has only side-call transcripts (review seats /"
            " compaction); --json dumps their materialized records.",
            file=sys.stderr,
        )
        return 2
    if window is not None:
        lo, hi = window
        messages = [t for t in messages if lo <= t.seq <= hi]
    print(
        transcript_render.render_markdown(
            messages, session_id=layout.session_id, show_thinking=not no_thinking, tools=tools
        ),
        end="",
    )
    return 0
