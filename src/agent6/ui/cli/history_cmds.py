# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` and the `sessions graph` and `sessions transcript` verbs."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from agent6.graph.storage import load_graph
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.sessions.layout import LOGS_NAME, SESSION_BUCKETS, SESSIONS_ROOT, SessionLayout
from agent6.ui.cli._common import (
    all_session_dirs,
    error,
    newest_layout_holding,
    resolve_session_layout,
    sgr,
)
from agent6.ui.cli._task_tree import task_tree_lines
from agent6.viewmodel.transcript_render import (
    conversation_transcripts,
    fold_conversation,
    load_transcripts,
    render_markdown,
    transcript_seq,
    window_turns,
)


def _cmd_history_search(query: str, *, fixed: bool, session_id: str) -> int:
    """Search a session's record, or every session's, with rg and print the hits.

    Args:
        query: The pattern.
        fixed: Match it literally.
        session_id: The session, or "" for every session in every bucket.

    Returns:
        The exit code: 0 with hits, 1 with none, 2 when the session cannot be resolved.
    """
    rg = shutil.which("rg")
    if rg is None:
        error(
            "`rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry."
        )
        return 2
    cwd = Path.cwd()
    if session_id:
        # Resolve across every bucket so an ask's logs/transcript are searchable.
        try:
            targets = [resolve_session_layout(cwd, session_id).session_dir]
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    else:
        # No id: every session in every bucket, so a search right after an `ask` finds it.
        targets = all_session_dirs(cwd)
        if not targets:
            print("[agent6] no sessions to search yet.")
            return 1
    argv: list[str] = [rg, "--json"]
    if fixed:
        argv.append("--fixed-strings")
    argv.extend(["--", query, *(str(t) for t in targets)])
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode not in (0, 1):  # 1 == no matches, not an error
        sys.stderr.write(completed.stderr)
        return completed.returncode
    hits = _parse_rg_matches(completed.stdout)
    _render_history_hits(hits, targets[0] if session_id else state_dir(cwd) / SESSIONS_ROOT)
    return 0 if hits else 1


@dataclass(frozen=True, slots=True)
class _SearchHit:
    """A search hit rendered readably: never the whole JSON event line.

    Attributes:
        session_id: The session the match file belongs to.
        when: The short clock time, or "" when the line is not a timestamped event.
        kind: The event type, or the file's basename for a non-event file.
        snippet: The window around the match.
        key: The content identity for collapsing the same text across storage encodings:
            the whole matched JSON value (a transcript's `TASK:` prefix stripped), else the
            snippet window, normalized.
    """

    session_id: str
    when: str
    kind: str
    snippet: str
    key: str


_SNIPPET_HALF = 70  # chars kept either side of the match in a hit's snippet


def _normalize(text: str) -> str:
    """Return the text decoded, lowercased and reduced to alphanumerics and spaces."""
    decoded = _collapse_escapes(text).lower()
    return "".join(ch for ch in decoded if ch.isalnum() or ch == " ").strip()


def _match_core(text: str, start: int, end: int) -> str:
    """Return a hit's content identity, including the whole matched JSON value.

    The same task is stored bare and behind a `TASK:` transcript prefix, which is storage
    syntax rather than content. A non-JSON line uses the window the operator sees, so
    distinct prose before a match stays distinct.

    Args:
        text: The matched line.
        start: The match's start offset.
        end: The match's end offset.
    """
    matched = _collapse_escapes(text[start:end])
    if matched.strip() and (event := _json_object(text)) is not None:
        for value in _strings_in(event):
            if matched not in value:
                continue
            core = value.strip()
            while core.casefold().startswith("task:"):
                core = core[5:].lstrip()
            return _normalize(core)
    lo = max(0, start - _SNIPPET_HALF)
    hi = min(len(text), end + _SNIPPET_HALF)
    return _normalize(text[lo:hi])


def _json_object(raw: str) -> dict[object, object] | None:
    """Decode a JSON object, or one line of a pretty-printed object as rg emits it.

    Returns:
        The object, or None when the line is neither.
    """
    for candidate in (raw, "{" + raw.rstrip().rstrip(",") + "}"):
        try:
            event = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict):
            return event
    return None


def _strings_in(obj: object) -> Iterator[str]:
    """Yield every string value nested anywhere in a decoded JSON object.

    Iterative: json.loads accepts nesting up to the interpreter's stack ceiling, so a
    recursive walk over what it returns could be the thing that blows up.

    Yields:
        Each string, in document order.
    """
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))


def _field_snippet(raw: str, start: int, end: int) -> str | None:
    """Window inside the string field holding the match when the line is a JSON object.

    The snippet then reads as prose instead of a raw `"type": "...", "text": " ...` fragment.

    Args:
        raw: The matched line.
        start: The match's start offset.
        end: The match's end offset.

    Returns:
        The window, or None when the line is not a JSON object or the match sits on syntax
        or keys; the caller falls back to the raw-line window.
    """
    event = _json_object(raw)
    if event is None:
        return None
    matched = _collapse_escapes(raw[start:end])
    if not matched.strip():
        return None
    for value in _strings_in(event):
        idx = value.find(matched)
        if idx >= 0:
            return _window(value, idx)
    return None


def _parse_rg_matches(rg_json: str) -> list[_SearchHit]:
    """Turn `rg --json` output into readable hits.

    A match line is parsed as a logs.jsonl event when it is one, for its type and
    timestamp; the snippet is a whitespace-collapsed window around the first match, inside
    the matched string field when the line is JSON, so a match buried in a huge tool or
    diff blob prints a short excerpt.

    Args:
        rg_json: The `rg --json` output.

    Returns:
        The hits, in rg's order.
    """
    hits: list[_SearchHit] = []
    for line in rg_json.splitlines():
        try:
            rec = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if rec.get("type") != "match":
            continue
        data = rec.get("data", {})
        path = Path(_rg_bytes(data.get("path")).decode("utf-8", "replace"))
        raw_bytes = _rg_bytes(data.get("lines")).rstrip(b"\n")
        raw = raw_bytes.decode("utf-8", "replace")
        subs = data.get("submatches") or []
        b_start = subs[0].get("start", 0) if subs else 0
        b_end = subs[0].get("end", b_start) if subs else b_start
        start, end = _char_span(raw_bytes, b_start, b_end)
        session_id = _session_id_from_path(path)
        when, kind = _event_when_kind(path, raw)
        snippet = _field_snippet(raw, start, end) or _window(raw, start)
        hits.append(
            _SearchHit(
                session_id=session_id,
                when=when,
                kind=kind,
                snippet=snippet,
                key=_match_core(raw, start, end) or snippet,
            )
        )
    return hits


def _kind_rank(hit: _SearchHit) -> int:
    """Return the readability order when the same content collapses across encodings.

    A timestamped event line beats the transcript record, which beats a rendered .md,
    which beats raw internals (manifest.json, per-call JSON).
    """
    if hit.when:
        return 0
    if hit.kind == "transcript":
        return 1
    if hit.kind.endswith(".md"):
        return 2
    return 3


def _rg_bytes(field: object) -> bytes:
    """Return an rg `{"text": ...}` or `{"bytes": ...}` field as bytes, so rg's offsets apply."""
    if isinstance(field, dict):
        if "bytes" in field:
            return base64.b64decode(str(field["bytes"]))
        return str(field.get("text", "")).encode("utf-8")
    return str(field or "").encode("utf-8")


def _char_span(line: bytes, b_start: int, b_end: int) -> tuple[int, int]:
    """Return rg's byte offsets as character offsets into the line decoded with U+FFFD.

    The prefix decodes to the same characters the whole line does, so non-ASCII prose and a
    stray byte before the match shift nothing.

    Args:
        line: The matched line.
        b_start: The match's start byte.
        b_end: The match's end byte.
    """
    start = len(line[:b_start].decode("utf-8", "replace"))
    return start, start + len(line[b_start:b_end].decode("utf-8", "replace"))


def _session_id_from_path(path: Path) -> str:
    """Return the session id owning a match file: `<sessions>/<bucket>/<id>`."""
    parts = path.parts
    anchors = set(SESSION_BUCKETS)
    for i in range(len(parts) - 3, -1, -1):
        if parts[i] == SESSIONS_ROOT and parts[i + 1] in anchors:
            return parts[i + 2]
    return path.parent.name


def _event_when_kind(path: Path, raw: str) -> tuple[str, str]:
    """Return `(clock time, event type)` when the matched line is a logs.jsonl event.

    Otherwise `("", label)`: a transcript snapshot, plan.md and the like. The snapshots are
    cumulative, so they share one "transcript" label to collapse repeated text.

    Args:
        path: The match file.
        raw: The matched line.
    """
    if path.name == LOGS_NAME:
        try:
            event = json.loads(raw)
        except (ValueError, RecursionError):
            return "", path.name
        if not isinstance(event, dict):
            return "", path.name
        ts = str(event.get("ts", ""))
        return ts[11:19] if len(ts) >= 19 else "", str(event.get("type", "event"))
    if path.parent.name == "transcripts":
        return "", "transcript"
    return "", path.name


def _collapse_escapes(s: str) -> str:
    r"""Render a JSON-encoded fragment readably.

    Scans left to right, so an escaped backslash is never mistaken for the start of `\n`.
    The whitespace escapes become spaces, `\\`, `\"` and `\/` decode to their character,
    and `\uXXXX` decodes with surrogate pairs combined: transcripts are written
    ascii-escaped while logs.jsonl is raw UTF-8, and the identity key must see one form.
    An unknown, clipped or lone-surrogate escape keeps its literal text.

    Args:
        s: The fragment.

    Returns:
        The rendered text.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in "ntr":
                out.append(" ")
                i += 2
                continue
            if nxt in '\\"/':
                out.append(nxt)
                i += 2
                continue
            if nxt == "u":
                decoded = _decode_u_escape(s, i)
                if decoded is not None:
                    ch, consumed = decoded
                    out.append(ch)
                    i += consumed
                    continue
        out.append(c)
        i += 1
    return "".join(out)


def _hex4(s: str, i: int) -> int | None:
    """Return `int(s[i:i+4], 16)`, or None when truncated or not hex."""
    if i + 4 > len(s):
        return None
    try:
        return int(s[i : i + 4], 16)
    except ValueError:
        return None


def _decode_u_escape(s: str, i: int) -> tuple[str, int] | None:
    r"""Decode the `\uXXXX` escape at `s[i]`, combining a surrogate pair.

    Returns:
        The character and the index after the escape, or None (the literal text is kept)
        for malformed hex, a truncation or a lone surrogate.
    """
    cp = _hex4(s, i + 2)
    if cp is None or 0xDC00 <= cp <= 0xDFFF:
        return None  # malformed/truncated, or a lone low surrogate
    if 0xD800 <= cp <= 0xDBFF:
        lo = _hex4(s, i + 8) if s[i + 6 : i + 8] == "\\u" else None
        if lo is None or not 0xDC00 <= lo <= 0xDFFF:
            return None  # a high surrogate without its pair
        return chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)), 12
    return chr(cp), 6


def _window(text: str, start: int) -> str:
    """Return a cleaned excerpt around an offset, with ellipses where it was clipped.

    JSON escapes inside transcript strings are decoded and whitespace collapses to single
    spaces, so the snippet reads as one line.

    Args:
        text: The matched line.
        start: The match's start offset.
    """
    lo = max(0, start - _SNIPPET_HALF)
    hi = min(len(text), start + _SNIPPET_HALF)
    excerpt = " ".join(_collapse_escapes(text[lo:hi]).split())
    return f"{'…' if lo > 0 else ''}{excerpt}{'…' if hi < len(text) else ''}"


def _render_history_hits(hits: list[_SearchHit], target: Path) -> None:
    """Print the hits grouped by session, a faded header once, then one line per hit.

    Identical snippets within a session (the same system-prompt boilerplate matched in
    every transcript) collapse to one line with an `(xN)` count.
    """
    if not hits:
        print(f"[agent6] no matches under {target}.")
        return
    grouped: dict[str, list[_SearchHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.session_id, []).append(hit)
    total = 0
    for i, (session_id, run_hits) in enumerate(grouped.items()):
        print("" if i == 0 else "\n", end="")
        print(sgr(session_id, "1"))
        # Dedup by content identity: one task string lives in many storage encodings.
        counts: dict[str, int] = {}
        best: dict[str, _SearchHit] = {}
        for hit in run_hits:
            counts[hit.key] = counts.get(hit.key, 0) + 1
            cur = best.get(hit.key)
            if cur is None or _kind_rank(hit) < _kind_rank(cur):
                best[hit.key] = hit
        for key, hit in best.items():
            n = counts[key]
            # A collapsed group spans several times, so it shows a count, not a timestamp.
            meta = "  ".join(p for p in (hit.when, hit.kind) if p) if n == 1 else hit.kind
            tag = f" {sgr(f'(x{n})', '2')}" if n > 1 else ""
            print(f"  {sgr(meta, '2')}  {hit.snippet}{tag}")
        total += len(run_hits)
    print(
        sgr(
            f"\n{total} matching line{'s' if total != 1 else ''} in {len(grouped)} "
            f"session{'s' if len(grouped) != 1 else ''}",
            "2",
        )
    )


def _cmd_history_graph(session_id: str) -> int:
    """Print a session's persisted task tree as a DFS-ordered listing.

    Returns:
        The exit code; 2 when the session cannot be resolved.
    """
    cwd = Path.cwd()
    if session_id:
        # Resolve across runs/ + asks/ so an ask's graph is findable too.
        try:
            layout = resolve_session_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    else:
        found = newest_layout_holding(cwd, "graph")
        if found is None:
            error(f"no sessions with a graph under {state_dir(cwd)}")
            return 2
        layout = found
        print(
            f"[agent6] showing graph for most recent session: {layout.session_id}",
            file=sys.stderr,
        )

    target_id = layout.session_id
    nodes = load_graph(layout)
    if not nodes:
        error(f"session {target_id} has no persisted graph nodes")
        return 2

    print(f"Session id: {target_id}")
    print()
    for line in task_tree_lines({nid: nodes[nid].model_dump() for nid in sorted(nodes)}):
        print(line)
    return 0


def _parse_seq_window(spec: str) -> tuple[int, int] | None:
    """Parse a `--seq` window: "" for all, "5" for one call, "3-7" for a range.

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


def _transcript_layout(cwd: Path, session_id: str) -> SessionLayout | int:
    """Resolve the session whose transcripts to render.

    By id, else the most recent session that has a transcripts dir.

    Returns:
        The layout, or the exit code of a printed error.
    """
    if session_id:
        try:
            return resolve_session_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
    found = newest_layout_holding(cwd, "transcripts")
    if found is None:
        error(f"no sessions with transcripts under {state_dir(cwd)}")
        return 2
    print(f"[agent6] transcript for most recent session: {found.session_id}", file=sys.stderr)
    return found


def _cmd_history_transcript(
    session_id: str, *, as_json: bool, no_thinking: bool, tools: str, seq: str
) -> int:
    """Render a session's full LLM conversation from its lossless per-call transcripts.

    The transcripts (`<run>/transcripts/*.json`) are the complete record, needing no join
    with logs.jsonl: assistant text and thinking plus every tool call with its full I/O.
    `agent6 attach` and `history search` are the terse event timeline.

    Args:
        session_id: The session, or "" for the newest with transcripts.
        as_json: Print the folded turns as JSON.
        no_thinking: Leave the thinking blocks out.
        tools: How much of each tool call to show.
        seq: The `--seq` call window.

    Returns:
        The exit code; 2 for a bad window or an unresolvable session.
    """
    layout = _transcript_layout(Path.cwd(), session_id)
    if isinstance(layout, int):
        return layout

    try:
        window = _parse_seq_window(seq)
    except ValueError:
        error(f"--seq expects N or N-M with N <= M, got {seq!r}")
        return 2

    transcripts = load_transcripts(layout.transcripts_dir)
    if not transcripts:
        error(f"session {layout.session_id} has no transcripts")
        return 2

    if as_json:
        if window is not None:
            lo, hi = window
            transcripts = [t for t in transcripts if lo <= transcript_seq(t) <= hi]
        print(json.dumps(transcripts, indent=2, ensure_ascii=False))
        return 0

    if not conversation_transcripts(transcripts):
        # A review-only dir: every round-trip is a side-call seat, so the fold would print nothing.
        print(
            f"session {layout.session_id} has only side-call transcripts (review seats /"
            " compaction); --json dumps them raw.",
            file=sys.stderr,
        )
        return 2

    # Fold the full set (the per-seq walk needs every call), then window the turns.
    turns = fold_conversation(transcripts)
    if window is not None:
        lo, hi = window
        turns = window_turns(turns, lo, hi)
    print(
        render_markdown(
            turns, session_id=layout.session_id, show_thinking=not no_thinking, tools=tools
        ),
        end="",
    )
    return 0
