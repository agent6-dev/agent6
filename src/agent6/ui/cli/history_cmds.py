# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` and the `sessions graph` verb."""

from __future__ import annotations

import base64
import bisect
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Sequence

from agent6 import paths
from agent6.graph import storage
from agent6.sessions import id
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import common, history_transcripts, task_tree


def _cmd_history_search(query: str, *, fixed: bool, session_id: str) -> int:
    """Search a session's record, or every session's, with rg and print the hits.

    Args:
        query: The pattern.
        fixed: Match it literally.
        session_id: The session, or "" for every session in every bucket.

    Returns:
        The exit code: 0 with hits, 1 with none, 2 on resolution or search errors.
    """
    rg = shutil.which("rg")
    if rg is None:
        common.error(
            "`rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry."
        )
        return 2
    cwd = pathlib.Path.cwd()
    if session_id:
        # Resolve across every bucket so an ask's logs/transcript are searchable.
        try:
            targets = [common.resolve_session_layout(cwd, session_id).session_dir]
        except id.SessionIdError as exc:
            common.error(f"{exc}")
            return 2
    else:
        # No id: every session in every bucket, so a search right after an `ask` finds it.
        targets = common.all_session_dirs(cwd)
        if not targets:
            print("[agent6] no sessions to search yet.")
            return 1
    argv: list[str] = [rg, "--json"]
    if fixed:
        argv.append("--fixed-strings")
    # The conversation is searched below through the filtered transcripts; the harness's own
    # copies of it (the snapshots, the checkpoints, the object store) would match the prompt too.
    excluded = (
        "!**/transcripts/*.json",
        "!**/checkpoints/*.json",
        "!**/objects/*.json",
        "!**/loop_state.json",
        "!**/turn_in_flight.json",
    )
    globs = [flag for pattern in excluded for flag in ("--glob", pattern)]
    completed = subprocess.run(
        [*argv, *globs, "--", query, *(str(t) for t in targets)],
        check=False,
        capture_output=True,
        text=True,
    )
    hits = _parse_rg_matches(completed.stdout)
    transcript_hits, errors = _search_transcripts([*argv, "--", query], targets)
    hits.extend(transcript_hits)
    incomplete = bool(errors) or completed.returncode not in (0, 1)
    if incomplete:
        common.error("search incomplete: some files were not read.")
        sys.stderr.write(completed.stderr)
        for error in errors:
            common.error(error)
    if hits or not incomplete:
        _render_history_hits(
            hits, targets[0] if session_id else paths.state_dir(cwd) / sessions_layout.SESSIONS_ROOT
        )
    return 2 if incomplete else 0 if hits else 1


@dataclasses.dataclass(frozen=True, slots=True)
class _SearchHit:
    """A search hit rendered readably: never the whole JSON event line.

    Attributes:
        session_id: The session the match file belongs to.
        when: The short clock time, or "" when the line is not a timestamped event.
        kind: The event type, or the file's basename for a non-event file.
        snippet: The window around the match.
        key: The hash of the normalized content identity across storage encodings.
    """

    session_id: str
    when: str
    kind: str
    snippet: str
    key: str


def _search_transcripts(
    argv: list[str], targets: list[pathlib.Path]
) -> tuple[list[_SearchHit], list[str]]:
    """Feed one transcript at a time while draining ripgrep's matches.

    Args:
        argv: The operator's ripgrep query.
        targets: The session directories.

    Returns:
        The hits and read errors, including a failed ripgrep's stderr.
    """
    sources: list[tuple[int, pathlib.Path]] = []
    errors: list[str] = []
    hits: list[_SearchHit] = []
    with (
        tempfile.TemporaryFile(mode="w+t") as stderr,
        subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True
        ) as process,
    ):
        assert process.stdin is not None and process.stdout is not None
        stdin = process.stdin

        def feed() -> None:
            first_line = 1
            try:
                for chunk in history_transcripts.search_text(targets):
                    if chunk.error:
                        errors.append(chunk.error)
                        continue
                    sources.append((first_line, chunk.path))
                    first_line += chunk.text.count("\n")
                    stdin.write(chunk.text)
            except OSError as exc:
                errors.append(f"transcript search input: {exc}")
            finally:
                with contextlib.suppress(OSError):
                    stdin.close()

        writer = threading.Thread(target=feed, name="transcript-search")
        writer.start()
        try:
            for line in process.stdout:
                hits.extend(_parse_rg_matches(line, sources=sources))
            process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
            writer.join()
        if process.wait() not in (0, 1):
            stderr.seek(0)
            errors.append(stderr.read() or "transcript search failed")
    return hits, errors


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

    Returns:
        The identity, normalized; "" when no letter or digit survives.
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

    Args:
        raw: The matched line.

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

    Args:
        obj: The value json.loads returned.

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


def _parse_rg_matches(
    rg_json: str, *, sources: Sequence[tuple[int, pathlib.Path]] = ()
) -> list[_SearchHit]:
    """Turn `rg --json` output into readable hits.

    A match line is parsed as a logs.jsonl event when it is one, for its type and
    timestamp; the snippet is a whitespace-collapsed window around the first match, inside
    the matched string field when the line is JSON, so a match buried in a huge tool or
    diff blob prints a short excerpt.

    Args:
        rg_json: The `rg --json` output.
        sources: Each transcript's first stdin line number and source path.

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
        path = (
            sources[bisect.bisect_right(sources, data["line_number"], key=lambda s: s[0]) - 1][1]
            if sources
            else pathlib.Path(_rg_bytes(data.get("path")).decode("utf-8", "replace"))
        )
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
                key=hashlib.sha256((_match_core(raw, start, end) or snippet).encode()).hexdigest(),
            )
        )
    return hits


def _kind_rank(hit: _SearchHit) -> int:
    """Return the readability order when the same content collapses across encodings.

    A timestamped event line beats the transcript record, which beats a rendered .md,
    which beats raw internals (manifest.json, per-call JSON).

    Args:
        hit: The hit to rank.

    Returns:
        The rank, from 0 (most readable) to 3.
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

    Returns:
        The match's start and end as character offsets.
    """
    start = len(line[:b_start].decode("utf-8", "replace"))
    return start, start + len(line[b_start:b_end].decode("utf-8", "replace"))


def _session_id_from_path(path: pathlib.Path) -> str:
    """Return the session id owning a match file: `<sessions>/<bucket>/<id>`."""
    parts = path.parts
    anchors = set(sessions_layout.SESSION_BUCKETS)
    for i in range(len(parts) - 3, -1, -1):
        if parts[i] == sessions_layout.SESSIONS_ROOT and parts[i + 1] in anchors:
            return parts[i + 2]
    return path.parent.name


def _event_when_kind(path: pathlib.Path, raw: str) -> tuple[str, str]:
    """Return `(clock time, event type)` when the matched line is a logs.jsonl event.

    Transcript snapshots are cumulative, so they share one label to collapse repeated text.

    Args:
        path: The match file.
        raw: The matched line.

    Returns:
        The event's `HH:MM:SS` time and type; otherwise "" and the file's name, or
        "transcript" for a transcript snapshot.
    """
    if path.name == sessions_layout.LOGS_NAME:
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

    Args:
        s: The fragment.
        i: The offset of the escape's backslash.

    Returns:
        The character and the length of the text it replaces, or None (the literal text is kept)
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
        text: The matched line, or the string field holding the match.
        start: The match's start offset.

    Returns:
        The excerpt of up to `_SNIPPET_HALF` characters either side of the offset.
    """
    lo = max(0, start - _SNIPPET_HALF)
    hi = min(len(text), start + _SNIPPET_HALF)
    excerpt = " ".join(_collapse_escapes(text[lo:hi]).split())
    return f"{'…' if lo > 0 else ''}{excerpt}{'…' if hi < len(text) else ''}"


def _render_history_hits(hits: list[_SearchHit], target: pathlib.Path) -> None:
    """Print the hits grouped by session, a faded header once, then one line per hit.

    Identical snippets within a session's cumulative transcripts collapse to one line
    with an `(xN)` count.

    Args:
        hits: The hits, in rg's order.
        target: The searched directory, named when nothing matched.
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
        print(common.sgr(session_id, "1"))
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
            tag = f" {common.sgr(f'(x{n})', '2')}" if n > 1 else ""
            print(f"  {common.sgr(meta, '2')}  {hit.snippet}{tag}")
        total += len(run_hits)
    print(
        common.sgr(
            f"\n{total} matching line{'s' if total != 1 else ''} in {len(grouped)} "
            f"session{'s' if len(grouped) != 1 else ''}",
            "2",
        )
    )


def _cmd_history_graph(session_id: str) -> int:
    """Print a session's persisted task tree as a DFS-ordered listing.

    Args:
        session_id: The session, or "" for the newest with a graph.

    Returns:
        The exit code; 2 when the session cannot be resolved or has no graph.
    """
    cwd = pathlib.Path.cwd()
    if session_id:
        # Resolve across runs/ + asks/ so an ask's graph is findable too.
        try:
            layout = common.resolve_session_layout(cwd, session_id)
        except id.SessionIdError as exc:
            common.error(f"{exc}")
            return 2
    else:
        found = common.newest_layout_holding(cwd, "graph")
        if found is None:
            common.error(f"no sessions with a graph under {paths.state_dir(cwd)}")
            return 2
        layout = found
        print(
            f"[agent6] showing graph for most recent session: {layout.session_id}",
            file=sys.stderr,
        )

    target_id = layout.session_id
    nodes = storage.load_graph(layout)
    if not nodes:
        common.error(f"session {target_id} has no persisted graph nodes")
        return 2

    print(f"Session id: {target_id}")
    print()
    for line in task_tree.task_tree_lines({nid: nodes[nid].model_dump() for nid in sorted(nodes)}):
        print(line)
    return 0
