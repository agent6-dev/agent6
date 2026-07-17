# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search/graph/transcript` commands."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.graph.storage import load_graph
from agent6.runs.id import RunIdError
from agent6.runs.layout import RunLayout
from agent6.ui.cli._common import (
    _runs_dir,
    _state_dir,
    all_run_dirs,
    resolve_run_layout,
    sgr,
)
from agent6.ui.cli._task_tree import task_tree_lines
from agent6.viewmodel import run_mtime
from agent6.viewmodel.transcript_render import (
    fold_conversation,
    load_transcripts,
    render_markdown,
)


def _cmd_history_search(query: str, *, fixed: bool, run_id: str) -> int:
    rg = shutil.which("rg")
    if rg is None:
        print(
            "ERROR: `rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry.",
            file=sys.stderr,
        )
        return 2
    cwd = Path.cwd()
    if run_id:
        # Resolve across every bucket so an ask's logs/transcript are searchable.
        try:
            targets = [resolve_run_layout(cwd, run_id).run_dir]
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        # No id: search every run across runs/ + asks/ + machine-drafts/, so a
        # search right after an `ask` finds it (matching what `agent6 runs` lists).
        targets = all_run_dirs(cwd)
        if not targets:
            print("[agent6] no runs to search yet.")
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
    _render_history_hits(hits, _state_dir(cwd))
    return 0 if hits else 1


# A search hit rendered readably: which run, when, the event type, and a snippet
# windowed around the match -- never the whole (possibly 400KB) JSON event line.
@dataclass(frozen=True, slots=True)
class _SearchHit:
    run_id: str
    when: str  # short clock time, or "" if the line is not a timestamped event
    kind: str  # event type, or the file's basename for non-event files
    snippet: str


_SNIPPET_HALF = 70  # chars kept either side of the match in a hit's snippet


def _parse_rg_matches(rg_json: str) -> list[_SearchHit]:
    """Turn `rg --json` output into readable hits. Each match line is parsed as a
    logs.jsonl event when it is one (for its type + timestamp), and the snippet is
    a whitespace-collapsed window around the first match, so a match buried in a
    huge tool/diff blob prints a short excerpt, not the entire line."""
    hits: list[_SearchHit] = []
    for line in rg_json.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") != "match":
            continue
        data = rec.get("data", {})
        path = Path(_rg_text(data.get("path")))
        raw = _rg_text(data.get("lines")).rstrip("\n")
        subs = data.get("submatches") or []
        start = subs[0].get("start", 0) if subs else 0
        run_id = _run_id_from_path(path)
        when, kind = _event_when_kind(path, raw)
        hits.append(_SearchHit(run_id=run_id, when=when, kind=kind, snippet=_window(raw, start)))
    return hits


def _rg_text(field: object) -> str:
    """rg --json encodes a path/line as {"text": ...} (or {"bytes": ...} for
    non-UTF8); return the text, empty for the bytes case."""
    if isinstance(field, dict):
        return str(field.get("text", ""))
    return str(field or "")


def _run_id_from_path(path: Path) -> str:
    """The run/ask id owning a match file: the child of a runs/ or asks/ dir."""
    parts = path.parts
    for anchor in ("runs", "asks", "machine-drafts"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return parts[i + 1]
    return path.parent.name


def _event_when_kind(path: Path, raw: str) -> tuple[str, str]:
    """(clock-time, event-type) when the matched line is a logs.jsonl event;
    otherwise ("", a short label) for a transcript snapshot, plan.md, etc. The
    transcript snapshots are cumulative, so they get one shared "transcript"
    label to collapse the same text repeated across snapshots."""
    if path.name == "logs.jsonl":
        try:
            event = json.loads(raw)
        except ValueError:
            return "", path.name
        ts = str(event.get("ts", ""))
        return ts[11:19] if len(ts) >= 19 else "", str(event.get("type", "event"))
    if path.parent.name == "transcripts":
        return "", "transcript"
    return "", path.name


def _window(text: str, start: int) -> str:
    """A cleaned excerpt of *text* around byte offset *start*, capped at
    ~2*_SNIPPET_HALF chars with leading/trailing ellipses when it was clipped.
    Literal JSON escapes (``\\n`` etc. inside transcript strings) and real
    whitespace both collapse to single spaces so the snippet reads as one line."""
    lo = max(0, start - _SNIPPET_HALF)
    hi = min(len(text), start + _SNIPPET_HALF)
    excerpt = text[lo:hi]
    for esc in ("\\n", "\\t", "\\r"):
        excerpt = excerpt.replace(esc, " ")
    excerpt = " ".join(excerpt.split())
    return f"{'…' if lo > 0 else ''}{excerpt}{'…' if hi < len(text) else ''}"


def _render_history_hits(hits: list[_SearchHit], target: Path) -> None:
    """Group hits by run, print a faded run header once, then one line per hit.
    Identical snippets within a run (the same system-prompt boilerplate matched
    in every transcript) collapse to one line with an ``(xN)`` count."""
    if not hits:
        print(f"[agent6] no matches under {target}.")
        return
    grouped: dict[str, list[_SearchHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.run_id, []).append(hit)
    total = 0
    for i, (run_id, run_hits) in enumerate(grouped.items()):
        print("" if i == 0 else "\n", end="")
        print(sgr(run_id, "1"))
        # Dedup by snippet text (not by kind): the same boilerplate repeated
        # across cumulative transcript snapshots collapses to one line with a count.
        counts: dict[str, int] = {}
        first: dict[str, _SearchHit] = {}
        for hit in run_hits:
            counts[hit.snippet] = counts.get(hit.snippet, 0) + 1
            first.setdefault(hit.snippet, hit)
        for snippet, hit in first.items():
            n = counts[snippet]
            # Show the timestamp only for a unique hit; a collapsed group spans
            # several times, so a count is clearer than any one of them.
            meta = "  ".join(p for p in (hit.when, hit.kind) if p) if n == 1 else hit.kind
            tag = f" {sgr(f'(x{n})', '2')}" if n > 1 else ""
            print(f"  {sgr(meta, '2')}  {snippet}{tag}")
        total += len(run_hits)
    print(sgr(f"\n{total} match{'es' if total != 1 else ''} in {len(grouped)} run(s)", "2"))


def _cmd_history_graph(run_id: str) -> int:
    """Render the persisted TaskNode tree for a run as a DFS-ordered listing."""

    cwd = Path.cwd()
    if run_id:
        # Resolve across runs/ + asks/ so an ask's graph is findable too.
        try:
            layout = resolve_run_layout(cwd, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        runs_dir = _runs_dir(cwd)
        if not runs_dir.is_dir():
            print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
            return 2
        candidates = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir() and (p / "graph").is_dir()),
            key=run_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"ERROR: no runs with a graph under {runs_dir}", file=sys.stderr)
            return 2
        layout = RunLayout(state_dir=_state_dir(cwd), run_id=candidates[0].name)
        print(f"[agent6] showing graph for most recent run: {layout.run_id}", file=sys.stderr)

    target_id = layout.run_id
    nodes = load_graph(layout)
    if not nodes:
        print(f"ERROR: run {target_id} has no persisted graph nodes", file=sys.stderr)
        return 2

    print(f"Run id: {target_id}")
    print()
    for line in task_tree_lines(nodes, show_commit=True):
        print(line)
    return 0


def _parse_seq_window(spec: str) -> tuple[int, int] | None:
    """`""` -> None (all); `"5"` -> (5,5); `"3-7"` -> (3,7). Raises ValueError on junk."""
    spec = spec.strip()
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


def _cmd_history_transcript(
    run_id: str, *, as_json: bool, no_thinking: bool, tools: str, seq: str
) -> int:
    """Render a run's full LLM conversation from its lossless per-call transcripts.

    The transcripts (``<run>/transcripts/*.json``) are the complete, self-
    contained record -- no join with logs.jsonl is needed. This is the CONVERSATION
    view (assistant text/thinking + every tool call with full I/O); for the terse
    EVENT timeline use `agent6 attach` / `agent6 history search`.
    """
    cwd = Path.cwd()
    if run_id:
        try:
            layout = resolve_run_layout(cwd, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        runs_dir = _runs_dir(cwd)
        candidates = (
            sorted(
                (p for p in runs_dir.iterdir() if p.is_dir() and (p / "transcripts").is_dir()),
                key=run_mtime,
                reverse=True,
            )
            if runs_dir.is_dir()
            else []
        )
        if not candidates:
            print(f"ERROR: no runs with transcripts under {runs_dir}", file=sys.stderr)
            return 2
        layout = RunLayout(state_dir=_state_dir(cwd), run_id=candidates[0].name)
        print(f"[agent6] transcript for most recent run: {layout.run_id}", file=sys.stderr)

    try:
        window = _parse_seq_window(seq)
    except ValueError:
        print(f"ERROR: --seq expects N or N-M, got {seq!r}", file=sys.stderr)
        return 2

    transcripts = load_transcripts(layout.transcripts_dir)
    if not transcripts:
        print(f"ERROR: run {layout.run_id} has no transcripts", file=sys.stderr)
        return 2

    if as_json:
        if window is not None:
            lo, hi = window
            transcripts = [t for t in transcripts if lo <= int(t.get("seq", 0)) <= hi]
        print(json.dumps(transcripts, indent=2, ensure_ascii=False))
        return 0

    # Fold the FULL set (the per-seq walk needs every call), then window the turns.
    turns = fold_conversation(transcripts)
    if window is not None:
        lo, hi = window
        turns = [t for t in turns if lo <= t.seq <= hi]
    print(
        render_markdown(turns, run_id=layout.run_id, show_thinking=not no_thinking, tools=tools),
        end="",
    )
    return 0
