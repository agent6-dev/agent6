# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search/graph/transcript` commands."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from agent6.cli._common import (
    _runs_dir,
    _state_dir,
    resolve_run_layout,
)
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, load_graph
from agent6.run_id import RunIdError
from agent6.transcript_render import fold_conversation, load_transcripts, render_markdown


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
        # Resolve across runs/ + asks/ so an ask's logs/transcript are searchable.
        try:
            target = resolve_run_layout(cwd, run_id).run_dir
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        target = _runs_dir(cwd)
    if not target.is_dir():
        print(f"ERROR: no such directory: {target}", file=sys.stderr)
        return 2
    argv: list[str] = [
        rg,
        "--color=never",
        "--with-filename",
        "--line-number",
    ]
    if fixed:
        argv.append("--fixed-strings")
    argv.extend(["--", query, str(target)])
    completed = subprocess.run(argv, check=False)
    # rg returns 1 if no matches; that's not an error for us.
    if completed.returncode in (0, 1):
        return completed.returncode
    return completed.returncode


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
            key=lambda p: p.stat().st_mtime,
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

    roots = sorted(
        (n for n in nodes.values() if n.parent_id is None),
        key=lambda n: n.created_at,
    )
    print(f"Run id: {target_id}")
    print()
    for root in roots:
        _print_node_dfs(root, nodes, depth=0)
    return 0


def _print_node_dfs(node: TaskNode, nodes: dict[str, TaskNode], *, depth: int) -> None:
    """Depth-first, left-to-right print of one TaskNode subtree."""

    indent = "  " * depth
    status = f"[{node.status}]"
    commit = f"  (commit: {node.commit_sha[:7]})" if node.commit_sha else ""
    print(f"{indent}{status} {node.title}{commit}")
    # Children are ordered by insertion (curator preserves order); walk them
    # left-to-right, recursing fully into each before moving to the next.
    for child_id in node.children:
        child = nodes.get(child_id)
        if child is None:
            print(f"{indent}  [MISSING] <child id {child_id} not found>")
            continue
        _print_node_dfs(child, nodes, depth=depth + 1)


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
    EVENT timeline use `agent6 watch` / `agent6 history search`.
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
                key=lambda p: p.stat().st_mtime,
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
