# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory/history/init/diff/mcp-serve/review` commands."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.cli._common import _agent6_dir, _check_provider_keys, _runs_dir
from agent6.cli.plan_watch import _most_recent_run_id
from agent6.cli.providers import _build_role_provider
from agent6.config import (
    ConfigError,
)
from agent6.config_layer import (
    load_effective,
    repo_config_path_for,
)
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, load_graph
from agent6.init import init_workspace
from agent6.mcp_server import run_server as _mcp_run_server
from agent6.memory import (
    MemoryError as Agent6MemoryError,
)
from agent6.memory import (
    MemoryScope,
)
from agent6.memory import (
    add as memory_add,
)
from agent6.memory import (
    invalidate as memory_invalidate,
)
from agent6.memory import (
    list_entries as memory_list,
)
from agent6.paths import (
    chown_to_real_user,
)
from agent6.providers import (
    ProviderError,
    TranscriptSink,
)
from agent6.run_id import RunIdError, resolve_run_id
from agent6.workflows.review import CodeReviewError, run_review


def _cmd_memory_add(scope: MemoryScope, body: str) -> int:
    try:
        entry = memory_add(_agent6_dir(Path.cwd()), scope, body)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{entry.scope} {entry.id} created at {entry.created_at}")
    return 0


def _cmd_memory_list(scope: MemoryScope | None, *, include_invalidated: bool) -> int:
    try:
        entries = memory_list(_agent6_dir(Path.cwd()), scope)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print("(no memories)")
        return 0
    for e in entries:
        if not include_invalidated and not e.is_active:
            continue
        flag = "" if e.is_active else " [INVALIDATED]"
        print(f"[{e.scope}] {e.id} {e.created_at}{flag}")
        if not e.is_active and e.invalidation_reason:
            print(f"    invalidated_at: {e.invalidated_at}  reason: {e.invalidation_reason}")
        for line in e.body.splitlines():
            print(f"    {line}")
        print()
    return 0


def _cmd_memory_invalidate(memory_id: str, reason: str) -> int:
    try:
        entry = memory_invalidate(_agent6_dir(Path.cwd()), memory_id, reason)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"invalidated {entry.scope} {entry.id} at {entry.invalidated_at}")
    return 0


def _cmd_history_search(query: str, *, fixed: bool, run_id: str) -> int:
    rg = shutil.which("rg")
    if rg is None:
        print(
            "ERROR: `rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry.",
            file=sys.stderr,
        )
        return 2
    runs_root = _runs_dir(Path.cwd())
    if run_id:
        try:
            run_id = resolve_run_id(runs_root, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    target = runs_root / run_id if run_id else runs_root
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

    runs_dir = _runs_dir(Path.cwd())
    if run_id:
        try:
            target_id = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
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
        target_id = candidates[0].name
        print(f"[agent6] showing graph for most recent run: {target_id}", file=sys.stderr)

    layout = RunLayout(state_dir=_agent6_dir(Path.cwd()), run_id=target_id)
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


def _cmd_init(*, force: bool, profile: str, assume_yes: bool = False) -> int:
    cwd = Path.cwd()
    target = repo_config_path_for(cwd)
    interactive = not assume_yes and not force and sys.stdin.isatty()
    rc = init_workspace(
        cwd,
        force=force,
        profile=profile,
        repo_config_target=target,
        interactive=interactive,
    )
    # Don't leave root-owned scaffolding in the user's repo (sudo case).
    chown_to_real_user(target.parent)
    return rc


def _cmd_diff(*, run_id: str, stat: bool, paths: tuple[str, ...]) -> int:  # noqa: PLR0911
    """Print the git diff a run produced (manifest.base_sha -> branch HEAD).

    Resolves the run id (or unique prefix; empty string means most-recent),
    reads ``manifest.json`` for ``base_sha`` and ``run_branch``, then shells
    out to ``git diff`` with operator-controlled argv (no LLM input).
    """
    cwd = Path.cwd()
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return 2

    target_id = run_id
    if not target_id:
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}", file=sys.stderr)
            return 2
        target_id = latest
        print(f"[agent6] diffing most recent run: {target_id}", file=sys.stderr)
    else:
        try:
            target_id = resolve_run_id(runs_dir, target_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    layout = RunLayout(state_dir=_agent6_dir(cwd), run_id=target_id)
    if not layout.manifest_path.is_file():
        print(
            f"ERROR: run {target_id} has no manifest.json "
            "(predates manifest support, or was killed before setup)",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2

    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    if not base_sha:
        print("ERROR: manifest has no base_sha; nothing to diff against", file=sys.stderr)
        return 2

    head_ref = str(run_branch) if run_branch else "HEAD"
    argv: list[str] = ["git", "diff"]
    if stat:
        argv.append("--stat")
    argv.extend([f"{base_sha}..{head_ref}"])
    if paths:
        argv.append("--")
        argv.extend(paths)
    print(
        f"[agent6] {' '.join(argv)}  (base_branch={manifest.get('base_branch')!r})",
        file=sys.stderr,
    )
    proc = subprocess.run(argv, cwd=cwd, check=False)
    return proc.returncode


def _cmd_mcp_serve(config_path: Path | None) -> int:
    """Spawn an MCP stdio server against ``config_path``'s
    workspace. Thin wrapper so dispatch stays uniform with the other
    ``_cmd_*`` helpers."""
    return _mcp_run_server(config_path)


def _cmd_review(  # noqa: PLR0911
    config_path: Path | None,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    model_override: str = "",
) -> int:
    """Print a freeform code review of a diff to stdout. Read-only; no jail."""
    try:
        cfg = load_effective(Path.cwd(), config_path).config
        cfg.require_runnable("reviewer", need_verify=False)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    err = _check_provider_keys(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    root = Path.cwd()
    git = shutil.which("git")
    if git is None:
        print("ERROR: git not found on PATH.", file=sys.stderr)
        return 2

    if base:
        diff_args = [git, "diff", f"{base}..{head}"]
    else:
        # Working tree vs HEAD, including untracked files (intent-to-add).
        subprocess.run([git, "add", "-N", "--", "."], cwd=root, check=False)
        diff_args = [git, "diff", "HEAD"]
    if paths:
        diff_args.extend(["--", *paths])
    diff_proc = subprocess.run(diff_args, cwd=root, capture_output=True, text=True, check=False)
    if diff_proc.returncode != 0:
        print(f"ERROR: git diff failed: {diff_proc.stderr.strip()}", file=sys.stderr)
        return 2
    diff = diff_proc.stdout
    if not diff.strip():
        print("(no diff to review)", file=sys.stderr)
        return 0

    log_proc = subprocess.run(
        [git, "log", "-n", "10", "--oneline"], cwd=root, capture_output=True, text=True, check=False
    )
    recent_log = log_proc.stdout if log_proc.returncode == 0 else ""

    agents_md_path = root / "AGENTS.md"
    agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""

    # Reviewer-only: route the "reviewer" role per [models.reviewer]. Budget
    # is per-invocation since this command is a one-shot.
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )
    layout_root = _agent6_dir(root) / "reviews"
    layout_root.mkdir(parents=True, exist_ok=True)
    transcript_sink = TranscriptSink(layout_root)

    try:
        reviewer = _build_role_provider(
            cfg,
            "reviewer",
            transcript_sink=transcript_sink,
            budget=budget,
            model_override=model_override,
        )
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    label = (
        "working tree vs HEAD"
        if not base
        else f"{base}..{head}" + (f" -- {' '.join(paths)}" if paths else "")
    )
    print(f"[agent6] reviewing: {label}", file=sys.stderr)
    try:
        text = run_review(
            reviewer,
            diff=diff,
            agents_md=agents_md,
            recent_log=recent_log,
        )
    except CodeReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3

    print(text)
    print(budget.format_summary(), file=sys.stderr)
    return 0
