# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 ask` read-only Q&A flow: listing past asks, building a run
digest for context, the interactive ask REPL, and saving ask transcripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent6.budget import BudgetTracker
from agent6.cli._common import (
    _runs_dir,
    _state_dir,
)
from agent6.cli.plan_watch import _most_recent_run_id
from agent6.graph.storage import RunLayout
from agent6.run_id import (
    RunIdError,
    resolve_run_id,
)
from agent6.workflows.loop import (
    RunResult,
    Workflow,
)


def ask_question_snippet(transcript: str) -> str:
    """First non-tag line of the `## Question` section of an ask transcript."""
    lines = transcript.splitlines()
    try:
        start = lines.index("## Question") + 1
    except ValueError:
        return "(no question)"
    for line in lines[start:]:
        s = line.strip()
        if s == "## Answer":
            break
        if s and not s.startswith("<"):  # skip blank lines + digest/file tags
            return s
    return "(question)"


def cmd_ask_list() -> int:
    """`agent6 ask list`: enumerate saved asks under the per-repo state dir (asks subdir)."""
    asks_dir = _state_dir(Path.cwd()) / "asks"
    if not asks_dir.is_dir():
        print("No asks yet (the asks subdir under the per-repo state dir does not exist).")
        return 0
    dirs = sorted(
        (d for d in asks_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        print("No asks yet.")
        return 0
    for d in dirs:
        tp = d / "transcript.md"
        snippet = (
            ask_question_snippet(tp.read_text(encoding="utf-8", errors="replace"))
            if tp.is_file()
            else "(no transcript)"
        )
        print(f"{d.name}  {snippet[:90]}")
    return 0


def summarize_run_log(logs_path: Path) -> str:
    """Compact prose summary of a run's logs.jsonl: outcome + event counts +
    recent notable events. Used to seed `agent6 ask --run`."""
    if not logs_path.is_file():
        return "(no logs.jsonl for this run)"
    events: list[dict[str, Any]] = []
    for line in logs_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    if not events:
        return "(empty log)"
    counts: dict[str, int] = {}
    for e in events:
        counts[str(e.get("type", ""))] = counts.get(str(e.get("type", "")), 0) + 1
    out: list[str] = []
    end = next((e for e in reversed(events) if e.get("type") == "run.end"), None)
    if end is not None:
        out.append(f"Ended: reason={end.get('reason')!r} iterations={end.get('iterations')}")
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    out.append("Event counts: " + ", ".join(f"{t}={n}" for t, n in top))
    notable_types = {"tool.call", "verify.end", "run.end", "loop.auto_commit", "loop.metric.sample"}
    notable = [e for e in events if e.get("type") in notable_types][-15:]
    if notable:
        out.append("Recent notable events:")
        out.extend(f"  - {fmt_run_event(e)}" for e in notable)
    return "\n".join(out)


def fmt_run_event(e: dict[str, Any]) -> str:
    """One-line summary of a logs.jsonl event for the ask `--run` digest."""
    t = str(e.get("type", ""))
    if t == "tool.call":
        return f"tool.call {e.get('name', '')} {str(e.get('args', ''))[:80]}".rstrip()
    if t == "verify.end":
        return f"verify.end exit={e.get('exit_code')}"
    if t == "run.end":
        return f"run.end reason={e.get('reason')}"
    if t == "loop.metric.sample":
        return f"loop.metric.sample score={e.get('score')}"
    return t


def build_ask_run_digest(cwd: Path, run_id: str, *, latest: bool) -> str | None:
    """Markdown digest of a prior run to seed an `ask`, or None (after printing
    an error) when the run can't be resolved."""
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return None
    if latest:
        target = _most_recent_run_id(runs_dir)
        if target is None:
            print(f"ERROR: --continue: no runs under {runs_dir}", file=sys.stderr)
            return None
    else:
        try:
            target = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return None
    layout = RunLayout(state_dir=_state_dir(cwd), run_id=target)
    if not layout.manifest_path.is_file():
        print(f"ERROR: run {target} has no manifest.json", file=sys.stderr)
        return None
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest for {target}: {exc}", file=sys.stderr)
        return None
    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    head_ref = str(run_branch) if run_branch else "HEAD"
    diff = ""
    if base_sha:
        # operator-controlled argv, no LLM input (same as `agent6 runs diff`).
        proc = subprocess.run(
            ["git", "diff", f"{base_sha}..{head_ref}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = proc.stdout
    cap = 8000
    diff_excerpt = diff[:cap]
    if len(diff) > cap:
        diff_excerpt += "\n... (diff truncated; read more with git)"
    return (
        f'<prior-run id="{target}">\n'
        "This question is about a PRIOR agent6 run. Its run state lives outside the"
        " workspace and is not reachable with read_file, so everything you have"
        " about it is in this digest.\n\n"
        f"## Run task\n{manifest.get('user_task', '')}\n\n"
        f"## Outcome / key events\n{summarize_run_log(layout.logs_path)}\n\n"
        f"## Diff base_sha..{head_ref} (truncated)\n```diff\n{diff_excerpt}\n```\n"
        f"</prior-run>"
    )


def seed_files(cwd: Path, files: list[str]) -> str:
    """Wrap explicit --file seeds for an `ask` (a non-fatal, capped read)."""
    parts: list[str] = []
    for f in files:
        try:
            content = (cwd / f).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARNING: --file {f}: {exc}", file=sys.stderr)
            continue
        cap = 64 * 1024
        if len(content) > cap:
            content = content[:cap] + "\n... (truncated)"
        parts.append(f'<file path="{f}">\n{content}\n</file>')
    return "\n".join(parts)


def save_ask_transcript(layout: RunLayout, *, question: str, answer: str) -> None:
    """Write the human-readable `ask` transcript (question + markdown answer)."""
    out = layout.run_dir / "transcript.md"
    out.write_text(
        f"# agent6 ask\n\n## Question\n\n{question}\n\n## Answer\n\n{answer}\n",
        encoding="utf-8",
    )


def save_ask_repl_transcript(layout: RunLayout, conversation: list[tuple[str, str]]) -> None:
    """Write the cumulative transcript for an interactive ask session."""
    parts = ["# agent6 ask (interactive)\n"]
    for i, (q, a) in enumerate(conversation, 1):
        parts.append(f"## Q{i}\n\n{q}\n\n## A{i}\n\n{a}\n")
    (layout.run_dir / "transcript.md").write_text("\n".join(parts), encoding="utf-8")


def run_ask_repl(
    wf: Workflow, budget: BudgetTracker, layout: RunLayout, *, first_question: str
) -> RunResult:
    """Interactive multi-turn ask. Each follow-up re-enters the loop with the
    prior Q&A carried as context, reusing the one provider/jail/budget setup.
    The agent re-reads what it needs per turn (prompt-cached); the conversation
    text is what gives continuity."""
    print(
        "[agent6] ask REPL — follow-up, /cost, /reset, or /quit (Ctrl-D exits).",
        file=sys.stderr,
    )
    conversation: list[tuple[str, str]] = []
    pending = first_question.strip()
    result: RunResult | None = None
    while True:
        if pending:
            question = pending
            pending = ""
        else:
            try:
                question = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
        if not question:
            continue
        if question in ("/quit", "/q", "/exit"):
            break
        if question == "/cost":
            print(budget.format_summary(), file=sys.stderr)
            continue
        if question == "/reset":
            conversation = []
            print("[agent6] conversation reset.", file=sys.stderr)
            continue
        if conversation:
            ctx = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in conversation)
            augmented = (
                f"<conversation-so-far>\n{ctx}\n</conversation-so-far>\n\nFollow-up: {question}"
            )
        else:
            augmented = question
        result = wf.run(augmented)
        print(result.summary)
        conversation.append((question, result.summary))
        save_ask_repl_transcript(layout, conversation)
        if budget.is_exhausted():
            print("[agent6] budget exhausted; ending the REPL.", file=sys.stderr)
            break
    if result is None:
        return RunResult(
            completed=True, reason="ask_repl_empty", summary="", iterations=0, tool_calls=0
        )
    return result
