# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 ask` flow: the seed digest, the interactive ask REPL and the transcripts."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
from typing import Any

from agent6 import budget as agent6_budget
from agent6 import errors, git_ops, paths
from agent6.harness import _snapshot, loop
from agent6.sessions import id
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import _common, _steer
from agent6.viewmodel import newest_session_dir


def summarize_session_log(logs_path: pathlib.Path) -> str:
    """Return a compact summary of a run's log: outcome, event counts and recent events."""
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
    end = next((e for e in reversed(events) if e.get("type") == "session.end"), None)
    if end is not None:
        out.append(f"Ended: reason={end.get('reason')!r} iterations={end.get('iterations')}")
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    out.append("Event counts: " + ", ".join(f"{t}={n}" for t, n in top))
    notable_types = {
        "tool.call",
        "verify.end",
        "session.end",
        "loop.auto_commit",
        "loop.metric.sample",
    }
    notable = [e for e in events if e.get("type") in notable_types][-15:]
    if notable:
        out.append("Recent notable events:")
        out.extend(f"  - {fmt_run_event(e)}" for e in notable)
    return "\n".join(out)


def fmt_run_event(e: dict[str, Any]) -> str:
    """Return a one-line summary of a log event for the seed digest."""
    t = str(e.get("type", ""))
    if t == "tool.call":
        return f"tool.call {e.get('name', '')} {str(e.get('args', ''))[:80]}".rstrip()
    if t == "verify.end":
        return f"verify.end exit={e.get('exit_code')}"
    if t == "session.end":
        return f"session.end reason={e.get('reason')}"
    if t == "loop.metric.sample":
        return f"loop.metric.sample score={e.get('score')}"
    return t


def _git_diff_text(cwd: pathlib.Path, range_spec: str) -> tuple[int, str, str]:
    """Return the rc, stdout and stderr of a hardened `git diff <range>`.

    Bytes are decoded lossily, since a valid diff can be non-UTF-8. Fixed argv from
    operator input; the hardening flags keep a poisoned `.git/config` from running
    `diff.external` or a textconv on the host.
    """
    proc = subprocess.run(
        [
            "git",
            *git_ops.git_hardening_flags(cwd),
            "diff",
            *git_ops.DIFF_SHOW_SAFETY_FLAGS,
            range_spec,
        ],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return (
        proc.returncode,
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
    )


def _diff_via_merge_stamp(
    cwd: pathlib.Path,
    manifest: sessions_manifest.SessionManifest,
    base_sha: str,
    run_branch: str | None,
) -> tuple[str, int, str, str] | None:
    """Return the diff through the manifest's merge stamp when the primary range is unreachable.

    Args:
        cwd: The repo.
        manifest: The session's manifest.
        base_sha: The run's base.
        run_branch: The run's branch, when it had one.

    Returns:
        `(label, rc, diff, err)`, the label naming the range and why; None without a stamp.
    """
    merged = manifest.merged
    if merged is None or not run_branch or not merged.sha:
        return None
    gone = git_ops.branch_tip_sha(cwd, run_branch) is None
    why = "run branch pruned" if gone else "base unreachable"
    if merged.sha == sessions_manifest.NO_MERGE_COMMIT:
        # A merge that added nothing names no commit; the run's work is its stamped tip's.
        if not merged.tip:
            return None
        label = f"{base_sha[:12]}..{merged.tip[:12]} ({why}; merged without a commit)"
        return label, *_git_diff_text(cwd, f"{base_sha}..{merged.tip}")
    merged_sha = merged.sha
    is_ff = merged_sha == merged.tip
    if is_ff:
        # Fast-forwarded: the full run is base..merged, both in the base branch's history.
        label = f"{base_sha[:12]}..{merged_sha[:12]} ({why}; fast-forward merge)"
        rc, diff, err = _git_diff_text(cwd, f"{base_sha}..{merged_sha}")
        if rc == 0:
            return label, rc, diff, err
    partial = "; last run commit only" if is_ff else ""
    label = f"{merged_sha[:12]}^..{merged_sha[:12]} ({why}; merge commit{partial})"
    return label, *_git_diff_text(cwd, f"{merged_sha}^..{merged_sha}")


@dataclasses.dataclass(frozen=True, slots=True)
class SessionSeed:
    """A prior session's resolved id and markdown context."""

    source_session_id: str
    text: str


def build_session_seed(cwd: pathlib.Path, session_id: str, *, latest: bool) -> SessionSeed | None:
    """Return the resolved source and its markdown context for a new session.

    Any session kind seeds any other: a run, a plan and an ask record the same shape.

    Args:
        cwd: The repo.
        session_id: The source, or "" with `latest`.
        latest: Take the newest run or ask.

    Returns:
        The seed, or None after printing why the source could not be resolved.
    """
    state = paths.state_dir(cwd)
    if latest:
        # A machine draft is an authoring log, not a session with a task and an outcome.
        newest = newest_session_dir(
            [sessions_layout.bucket_dir(state, "runs"), sessions_layout.bucket_dir(state, "asks")]
        )
        if newest is None:
            _common.error(f"--from-latest: no run or ask under {state}")
            return None
        session_id = newest.name
    try:
        layout = id.resolve_session(state, session_id)
    except id.SessionIdError as exc:
        _common.error(f"{exc}")
        return None
    target = layout.session_id
    if not layout.manifest_path.is_file():
        _common.error(f"run {target} has no manifest.json")
        return None
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError as exc:
        _common.error(f"could not read manifest for {target}: {exc}")
        return None
    base_sha = manifest.base_sha
    run_branch = manifest.run_branch
    diff_label = f"{base_sha}..{run_branch}"
    diff_body = "(no diff: the run recorded no base_sha)"
    if not run_branch:
        # A plan and an ask commit nothing; diffing HEAD would label the operator's work as theirs.
        diff_label = "(none)"
        diff_body = "(no diff: this session wrote no code)"
    elif base_sha:
        rc, diff, err = _git_diff_text(cwd, f"{base_sha}..{run_branch}")
        if rc != 0:
            fallback = _diff_via_merge_stamp(cwd, manifest, base_sha, run_branch)
            if fallback is not None:
                diff_label, rc, diff, err = fallback
        if rc != 0:
            # Loud: an empty diff block reads as "no changes".
            diff_body = f"(diff unavailable: git diff exited {rc}: {err.strip()[:300]})"
        else:
            cap = 8000
            tail = "\n... (diff truncated; read more with git)" if len(diff) > cap else ""
            diff_body = f"```diff\n{diff[:cap]}{tail}\n```"
    plan_path = layout.session_dir / "plan.md"
    plan_section = (
        f"\n## Plan\n{errors.read_operator_file(plan_path)}\n" if plan_path.is_file() else ""
    )
    transcript_path = layout.session_dir / "transcript.md"
    ask_section = (
        f"\n## Ask transcript\n{errors.read_operator_file(transcript_path)}\n"
        if layout.subdir == "asks" and transcript_path.is_file()
        else ""
    )
    return SessionSeed(
        source_session_id=target,
        text=(
            f'<prior-run id="{target}">\n'
            "This question is about a PRIOR agent6 run. Its run state lives outside the"
            " workspace and is not reachable with read_file, so everything you have"
            " about it is in this digest.\n\n"
            f"## Run task\n{manifest.user_task}\n\n"
            f"## Outcome / key events\n{summarize_session_log(layout.logs_path)}\n\n"
            f"## Diff {diff_label}\n{diff_body}\n"
            f"{plan_section}"
            f"{ask_section}"
            f"</prior-run>"
        ),
    )


def seed_files(cwd: pathlib.Path, files: list[str]) -> str:
    """Return the `--file` seeds wrapped for an ask; a capped, non-fatal read each."""
    parts: list[str] = []
    for f in files:
        try:
            content = (cwd / f).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _common.warn(f"--file {f}: {exc}")
            continue
        cap = 64 * 1024
        if len(content) > cap:
            content = content[:cap] + "\n... (truncated)"
        parts.append(f'<file path="{f}">\n{content}\n</file>')
    return "\n".join(parts)


def save_ask_transcript(
    layout: sessions_layout.SessionLayout, *, question: str, answer: str
) -> None:
    """Append the question and its markdown answer to the ask's transcript.

    A resumed ask appends both halves: an answer alone under the first question
    would read as its continuation.

    Args:
        layout: The ask's layout.
        question: The question.
        answer: The answer.
    """
    out = layout.session_dir / "transcript.md"
    if out.is_file():
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## Question (continued)\n\n{question}\n\n## Answer\n\n{answer}\n")
        return
    out.write_text(
        f"# agent6 ask\n\n## Question\n\n{question}\n\n## Answer\n\n{answer}\n",
        encoding="utf-8",
    )


def save_ask_repl_transcript(
    layout: sessions_layout.SessionLayout, conversation: list[tuple[str, str]]
) -> None:
    """Write the cumulative transcript for an interactive ask session."""
    parts = ["# agent6 ask (interactive)\n"]
    for i, (q, a) in enumerate(conversation, 1):
        parts.append(f"## Q{i}\n\n{q}\n\n## A{i}\n\n{a}\n")
    (layout.session_dir / "transcript.md").write_text("\n".join(parts), encoding="utf-8")


def run_ask_repl(
    wf: loop.Harness,
    budget: agent6_budget.BudgetTracker,
    layout: sessions_layout.SessionLayout,
    *,
    first_question: str,
) -> _snapshot.SessionResult:
    """Run a multi-turn ask, each follow-up re-entering the loop with the prior Q&A as context.

    Args:
        wf: The harness, with its provider, jail and budget set up once.
        budget: The budget tracker.
        layout: The ask's layout.
        first_question: The question the command line carried.

    Returns:
        The last turn's result.
    """
    print(
        "[agent6] ask REPL: type a follow-up, or /cost /reset /quit (Ctrl-D exits).",
        file=sys.stderr,
    )
    conversation: list[tuple[str, str]] = []
    pending = first_question.strip()
    result: _snapshot.SessionResult | None = None
    while True:
        if pending:
            question = pending
            pending = ""
        else:
            try:
                with _steer.idle_prompt_sigint():
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
        print(result.summary, flush=True)
        conversation.append((question, result.summary))
        save_ask_repl_transcript(layout, conversation)
        if budget.is_exhausted():
            print("[agent6] budget exhausted; ending the REPL.", file=sys.stderr)
            break
    if result is None:
        return _snapshot.SessionResult(
            completed=True, reason="ask_repl_empty", summary="", iterations=0, tool_calls=0
        )
    return result
