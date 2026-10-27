# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build squash messages from the landed diff and the run's execution summaries."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6 import errors, event_log, harness, secret_store, viewmodel
from agent6.app import _setup, providers
from agent6.config import Config, ConfigError
from agent6.providers import ProviderError, TranscriptSink, call_for_text
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import commit_message, git_read


def squash_message(
    cwd: pathlib.Path,
    cfg: Config,
    manifest: sessions_manifest.SessionManifest,
    *,
    base_sha: str,
    run_branch: str,
    finish_summaries: tuple[str, ...],
    transcript_dir: pathlib.Path | None,
    budget: agent6_budget.BudgetTracker | None,
    events: event_log.EventSink | None,
    warn: Callable[[str], None],
) -> str | None:
    """Return the squash message per `[git.commit.squash].message`; None lets git combine."""
    style = cfg.git.commit.squash.message
    rows = git_read.list_run_commits(cwd, base_sha, run_branch)
    if style == "combine":
        # Git's own SQUASH_MSG shape; the plumbing merge never runs `merge --squash`.
        parts = ["Squashed commit of the following:\n"]
        parts += [
            f"commit {r.sha}\n\n    {r.message.rstrip().replace(chr(10), chr(10) + '    ')}"
            for r in rows
        ]
        return "\n".join(parts) if rows else None
    task = manifest.user_task or "agent6 run"
    summary = finish_summaries[-1] if finish_summaries else ""
    if style == "conventional":
        # The prefix goes on the uncut clause, so the subject is cut once.
        whole = commit_message.condense_commit_message(
            rows, subject=task, summary=summary, fit=False
        ).splitlines()
        subject = commit_message.conventional_commit_subject(
            git_read.range_name_status(cwd, base_sha, run_branch), summary=whole[0]
        )
        return "\n".join([subject, *whole[1:]])
    base_msg = commit_message.condense_commit_message(rows, subject=task, summary=summary)
    if style == "model":
        msg = model_squash_message(
            cwd,
            cfg,
            rows,
            base_sha=base_sha,
            run_branch=run_branch,
            task=task,
            finish_summaries=finish_summaries,
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
            warn=warn,
        )
        if msg:
            return msg
    return base_msg


def model_squash_message(
    cwd: pathlib.Path,
    cfg: Config,
    rows: tuple[commit_message.CommitRow, ...],
    *,
    base_sha: str,
    run_branch: str,
    task: str,
    finish_summaries: tuple[str, ...] = (),
    transcript_dir: pathlib.Path | None,
    budget: agent6_budget.BudgetTracker | None,
    events: event_log.EventSink | None,
    warn: Callable[[str], None] = lambda _m: None,
) -> str | None:
    """Write the squash message from the landed diff and the run's context.

    Args:
        cwd: The repository.
        cfg: The resolved config.
        rows: The run's commits.
        base_sha: The commit the run started from.
        run_branch: The run's branch or chain ref.
        task: The run's task.
        finish_summaries: Completed executions' summaries, subordinate to the diff.
        transcript_dir: Where the call is transcribed; None skips the call.
        budget: The run's tracker when auto_merge runs inside a run, so the call spends the
            run's remainder; None takes a fresh per-invocation cap.
        events: The run's event sink; with it the call's spend reaches the log.
        warn: Where a failure is reported.

    Returns:
        The message, or None on any failure (the caller degrades to the agent6 style).
    """
    if transcript_dir is None:
        return None
    try:
        tracker = budget if budget is not None else _setup.budget_tracker(cfg)
        provider = providers.build_role_provider(
            cfg,
            "worker",
            transcript_sink=TranscriptSink(transcript_dir),
            budget=tracker,
            events=events,
        )
        if events is not None:
            rm = cfg.models.resolve("worker")
            provider = providers.InstrumentedProvider(
                inner=provider,
                role="squash",
                model=rm.model if rm is not None else "",
                provider_name=rm.provider if rm is not None else "",
                events=events,
                budget=tracker,
            )
        steps = "\n".join(f"- {r.subject}" for r in rows[:100])
        files = "\n".join(
            f"{s}\t{p}" for s, p in git_read.range_name_status(cwd, base_sha, run_branch)[:200]
        )
        diff = git_read.diff_range(cwd, base_sha, run_branch)
        diff_bytes = diff.encode("utf-8")
        if len(diff_bytes) > 65536:
            # Keep whole lines, including when one line alone exceeds the byte budget.
            diff = diff_bytes[:65536].rpartition(b"\n")[0].decode("utf-8")
            diff += f"\n[diff truncated after {len(diff.splitlines())} lines]"
        conventions = harness.agents_md_text(cwd, cfg.sandbox.hide_paths)
        system = (
            "Write a git commit message describing the entire base-to-tip diff being landed,"
            " not just the latest execution or review feedback. The diff is authoritative;"
            " the task, finish summaries and per-step subjects are context only. Use only"
            " the facts given. Write one imperative subject line under 72 characters, a blank"
            " line, then a short body. Output the message text only."
        )
        if conventions:
            system += (
                "\nFollow the repository's commit-message conventions stated in AGENTS.md."
                f"\n\nAGENTS.md (project conventions):\n{conventions}"
            )
        summaries = "\n\n".join(finish_summaries)
        msg = call_for_text(
            provider,
            system=system,
            user=(
                f"Task: {task}\nFinish summaries (execution order):\n{summaries}\n"
                f"Per-step subjects:\n{steps}\nChanged files (status\tpath):\n{files}\n"
                f"Base-to-tip diff ({base_sha}..{run_branch}):\n{diff}"
            ),
            max_tokens=500,
        )
    except (
        agent6_budget.BudgetExceededError,
        ConfigError,
        errors.GitError,
        OSError,
        ProviderError,
        secret_store.SecretsError,
    ) as exc:
        # auto_merge runs the draft in a finished run's teardown, which must not crash on it.
        warn(f"model squash message failed ({exc}); using the agent6 style")
        return None
    if not msg:
        warn("model squash message failed (the model returned nothing); using the agent6 style")
    return msg or None


def finish_summaries(layout: sessions_layout.SessionLayout) -> tuple[str, ...]:
    """Return the summaries of executions that ended on a successful finish call."""
    state = viewmodel.initial_state()
    summaries: list[str] = []
    for event in viewmodel.tail_events(layout.logs_path, follow=False):
        state = viewmodel.apply_event(state, event)
        if (
            event.get("type") == "session.end"
            and state.end_reason == "finish_session"
            and state.finish_summary
        ):
            summaries.append(state.finish_summary)
    return tuple(summaries)
