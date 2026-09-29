# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Revise the task once before the first worker call.

The reviser model rewrites a terse task into an explicit one and surfaces clarifying
questions. This module parses its output, builds the repo-context block it reads and folds the
revision with the original; `revise_prompt` runs the call.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Literal

from agent6 import budget, kinds
from agent6.prompts import revision as prompts_revision
from agent6.providers import Provider, ProviderError

# One leading list marker; the numeric form needs trailing whitespace so "0.5s" keeps its digits.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]\s)\s*")


@dataclasses.dataclass(frozen=True, slots=True)
class RevisionSettings:
    """The `[prompt].revise_prompt` settings.

    Attributes:
        reviser: The reviewer role's provider; None when the pass is off.
        mode: Off, automatic, or interactive (the operator chooses).
        temperature: The reviser call's temperature.
        max_tokens: The reviser call's output cap.
        selector: In interactive mode, takes the original, the revision and the questions and
            returns the operator's choice, or None when they quit.
    """

    reviser: Provider | None = None
    mode: Literal["off", "auto", "interactive"] = "off"
    temperature: float | None = 0.0
    max_tokens: int = 2048
    selector: Callable[[str, str, tuple[str, ...]], str | None] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PromptRevision:
    """The reviser's answer: the rewritten task and up to three clarifying questions."""

    revised_task: str
    clarifying_questions: tuple[str, ...] = ()


class PromptRevisionError(Exception):
    """The revision pass could not produce a task."""


class PromptRevisionDeclined(PromptRevisionError):  # noqa: N818  # a signal, not an error
    """The operator quit at the interactive choice."""


def clip_text(text: str, max_chars: int) -> str:
    """Clip text to at most `max_chars`, marking the cut.

    Args:
        text: The text.
        max_chars: The cap, including the marker.

    Returns:
        The text, or its head with a truncation marker.
    """
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 40)].rstrip() + "\n...[truncated for prompt revision]"


def tag_body(text: str, tag: str) -> str:
    """Return the stripped body of the first `<tag>...</tag>` pair in the text, or "".

    Args:
        text: The text to search.
        tag: The tag's name.

    Returns:
        The body between the tags, or "" when either tag is missing.
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def parse_prompt_revision(text: str) -> PromptRevision:
    """Parse the reviser's reply.

    Args:
        text: The reply; a `<revised_task>` body, else the whole text, plus an optional
            `<clarifying_questions>` list.

    Returns:
        The revision, with at most three questions.
    """
    revised = tag_body(text, "revised_task") if "<revised_task>" in text else text.strip()
    questions_raw = tag_body(text, "clarifying_questions")
    questions: list[str] = []
    for raw_line in questions_raw.splitlines():
        line = _LIST_MARKER_RE.sub("", raw_line).strip()
        if not line or line.lower() in {"none", "n/a", "no questions"}:
            continue
        questions.append(line)
    return PromptRevision(revised_task=revised.strip(), clarifying_questions=tuple(questions[:3]))


def format_prompt_revision_context(repo: kinds.RepoSummary) -> str:
    """Return the repo-context block the reviser reads, clipped to 20,000 characters.

    Args:
        repo: The repository summary.

    Returns:
        The header, top-level listing, AGENTS.md, repo map and recent commits, each clipped.
    """
    if repo.is_git:
        repo_line = (
            f"Repository: branch={repo.branch}, head={repo.head_sha[:12]}, files={repo.file_count}"
        )
    else:
        # Outside git a fake empty header would send the model after a history that does not exist.
        repo_line = "Directory (not a git repository; no branch, history, or tracked-file map)."
    parts = [
        repo_line,
        f"Top-level: {', '.join(repo.top_level)}",
    ]
    if repo.agents_md:
        parts.append("AGENTS.md:\n" + clip_text(repo.agents_md, 5000))
    if repo.repo_map:
        parts.append("Repo map:\n" + clip_text(repo.repo_map, 4000))
    if repo.recent_log:
        parts.append("Recent commits:\n" + clip_text(repo.recent_log, 2000))
    return clip_text("\n\n".join(parts), 20_000)


def format_effective_task(raw_task: str, revision: PromptRevision) -> str:
    """Fold the revision with the original task, the original authoritative.

    Args:
        raw_task: The task as the operator gave it.
        revision: The reviser's answer.

    Returns:
        The task text the worker gets.
    """
    pieces = [
        "Revised task prompt:",
        revision.revised_task,
        "Original user task (authoritative if anything conflicts):",
        raw_task,
    ]
    if revision.clarifying_questions:
        pieces.extend(
            [
                "Clarifying questions raised by the revision pass:",
                "\n".join(f"- {q}" for q in revision.clarifying_questions),
                (
                    "Proceed under conservative assumptions if these cannot be answered from"
                    " repository context; do not stop solely because questions exist."
                ),
            ]
        )
    return "\n\n".join(pieces)


def revise_prompt(
    settings: RevisionSettings,
    user_task: str,
    repo: kinds.RepoSummary,
    *,
    log: Callable[[str], None],
    emit: Callable[..., None],
) -> str:
    """Return the task the worker gets: the task as given, or its one-shot revision.

    In interactive mode the operator chooses between the two.

    Args:
        settings: The revision settings; mode `off` returns the task as given.
        user_task: The task as the operator gave it.
        repo: The repository summary the reviser reads.
        log: The run's text logger.
        emit: The run's event sink.

    Returns:
        The task as given, or the revision folded with the original.

    Raises:
        PromptRevisionError: The reviser is missing, failed, or returned no task.
        PromptRevisionDeclined: The operator quit the interactive choice.
    """
    if settings.mode == "off":
        return user_task
    if settings.reviser is None:
        raise PromptRevisionError(
            "prompt.revise_prompt is enabled but no reviser provider is wired"
        )

    context = format_prompt_revision_context(repo)
    user_msg = f"RAW_TASK:\n{user_task}\n\nREPO_CONTEXT:\n{context}\n\nRewrite the raw task now."
    log(f"LOOP: prompt revision ({settings.mode})")
    emit("loop.prompt_revision.call", mode=settings.mode)
    try:
        resp = settings.reviser.call(
            system=prompts_revision.PROMPT_REVISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    except (ProviderError, budget.BudgetExceededError) as exc:
        emit("loop.prompt_revision.failed", error=str(exc)[:200])
        raise PromptRevisionError(str(exc)) from exc

    revision = parse_prompt_revision(resp.text or "")
    if not revision.revised_task:
        emit("loop.prompt_revision.failed", error="empty revised task")
        raise PromptRevisionError("reviser returned an empty task")

    emit(
        "loop.prompt_revision.result",
        raw_chars=len(user_task),
        revised_chars=len(revision.revised_task),
        questions=len(revision.clarifying_questions),
    )
    log(
        "PROMPT REVISION\n"
        "--- original ---\n"
        f"{clip_text(user_task, 4000)}\n"
        "--- revised ---\n"
        f"{clip_text(revision.revised_task, 6000)}"
    )
    if revision.clarifying_questions:
        log(
            "PROMPT REVISION QUESTIONS\n"
            + "\n".join(f"- {q}" for q in revision.clarifying_questions)
        )

    if settings.mode == "interactive":
        if settings.selector is None:
            raise PromptRevisionError(
                "prompt.revise_prompt='interactive' needs an interactive selector"
            )
        selected = settings.selector(
            user_task,
            revision.revised_task,
            revision.clarifying_questions,
        )
        if selected is None or not selected.strip():
            raise PromptRevisionDeclined("operator quit at the revise_prompt choice")
        selected_task = selected.strip()
        if selected_task == user_task.strip():
            return user_task
        return format_effective_task(
            user_task,
            PromptRevision(
                revised_task=selected_task,
                clarifying_questions=revision.clarifying_questions,
            ),
        )

    return format_effective_task(user_task, revision)
