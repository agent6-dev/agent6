# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The optional pre-loop prompt-revision pass.

Before the worker loop starts, the reviser model can rewrite a terse task into
an explicit one and surface clarifying questions. This module holds the parse
of its output, the repo-context block fed to it, the effective-task assembly,
and the small text helpers they use. The loop owns running the reviser call.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agent6.budget import BudgetExceeded
from agent6.prompts.revision import PROMPT_REVISION_SYSTEM_PROMPT
from agent6.providers import Provider, ProviderError
from agent6.types import RepoSummary

# One leading list marker ("- ", "* ", "1. ", "2) "). A charset lstrip would
# also eat leading digits of the question itself ("- 32-bit ..." -> "bit ...").
# The numeric marker requires trailing whitespace so a bare decimal that opens a
# question keeps it ("0.5s latency budget OK?" must not become "5s ...").
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]\s)\s*")


@dataclass(frozen=True, slots=True)
class RevisionSettings:
    """The one-shot prompt revision before the first worker call
    (`prompt.revise_prompt`): `reviser` (the reviewer role) rewrites the
    task once, with no tools and no iteration, at `temperature` and within
    `max_tokens`; `interactive` hands the original, the revision and the
    reviser's questions to `selector` for the operator to choose."""

    reviser: Provider | None = None
    mode: Literal["off", "auto", "interactive"] = "off"
    temperature: float | None = 0.0
    max_tokens: int = 2048
    selector: Callable[[str, str, tuple[str, ...]], str | None] | None = None


@dataclass(frozen=True, slots=True)
class PromptRevision:
    revised_task: str
    clarifying_questions: tuple[str, ...] = ()


class PromptRevisionError(Exception):
    """Raised when the optional prompt-revision pass cannot produce a task."""


class PromptRevisionDeclined(PromptRevisionError):
    """The operator quit at the interactive choice: their stop, not a failure."""


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 40)].rstrip() + "\n...[truncated for prompt revision]"


def tag_body(text: str, tag: str) -> str:
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
    revised = tag_body(text, "revised_task") if "<revised_task>" in text else text.strip()
    questions_raw = tag_body(text, "clarifying_questions")
    questions: list[str] = []
    for raw_line in questions_raw.splitlines():
        line = _LIST_MARKER_RE.sub("", raw_line).strip()
        if not line or line.lower() in {"none", "n/a", "no questions"}:
            continue
        questions.append(line)
    return PromptRevision(revised_task=revised.strip(), clarifying_questions=tuple(questions[:3]))


def format_prompt_revision_context(repo: RepoSummary) -> str:
    if repo.is_git:
        repo_line = (
            f"Repository: branch={repo.branch}, head={repo.head_sha[:12]}, files={repo.file_count}"
        )
    else:
        # Same degrade as the worker prompt: outside git a fake empty header
        # would send the model after branch and history that do not exist.
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
    repo: RepoSummary,
    *,
    log: Callable[[str], None],
    emit: Callable[..., None],
) -> str:
    """The task the worker gets: *user_task* as given, or its one-shot
    revision by the reviser (`RevisionSettings.mode`), folded with the
    original (`format_effective_task`); in interactive mode the operator
    chooses. Raises `PromptRevisionError` when the reviser is missing or
    fails, `PromptRevisionDeclined` when the operator quits the choice."""
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
            system=PROMPT_REVISION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    except (ProviderError, BudgetExceeded) as exc:
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
