# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose commit messages.

The trailer line, the condensed message a squash carries, and a checkpoint's
subject in the agent6 or the Conventional Commits style. Pure string work;
`git_ops` runs git.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from agent6.task_text import task_headline


def render_commit_trailer(fmt: str, *, models: Sequence[str]) -> str | None:
    """Render the `[git.commit].trailer` format as a trailer line.

    Args:
        fmt: The format; the config validator pins its placeholders and shape.
        models: The models that wrote the code, first seen first; the model that wrote
            a commit message never appears.

    Returns:
        The line, or None when the format is unset.
    """
    if not fmt:
        return None
    return fmt.format(model=", ".join(dict.fromkeys(m for m in models if m)))


@dataclass(frozen=True, slots=True)
class CommitRow:
    """One commit on a run branch.

    Attributes:
        sha: The commit.
        subject: Its subject line.
        message: Its whole message.
    """

    sha: str
    subject: str
    message: str


_ITER_SUBJECT_RE = re.compile(r"^agent6 iter \d+:\s*", re.IGNORECASE)


def condense_commit_message(rows: tuple[CommitRow, ...], *, subject: str) -> str:
    """Fold per-step commits into one message that reads as a single authored commit.

    Args:
        rows: The commits, oldest first.
        subject: The run's task, headlined to its first clause.

    Returns:
        The headline, then the distinct per-step subjects as bullets with the
        `agent6 iter N:` prefix and checkpoint noise stripped.
    """
    bullets: list[str] = []
    seen: set[str] = set()
    for row in rows:
        s = _ITER_SUBJECT_RE.sub("", row.subject).strip()
        if not s or s.lower().startswith("checkpoint") or s.lower() in seen:
            continue
        seen.add(s.lower())
        bullets.append(s)
    headline = _headline_subject(subject) or (bullets[0] if bullets else "agent6 run")
    parts = [headline]
    if bullets:
        parts.append("")
        parts.extend(f"- {b}" for b in bullets)
    return "\n".join(parts)


_SUBJECT_LIMIT = 72  # git's soft subject cap


def first_prose_line(text: str, *, fallback: str) -> str:
    """Return the agent's first prose line, thinking blocks and markers dropped, or the fallback."""
    cleaned = text
    while cleaned.lstrip().startswith("<thinking>"):
        end = cleaned.find("</thinking>")
        if end == -1:
            cleaned = ""
            break
        cleaned = cleaned[end + len("</thinking>") :]
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line
    return fallback


def agent6_subject(text: str, iteration: int, *, fallback: str = "verify passed") -> str:
    """Return `agent6 iter N: <first line>`, the line cut at the subject limit."""
    return f"agent6 iter {iteration}: {first_prose_line(text, fallback=fallback)[:_SUBJECT_LIMIT]}"


def _is_testish(p: str) -> bool:
    """Return whether a path is a test file."""
    parts = PurePosixPath(p).parts
    name = parts[-1] if parts else ""
    return parts[:1] == ("tests",) or name.startswith("test_") or name == "conftest.py"


def _is_docish(p: str) -> bool:
    """Return whether a path is documentation."""
    pp = PurePosixPath(p)
    return pp.suffix.lower() in (".md", ".rst") or pp.parts[:1] == ("docs",)


def _conventional_scope(paths: Sequence[str]) -> str:
    """Return the one area every path shares, or "".

    The package dir under `src/<pkg>/`, the module stem for a file directly under the
    package, else a second-level dir every path shares.
    """
    parts = [PurePosixPath(p).parts for p in paths if p]
    if not parts:
        return ""
    src_pkgs = [pp for pp in parts if len(pp) >= 3 and pp[0] == "src"]
    if src_pkgs:
        names = {pp[2] if len(pp) > 3 else str(PurePosixPath(pp[2]).stem) for pp in src_pkgs}
        return names.pop() if len(names) == 1 else ""
    tops = {pp[0] for pp in parts}
    if len(tops) != 1:
        return ""
    seconds = {pp[1] for pp in parts if len(pp) >= 3}
    return seconds.pop() if len(seconds) == 1 else ""


def conventional_commit_subject(changes: Sequence[tuple[str, str]], *, summary: str) -> str:
    """Derive a Conventional Commits subject from status and path pairs, with no model call.

    All tests is `test`, all docs is `docs`, any added file is `feat`, else `fix`;
    `chore` when nothing changed.

    Args:
        changes: The (status, path) pairs.
        summary: The change's summary; its head is lowercased and a trailing period dropped.

    Returns:
        The subject, capped at the subject limit.
    """
    paths = [p for _, p in changes]
    if not paths:
        ctype = "chore"
    elif all(_is_testish(p) for p in paths):
        ctype = "test"
    elif all(_is_docish(p) for p in paths):
        ctype = "docs"
    elif any(status.startswith("A") for status, _ in changes):
        ctype = "feat"
    else:
        ctype = "fix"
    scope = _conventional_scope(paths)
    head = f"{ctype}({scope}): " if scope else f"{ctype}: "
    subject = " ".join(summary.split()).rstrip(".")
    subject = (subject[:1].lower() + subject[1:]) if subject else "update"
    return (head + subject)[:_SUBJECT_LIMIT]


def _headline_subject(task: str, *, limit: int = _SUBJECT_LIMIT) -> str:
    """Return a subject from the task's first clause, capped at the limit with an ellipsis."""
    first_line = _ITER_SUBJECT_RE.sub("", task_headline(task)).strip()
    match = re.search(r"[.!?](?:\s|$)", first_line)
    clause = first_line[: match.start()] if match else first_line
    clause = " ".join(clause.split())
    if len(clause) <= limit:
        return clause
    return clause[: limit - 1].rstrip() + "…"
