# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's checkpoints: the per-step commit on its chain, with the subject
`[git.commit.checkpoint].message` asks for and the events every fold counts
a commit by, the report of a commit that failed, and the final checkpoint a
run's end takes of a dirty tree. The loop decides when a step commits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent6.commit_message import agent6_subject, conventional_commit_subject, first_prose_line
from agent6.git_ops import GitError, commit_diff
from agent6.git_ops import status as git_status
from agent6.providers import Provider, call_for_text
from agent6.workflows._chain import RunChain

if TYPE_CHECKING:
    from agent6.workflows._loop_state import TurnState


@dataclass(frozen=True, slots=True)
class Checkpoints:
    """`style` is `[git.commit.checkpoint].message`; `enabled` is run mode
    (plan and ask never commit); `provider` drafts the `model` style."""

    chain: RunChain
    style: str
    enabled: bool
    provider: Provider
    log: Callable[[str], None]
    emit: Callable[..., None]

    def subject(self, turn: TurnState, *, fallback: str) -> str:
        """The step's commit subject in the configured style; the `model`
        style degrades to the agent6 style, with a warning, when the draft
        fails."""
        text = turn.resp.text or ""
        default = agent6_subject(text, turn.iteration, fallback=fallback)
        if self.style == "agent6":
            return default
        summary = first_prose_line(text, fallback=fallback)
        changes = self.chain.name_status()
        if self.style == "conventional":
            return conventional_commit_subject(changes, summary=summary)
        msg = self._model_subject(changes, hint=summary)
        if msg:
            return msg
        self.log("WARNING: model commit message failed; using the agent6 style")
        return default

    def commit(self, subject: str, *, iteration: int, label: str = "auto-commit") -> str:
        """The step's commit, with the `loop.auto_commit` and `diff.updated`
        events a live viewer and every fold read it from; "" when the tree
        held nothing new (then no event claims a commit that never happened).
        A git or filesystem fault raises."""
        sha = self.chain.commit(subject)
        if not sha:
            return ""
        self.log(f"  {label}: {sha[:12]}")
        self.emit("loop.auto_commit", iteration=iteration, sha=sha, subject=subject)
        # The commit is COUNTED by this event: every fold tallies commits and
        # the latest diff from diff.updated alone. Capped; best-effort.
        self.emit("diff.updated", sha=sha, patch=commit_diff(self.chain.root, sha, max_bytes=8000))
        return sha

    def report_failure(self, exc: GitError | OSError, subject: str, *, iteration: int) -> None:
        """Log and emit a commit that failed, with a worktree status snapshot
        so the event says what the tree held. A "nothing to commit" variant
        is benign and stays silent: the phrase arrives in either half of the
        detail string, "no changes added" covers only-ignored changes, and
        "working tree clean" a green verify without a file mutation."""
        msg = str(exc).lower()
        if "nothing to commit" in msg or "no changes added" in msg or "working tree clean" in msg:
            return
        self.log(f"  auto-commit failed: {exc}")
        worktree_status = ""
        try:
            st = git_status(self.chain.root, exclude=self.chain.untracked_at_start)
            worktree_status = (
                f"branch={st.branch}"
                f" head={st.head_sha[:12]}"
                f" clean={st.is_clean}"
                f" modified={st.modified_count}"
                f" untracked={st.untracked_count}"
            )
        except (GitError, OSError):
            pass  # the status itself failed: the event goes without the snapshot
        self.emit(
            "loop.auto_commit.failed",
            iteration=iteration,
            error=str(exc)[:2000],
            worktree_status=worktree_status,
            commit_subject=subject[:200],
        )

    def final(self, *, iteration: int) -> None:
        """A dirty tree at a successful exit commits, so a run_command-authored
        edit after the last green verify (which the per-step commit never
        saw) stays in git history, where resume, the diff viewer and the
        scorers read. Best-effort."""
        if not self.enabled or not self.chain.per_step or not self.chain.dirty():
            return
        try:
            self.commit(
                f"checkpoint (iter {iteration})", iteration=iteration, label="final checkpoint"
            )
        except (GitError, OSError) as exc:
            self.log(f"  final checkpoint commit failed: {exc}")

    def _model_subject(self, changes: Sequence[tuple[str, str]], *, hint: str) -> str | None:
        """A model-drafted subject from git facts only; None on any failure."""
        listing = "\n".join(f"{s}\t{p}" for s, p in changes[:200])
        return call_for_text(
            self.provider,
            system=(
                "Write a git commit message for the change set: one"
                " imperative subject line under 72 characters, optionally a"
                " blank line and a short body. Use only the facts given."
                " Output the message text only."
            ),
            user=f"Summary hint: {hint}\nChanged files (status\tpath):\n{listing}",
            max_tokens=400,
        )
