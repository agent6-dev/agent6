# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's checkpoints: the per-step commit on its chain.

The subject follows `[git.commit.checkpoint].message`; the events are what
every fold counts a commit by. The loop decides when a step commits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from agent6 import commit_message, git_ops
from agent6.harness import _chain
from agent6.providers import Provider, call_for_text

if TYPE_CHECKING:
    from agent6.harness import _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoints:
    """The checkpoint policy and the sinks a commit reports through.

    Attributes:
        chain: The run's commit chain.
        style: `[git.commit.checkpoint].message`: agent6, conventional or model.
        enabled: True in run mode; plan and ask never commit.
        provider: The provider that drafts the model style's subject.
        log: The run log line sink.
        emit: The run event sink.
    """

    chain: _chain.RunChain
    style: str
    enabled: bool
    provider: Provider
    log: Callable[[str], None]
    emit: Callable[..., None]

    def subject(self, turn: _loop_state.TurnState, *, fallback: str) -> str:
        """Return the step's commit subject in the configured style.

        The model style degrades to the agent6 style, with a warning, when the
        draft fails.

        Args:
            turn: The turn whose response text names the step.
            fallback: The subject when the response holds no prose.

        Returns:
            The subject line.
        """
        text = turn.resp.text or ""
        default = commit_message.agent6_subject(text, turn.iteration, fallback=fallback)
        if self.style == "agent6":
            return default
        summary = commit_message.first_prose_line(text, fallback=fallback)
        changes = self.chain.name_status()
        if self.style == "conventional":
            return commit_message.conventional_commit_subject(changes, summary=summary)
        msg = self._model_subject(changes, hint=summary)
        if msg:
            return msg
        self.log("WARNING: model commit message failed; using the agent6 style")
        return default

    def commit(self, subject: str, *, iteration: int, label: str = "auto-commit") -> str:
        """Commit the step and emit the two events every fold counts a commit by.

        Args:
            subject: The commit subject.
            iteration: The loop iteration the events carry.
            label: The log line's label.

        Returns:
            The commit's sha, or "" when the tree held nothing new (then no event
            claims a commit); a git or filesystem fault propagates.
        """
        sha = self.chain.commit(subject)
        if not sha:
            return ""
        self.log(f"  {label}: {sha[:12]}")
        self.emit("loop.auto_commit", iteration=iteration, sha=sha, subject=subject)
        # Every fold tallies commits and the latest diff from diff.updated alone.
        self.emit(
            "diff.updated", sha=sha, patch=git_ops.commit_diff(self.chain.root, sha, max_bytes=8000)
        )
        return sha

    def report_failure(
        self, exc: git_ops.GitError | OSError, subject: str, *, iteration: int
    ) -> None:
        """Log and emit a commit that failed, with a worktree status snapshot.

        A "nothing to commit" variant is benign and stays silent: the phrase arrives
        in either half of the detail string, "no changes added" covers only-ignored
        changes, and "working tree clean" a green verify without a file mutation.

        Args:
            exc: The fault the commit raised.
            subject: The subject the commit was to carry.
            iteration: The loop iteration the event carries.
        """
        msg = str(exc).lower()
        if "nothing to commit" in msg or "no changes added" in msg or "working tree clean" in msg:
            return
        self.log(f"  auto-commit failed: {exc}")
        worktree_status = ""
        try:
            st = git_ops.status(self.chain.root, exclude=self.chain.untracked_at_start)
            worktree_status = (
                f"branch={st.branch}"
                f" head={st.head_sha[:12]}"
                f" clean={st.is_clean}"
                f" modified={st.modified_count}"
                f" untracked={st.untracked_count}"
            )
        except (git_ops.GitError, OSError):
            pass  # the status itself failed: the event goes without the snapshot
        self.emit(
            "loop.auto_commit.failed",
            iteration=iteration,
            error=str(exc)[:2000],
            worktree_status=worktree_status,
            commit_subject=subject[:200],
        )

    def final(self, *, iteration: int) -> None:
        """Commit a dirty tree at a successful exit, best-effort.

        A run_command-authored edit after the last green verify, which the per-step
        commit never saw, stays in git history where resume, the diff viewer and
        the scorers read.

        Args:
            iteration: The loop iteration the checkpoint names.
        """
        if not self.enabled or not self.chain.per_step or not self.chain.dirty():
            return
        try:
            self.commit(
                f"checkpoint (iter {iteration})", iteration=iteration, label="final checkpoint"
            )
        except (git_ops.GitError, OSError) as exc:
            self.log(f"  final checkpoint commit failed: {exc}")

    def _model_subject(self, changes: Sequence[tuple[str, str]], *, hint: str) -> str | None:
        """Return a model-drafted subject from git facts only, or None on any failure.

        Args:
            changes: The (status, path) pairs of the change set.
            hint: The one-line summary the draft is anchored on.

        Returns:
            The drafted message, or None when the call failed.
        """
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
