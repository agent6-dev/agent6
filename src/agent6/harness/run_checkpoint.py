# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's checkpoints: the per-turn commit on its chain.

The subject follows `[git.commit].checkpoint_message`; the events are what
every fold counts a commit by. The loop decides when a turn commits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from agent6 import errors
from agent6.harness import run_chain
from agent6.providers import Provider, ProviderError, call_for_text
from agent6.sessions import journal
from agent6.vcs import commit_message
from agent6.vcs.git import read as git_read

if TYPE_CHECKING:
    from agent6.harness import loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoints:
    """The checkpoint policy and the sinks a commit reports through.

    Attributes:
        chain: The run's commit chain.
        style: `[git.commit].checkpoint_message`: agent6, conventional or model.
        enabled: True in run mode; plan and ask never commit.
        provider: The provider that drafts the model style's subject.
        log: The run log line sink.
        emit: The run event sink.
    """

    chain: run_chain.RunChain
    style: str
    enabled: bool
    provider: Provider
    log: Callable[[str], None]
    emit: Callable[..., None]

    def subject(self, turn: loop_state.TurnState, *, fallback: str) -> str:
        """Return the step's commit subject in the configured style.

        The model style degrades to the agent6 style, with a warning, when the
        draft fails.

        Args:
            turn: The turn whose response text names the step.
            fallback: The subject when the response holds no prose and no paths changed.

        Returns:
            The subject line.
        """
        text = turn.resp.text or ""
        if self.style == "agent6" and commit_message.first_prose_line(text, fallback=""):
            return commit_message.agent6_subject(text, turn.number, fallback=fallback)
        # The chain's delta names the fallback and feeds the conventional and model styles.
        changes = self.chain.name_status()
        if len(changes) == 1:
            path = changes[0][1]
            fallback = f"update {path if path.isprintable() else repr(path)}"
        elif changes:
            fallback = f"update {len(changes)} files"
        default = commit_message.agent6_subject(text, turn.number, fallback=fallback)
        if self.style == "agent6":
            return default
        if self.style == "conventional":
            # The delta words it: the response's first line is talk, not a subject.
            return commit_message.conventional_commit_subject(changes, summary=fallback)
        try:
            msg = self._model_subject(
                changes, hint=commit_message.first_prose_line(text, fallback=fallback)
            )
        except ProviderError as exc:
            self.log(f"WARNING: model commit message failed ({exc}); using the agent6 style")
            return default
        if msg:
            return msg
        self.log(
            "WARNING: model commit message failed (the model returned nothing);"
            " using the agent6 style"
        )
        return default

    def commit(
        self, subject: str, *, turn: int, label: str = "auto-commit", task_id: str | None = None
    ) -> str:
        """Commit the step and emit the two events every fold counts a commit by.

        Args:
            subject: The commit subject.
            turn: The loop turn the events carry.
            label: The log line's label.
            task_id: The subtask the turn worked on, named in the commit's trailer and the
                event; None when the run has no current subtask.

        Returns:
            The commit's sha, or "" when the tree held nothing new (then no event
            claims a commit); a git or filesystem fault propagates.
        """
        sha = self.chain.commit(subject, task_id=task_id)
        if not sha:
            return ""
        self.log(f"  {label}: {sha[:12]}")
        self.emit(
            journal.CheckpointCommitted(turn=turn, sha=sha, subject=subject, task_id=task_id or "")
        )
        # Every fold tallies commits and the latest diff from diff.updated alone.
        self.emit(
            journal.DiffUpdated(
                sha=sha,
                patch=git_read.commit_diff(self.chain.root, sha, max_bytes=8000),
            )
        )
        return sha

    def report_failure(self, exc: errors.GitError | OSError, subject: str, *, turn: int) -> None:
        """Log and emit a commit that failed, with a worktree status snapshot.

        A "nothing to commit" variant is benign and stays silent: the phrase arrives
        in either half of the detail string, "no changes added" covers only-ignored
        changes, and "working tree clean" a green verify without a file mutation.

        Args:
            exc: The fault the commit raised.
            subject: The subject the commit was to carry.
            turn: The loop turn the event carries.
        """
        msg = str(exc).lower()
        if "nothing to commit" in msg or "no changes added" in msg or "working tree clean" in msg:
            return
        self.log(f"  auto-commit failed: {exc}")
        worktree_status = ""
        try:
            st = git_read.status(self.chain.root, exclude=self.chain.excluded)
            worktree_status = (
                f"branch={st.branch}"
                f" head={st.head_sha[:12]}"
                f" clean={st.is_clean}"
                f" modified={st.modified_count}"
                f" untracked={st.untracked_count}"
            )
        except (errors.GitError, OSError):
            pass  # the status itself failed: the event goes without the snapshot
        self.emit(
            journal.CheckpointFailed(
                turn=turn,
                error=str(exc)[:2000],
                worktree_status=worktree_status,
                commit_subject=subject[:200],
            )
        )

    def final(self, *, turn: int, task_id: str | None = None) -> None:
        """Commit a dirty tree at a successful exit, best-effort.

        A run_command-authored edit after the last green verify, which the per-turn
        commit never saw, stays in git history where resume, the diff viewer and
        the scorers read.

        Args:
            turn: The loop turn the checkpoint names.
            task_id: The subtask the last turn worked on, or None.
        """
        if not self.enabled or not self.chain.per_turn or not self.chain.dirty():
            return
        try:
            self.commit(
                f"checkpoint (turn {turn})", turn=turn, label="final checkpoint", task_id=task_id
            )
        except (errors.GitError, OSError) as exc:
            self.log(f"  final checkpoint commit failed: {exc}")

    def _model_subject(self, changes: Sequence[tuple[str, str]], *, hint: str) -> str | None:
        """Return a model-drafted subject from git facts only.

        Args:
            changes: The (status, path) pairs of the change set.
            hint: The one-line summary the draft is anchored on.

        Returns:
            The drafted message, or None for an empty reply.

        Raises:
            ProviderError: The call failed, with its reason.
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
