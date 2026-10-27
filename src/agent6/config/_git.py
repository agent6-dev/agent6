# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git]` model.

Worktree policy, the run's detached chain, merge and message styles.
"""

from __future__ import annotations

import re
import string
from typing import Literal

import pydantic

from agent6.config import _base


class GitCommitCheckpointConfig(pydantic.BaseModel):
    """Message style for the per-turn commits a run makes on its branch."""

    model_config = _base.MODEL_CONFIG

    message: Literal["agent6", "conventional", "model"] = pydantic.Field(
        default="agent6",
        description=(
            "The message of each per-turn commit: `agent6` (`agent6 turn N: <summary>`), "
            "`conventional` (a `type(scope): subject` derived from the diff, no model call), or "
            "`model` (the model writes it from the git facts, falling back to `agent6` with a "
            "warning on any failure)."
        ),
    )


class GitCommitSquashConfig(pydantic.BaseModel):
    """Message style for the one commit a squash merge produces."""

    model_config = _base.MODEL_CONFIG

    message: Literal["agent6", "conventional", "combine", "model"] = pydantic.Field(
        default="agent6",
        description=(
            "The message of the one commit a squash merge produces: `agent6` (the task's first "
            "clause as the subject, the finish summary as the body, else the steps as bullets), "
            "`conventional` (the same body under a `type(scope): subject` derived from the diff, "
            "no model call), `combine` (git's own squash message: the per-step log concatenated), "
            "or "
            "`model` (model-written from up to 64 KiB of the base-to-tip diff, marked when "
            "truncated, with the task and execution finish "
            "summaries as context and commit conventions from AGENTS.md; falls back to `agent6` "
            "with a warning on any failure)."
        ),
    )


class GitCommitConfig(pydantic.BaseModel):
    """The commit identity, the provenance trailer and the per-kind message styles.

    `name` and `email` unset mean the project's own `git config` identity; a run with
    neither an override nor a resolvable identity refuses at startup.
    """

    model_config = _base.MODEL_CONFIG

    name: str | None = pydantic.Field(
        default=None,
        description=(
            "author and committer name for agent6 commits; also passed as `GIT_AUTHOR_NAME` and "
            '`GIT_COMMITTER_NAME` to jailed `git commit` under `git.control = "model"`; repo\'s '
            "own `git config` identity when unset; startup refused without a resolvable "
            "identity"
        ),
    )
    email: str | None = pydantic.Field(
        default=None,
        description=(
            "author and committer email for agent6 commits; also passed as `GIT_AUTHOR_EMAIL` "
            'and `GIT_COMMITTER_EMAIL` to jailed `git commit` under `git.control = "model"`; '
            "the `GIT_COMMITTER_*` variables, then the repo's own `git config` identity, when"
            " unset; startup refused without a resolvable identity"
        ),
    )
    trailer: str = pydantic.Field(
        default="",
        description=(
            'git trailer line (`Key: value`) on every agent6 commit, such as `"Assisted-by: '
            'agent6:{model}"` or `"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`; '
            "`{model}`: the code-writing model, several joined with `, `; no trailer when empty"
        ),
    )
    checkpoint: GitCommitCheckpointConfig = pydantic.Field(
        default_factory=GitCommitCheckpointConfig
    )
    squash: GitCommitSquashConfig = pydantic.Field(default_factory=GitCommitSquashConfig)

    @pydantic.field_validator("trailer")
    @classmethod
    def _trailer_is_a_trailer_line(cls, v: str) -> str:
        """Refuse a trailer that is not a `Key: value` line with only `{model}` as a placeholder.

        Args:
            v: The trailer template.

        Returns:
            The template unchanged.

        Raises:
            ValueError: An unknown placeholder, or the rendered line is not a trailer.
        """
        if not v:
            return v
        fields = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
        unknown = fields - {"model"}
        if unknown:
            raise ValueError(
                f"unknown placeholder {sorted(unknown)} in git.commit.trailer (known: {{model}})"
            )
        rendered = v.format(model="m")
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*: .+", rendered, re.DOTALL):
            raise ValueError(
                'git.commit.trailer must be a git trailer line, "Key: value"'
                ' (e.g. "Assisted-by: agent6:{model}")'
            )
        return v


class GitConfig(pydantic.BaseModel):
    """The `[git]` table."""

    model_config = _base.MODEL_CONFIG

    # A run records the untracked files present at its start and never commits them.
    dirty_tree: Literal["ask", "stash", "include"] = pydantic.Field(
        default="ask",
        description=(
            "handling of tracked files' uncommitted changes at run start; `git.dirty_tree = "
            '"ask"`: an `ask_user` choice of stash for the run, include in its commits, or '
            "cancel (park for later resume), startup refused without someone to answer; "
            '`git.dirty_tree = "stash"`: stash without asking, restored at the end per '
            "`git.auto_stash_pop`, otherwise a printed restore command naming the stash by sha; "
            "worktree restore command: its directory, no branch switch; `git.dirty_tree = "
            '"include"`: start without asking, changes in the first commit, an unmerged run '
            "marked merged on landing if its work matches those changes exactly; untracked "
            "files never counted or committed; `--parallel` allowed with `git.dirty_tree = "
            '"stash"` or `"include"`, refused with `"ask"`; same policy on resume for files '
            "changed since the last execution, stash refused if the run also changed one of "
            "them"
        ),
    )
    auto_stash_pop: bool = pydantic.Field(
        default=False,
        description=(
            "pre-run stash applied over the working tree at run or resume end, even over the "
            "run's uncommitted work; worktree sessions restored in their own directory without "
            "a branch switch; stash dropped only after a clean apply; failed apply: stash "
            "preserved, restore command printed, conflicts visible in `git status`; no `reset "
            '--hard`; requires `git.dirty_tree = "stash"`'
        ),
    )
    # The chain is refs/agent6/<session>/head, parented on HEAD at run start.
    control: Literal["agent6", "model"] = pydantic.Field(
        default="agent6",
        description=(
            'git ownership during a run; `git.control = "agent6"`: each editing turn on the '
            'run\'s own commit chain and branch, HEAD untouched; `git.control = "model"`: '
            "model-managed commits and branches, no per-turn chain or run branch, `sessions "
            "diff`/`merge`, `/undo` and `fork` refused; with model control: "
            '`sandbox.protect_git = "auto"` disabled, `sandbox.protect_git = "on"` refused'
        ),
    )
    branch_per_run: bool = pydantic.Field(
        default=True,
        description=(
            "visible `agent6/<run-id>` branch advanced to the run's chain tip; "
            "`git.branch_per_run = false`: only the hidden `refs/agent6/<run-id>/head` ref; "
            "forced on for `--parallel` lanes, whose work is imported by branch"
        ),
    )
    # Off, resume still works from snapshots; the step-history surfaces degrade.
    commit_per_turn: bool = pydantic.Field(
        default=True,
        description=(
            "each editing turn committed to the run's detached chain through a temp index; HEAD "
            "never moved; index and working tree touched only with the run's own branch checked "
            "out; `git.commit_per_turn = false`: no agent6 commits, work only in the worktree, "
            "degraded resume-from-git, `sessions diff`/`merge` and `/parallel` dispatch from a "
            "changed tree"
        ),
    )
    merge_method: Literal["squash", "merge", "ff_only"] = pydantic.Field(
        default="squash",
        description=(
            "method for `agent6 sessions merge` to land a run on its base; `git.merge_method = "
            '"squash"`: one commit; `git.merge_method = "merge"`: a merge commit keeping '
            'per-step history; `git.merge_method = "ff_only"`: fast-forward only; consolidation '
            "only, per-turn commits always on the run's chain"
        ),
    )
    # With auto_stash_pop the merge lands first, then the stash goes back on top.
    auto_merge: bool = pydantic.Field(
        default=False,
        description=(
            "automatic merge into the base branch only for end reason `finish_session` with a "
            "passed or absent verify gate; other ends: no auto-merge, reason printed; hidden "
            "chain ref merged with `git.branch_per_run = false`; conflict: nothing moved, "
            "instructions printed"
        ),
    )
    # The hidden chain ref stays as the run's record until `sessions rm`.
    auto_prune: bool = pydantic.Field(
        default=False,
        description=(
            "run branch deleted after auto-merge only when `git branch -d` allows it "
            '(`git.merge_method = "merge"` or `"ff_only"`); squash-merged branch: `-D` command '
            "printed, never force-deleted; requires `git.auto_merge = true`; no action without "
            "a run branch"
        ),
    )
    # A hook runs on the host outside the jail: host RCE for an adversarial repo.
    # `core.fsmonitor` and `diff.external` fire on status/diff and have no legitimate use here.
    run_repo_hooks: bool = pydantic.Field(
        default=False,
        description=(
            "repo's `.git/hooks/*` during agent6 git operations; `git.run_repo_hooks = false`: "
            "hooks skipped to prevent repo-controlled code running on the host; "
            "`core.fsmonitor` and `diff.external` always neutralized"
        ),
    )
    # A driver in a poisoned `.git/config` runs on the host at every stage or merge.
    run_repo_drivers: bool = pydantic.Field(
        default=False,
        description=(
            "repo's filter and merge drivers (`filter.<name>.clean/smudge/process`, "
            "`merge.<name>.driver`) during agent6 git operations; `git.run_repo_drivers = "
            "false`: each driver neutralized by name to prevent repo-controlled `.git/config` "
            "code running on the host at every commit; `git.run_repo_drivers = true` required "
            "for Git LFS clean/smudge filters"
        ),
    )
    commit: GitCommitConfig = pydantic.Field(default_factory=GitCommitConfig)

    @pydantic.model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        """Refuse the stash and prune settings without the setting each depends on.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `auto_stash_pop` without `dirty_tree = "stash"`, or `auto_prune` without
                `auto_merge`.
        """
        if self.auto_stash_pop and self.dirty_tree != "stash":
            raise ValueError(
                'git.auto_stash_pop requires git.dirty_tree = "stash": with nothing stashed '
                "pre-run there is nothing to restore at run end."
            )
        if self.auto_prune and not self.auto_merge:
            raise ValueError(
                "git.auto_prune requires git.auto_merge: pruning a run branch only makes "
                "sense once it has been merged."
            )
        return self
