# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git]` model.

Worktree policy, the run's detached chain, merge and message styles, and the worktrees
agent6 makes.
"""

from __future__ import annotations

import re
import string
from typing import Literal

import pydantic

from agent6.config import validation


class GitCommitConfig(pydantic.BaseModel):
    """The commit identity, the provenance trailer and the two message styles.

    `name` and `email` unset read `GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL`, then
    the project's own `git config` identity; a run with
    neither an override nor a resolvable identity refuses at startup.
    """

    model_config = validation.MODEL_CONFIG

    name: str | None = pydantic.Field(
        default=None,
        description=(
            "author and committer name for agent6 commits; also passed as `GIT_AUTHOR_NAME` and "
            '`GIT_COMMITTER_NAME` to jailed `git commit` under `git.control = "model"`; '
            "`GIT_COMMITTER_NAME`, then the repo's own `git config` identity, when unset; "
            "startup refused without a resolvable identity"
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
    checkpoint_message: Literal["agent6", "conventional", "model"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "agent6": "Generate turn number and summary message",
                "conventional": "Derive conventional subject from diff without model",
                "model": "Ask model to write commit message",
            }
        },
        default="agent6",
        description=(
            "Per-turn commit message style; generated messages use `agent6 turn N: <summary>` "
            "or `type(scope): subject`; model messages use git facts and fall back to the "
            "generated agent6 message with a warning on failure"
        ),
    )
    squash_message: Literal["agent6", "conventional", "combine", "model"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "agent6": "Generate message from task and run summary",
                "conventional": "Derive conventional subject from diff without model",
                "combine": "Concatenate the git commit log",
                "model": "Ask model to write commit message",
            }
        },
        default="agent6",
        description=(
            "Squash commit message style; generated messages use the task's first clause, "
            "then `continued: <steer>` for a run resumed with `--steer`, with the finish "
            "summary as body or commit subjects as bullets; conventional subjects use "
            "`type(scope): subject` with the same body; model messages use up to 64 KiB of "
            "base-to-tip diff (marked if truncated), the task, continuation and execution "
            "finish summaries, and AGENTS.md commit conventions; failure warns and falls "
            "back to the generated agent6 message"
        ),
    )

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


class GitWorktreeConfig(pydantic.BaseModel):
    """The `[git.worktree]` table: where agent6 puts the worktrees it makes, and their setup."""

    model_config = validation.MODEL_CONFIG

    # A lane: `<base_dir>/<repo-id>/<fanout-id>/lane-<i>`; a fork or a `run --worktree` run:
    # `<base_dir>/<repo-id>/<session-id>`.
    base_dir: str = pydantic.Field(
        default="",
        description=(
            "base directory for the worktrees agent6 makes, grouped by repository: lane and "
            "machine-state clones, fork and `run --worktree` worktrees; empty: "
            "`<cache_dir>/worktrees`; lane clone removed after work import; fork and `run "
            "--worktree` worktrees removed by `sessions prune` once merged or holding no work"
        ),
    )
    # Measured on a 1,546-file repository: a clone takes 0.6 to 0.8 s, a linked worktree 0.3 to
    # 0.7 s; mirroring one commit into the repository takes 20 ms.
    kind: Literal["clone", "linked"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "clone": "Clone of the repository with its own git data",
                "linked": "Linked git worktree sharing the repository's git data",
            }
        },
        default="clone",
        description=(
            "Worktree a fork or a `run --worktree` run works in; a clone runs at any path, in "
            "any container, and agent6 copies its commits into the repository as the run makes "
            "them; a linked worktree is smaller for a large repository whose base is on "
            "another filesystem, commits straight into the repository, and under a container "
            "needs the workspace mounted at its host path; lanes and review seats are clones."
        ),
    )
    setup_command: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "environment-setup command for a new worktree as argv, no shell (pipeline "
            'form: `["sh", "-c", "a && b"]`); `run --worktree` and fork worktrees, lane and '
            "machine-state clones: tracked files only, no untracked `.venv` or `node_modules`; "
            "command run once in each worktree before the first turn, under the run's command "
            "sandbox (same isolation, network and grants); nonzero exit or 600-second timeout: "
            "session ended before any turn; unset: AGENTS.md `## Setup command` fenced block, "
            "parsed like `## Verify command`, command printed; no configured or inferred "
            "command: nothing run"
        ),
    )


class GitConfig(pydantic.BaseModel):
    """The `[git]` table."""

    model_config = validation.MODEL_CONFIG

    # A run records the untracked files present at its start and never commits them.
    dirty_tree: Literal["ask", "stash", "include"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "ask": "Ask whether to stash, include or cancel",
                "stash": "Stash tracked changes for the run",
                "include": "Include tracked changes in first commit",
            }
        },
        default="ask",
        description=(
            "Tracked files' uncommitted changes at run start; asking requires someone to "
            "answer, with cancellation parking the run for resume; stashes are restored per "
            "`git.auto_stash_pop`, otherwise a printed restore command names the stash by "
            "sha and the worktree directory without switching branches; included changes "
            "mark an unmerged run merged on landing if its work matches exactly; files "
            "already untracked at the start stay out of the run's commits, and files the run "
            "creates are its own; `--parallel` refuses asking; a resume applies the same "
            "policy to changes since the last execution, the run's own files included, "
            "refuses stash if the run also changed a file, and names included files to the "
            "model"
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
        json_schema_extra={
            "choice_meanings": {
                "agent6": "Harness owns per-turn commits and run branch",
                "model": "Model owns commits and branches",
            }
        },
        default="agent6",
        description=(
            "Git ownership during a run; harness control leaves HEAD untouched; model control "
            "has no per-turn chain or run branch and refuses `sessions diff`/`merge`, `/undo` "
            "and `fork`; automatic git protection is disabled and explicit protection refused."
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
            "degraded resume-from-git and `sessions diff`/`merge`; `/parallel` dispatch from a "
            "changed tree refused; machine run states require true because only the committed "
            "chain leaves a clone"
        ),
    )
    merge_method: Literal["tasks", "squash", "merge", "ff_only"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "tasks": "Squash consecutive checkpoints by task",
                "squash": "Land as one commit",
                "merge": "Merge while preserving per-turn history",
                "ff_only": "Require a fast-forward merge",
            }
        },
        default="tasks",
        description=(
            "Method for `agent6 sessions merge` to land a run on its base; a task's commit "
            "follows `git.commit.squash_message` under the `model` style with the subtask's "
            "title and acceptance as its body, else the subtask's title; checkpoints outside "
            "subtasks follow `git.commit.squash_message`, so a run that never decomposed lands "
            "as one commit; "
            "consolidation only, per-turn commits stay on the run's chain; `sessions merge "
            "-m` lands one commit with that message"
        ),
    )
    # With auto_stash_pop the merge lands first, then the stash goes back on top.
    auto_merge: bool = pydantic.Field(
        default=False,
        description=(
            "automatic merge into the base branch only for end reason `finish_session`, or "
            "`lanes_ranked` for a `run --parallel` run that joined its best-ranked lane, with a "
            "passed or absent verify gate; other ends: no auto-merge, reason printed; hidden "
            "chain ref merged with `git.branch_per_run = false`; conflict: nothing moved, "
            "instructions printed"
        ),
    )
    # The hidden chain ref stays as the run's record until `sessions remove`.
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
    worktree: GitWorktreeConfig = pydantic.Field(default_factory=GitWorktreeConfig)

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
