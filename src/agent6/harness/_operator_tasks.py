# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Write the tasks the operator gives a run to its task graph.

The root task a fresh run starts on, the standing goal (`run --standing`, `/standing`) and
what `/task` and `/retire` queue. Nothing enters the conversation: the turn in flight never
sees a queued request, and the next turn's focus banner names the work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import ValidationError

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent
from agent6.graph.order import OPEN_STATUSES
from agent6.harness._context import load_repo_summary
from agent6.harness._prompt_revision import (
    PromptRevisionError,
    RevisionSettings,
    revise_prompt,
)
from agent6.sessions.ipc import OperatorRequest
from agent6.task_text import task_headline


@dataclass(frozen=True, slots=True)
class OperatorTasks:
    """The operator's writes to the run's task graph.

    Attributes:
        curator: The graph, or None for a run without one.
        take_requests: The bridge's queue: `/task`, `/standing`, `/retire`, or `agent6 steer`
            with the same words.
        revision: The `[prompt].revise_prompt` settings, applied to a queued task as to the
            initial one.
        root: The repository.
        log: The run's text logger.
        emit: The run's event sink.
        emit_graph_snapshot: Publishes the graph after a write.
    """

    curator: GraphCurator | None
    take_requests: Callable[[], list[OperatorRequest]]
    revision: RevisionSettings
    root: Path
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def take(self, root_task_id: str | None) -> bool:
        """Apply what the operator asked since the last turn boundary, in the order asked.

        A refused request is logged and emitted (`loop.request.refused`); the rest still land.

        Args:
            root_task_id: The run's root task, or None before the graph is seeded.

        Returns:
            Whether anything landed.
        """
        if self.curator is None or root_task_id is None:
            return False
        landed = False
        for request in self.take_requests():
            try:
                if request.kind == "task":
                    self.queue(root_task_id, request.text)
                elif request.kind == "standing":
                    self.set_standing(root_task_id, request.text)
                else:
                    self.retire(request.text)
            except (CuratorError, OSError, ValidationError) as exc:
                self.log(f"LOOP: {request.kind} request refused: {exc}")
                self.emit(
                    "loop.request.refused",
                    kind=request.kind,
                    text=request.text[:200],
                    error=str(exc),
                )
                continue
            landed = True
        if landed:
            self.emit_graph_snapshot()
        return landed

    def seed_root(self, user_task: str) -> str | None:
        """Seed the run's root task, the user's task itself.

        The model's `add_task` calls with `parent_id=None` attach under it.

        Args:
            user_task: The task as the operator gave it.

        Returns:
            The root's id, or None without a curator or when the graph refused the write.
        """
        if self.curator is None:
            return None
        # TaskNodeDraft.title has min_length=1: "(run)" when the task is blank.
        title = task_headline(user_task)[:200] or "(run)"
        try:
            draft = TaskNodeDraft(
                title=title,
                rationale="single-loop run; root task seeded by Harness",
                acceptance="",
                relevant_paths=(),
                created_by="user",
            )
            node = self.curator.add_subtask(AddSubtaskIntent(parent_id=None, draft=draft))
            return node.id
        except (CuratorError, OSError, ValidationError) as exc:
            self.log(f"LOOP: failed to seed root task: {exc}")
            return None

    def seed_standing(self, root_id: str, goal: str) -> None:
        """Seed the operator's `run --standing` goal, as `/standing` sets one.

        Only a fresh run seeds one; `/standing` changes the goal of a run already going.

        Args:
            root_id: The run's root task.
            goal: The goal's text; blank seeds nothing.
        """
        goal = goal.strip()
        if not goal or self.curator is None:
            return
        try:
            self.set_standing(root_id, goal)
        except (CuratorError, OSError, ValidationError) as exc:
            self.log(f"LOOP: standing goal not seeded: {exc}")

    def queue(self, root_id: str, text: str) -> None:
        """Queue a task as the root's last ordinary child, reached once the open work drains.

        The title stays the operator's own first line even when the revision rewrites the body,
        so the task tree reads in their words; the whole text is the rationale.

        Args:
            root_id: The run's root task.
            text: The task as the operator typed it.
        """
        title = task_headline(text)[:200] or text.strip()[:200]
        spec = self._revised(text)
        node = self._graph.add_subtask(
            AddSubtaskIntent(
                parent_id=root_id,
                draft=TaskNodeDraft(
                    title=title,
                    rationale=spec if spec.strip() != title else "",
                    created_by="user",
                ),
            )
        )
        self.log(f"LOOP: operator queued task {node.id}: {title}")
        self.emit("loop.task.queued", id=node.id, title=title)

    def retire(self, task_id: str) -> None:
        """Retire a task the operator no longer wants: obsolete, not skipped.

        The route is the curator itself, so a task the operator queued is retirable here even
        though `update_task` refuses it to the model.

        Args:
            task_id: The task to retire.
        """
        node = self._graph.update_status(
            UpdateStatusIntent(id=task_id, new_status="obsolete", note="retired by the operator")
        )
        self.log(f"LOOP: operator retired task {node.id}")
        self.emit("loop.task.retired", id=node.id, title=node.title)

    def set_standing(self, root_id: str, goal: str) -> None:
        """Set the standing goal, retiring the one it replaces.

        Retired, not made ordinary: a goal reads as an activity ("keep hunting for defects"),
        and an ordinary task of that shape is worked once and marked passed.

        Args:
            root_id: The run's root task.
            goal: The goal's text.
        """
        curator = self._graph
        for node in curator.nodes().values():
            if node.standing and node.status in OPEN_STATUSES:
                curator.update_status(
                    UpdateStatusIntent(
                        id=node.id, new_status="obsolete", note="replaced by the operator"
                    )
                )
                self.log(f"LOOP: standing goal {node.id} retired for a new one")
        node = curator.add_subtask(
            AddSubtaskIntent(
                parent_id=root_id,
                draft=TaskNodeDraft(title=goal, standing=True, created_by="steering"),
            )
        )
        self.log(f"LOOP: standing goal set: {node.id}")
        self.emit("loop.standing.set", id=node.id, title=goal)

    @property
    def _graph(self) -> GraphCurator:
        """The curator, for a write that needs one.

        Raises:
            CuratorError: The run has no task graph.
        """
        if self.curator is None:
            raise CuratorError("this run has no task graph")
        return self.curator

    def _revised(self, text: str) -> str:
        """Run a queued task through `[prompt].revise_prompt` when the operator turned it on.

        The same one-shot pass the initial task gets. Nobody is at a terminal at a turn
        boundary, so `interactive` revises the way `auto` does; a failed pass keeps the text as
        written.

        Args:
            text: The task as the operator typed it.

        Returns:
            The revised text folded with the original, or the text unchanged.
        """
        if self.revision.mode == "off":
            return text
        try:
            return revise_prompt(
                replace(self.revision, mode="auto"),
                text,
                load_repo_summary(self.root),
                log=self.log,
                emit=self.emit,
            )
        except PromptRevisionError as exc:
            self.log(f"LOOP: queued task kept as written: {exc}")
            return text
