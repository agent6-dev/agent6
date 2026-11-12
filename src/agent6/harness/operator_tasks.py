# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Write the tasks the operator gives a run to its task graph.

The root task a fresh run starts on, the standing goal (`run --standing`, `/standing`) and
what `/task` and `/retire` queue. Nothing enters the conversation: the turn in flight never
sees a queued request, and the next turn's focus banner names the work.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pydantic

from agent6 import task_text
from agent6.graph import curator as graph_curator
from agent6.graph import models, order
from agent6.sessions import ipc, journal


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorTasks:
    """The operator's writes to the run's task graph.

    Attributes:
        curator: The graph, or None for a run without one.
        take_requests: The bridge's queue: `/task`, `/standing`, `/retire`, or `agent6 steer`
            with the same words.
        log: The run's text logger.
        emit: The run's event sink.
        emit_graph_snapshot: Publishes the graph after a write.
    """

    curator: graph_curator.GraphCurator | None
    take_requests: Callable[[], list[ipc.OperatorRequest]]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def take(self, root_task_id: str | None) -> bool:
        """Apply what the operator asked since the last turn boundary, in the order asked.

        A refused request is logged and emitted (`task.request_refused`); the rest still land.

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
            except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
                self.log(f"LOOP: {request.kind} request refused: {exc}")
                self.emit(
                    journal.TaskRequestRefused(
                        kind=request.kind,
                        text=request.text[:200],
                        error=str(exc),
                    )
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
        title = task_text.clip_title(task_text.task_headline(user_task)) or "(run)"
        try:
            draft = models.TaskNodeDraft(
                title=title,
                rationale="single-loop run; root task seeded by Harness",
                acceptance="",
                relevant_paths=(),
                created_by="user",
            )
            node = self.curator.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=draft))
            return node.id
        except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
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
        except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
            self.log(f"LOOP: standing goal not seeded: {exc}")

    def queue(self, root_id: str, text: str) -> None:
        """Queue a task as the root's last ordinary child, reached once the open work drains.

        The title is the operator's first line; the whole text is the rationale.

        Args:
            root_id: The run's root task.
            text: The task as the operator typed it.
        """
        title = task_text.clip_title(task_text.task_headline(text) or text.strip())
        node = self._graph.add_subtask(
            models.AddSubtaskIntent(
                parent_id=root_id,
                draft=models.TaskNodeDraft(
                    title=title,
                    rationale=text if text.strip() != title else "",
                    created_by="user",
                ),
            )
        )
        self.log(f"LOOP: operator queued task {node.id}: {title}")
        self.emit(journal.TaskQueued(task_id=node.id, title=title))

    def retire(self, task_id: str) -> None:
        """Retire a task the operator no longer wants: obsolete, not skipped.

        The route is the curator itself, so a task the operator queued is retirable here even
        though `update_task` refuses it to the model.

        Args:
            task_id: The task to retire.
        """
        node = self._graph.update_status(
            models.UpdateStatusIntent(
                id=task_id, new_status="obsolete", note="retired by the operator"
            )
        )
        self.log(f"LOOP: operator retired task {node.id}")
        self.emit(journal.TaskRetired(task_id=node.id, title=node.title))

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
            if node.standing and node.status in order.OPEN_STATUSES:
                curator.update_status(
                    models.UpdateStatusIntent(
                        id=node.id, new_status="obsolete", note="replaced by the operator"
                    )
                )
                self.log(f"LOOP: standing goal {node.id} retired for a new one")
        node = curator.add_subtask(
            models.AddSubtaskIntent(
                parent_id=root_id,
                draft=models.TaskNodeDraft(title=goal, standing=True, created_by="steering"),
            )
        )
        self.log(f"LOOP: standing goal set: {node.id}")
        self.emit(journal.StandingSet(task_id=node.id, title=goal))

    @property
    def _graph(self) -> graph_curator.GraphCurator:
        """The curator, for a write that needs one.

        Raises:
            CuratorError: The run has no task graph.
        """
        if self.curator is None:
            raise graph_curator.CuratorError("this run has no task graph")
        return self.curator
