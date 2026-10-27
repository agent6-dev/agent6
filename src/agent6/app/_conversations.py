# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The opaque conversation ids and turn state a session's model calls carry.

Each role's calls to one provider form a conversation. Its id is minted here, never
derived from a path, a name or a host, and lives in `conversations.json` beside the
session: a resume continues the same conversations; a fork, which copies none of this,
starts new ones.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import threading
import uuid
from typing import Any

from agent6 import portable
from agent6.providers import _headers

FILE_NAME = "conversations.json"


@dataclasses.dataclass(slots=True)
class _Route:
    """One conversation: its opaque id and how many of its calls have completed."""

    id: str
    completed: int = 0


@dataclasses.dataclass(slots=True)
class Conversations:
    """A session's conversations, one per role and provider.

    Attributes:
        path: Where the state persists; None keeps it in memory for one process.
        is_subagent: The session is a fan-out lane.
    """

    path: pathlib.Path | None = None
    is_subagent: bool = False
    _routes: dict[str, _Route] = dataclasses.field(default_factory=dict)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Load the persisted state; a missing or unreadable file starts empty."""
        if self.path is None or not self.path.is_file():
            return
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
            routes = data.get("routes") if isinstance(data, dict) else None
            if isinstance(routes, dict):
                self._routes = {
                    str(k): _Route(str(v["id"]), int(v.get("completed", 0)))
                    for k, v in routes.items()
                    if isinstance(v, dict) and v.get("id")
                }
        except (OSError, ValueError, TypeError):
            self._routes = {}

    def context(
        self,
        role: str,
        provider: str,
        *,
        initiator: str,
        observe: Any = None,
    ) -> _headers.RequestContext:
        """Return the context of one model call, minting the conversation on first use.

        Args:
            role: The calling role.
            provider: The provider entry's name.
            initiator: `user` or `agent`.
            observe: Receives each HTTP attempt's facts; None observes nothing.

        Returns:
            The context.
        """
        with self._lock:
            route = self._routes.get(f"{role}@{provider}")
            if route is None:
                route = _Route(uuid.uuid4().hex)
                self._routes[f"{role}@{provider}"] = route
                self._persist()
            return _headers.RequestContext(
                session_id=route.id,
                is_first_turn=route.completed == 0,
                is_subagent=self.is_subagent,
                initiator=initiator,
                observe=observe,
            )

    def completed(self, role: str, provider: str) -> None:
        """Record one answered call of a conversation.

        Args:
            role: The calling role.
            provider: The provider entry's name.
        """
        with self._lock:
            route = self._routes.get(f"{role}@{provider}")
            if route is not None:
                route.completed += 1
                self._persist()

    def _persist(self) -> None:
        """Write the state under the lock; nothing without a path."""
        if self.path is not None:
            routes = {k: dataclasses.asdict(r) for k, r in self._routes.items()}
            blob = json.dumps({"routes": routes}, indent=1).encode("utf-8")
            portable.atomic_write(self.path, blob)
