"""Answer the parent and each detached child through a loopback provider."""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
from collections.abc import Generator
from typing import Any

from bench.handoff import job
from tests import launch_env


def reply(body: dict[str, Any]) -> launch_env.Reply:
    """Read the assigned file or advance the parent's observed queue."""
    messages = body["messages"]
    opening = str(next(m["content"] for m in messages if m["role"] == "user"))
    answers = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if "Collect the six file reports." not in opening:
        name = next(n for n in job.FILES if n in opening)
        if not answers:
            return "read_file", {"path": name}
        return "finish_session", {"summary": answers[-1]["content"].strip()}

    return parent_reply(opening, answers)


def parent_reply(opening: str, answers: list[dict[str, Any]]) -> launch_env.Reply:
    """Use returned child ids to refill the queue, then accept the reports."""
    lists = [a["tasks"] for a in answers if "tasks" in a]
    if not lists:
        return "list_tasks", {}
    tasks = [r for r in lists[0] if r["title"] in job.TITLES]
    started = [child for a in answers if a.get("dispatched") for child in a["children"]]
    ended = {a["id"] for a in answers if a.get("running") is False}
    outstanding = [child for child in started if child not in ended]
    arm = "text" if job.INSTRUCTION["text"] in opening else "id"
    room = job.CHILDREN - len(outstanding)
    if room and (ready := tasks[len(started) : len(started) + room]):
        return [
            ("parallel", {"task": r["title"]} if arm == "text" else {"task_id": r["id"]})
            for r in ready
        ]
    if outstanding:
        return "read_background", {"id": outstanding[0], "wait_s": 60, "tail_lines": 3}
    if len(lists) == 1:
        return "list_tasks", {}
    if answers[-1].get("tasks") and (
        unaccepted := [
            # A text dispatch adds a child node. Mark descendants before their parents.
            r
            for r in reversed(lists[-1])
            if r["title"] in job.TITLES and r["status"] != "passed"
        ]
    ):
        return [("update_task", {"id": r["id"], "status": "passed"}) for r in unaccepted]
    reports = [a["output"] for a in answers if a.get("running") is False]
    return "finish_session", {"summary": "\n".join(reports)}


@contextlib.contextmanager
def profile() -> Generator[pathlib.Path]:
    """Serve a dedicated provider config without opening a live profile."""
    with (
        tempfile.TemporaryDirectory(prefix="handoff-provider-") as temp,
        launch_env.scripted_provider(
            reply_for=reply,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        ) as (url, _calls),
    ):
        directory = pathlib.Path(temp)
        (directory / "config.toml").write_text(
            f'[providers.scripted]\nbackend = "openai"\nauth_style = "none"\nbase_url = "{url}"\n',
            encoding="utf-8",
        )
        yield directory
