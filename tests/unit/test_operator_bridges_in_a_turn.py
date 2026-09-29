# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the operator wrote reaches the graph from inside a running turn.

`/task`, `/retire` and `/standing` each write a file that the loop drains at
its pre-turn boundary, before the focus is computed. Every other test calls
those drains directly; this one runs the loop over a real curator and a real
journal, with the files written mid-run, so the hot path every turn takes is
exercised end to end.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

from agent6 import events as agent6_events
from agent6.graph import curator as graph_curator
from agent6.harness import _chain, _operator, loop
from agent6.providers import types as providers_types
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import results


def _repo(path: pathlib.Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True, capture_output=True
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> providers_types.ProviderResponse:
    block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
    return providers_types.ProviderResponse(
        text="",
        tool_uses=({"id": call_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [block]},
    )


def test_a_running_turn_takes_the_task_the_retirement_and_the_goal(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    head = _repo(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / ".agent6", session_id="live-run-AAAAAA"
    )
    curator = graph_curator.GraphCurator(layout)
    session_dir = layout.session_dir
    events = agent6_events.EventSink(session_dir / "logs.jsonl")

    provider = mock.MagicMock()
    dispatcher = mock.MagicMock()
    dispatcher.dispatch.return_value = results.RawResult({"content": "hi\n"})

    def _turn(*_args: Any, **_kwargs: Any) -> providers_types.ProviderResponse:
        """Between the turns the operator writes to all three bridges."""
        if provider.call.call_count == 1:
            ipc.queue_request(session_dir, "task", "Add a --json flag to the stats report")
            ipc.queue_request(session_dir, "standing", "keep the suite green")
        elif provider.call.call_count == 2:
            # The task queued above is 0002, the run's second node.
            ipc.queue_request(session_dir, "retire", "0002")
        return _tool_call("read_file", {"path": "README.md"}, f"t{provider.call.call_count}")

    provider.call.side_effect = _turn

    wf = loop.Harness(
        chain=_chain.RunChain(repo, ref="refs/agent6/bridges", fallback_parent=head),
        config=mock.MagicMock(
            budget=types.SimpleNamespace(max_usd=10.0, max_tokens_fallback=2_000_000),
            prompt=mock.MagicMock(system_prompt_file="", decompose="off", revise_prompt="off"),
            harness=mock.MagicMock(
                standing_patience=-1,
                went_quiet_max_nudges=4,
                loop_guard_kill_threshold=10,
                stagnation_notice_after_s=300.0,
                verify_command=(),
                verify_when="never",
                verify_retries=2,
            ),
            parallel=types.SimpleNamespace(max_lanes=4),
        ),
        provider=provider,
        dispatcher=dispatcher,
        logger=lambda _m: None,
        events=events,
        curator=curator,
        bridge=_operator.OperatorBridge(take_requests=lambda: ipc.drain_requests(session_dir)),
        max_iterations=3,
    )

    wf.run("Implement parse_csv_row")

    nodes = curator.nodes()
    by_title = {n.title: n for n in nodes.values()}
    assert sorted(nodes) == ["0001", "0002", "0003"]
    # The queued task landed under the root and was retired a turn later.
    queued = by_title["Add a --json flag to the stats report"]
    assert queued.created_by == "user" and queued.status == "obsolete"
    # The goal is the operator's, and it sorts last, after the ordinary work.
    goal = by_title["keep the suite green"]
    assert goal.standing and goal.created_by == "steering"
    assert curator.get("0001").children[-1] == goal.id

    kinds = [json.loads(line)["type"] for line in events.path.read_text().splitlines()]
    assert "loop.task.queued" in kinds
    assert "loop.task.retired" in kinds
    assert "loop.standing.set" in kinds
