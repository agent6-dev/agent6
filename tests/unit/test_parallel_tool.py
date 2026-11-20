# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The opt-in `parallel` tool: two gates, a resolved description, the dispatcher's own path."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import child_records
from agent6.config import Config
from agent6.harness import (
    loop,
    operator_bridge,
    parallel_join,
    provider_call,
    run_chain,
    toolset,
)
from agent6.providers import ProviderResponse
from agent6.sessions import child_dispatch, journal
from agent6.tools import child_tools, dispatch, schema, tool_errors
from agent6.tools import results as tools_results
from tests import child_env


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _head(repo: pathlib.Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _make_branch(repo: pathlib.Path, branch: str, base: str, fname: str, wt: pathlib.Path) -> str:
    _git(repo, "worktree", "add", "-b", branch, str(wt), base)
    (wt / fname).write_text(f"{fname}\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"{branch} work")
    _git(repo, "worktree", "remove", "--force", str(wt))
    return _git(repo, "rev-parse", branch)


def _config(*, tool: bool, extra: dict[str, Any] | None = None) -> Config:
    return Config.model_validate(
        {
            "providers": {
                "p": {"backend": "openai", "base_url": "http://localhost:1/v1"},
                "q": {"backend": "openai", "base_url": "http://localhost:2/v1"},
            },
            "models": {
                "default": {"model": "p/base", "effort": "high", "temperature": 0.7},
                "reviewer": {"model": "q/critic", "effort": "max"},
            },
            "harness": {"parallel": {"tool": tool}},
            **(extra or {}),
        }
    )


class _Work:
    """The loop's children as the tools reach them: a scripted roster, and what was asked of it."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.waits: list[tuple[str, float]] = []
        self.stopped: list[str] = []
        self.lines: list[child_tools.ChildLine] = []

    def start(self, task: str, spec: str) -> tools_results.StartedResult:
        self.started.append((task, spec))
        child = f"run-p{len(self.started)}-l1"
        self.lines.append(child_tools.ChildLine(child, "running", task, True))
        return tools_results.StartedResult((child,), f"[parallel] started child {child}.")

    def roster(self) -> tuple[child_tools.ChildLine, ...]:
        return tuple(self.lines)

    def wait(self, child: str, wait_s: float, interrupted: Any) -> None:
        self.waits.append((child, wait_s))

    def stop(self, child: str) -> None:
        self.stopped.append(child)
        self.lines = [
            dataclasses.replace(line, state="stopping") if line.id == child else line
            for line in self.lines
        ]


def _dispatcher(
    tmp_path: pathlib.Path,
    cfg: Config,
    *,
    mode: str = "run",
    work: Any = None,
    state_dir: pathlib.Path | None = None,
) -> dispatch.ToolDispatcher:
    repo = tmp_path / "repo"
    if not (repo / ".git").exists():
        _init_repo(repo)
    d = dispatch.ToolDispatcher(
        root=repo,
        config=cfg,
        isolation="none",
        mode=mode,  # type: ignore[arg-type]
        session_dir=tmp_path / "session",
        state_dir=state_dir,
    )
    d.children = work
    return d


def _names(d: dispatch.ToolDispatcher, mode: str = "run") -> list[str]:
    return [t.name for t in toolset.tool_definitions(d, mode=mode)]  # type: ignore[arg-type]


def _description(d: dispatch.ToolDispatcher, name: str) -> str:
    (definition,) = [t for t in toolset.tool_definitions(d) if t.name == name]
    return definition.description


def test_the_tool_is_off_by_default_for_exposure_and_execution(tmp_path: pathlib.Path) -> None:
    """`harness.parallel.tool = false` offers nothing; a hallucinated call refuses, naming it."""
    d = _dispatcher(tmp_path, _config(tool=False), work=_Work())
    assert "parallel" not in _names(d)
    with pytest.raises(tool_errors.ToolError, match=r"parallel\.tool"):
        d.dispatch("parallel", {"task": "consult"})


def test_enabled_in_run_mode_the_tool_shows_and_answers_with_the_childs_id(
    tmp_path: pathlib.Path,
) -> None:
    """The handle is the child's session id: one call, one task, the answer at once."""
    work = _Work()
    d = _dispatcher(tmp_path, _config(tool=True), work=work)
    assert "parallel" in _names(d)
    result = d.dispatch("parallel", {"task": "consult the reviewer", "spec": "reviewer"})
    assert result.to_wire() == {
        "dispatched": True,
        "children": ["run-p1-l1"],
        "message": "[parallel] started child run-p1-l1.",
    }
    assert work.started == [("consult the reviewer", "reviewer")]
    refused = tools_results.StartedResult((), "[parallel] nothing dispatched.")
    assert refused.to_wire() == {
        "dispatched": False,
        "children": [],
        "message": "[parallel] nothing dispatched.",
    }


@pytest.mark.parametrize("mode", ["plan", "ask", "machine"])
def test_other_modes_never_offer_or_run_the_tool(tmp_path: pathlib.Path, mode: str) -> None:
    d = _dispatcher(tmp_path, _config(tool=True), mode=mode, work=_Work())
    assert "parallel" not in _names(d, mode)
    with pytest.raises(tool_errors.ToolError, match="not available"):
        d.dispatch("parallel", {"task": "consult"})


def test_a_child_and_a_front_end_without_a_dispatcher_refuse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth stays one, and a harness with no child dispatcher cannot pretend otherwise."""
    monkeypatch.setenv("AGENT6_CHILD", "1")
    inside = _dispatcher(tmp_path, _config(tool=True), work=_Work())
    assert "parallel" not in _names(inside)
    with pytest.raises(tool_errors.ToolError, match="inside a child"):
        inside.dispatch("parallel", {"task": "consult"})
    monkeypatch.delenv("AGENT6_CHILD")
    headless = _dispatcher(tmp_path, _config(tool=True), work=None)
    assert "parallel" not in _names(headless)
    with pytest.raises(tool_errors.ToolError, match="no child dispatcher"):
        headless.dispatch("parallel", {"task": "consult"})


def test_a_blank_task_is_refused_before_any_handler(tmp_path: pathlib.Path) -> None:
    work = _Work()
    d = _dispatcher(tmp_path, _config(tool=True), work=work)
    with pytest.raises(tool_errors.ToolError, match="task"):
        d.dispatch("parallel", {"task": "   "})
    assert work.started == []
    with pytest.raises(tool_errors.ToolError, match="spec"):
        d.dispatch("parallel", {"task": "t", "spec": "x" * 300})


def test_a_default_run_reads_the_background_tools_text_unchanged(tmp_path: pathlib.Path) -> None:
    """The two background tools mention children only in a run that has the `parallel` tool."""
    plain = _dispatcher(tmp_path, _config(tool=False), work=_Work())
    assert _description(plain, "read_background") == schema.ReadBackgroundInput.TOOL_DESCRIPTION
    assert _description(plain, "stop_background") == schema.StopBackgroundInput.TOOL_DESCRIPTION
    assert "child" not in schema.ReadBackgroundInput.TOOL_DESCRIPTION
    assert "child" not in schema.StopBackgroundInput.TOOL_DESCRIPTION
    # A run with no child reads the same result too: no `children` key.
    assert plain.dispatch("read_background", {}).to_wire() == {"shells": []}

    offered = _dispatcher(tmp_path, _config(tool=True), work=_Work())
    read, stop = (_description(offered, name) for name in ("read_background", "stop_background"))
    assert read == (
        schema.ReadBackgroundInput.TOOL_DESCRIPTION + schema.ReadBackgroundInput.CHILDREN_NOTE
    )
    assert "With no `id`, a `wait_s` above 0 waits for the next child to end" in read
    assert stop.endswith(
        " A child's id stops that child at once; a stopped child's work does not land, and its"
        " branch holds what it committed."
    )
    # The arguments are the same either way: no new field addresses a child.
    for name in ("read_background", "stop_background"):
        ((one,), (other,)) = (
            [t.input_schema for t in toolset.tool_definitions(d) if t.name == name]
            for d in (plain, offered)
        )
        assert one == other


def test_read_background_reads_a_child_by_its_id_and_waits_for_the_next(
    tmp_path: pathlib.Path,
) -> None:
    """One `id` space per kind: a child's session id reads the child, `bg<N>` a command."""
    state_dir = tmp_path / "state"
    child_dir = state_dir / "sessions" / "runs" / "run-p1-l1"
    child_dir.mkdir(parents=True)
    (child_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": "run-p1-l1", "mode": "run", "user_task": "t"})
    )
    rows = [
        {"type": "session.start", "user_task": "write a"},
        {"type": "tool.call", "name": "apply_edit"},
        {"type": "model.result", "text": "a.txt is written"},
    ]
    (child_dir / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    work = _Work()
    d = _dispatcher(tmp_path, _config(tool=True), work=work, state_dir=state_dir)
    d.dispatch("parallel", {"task": "write a"})

    read = d.dispatch("read_background", {"id": "run-p1-l1", "wait_s": 3}).to_wire()

    assert work.waits == [("run-p1-l1", 3.0)]
    assert read["children"] == ["[run-p1-l1] running: write a"]
    assert read["output"].splitlines() == [
        "[run-p1-l1] running: write a",
        "",
        "user: write a",
        "",
        "[tool] apply_edit",
        "",
        "assistant: a.txt is written",
    ]
    tail = d.dispatch("read_background", {"id": "run-p1-l1", "wait_s": 0, "tail_lines": 1})
    assert tail.to_wire()["output"].splitlines()[2:] == [
        "... 4 earlier lines ...",
        "assistant: a.txt is written",
    ]
    # No id and a wait: the next child to end, whichever it is.
    waited = d.dispatch("read_background", {"wait_s": 5}).to_wire()
    assert work.waits[-1] == ("", 5.0)
    assert waited == {"shells": [], "children": ["[run-p1-l1] running: write a"]}
    # No id and no wait is the roster at once, as it always was.
    assert d.dispatch("read_background", {}).to_wire() == waited
    assert len(work.waits) == 3
    with pytest.raises(tool_errors.ToolError, match="no background command 'run-p9-l1'"):
        d.dispatch("read_background", {"id": "run-p9-l1"})


def test_stop_background_stops_a_child_even_where_commands_are_denied(
    tmp_path: pathlib.Path,
) -> None:
    """The parent can stop a child it started; `run_commands = "deny"` withholds commands only."""
    denied = {"sandbox": {"run_commands": "deny"}}
    work = _Work()
    d = _dispatcher(tmp_path, _config(tool=True, extra=denied), work=work)
    assert "stop_background" in _names(d) and "run_command" not in _names(d)
    d.dispatch("parallel", {"task": "write a"})

    stopped = d.dispatch("stop_background", {"id": "run-p1-l1"}).to_wire()

    assert work.stopped == ["run-p1-l1"]
    assert stopped == {"shells": [], "children": ["[run-p1-l1] stopping: write a"]}
    without = _dispatcher(tmp_path, _config(tool=False, extra=denied), work=_Work())
    assert "stop_background" not in _names(without)


def test_the_description_names_every_role_the_providers_and_the_limits(
    tmp_path: pathlib.Path,
) -> None:
    """The model reads what each selector would run: route, effort, temperature, inheritance."""
    d = _dispatcher(tmp_path, _config(tool=True), work=_Work())
    text = _description(d, "parallel")
    assert text.startswith("Start one task in a child session (a subagent) on a fresh clone")
    assert "return at once with the child's id. Call it once per task." in text
    assert "starts that many attempts at the same task" in text
    assert "blocks until" not in text
    assert "default: p/base, effort high, temperature 0.7" in text
    assert "worker: p/base, effort high, temperature 0.7 (as default)" in text
    assert "reviewer: q/critic, effort max, temperature 0.7 (temperature from default)" in text
    assert "Configured providers: p, q" in text
    assert "turns" not in text.split("Configured providers")[1] and "minutes" not in text
    assert "an ordinary run with this run's tools and limits that may edit its clone" in text
    assert "the children share this run's remaining budget" in text
    assert "at most 4 run at once (`harness.parallel.max_children`)" in text
    assert "secret" not in text and "localhost" not in text


# ---- through the real harness --------------------------------------------------------------


def _resp_tool(name: str, args: dict[str, Any], tu_id: str) -> ProviderResponse:
    block = {"type": "tool_use", "id": tu_id, "name": name, "input": args}
    return ProviderResponse(
        text="",
        tool_uses=({"id": tu_id, "name": name, "input": args},),
        stop_reason="tool_use",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={"content": [block]},
    )


class _Events:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: journal.Event) -> None:
        self.emitted.append((event.TYPE, event.model_dump(mode="json", exclude_unset=True)))

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [f for t, f in self.emitted if t == event_type]


class _Graph:
    def __init__(self) -> None:
        self.rationales: list[str] = []
        self.graph_version = 0

    def nodes(self) -> dict[str, Any]:
        return {}

    def cursor(self) -> str | None:
        return None

    def add_subtask(self, intent: Any) -> Any:
        self.rationales.append(intent.draft.rationale)
        return types.SimpleNamespace(id=f"N{len(self.rationales)}")

    def update_status(self, intent: Any) -> Any:
        return types.SimpleNamespace(id=intent.id)

    def record_commit(self, intent: Any) -> Any:
        return types.SimpleNamespace(id=intent.id)


def _child_result(sid: str, workdir: pathlib.Path, *, branch: str, tip: str, usd: float) -> Any:
    child = int(sid.rsplit("-l", 1)[1])
    return child_records.ChildResult(
        spec=child_records.ChildSpec(child, sid, workdir, None),
        session_dir=workdir,
        branch=branch,
        ok=True,
        error="",
        imported_ref=branch,
        imported_tip=tip,
        has_commits=True,
        usage=agent6_budget.UsageSettlement(usd=usd, usd_partial=False, tokens=0),
    )


def test_the_model_starts_attempts_and_the_call_returns_their_ids_at_once(
    tmp_path: pathlib.Path,
) -> None:
    """One call is one task: a model list starts attempts, and the answer is their ids.

    The call blocked until every child returned and merged each one in order.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _head(repo)
    session_dir = tmp_path / "run-abc"
    session_dir.mkdir()
    seen: list[child_records.ChildTask] = []

    @child_env.blocking
    def spawner(children: Any, group: str, *, at: str | None = None) -> list[Any]:
        seen.extend(children)
        out = []
        for i in range(1, len(children) + 1):
            sid = f"run-abc-{group}-l{i}"
            tip = _make_branch(repo, f"agent6/{sid}", base, f"child{i}.txt", tmp_path / f"wt{i}")
            out.append(
                _child_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tip, usd=0.5)
            )
        return out

    def first_ranked(results: Any, *, group: str, base: str) -> child_records.Ranking:
        order = tuple(res.spec.session_id for res in results)
        return child_records.Ranking(order=order, winner=order[0])

    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp_tool("parallel", {"task": "consult", "spec": "reviewer,default"}, "t1"),
        _resp_tool("read_file", {"path": "README.md"}, "t2"),
    ]
    cfg = _config(tool=True)
    disp = dispatch.ToolDispatcher(
        root=repo, config=cfg, isolation="none", session_dir=session_dir, events=None
    )
    budget = agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    graph = _Graph()
    events: Any = _Events(session_dir / "logs.jsonl")
    wf = loop.Harness(
        chain=run_chain.RunChain(repo, ref="refs/agent6/parent", fallback_parent=base),
        config=cfg,
        provider=provider,
        dispatcher=disp,
        logger=lambda _m: None,
        events=events,
        curator=graph,  # type: ignore[arg-type]
        budget=budget,
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(child_spawner=spawner, child_ranker=first_ranked),
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=2,
    )
    wf.run("start")
    # The second call's messages carry the first call's tool result: the handles, no joins.
    second = provider.call.call_args_list[1]
    messages = second.kwargs.get("messages") or second.args[1]
    (answer,) = [
        json.loads(block["content"])
        for message in messages
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    ids = ["run-abc-p1-l1", "run-abc-p1-l2"]
    assert answer == {
        "dispatched": True,
        "children": ids,
        "message": parallel_join.started_text(ids),
    }
    # The fake's children had ended, so the first boundary told their ends and landed the task.
    wire = json.dumps(messages)
    assert wire.count("[parallel] child run-abc-p1-l") == 2
    assert "run-abc-p1-l1 (agent6/run-abc-p1-l1): joined at" in wire
    assert "run-abc-p1-l2 (agent6/run-abc-p1-l2): ranked 2 of 2; not joined" in wire
    assert [child.model for child in seen] == ["reviewer", "default"]
    # A child the model starts runs under its parent's config: no cap of its own.
    assert [child.limits for child in seen] == [None, None]
    assert all(
        child.allowance is not None and child.allowance.usd == pytest.approx(10.0) for child in seen
    )
    (record,) = child_dispatch.read_records(session_dir)
    assert record.source == "model" and record.state == "complete" and record.delivered
    assert (record.join_policy, record.spec) == ("rank", "reviewer,default")
    assert [c.limits for c in record.children] == [None, None]
    assert graph.rationales[-1] == "dispatched through /parallel by the model"
    assert budget.estimate_usd()[0] == pytest.approx(1.0)
    # The first-ranked attempt alone is on the chain; the other keeps its branch.
    log = _git(repo, "log", "--oneline", "refs/agent6/parent")
    assert "run-abc-p1-l1 work" in log and "run-abc-p1-l2 work" not in log
    assert _git(repo, "rev-parse", "--verify", "agent6/run-abc-p1-l2")
    assert not os.environ.get("AGENT6_CHILD")
