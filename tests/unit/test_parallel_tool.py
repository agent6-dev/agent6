# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The opt-in `parallel` tool: two gates, a resolved description, the dispatcher's own path."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import lane_records
from agent6.config import Config
from agent6.harness import loop, operator_bridge, provider_call, run_chain, toolset
from agent6.providers import ProviderResponse
from agent6.sessions import journal, lane_dispatch
from agent6.tools import dispatch, tool_errors
from agent6.tools import results as tools_results


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
            "parallel": {"tool": tool},
            **(extra or {}),
        }
    )


def _dispatcher(
    tmp_path: pathlib.Path, cfg: Config, *, mode: str = "run", handler: Any = None
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
    )
    d.parallel_handler = handler
    return d


def _names(d: dispatch.ToolDispatcher, mode: str = "run") -> list[str]:
    return [t.name for t in toolset.tool_definitions(d, mode=mode)]  # type: ignore[arg-type]


def _handler(raw: dict[str, Any], origin: Any) -> tools_results.ToolResult:
    return tools_results.RawResult({"message": f"dispatched {raw['task']}"})


def test_the_tool_is_off_by_default_for_exposure_and_execution(tmp_path: pathlib.Path) -> None:
    """`parallel.tool = false` offers nothing; a hallucinated call refuses naming the knob."""
    d = _dispatcher(tmp_path, _config(tool=False), handler=_handler)
    assert "parallel" not in _names(d)
    with pytest.raises(tool_errors.ToolError, match=r"parallel\.tool"):
        d.dispatch("parallel", {"task": "consult"})


def test_enabled_in_run_mode_the_tool_shows_and_dispatches_through_its_handler(
    tmp_path: pathlib.Path,
) -> None:
    d = _dispatcher(tmp_path, _config(tool=True), handler=_handler)
    assert "parallel" in _names(d)
    result = d.dispatch("parallel", {"task": "consult the reviewer"})
    assert result.to_wire() == {"message": "dispatched consult the reviewer"}


@pytest.mark.parametrize("mode", ["plan", "ask", "machine"])
def test_other_modes_never_offer_or_run_the_tool(tmp_path: pathlib.Path, mode: str) -> None:
    d = _dispatcher(tmp_path, _config(tool=True), mode=mode, handler=_handler)
    assert "parallel" not in _names(d, mode)
    with pytest.raises(tool_errors.ToolError, match="not available"):
        d.dispatch("parallel", {"task": "consult"})


def test_a_subordinate_run_and_a_front_end_without_a_dispatcher_refuse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth stays one, and a harness with no lane dispatcher cannot pretend otherwise."""
    monkeypatch.setenv("AGENT6_SUBRUN", "1")
    inside = _dispatcher(tmp_path, _config(tool=True), handler=_handler)
    assert "parallel" not in _names(inside)
    with pytest.raises(tool_errors.ToolError, match="inside a lane"):
        inside.dispatch("parallel", {"task": "consult"})
    monkeypatch.delenv("AGENT6_SUBRUN")
    headless = _dispatcher(tmp_path, _config(tool=True), handler=None)
    assert "parallel" not in _names(headless)
    with pytest.raises(tool_errors.ToolError, match="no lane dispatcher"):
        headless.dispatch("parallel", {"task": "consult"})


def test_a_blank_task_is_refused_before_any_handler(tmp_path: pathlib.Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(raw: dict[str, Any], origin: Any) -> tools_results.ToolResult:
        calls.append(raw)
        return tools_results.RawResult({})

    d = _dispatcher(tmp_path, _config(tool=True), handler=handler)
    with pytest.raises(tool_errors.ToolError, match="task"):
        d.dispatch("parallel", {"task": "   "})
    assert calls == []
    with pytest.raises(tool_errors.ToolError, match="spec"):
        d.dispatch("parallel", {"task": "t", "spec": "x" * 300})


def test_the_description_names_every_role_the_providers_and_the_ceilings(
    tmp_path: pathlib.Path,
) -> None:
    """The model reads what each selector would run: route, effort, temperature, inheritance."""
    d = _dispatcher(tmp_path, _config(tool=True), handler=_handler)
    (definition,) = [t for t in toolset.tool_definitions(d) if t.name == "parallel"]
    text = definition.description
    assert "default: p/base, effort high, temperature 0.7" in text
    assert "worker: p/base, effort high, temperature 0.7 (as default)" in text
    assert "reviewer: q/critic, effort max, temperature 0.7 (temperature from default)" in text
    assert "Configured providers: p, q" in text
    assert "200 turns" in text and "30 minutes" in text
    assert "edit" in text and "this run's remaining budget, which the lanes share" in text
    assert "gives the summary it finished with" in text
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


def _lane_result(sid: str, workdir: pathlib.Path, *, branch: str, tip: str, usd: float) -> Any:
    lane = int(sid.rsplit("-l", 1)[1])
    return lane_records.LaneResult(
        spec=lane_records.LaneSpec(lane, sid, workdir, None),
        session_dir=workdir,
        branch=branch,
        ok=True,
        error="",
        imported_ref=branch,
        imported_tip=tip,
        has_commits=True,
        usage=agent6_budget.UsageSettlement(usd=usd, usd_partial=False, tokens=0),
    )


def test_the_model_dispatches_lanes_and_reads_the_result_as_the_tools_answer(
    tmp_path: pathlib.Path,
) -> None:
    """One call: the lanes run under the model's ceilings, join in order, and the result returns."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _head(repo)
    session_dir = tmp_path / "run-abc"
    session_dir.mkdir()
    seen: list[lane_records.LaneTask] = []

    def spawner(lanes: Any, group: str, *, at: str | None = None) -> list[Any]:
        seen.extend(lanes)
        out = []
        for i in range(1, len(lanes) + 1):
            sid = f"run-abc-{group}-l{i}"
            tip = _make_branch(repo, f"agent6/{sid}", base, f"lane{i}.txt", tmp_path / f"wt{i}")
            out.append(
                _lane_result(sid, tmp_path / f"w{i}", branch=f"agent6/{sid}", tip=tip, usd=0.5)
            )
        return out

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
        chain=run_chain.RunChain(repo, ref="refs/agent6/coord", fallback_parent=base),
        config=cfg,
        provider=provider,
        dispatcher=disp,
        logger=lambda _m: None,
        events=events,
        curator=graph,  # type: ignore[arg-type]
        budget=budget,
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(lane_spawner=spawner),
        call=provider_call.CallSettings(retry_count=0, retry_delay_s=0.0),
        max_turns=2,
    )
    wf.run("start")
    # The second call's messages carry the first call's tool result.
    second = provider.call.call_args_list[1]
    messages = second.kwargs.get("messages") or second.args[1]
    wire = json.dumps(messages)
    assert "tool_result" in wire and "joined at" in wire, wire[-2500:]
    assert [lane.model for lane in seen] == ["reviewer", "default"]
    assert all(
        lane.limits == lane_records.LaneLimits(max_turns=200, deadline_at=lane.limits.deadline_at)
        for lane in seen
        if lane.limits
    )
    assert all(lane.limits is not None and lane.limits.deadline_at is not None for lane in seen)
    assert all(
        lane.allowance is not None and lane.allowance.usd == pytest.approx(10.0) for lane in seen
    )
    (record,) = lane_dispatch.read_records(session_dir)
    assert record.source == "model" and record.state == "complete" and record.delivered
    assert all(c.limits is not None and c.limits.max_turns == 200 for c in record.children)
    assert graph.rationales[-1] == "dispatched through /parallel by the model"
    assert budget.estimate_usd()[0] == pytest.approx(1.0)
    log = _git(repo, "log", "--oneline", "refs/agent6/coord")
    assert "run-abc-p1-l1 work" in log and "run-abc-p1-l2 work" in log
    # Lane 1 joined first: the chain tip was its tip when lane 2's merge was made.
    l1, l2 = record.children
    assert l1.join is not None and l2.join is not None
    assert l2.join.target_before == _git(repo, "rev-parse", "agent6/run-abc-p1-l1")
    assert not os.environ.get("AGENT6_SUBRUN")
