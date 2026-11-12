# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A reviewer runs through the one Harness, as a child of its own kind."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds, lane_records, paths
from agent6.app import lane_launch, lane_watch, worktrees
from agent6.config import Config
from agent6.harness import loop, operator_bridge, provider_call, run_chain
from agent6.harness.compaction import window as compaction_window
from agent6.providers import ProviderResponse
from agent6.sessions import completion, journal, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import dispatch, tool_errors
from agent6.viewmodel import listing
from tests import lane_env


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


def _resp(text: str, tool_uses: tuple[dict[str, Any], ...] = ()) -> ProviderResponse:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks += [{"type": "tool_use", **tu} for tu in tool_uses]
    return ProviderResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason="tool_use" if tool_uses else "end_turn",
        input_tokens=1,
        output_tokens=7,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={"content": blocks},
    )


class _Events:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: journal.Event) -> None:
        self.emitted.append((event.TYPE, event.model_dump(mode="json", exclude_unset=True)))


def _harness(
    repo: pathlib.Path,
    session_dir: pathlib.Path,
    provider: mock.MagicMock,
    contract: kinds.ConversationContract,
    *,
    retries: int = 0,
    compaction: compaction_window.CompactionSettings | None = None,
) -> tuple[loop.Harness, dispatch.ToolDispatcher]:
    session_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "none", "run_commands": "allow", "protect_git": "off"}}
    )
    disp = dispatch.ToolDispatcher(
        root=repo,
        config=cfg,
        isolation="none",
        mode="reviewer",
        session_dir=session_dir,
    )
    events: Any = _Events(session_dir / "logs.jsonl")
    wf = loop.Harness(
        chain=run_chain.RunChain(repo, ref=None, fallback_parent=None, per_turn=False),
        config=cfg,
        provider=provider,
        dispatcher=disp,
        logger=lambda _m: None,
        events=events,
        resume_state_path=session_dir / "loop_state.json",
        bridge=operator_bridge.OperatorBridge(),
        call=provider_call.CallSettings(retry_count=retries, retry_delay_s=0.0),
        mode="reviewer",
        contract=contract,
        max_turns=50,
        compaction=compaction or compaction_window.CompactionSettings(),
    )
    return wf, disp


def test_a_contract_handed_no_tools_offers_none_and_records_its_text(
    tmp_path: pathlib.Path,
) -> None:
    """One provider call with no tools; a hallucinated tool is refused; the text is the record."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp(
            '{"verdict": "pass"}',
            ({"id": "t1", "name": "apply_edit", "input": {"path": "x", "edits": []}},),
        ),
        _resp("never asked"),
    ]
    contract = kinds.ConversationContract(
        system="You are a reviewer. Answer in JSON.",
        opening="Review this diff: +x",
        max_turns=1,
    )
    wf, disp = _harness(repo, tmp_path / "leaf", provider, contract)
    result = wf.run("review")
    assert provider.call.call_count == 1
    sent = provider.call.call_args_list[0].kwargs
    assert sent["system"] == contract.system and sent["tools"] == []
    assert result.reason == "max_turns" and not result.completed
    assert result.summary == '{"verdict": "pass"}'
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None
    assert record.text == '{"verdict": "pass"}' and record.stop_reason == "max_turns"
    assert record.calls == 1 and record.output_tokens == 7
    assert "turn limit 1 was reached: apply_edit" in " ".join(record.diagnostics)
    with pytest.raises(tool_errors.ToolError, match="not available in this conversation contract"):
        disp.dispatch("read_file", {"path": "README.md"})


def test_a_contract_stops_at_its_turn_limit_without_running_the_last_tools(
    tmp_path: pathlib.Path,
) -> None:
    """Three turns allowed: the third response's tool use is never executed; its text is kept."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    read = {"id": "t1", "name": "read_file", "input": {"path": "README.md"}}
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp("", (read,)),
        _resp("", ({**read, "id": "t2"},)),
        _resp("final answer", ({**read, "id": "t3"},)),
        _resp("never"),
    ]
    contract = kinds.ConversationContract(
        system="Read, do not write.",
        opening="What is in README?",
        max_turns=3,
        allowed_tools=("read_file",),
    )
    wf, disp = _harness(repo, tmp_path / "leaf", provider, contract)
    reads: list[str] = []
    real = disp.dispatch

    def counting(name: str, raw: dict[str, Any], **kw: Any) -> Any:
        reads.append(name)
        return real(name, raw, **kw)

    disp.dispatch = counting  # type: ignore[method-assign]
    result = wf.run("read")
    assert provider.call.call_count == 3
    assert reads == ["read_file", "read_file"]  # the third response's tool never ran
    assert result.reason == "max_turns" and result.summary == "final answer"
    tools = provider.call.call_args_list[0].kwargs["tools"]
    assert [t.name for t in tools] == ["read_file"]
    for forbidden, raw in (
        ("apply_edit", {"path": "x", "edits": []}),
        ("run_command", {"argv": ["true"]}),
        ("run_verify_command", {}),
        ("fetch", {"url": "http://127.0.0.1:1/"}),
    ):
        with pytest.raises(tool_errors.ToolError, match="not available"):
            disp.dispatch(forbidden, raw)


def test_a_contract_runs_the_commands_it_is_handed_and_never_edits(tmp_path: pathlib.Path) -> None:
    """A handed `run_command` runs; an edit refuses even when a parent hands its name down."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [_resp("done")]
    contract = kinds.ConversationContract(
        system="s",
        opening="o",
        max_turns=2,
        allowed_tools=("apply_edit", "parallel", "read_file", "run_command"),
    )
    wf, disp = _harness(repo, tmp_path / "leaf", provider, contract)
    wf.run("o")
    offered = [t.name for t in provider.call.call_args_list[0].kwargs["tools"]]
    assert offered == ["read_file", "run_command"]
    # The handler is reached: this bare dispatcher has no jail session to run it in.
    with pytest.raises(tool_errors.ToolError, match="no prepared command session"):
        disp.dispatch("run_command", {"argv": ["git", "log", "--oneline"]})
    for outside, raw in (
        ("apply_edit", {"path": "README.md", "edits": [{"old": "hi", "new": "bye"}]}),
        ("parallel", {"lanes": [{"task": "t"}]}),
    ):
        with pytest.raises(tool_errors.ToolError, match="not available"):
            disp.dispatch(outside, raw)
    assert (repo / "README.md").read_text(encoding="utf-8") == "hi\n"


def test_a_reviewer_has_every_base_tool_except_the_two_that_edit() -> None:
    """The reviewer mode's surface; a machine's reading loop, ask and plan keep theirs."""
    from agent6.tools import schema

    assert schema.mode_tools("reviewer").names == {
        "fetch",
        "find_definition",
        "find_references",
        "list_dir",
        "outline",
        "read_background",
        "read_file",
        "read_session",
        "run_command",
        "run_verify_command",
        "stop_background",
        "use_skill",
    }
    assert schema.mode_tools("read_only").names == {
        "find_definition",
        "find_references",
        "finish_session",
        "list_dir",
        "outline",
        "read_file",
    }
    # Every mode that runs commands owns its background commands; a machine state runs none.
    for mode in ("run", "ask", "plan"):
        assert {"read_background", "stop_background"} <= schema.mode_tools(mode).names
    assert "read_background" not in schema.mode_tools("machine").names


def test_a_reviewer_compacts_and_keeps_its_opening(tmp_path: pathlib.Path) -> None:
    """A long review restarts from a summary; the task, rules and diff stay, once."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "big.txt").write_text("line of filler text\n" * 400, encoding="utf-8")
    read = {"id": "t1", "name": "read_file", "input": {"path": "big.txt"}}
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _resp("", (read,)),
        _resp("", ({**read, "id": "t2"},)),
        _resp("SUMMARY: read big.txt twice"),
        _resp('{"verdict": "pass"}'),
    ]
    opening = "REVIEW OPENING: the task, the rules and the diff"
    contract = kinds.ConversationContract(
        system="s", opening=opening, max_turns=6, allowed_tools=("read_file",)
    )
    small = compaction_window.CompactionSettings(
        drop_at_chars=2_000, summarise_at_chars=4_000, keep_recent_chars=500, elision_gists=False
    )
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract, compaction=small)
    result = wf.run("review")
    assert result.reason == "answered" and result.summary == '{"verdict": "pass"}'
    summarise, final = (
        provider.call.call_args_list[2].kwargs,
        provider.call.call_args_list[3].kwargs,
    )
    assert summarise["tools"] == [] and "TRANSCRIPT" in json.dumps(summarise["messages"])
    sent = json.dumps(final["messages"])
    assert "SUMMARY: read big.txt twice" in sent and sent.count(opening) == 1


def test_a_retried_call_does_not_use_a_turn(tmp_path: pathlib.Path) -> None:
    """Two turns allowed: a transient error's retry leaves the second turn its answer."""
    from agent6.providers import ProviderError

    repo = tmp_path / "repo"
    _init_repo(repo)
    read = {"id": "t1", "name": "read_file", "input": {"path": "README.md"}}
    provider = mock.MagicMock()
    provider.call.side_effect = [
        ProviderError("transient", status_code=503),
        _resp("", (read,)),
        _resp("the answer"),
    ]
    contract = kinds.ConversationContract(
        system="s", opening="o", max_turns=2, allowed_tools=("read_file",)
    )
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract, retries=3)
    result = wf.run("o")
    assert provider.call.call_count == 3
    assert result.reason == "answered" and result.summary == "the answer"
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None and record.calls == 3


def test_an_answer_on_the_last_allowed_turn_is_an_answer(tmp_path: pathlib.Path) -> None:
    """A one-call reviewer that answers must not read as cut at its ceiling."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [_resp('{"verdict": "pass"}'), _resp("never")]
    contract = kinds.ConversationContract(system="s", opening="o", max_turns=1)
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract)
    result = wf.run("o")
    assert provider.call.call_count == 1
    assert result.reason == "answered" and result.completed
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None
    assert record.stop_reason == "answered" and record.diagnostics == ()


def test_the_reviewers_bucket_is_a_real_discoverable_one(tmp_path: pathlib.Path) -> None:
    assert kinds.session_bucket("reviewer") == "reviewers"
    assert "reviewers" in layout.SESSION_BUCKETS and "reviewers" in layout.HUB_BUCKETS
    assert "reviewer" not in kinds.OPERATOR_MODES
    state = tmp_path / "state"
    link = lane_watch.lane_dir(state, "co-p1-l1", mode="reviewer")
    assert link == state / "sessions" / "reviewers" / "co-p1-l1"
    assert lane_records.LaneSpec(1, "x", tmp_path, None).mode == "run"


@pytest.mark.parametrize(("policy", "handed"), [("allow", True), ("deny", True), ("allow", False)])
def test_a_real_reviewer_never_edits_and_runs_only_the_commands_it_may(
    short_repo: pathlib.Path, tmp_path: pathlib.Path, policy: str, handed: bool
) -> None:
    """A reviewer's edit refuses; its command and setup need the hand-down and the policy."""
    runs = handed and policy == "allow"
    clone = short_repo / "clone"
    _init_repo(clone)
    marker = tmp_path / "setup-ran"
    replies: list[tuple[str, dict[str, Any]] | str] = [
        ("apply_edit", {"path": "README.md", "edits": [{"old": "hi", "new": "bye"}]}),
        ("run_command", {"argv": ["touch", str(tmp_path / "command-ran")]}),
        ("read_file", {"path": "README.md"}),
        '{"verdict": "pass"}',
    ]
    with lane_env.scripted_provider(replies) as (base_url, calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/base"}},
                "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": policy},
                "git": {"worktree": {"setup_command": ["touch", str(marker)]}},
                "harness": {
                    "verify": {"when": "never"},
                },
                "review": {"when": "never"},
                "notify": {"on_event": ["touch", str(tmp_path / "hook-ran")]},
            }
        )
        into = lane_launch.launch_dir(paths.state_dir(short_repo), "leaf")
        spec = lane_records.LaneSpec(1, "leaf-l1", clone, None, mode="reviewer")
        config = worktrees.write_lane_config(
            cfg,
            spec,
            explicit_leaves=frozenset(
                {
                    "notify.on_event",
                    "sandbox.isolation",
                    "sandbox.protect_git",
                    "sandbox.run_commands",
                }
            ),
            into=into,
        )
        contract = kinds.ConversationContract(
            system="Review only.",
            opening="Try to edit, then run, then read.",
            max_turns=6,
            allowed_tools=("read_file", "run_command") if handed else ("read_file",),
        )
        path = lane_launch.write(
            into,
            lane_launch.LaneLaunch(
                worktree=clone,
                record_state=paths.state_dir(clone),
                session_id="leaf-l1",
                task="review",
                config=config.path,
                notes=config.notes,
                parent="leaf",
                group="leaf",
                lane=1,
                contract=contract,
            ),
        )
        env = {
            **os.environ,
            lane_launch.ENV: str(path),
            "AGENT6_CHILD": "1",
            "AGENT6_STREAM_TO_LOG": "1",
            "AGENT6_ALLOW_ROOT": "1",
        }
        proc = subprocess.run(
            lane_launch.bootstrap_argv(),
            cwd=str(clone),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert proc.returncode == 0, proc.stderr
    session_dir = paths.state_dir(clone) / "sessions" / "reviewers" / "leaf-l1"
    assert session_dir.is_dir(), list((paths.state_dir(clone) / "sessions").iterdir())
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    results = {r["name"]: r for r in rows if r.get("type") == "tool.result"}
    assert (
        results["apply_edit"]["ok"] is False and "not available" in results["apply_edit"]["summary"]
    )
    assert results["run_command"]["ok"] is runs
    assert results["read_file"]["ok"] is True
    assert (clone / "README.md").read_text(encoding="utf-8") == "hi\n"
    # A reviewer that may run commands gets the worktree a lane gets, its setup included.
    assert marker.exists() is runs
    assert (tmp_path / "command-ran").exists() is runs
    # The operator's hook hears of the parent's end, never of a reviewer's.
    assert not (tmp_path / "hook-ran").exists()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "reviewer"
    assert len(calls) == 4
    offered = sorted(t["function"]["name"] for t in calls[0]["tools"])
    assert offered == (["read_file", "run_command"] if runs else ["read_file"])
    record = completion.read_completion(session_dir)
    assert record is not None and record.stop_reason == "answered"
    assert record.text == '{"verdict": "pass"}'
    assert manifest["parallel"]["parent"] == "leaf"
    # Found by its bucket and its id; the resume gate refuses it.
    state = paths.state_dir(clone)
    assert session_dir in listing.session_dirs(state)
    summary = listing.summarize_session_dir(session_dir)
    assert summary.session_id == "leaf-l1" and summary.parent == "leaf"
    assert summary.mode == "reviewer" and summary.stop_reason == "answered"
    with pytest.raises(sessions_manifest.ManifestError, match="not resumable"):
        sessions_manifest.read_manifest(session_dir).session_mode()
