# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A bounded read-only conversation runs through the one Harness, as a lane of its own kind."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds, lane_records, paths
from agent6.app import lane_launch, lane_watch, workdirs
from agent6.config import Config
from agent6.harness import loop, operator_bridge, provider_call, run_chain
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
    max_calls_retry: int = 0,
) -> tuple[loop.Harness, dispatch.ToolDispatcher]:
    session_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "none", "run_commands": "allow", "protect_git": "off"}}
    )
    disp = dispatch.ToolDispatcher(
        root=repo,
        config=cfg,
        isolation="none",
        mode="read_only",
        session_dir=session_dir,
        no_tools=contract.kind == "no_tools",
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
        call=provider_call.CallSettings(retry_count=max_calls_retry, retry_delay_s=0.0),
        mode="read_only",
        contract=contract,
        max_turns=50,
    )
    return wf, disp


def test_a_no_tools_contract_makes_one_call_and_records_its_text(tmp_path: pathlib.Path) -> None:
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
        kind="no_tools",
        system="You are a reviewer. Answer in JSON.",
        opening="Review this diff: +x",
        max_calls=1,
        max_output_tokens=1500,
    )
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract)
    result = wf.run("review")
    assert provider.call.call_count == 1
    sent = provider.call.call_args_list[0].kwargs
    assert sent["system"] == contract.system and sent["tools"] == []
    assert sent["max_tokens"] == 1500
    assert result.reason == "calls_exhausted" and not result.completed
    assert result.summary == '{"verdict": "pass"}'
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None
    assert record.text == '{"verdict": "pass"}' and record.stop_reason == "calls_exhausted"
    assert record.calls == 1 and record.output_tokens == 7
    assert "no tools" in " ".join(record.diagnostics)


def test_a_read_only_contract_stops_at_its_call_ceiling_without_running_the_last_tools(
    tmp_path: pathlib.Path,
) -> None:
    """Three calls allowed: the third response's tool use is never executed; its text is kept."""
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
        kind="read_only",
        system="Read, do not write.",
        opening="What is in README?",
        max_calls=3,
        max_output_tokens=2000,
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
    assert result.reason == "calls_exhausted" and result.summary == "final answer"
    tools = provider.call.call_args_list[0].kwargs["tools"]
    names = {t.name for t in tools}
    assert "read_file" in names and "apply_edit" not in names and "run_command" not in names
    for forbidden, raw in (
        ("apply_edit", {"path": "x", "edits": []}),
        ("run_command", {"argv": ["true"]}),
        ("run_verify_command", {}),
        ("fetch", {"url": "http://127.0.0.1:1/"}),
    ):
        with pytest.raises(tool_errors.ToolError, match="not available"):
            disp.dispatch(forbidden, raw)


def test_retries_count_toward_the_ceiling(tmp_path: pathlib.Path) -> None:
    """A transient error's retry is an attempt: hidden retries cannot stretch the ceiling."""
    from agent6.providers import ProviderError

    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [
        ProviderError("transient", status_code=503),
        _resp("after one retry"),
        _resp("never"),
    ]
    contract = kinds.ConversationContract(
        kind="no_tools", system="s", opening="o", max_calls=2, max_output_tokens=100
    )
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract, max_calls_retry=3)
    result = wf.run("o")
    assert provider.call.call_count == 2
    assert result.reason == "answered" and result.summary == "after one retry"
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None and record.calls == 2


def test_an_answer_on_the_last_allowed_call_is_an_answer(tmp_path: pathlib.Path) -> None:
    """A one-call seat that answers must not read as cut at its ceiling."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = mock.MagicMock()
    provider.call.side_effect = [_resp('{"verdict": "pass"}'), _resp("never")]
    contract = kinds.ConversationContract(
        kind="no_tools", system="s", opening="o", max_calls=1, max_output_tokens=100
    )
    wf, _disp = _harness(repo, tmp_path / "leaf", provider, contract)
    result = wf.run("o")
    assert provider.call.call_count == 1
    assert result.reason == "answered" and result.completed
    record = completion.read_completion(tmp_path / "leaf")
    assert record is not None
    assert record.stop_reason == "answered" and record.diagnostics == ()


def test_the_read_only_bucket_is_a_real_discoverable_one(tmp_path: pathlib.Path) -> None:
    assert kinds.session_bucket("read_only") == "read_only"
    assert "read_only" in layout.SESSION_BUCKETS and "read_only" in layout.HUB_BUCKETS
    assert "read_only" not in kinds.OPERATOR_MODES
    state = tmp_path / "state"
    link = lane_watch.lane_link(state, "co-p1-l1", mode="read_only")
    assert link == state / "sessions" / "read_only" / "co-p1-l1"
    assert lane_records.LaneSpec(1, "x", tmp_path, None).mode == "run"


def test_a_real_read_only_lane_cannot_write_run_or_start_anything(
    short_repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Through the installed bootstrap: edits and commands refuse, no setup or MCP starts."""
    clone = short_repo / "clone"
    _init_repo(clone)
    marker = tmp_path / "setup-ran"
    replies: list[tuple[str, dict[str, Any]]] = [
        ("apply_edit", {"path": "README.md", "edits": [{"old": "hi", "new": "bye"}]}),
        ("run_command", {"argv": ["touch", str(tmp_path / "command-ran")]}),
        ("read_file", {"path": "README.md"}),
        lane_env.FINISH,
    ]
    with lane_env.scripted_provider(replies) as (base_url, calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/base"}},
                "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "allow"},
                "workspace": {"setup_command": ["touch", str(marker)]},
                "harness": {
                    "verify": {"when": "never"},
                },
                "review": {"when": "never"},
                "notify": {"on_event": ["touch", str(tmp_path / "hook-ran")]},
            }
        )
        into = lane_launch.launch_dir(paths.state_dir(short_repo), "leaf")
        spec = lane_records.LaneSpec(1, "leaf-l1", clone, None, mode="read_only")
        config = workdirs.write_lane_config(
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
            kind="read_only",
            system="Review only.",
            opening="Try to edit, then run, then read, then finish.",
            max_calls=6,
            max_output_tokens=2000,
        )
        path = lane_launch.write(
            into,
            lane_launch.LaneLaunch(
                workdir=clone,
                session_id="leaf-l1",
                task="review",
                config=config.path,
                notes=config.notes,
                coordinator="leaf",
                group="leaf",
                lane=1,
                contract=contract,
            ),
        )
        env = {
            **os.environ,
            lane_launch.ENV: str(path),
            "AGENT6_SUBRUN": "1",
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
    session_dir = paths.state_dir(clone) / "sessions" / "read_only" / "leaf-l1"
    assert session_dir.is_dir(), list((paths.state_dir(clone) / "sessions").iterdir())
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    results = {r["name"]: r for r in rows if r.get("type") == "tool.result"}
    assert (
        results["apply_edit"]["ok"] is False and "not available" in results["apply_edit"]["summary"]
    )
    assert results["run_command"]["ok"] is False
    assert results["read_file"]["ok"] is True
    assert (clone / "README.md").read_text(encoding="utf-8") == "hi\n"
    assert not marker.exists() and not (tmp_path / "command-ran").exists()
    # The operator's hook hears of the parent's end, never of a seat's.
    assert not (tmp_path / "hook-ran").exists()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "read_only"
    assert len(calls) == 4
    assert all("apply_edit" not in json.dumps(c.get("tools", [])) for c in calls)
    record = completion.read_completion(session_dir)
    assert record is not None and record.stop_reason == "finish_session"
    assert manifest["parallel"]["coordinator"] == "leaf"
    # Found by its bucket and its id; the resume gate refuses it.
    state = paths.state_dir(clone)
    assert session_dir in listing.session_dirs(state)
    summary = listing.summarize_session_dir(session_dir)
    assert summary.session_id == "leaf-l1" and summary.coordinator == "leaf"
    assert summary.mode == "read_only" and summary.stop_reason == "finish_session"
    with pytest.raises(sessions_manifest.ManifestError, match="not resumable"):
        sessions_manifest.read_manifest(session_dir).session_mode()
