# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subprocess entry: run ONE machine `agent` state, self-confined.

A machine run's engine is a thin supervisor that stays in the host network
namespace and makes no network calls itself. Each `agent` state runs *here*, in
its own fresh process, so it can confine its OWN egress per
`sandbox.agent_network` (the broker on `strict`, Landlock on `hardened`),
independently of the engine and of sibling `tool` states. That is what lets a
machine run a broker-confined agent alongside an operator-reviewed, network-carved-out
tool in the same run.

Invoked as ``python -m agent6.cli.machine_agent <request.json> <result.json>``.
It reads a request, sets up the sandbox while still single-threaded, runs the
agent loop to completion, and writes the result. The engine enforces the
timeout by killing this process, which gives true mid-call cancellation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agent6.budget import BudgetTracker
from agent6.cli.egress import (
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
)
from agent6.cli.providers import (
    _build_role_provider,
    _InstrumentedProvider,
    resolve_compaction_thresholds,
)
from agent6.config_layer import load_effective_with_overlay
from agent6.detect import detect
from agent6.events import EventSink
from agent6.git_ops import set_repo_hook_policy
from agent6.providers import TranscriptSink
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import SandboxProfile
from agent6.workflows.loop import Workflow


def _result(
    reason: str, payload: dict[str, Any] | None, budget: BudgetTracker | None
) -> dict[str, Any]:
    usd = 0.0
    inp = out = 0
    if budget is not None:
        usd, _ = budget.estimate_usd()
        snap = budget.snapshot()
        inp_v, out_v = snap["input_total"], snap["output_total"]
        assert isinstance(inp_v, int) and isinstance(out_v, int)
        inp, out = inp_v, out_v
    return {
        "reason": reason,
        "payload": payload,
        "usd": usd,
        "input_tokens": inp,
        "output_tokens": out,
    }


def _run_one(req: dict[str, Any]) -> dict[str, Any]:
    cwd = Path(req["cwd"])
    profile: SandboxProfile = req["profile"]
    root = Path(req["root"])
    transcript_dir = Path(req["transcript_dir"])
    r = req["request"]
    cfg = load_effective_with_overlay(cwd, req["overlay"]).config.with_machine_agent_overrides(
        provider=r["provider"],
        model=r["model"],
        thinking=r["thinking"],
        temperature=r["temperature"],
        max_usd=r["max_usd"],
        max_input_tokens=r["max_input_tokens"],
        max_output_tokens=r["max_output_tokens"],
    )
    set_repo_hook_policy(cfg.git.run_repo_hooks)
    # A mode="run" state commits its work, but this confined process can't read
    # ~/.gitconfig (not a Landlock read root). The engine resolved the identity on
    # the host and passed it down; export it so git uses it regardless of where
    # the config lives. None for read-only states (no commits).
    ident = req.get("commit_identity")
    if isinstance(ident, dict):
        if name := ident.get("name"):
            os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = str(name)
        if email := ident.get("email"):
            os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = str(email)
    # Confine THIS process's egress per sandbox.agent_network (single-threaded
    # here, as required by unshare). The engine already validated the combo, but
    # re-check defensively and fail closed.
    net_err = _check_network_profile(cfg, profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return _result("error", None, None)
    broker, sock_dir, egress_err = _maybe_start_egress(cfg, profile)
    if egress_err is not None:
        print(f"REFUSING: {egress_err}", file=sys.stderr)
        return _result("error", None, None)
    budget: BudgetTracker | None = None
    try:
        landlock_err = _maybe_apply_agent_landlock(cfg, profile, detect())
        if landlock_err is not None:
            print(f"REFUSING: {landlock_err}", file=sys.stderr)
            return _result("error", None, None)
        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=cfg.budget.best_effort_usd_limit,
        )
        inner_provider = _build_role_provider(
            cfg, "worker", transcript_sink=TranscriptSink(transcript_dir), budget=budget
        )
        # An EventSink only when the caller asked for one (events_log set): the
        # `machine create` driver points it at the draft dir's logs.jsonl so the
        # TUI can watch the authoring agent's reasoning + tool calls live, exactly
        # like a run. None for an `agent` state (no watchable log there yet).
        events_log = req.get("events_log")
        events_sink = EventSink(Path(events_log)) if isinstance(events_log, str) else None
        # stream_text: ALWAYS use the streaming transport. Machine agents run
        # headless (cron / nohup) and produce long generations; the
        # non-streaming path drops the connection mid-body on OpenRouter-style
        # gateways (SSE heartbeats corrupt it, observed as "incomplete chunked
        # read" on ~14k-token authoring calls). It is also what feeds the
        # role.*_delta events to the sink above.
        # console_stream: additionally echo reasoning + answer to stderr at a
        # TTY so `machine create` and live `agent` states are watchable.
        rm = cfg.models.resolve("worker")
        provider = _InstrumentedProvider(
            inner=inner_provider,
            role=r.get("role_label", "agent"),
            model=rm.model if rm is not None else "",
            provider_name=rm.provider if rm is not None else "",
            events=events_sink,
            budget=budget,
            stream_text=True,
            console_stream=sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1",
        )
        # Re-confirm the cwd-containment invariant at the subprocess boundary
        # (defense in depth, the engine already filtered these).
        root_r = root.resolve()
        protect = tuple(
            rp
            for p in req.get("protect_paths", [])
            if (rp := Path(p).resolve()).is_relative_to(root_r)
        )
        # "machine" (the `machine create` authoring agent) and "agent" (a
        # running machine's `agent` state, unless it opted into mode="run") are
        # read-only structured-output loops: the dispatcher refuses edits AND
        # run_command/run_verify (defense in depth alongside the read-only tool
        # list) and the loop uses a finish_run-focused prompt.
        mode = r.get("mode", "agent")
        read_only = mode in ("machine", "agent")
        dispatcher = ToolDispatcher(
            root=root,
            config=cfg,
            sandbox_profile=profile,
            approver=None,
            events=events_sink,
            graph_client=None,
            run_root_node_id=None,
            mcp_manager=None,
            extra_protect_paths=protect,
            mode="machine" if read_only else "run",
        )
        compact_drop, compact_summarise = resolve_compaction_thresholds(
            cfg, rm, log=lambda msg: print(msg, file=sys.stderr)
        )
        wf = Workflow(
            root=root,
            config=cfg,
            provider=provider,
            dispatcher=dispatcher,
            logger=lambda msg: print(msg, file=sys.stderr),
            mode=mode if mode in ("machine", "agent") else "run",
            compact_drop_at_chars=compact_drop,
            compact_summarise_at_chars=compact_summarise,
            context_summary_max_tokens=cfg.context.summary_max_tokens,
        )
        result = wf.run(r["prompt"])
        payload = result.finish_payload if result.reason == "finish_run" else None
        return _result(result.reason, payload, budget)
    finally:
        _stop_egress(broker, sock_dir)


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = _run_one(req)
    Path(sys.argv[2]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
