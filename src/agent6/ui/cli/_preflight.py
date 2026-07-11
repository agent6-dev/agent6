# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-loop guards shared by `agent6 run`/`resume`: refusals, startup
warnings, and per-run verify-command resolution."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.git_ops import is_git_repo
from agent6.models.pricing import lookup_price
from agent6.providers import TranscriptSink
from agent6.types import SandboxProfile
from agent6.ui.cli._steer import tty_prompt as _tty_prompt
from agent6.ui.cli.providers import (
    _build_role_provider,
    _InstrumentedProvider,
)
from agent6.verify_infer import VERIFY_INFER_SYSTEM_PROMPT, infer_verify_command


def warn_if_usd_unenforceable(cfg: Config) -> None:
    """Warn at startup when ``best_effort_usd_limit`` is set but a configured
    role model has no published price. The USD ceiling sums per-model estimated
    cost, and an unpriced model contributes $0, so ANY unpriced role that spends
    tokens (the worker, but also a distinct reviewer/critic/planner) makes the
    ceiling silently under-count and spend is bounded only by the token
    ceilings. We deliberately do NOT guess a price or terminate early (a wrong
    guess could kill a run mid-task); we just make the gap visible. Anthropic
    publishes no pricing, so this fires for Claude in any role."""
    usd = cfg.budget.best_effort_usd_limit
    if usd <= 0:
        return
    unpriced = sorted(
        {rm.model for rm in cfg.models.configured().values() if lookup_price(rm.model) is None}
    )
    if unpriced:
        print(
            f"[agent6] WARNING: best_effort_usd_limit=${usd:g} cannot be enforced "
            f"for {', '.join(repr(m) for m in unpriced)} (no published price); their "
            f"spend is invisible to the dollar ceiling, so it is bounded only by "
            f"max_input_tokens={cfg.budget.max_input_tokens:,} / "
            f"max_output_tokens={cfg.budget.max_output_tokens:,}. Set explicit "
            f"token ceilings if you need a precise dollar bound.",
            file=sys.stderr,
        )


def warn_if_prompt_override_incomplete(cfg: Config) -> None:
    """Warn when a custom ``prompt.system_prompt_file`` omits the core tool
    contracts the worker needs: ``finish_run`` is the only clean exit, and an
    edit primitive (``apply_edit``/``apply_patch``) is needed to do work. The
    override is advanced + operator-owned, so we don't block -- just flag the
    likely-broken case loudly and point at ``agent6 prompt show``."""
    path = cfg.prompt.system_prompt_file
    if not path:
        return
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return  # config validation already enforces existence; nothing to add
    missing = [t for t in ("finish_run",) if t not in text]
    if "apply_edit" not in text and "apply_patch" not in text:
        missing.append("apply_edit/apply_patch")
    if missing:
        # Name every capability that is actually absent, not just one of them, so
        # a prompt missing both finish_run AND an edit primitive reads correctly.
        actions = []
        if "finish_run" in missing:
            actions.append("terminate")
        if "apply_edit/apply_patch" in missing:
            actions.append("make edits")
        print(
            f"[agent6] WARNING: custom system_prompt_file ({path}) does not mention "
            f"{', '.join(missing)}; the worker may not know how to "
            f"{' or '.join(actions)}. The override "
            "replaces the built-in run-mode base -- you own preserving the tool "
            "contracts. Inspect the assembled prompt with `agent6 prompt show`.",
            file=sys.stderr,
        )


def require_git_repo(cwd: Path) -> bool:
    """Print a friendly error and return False when *cwd* is not a git repo.

    A clean early exit instead of the misleading "Git identity not configured"
    error (when there's no global identity) or an ugly failure deeper in the
    run. agent6 needs git to branch, commit per step, and let the user
    review/revert what the agent did.
    """
    if is_git_repo(cwd):
        return True
    print(
        f"ERROR: {cwd} is not a git repository.\n"
        "agent6 needs git here to create a run branch, commit each step, and let"
        " you review or revert what the agent did.\n"
        "  Fix: run `agent6 init` (it offers to set up git for you), or\n"
        '       `git init && git add -A && git commit -m "initial commit"`.',
        file=sys.stderr,
    )
    return False


def confirm_run_on_run_branch(base_branch: str) -> bool:
    """The checkout is on another run's branch (agent6/<id>); a new run would branch
    off it. Confirm before proceeding. A non-interactive caller (a detached TUI/web
    run) has no terminal to prompt, so it warns and proceeds."""
    warning = (
        f"[agent6] You are on run branch '{base_branch}', not a base branch. A new run\n"
        "  branches off it -- you may have meant to merge it (agent6 runs merge) or\n"
        "  switch back (git switch <base>) first."
    )
    if not sys.stdin.isatty():
        print(warning + " Proceeding (non-interactive).", file=sys.stderr)
        return True
    print(warning, file=sys.stderr)
    try:
        ans = input("  Start a new run here anyway? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


def confirm_unconfined_autorun(selected_profile: SandboxProfile, cfg: Config) -> bool:
    """The one genuinely dangerous combination: the sandbox is OFF and
    run_command is auto-approved, so the agent can run any command on the host
    with no confinement and no prompt. Get one explicit consent at startup when
    interactive; proceed with a loud warning when not (the explicit opt-outs
    are already the consent, and machines/CI must not block). Not a per-command
    guard -- once unconfined, guarding individual commands would be theatre.

    Returns True to proceed, False to abort.
    """
    if selected_profile != "none" or cfg.sandbox.run_commands != "yes":
        return True
    print(
        "[agent6] DANGER: the sandbox is DISABLED and run_command is"
        " AUTO-APPROVED. The agent can run ANY command on this host with no"
        " confinement and no prompt.",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        print("[agent6] proceeding (non-interactive).", file=sys.stderr)
        return True
    answer = _tty_prompt("Continue? [y/N]: ")
    return (answer or "").strip().lower() in {"y", "yes"}


def warn_if_headless_ask(cfg: Config, *, tui_enabled: bool) -> None:
    """Warn when run_commands='ask' but no approver is reachable.

    A headless run (no TUI, no controlling TTY) has nothing to answer the
    Allow/Deny prompt, so every ``run_command`` is auto-denied. Surface that up
    front instead of letting the agent hit confusing mid-run "denied" errors
    (observed dogfooding a headless run). run_verify_command is unaffected.
    """
    if cfg.sandbox.run_commands == "ask" and not tui_enabled and not sys.stdin.isatty():
        print(
            "[agent6] NOTE: sandbox.run_commands='ask' but this run is headless (no TUI,"
            " no TTY), so run_command calls will be auto-denied. Pass --auto-approve"
            " (or set sandbox.run_commands='yes') to auto-approve, or 'no' to withhold"
            " run_command, for headless/CI runs.",
            file=sys.stderr,
        )


def infer_verify_if_unset(
    cfg: Config,
    cwd: Path,
    *,
    mode: str,
    events: EventSink,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
) -> Config:
    """When ``workflow.verify_command`` is unset for a run/plan, infer one and
    inject it IN-MEMORY (never persisted -- runs do not mutate config).

    Layered cheapest-first (AGENTS.md -> repo signals -> a cheap reviewer-role
    LLM call); see ``agent6.verify_infer``. Emits ``loop.verify_inferred`` and
    prints what was picked + that it is per-run. If nothing can be inferred the
    run proceeds GATELESS (no verify gate; the loop commits each editing step).
    """
    if mode not in ("run", "plan") or cfg.workflow.verify_command:
        return cfg
    agents_md = ""
    agents_path = cwd / "AGENTS.md"
    if agents_path.is_file():
        try:
            agents_md = agents_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            agents_md = ""

    def _llm_call(context: str) -> str:
        inner = _build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
        )
        rm = cfg.models.resolve("reviewer")
        provider = _InstrumentedProvider(
            inner=inner,
            role="verify_inferer",
            model=rm.model if rm else "",
            provider_name=rm.provider if rm else "",
            events=events,
            budget=budget,
        )
        resp = provider.call(
            system=VERIFY_INFER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
            tools=[],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.text or ""

    inferred = infer_verify_command(cwd, agents_md, llm_call=_llm_call)
    if inferred is None:
        events.emit("loop.verify_inferred", command=[], source="none")
        if mode == "run":
            print(
                "[agent6] no verify_command set and none could be inferred; running"
                " gateless\n         (per-step commits, no green gate). Set"
                " workflow.verify_command to gate commits on a passing verify.",
                file=sys.stderr,
            )
        return cfg
    events.emit("loop.verify_inferred", command=list(inferred.argv), source=inferred.source)
    print(
        f"[agent6] verify_command not set; inferred from {inferred.source}:"
        f" {' '.join(inferred.argv)}\n         (this run only — set"
        " workflow.verify_command in your per-repo config to pin it)",
        file=sys.stderr,
    )
    return cfg.with_inferred_verify(inferred.argv)
