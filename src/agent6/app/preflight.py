# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-loop guards shared by `agent6 run`/`resume`: refusals, startup
warnings, branch-base resolution, and per-run verify-command resolution.
The interactive confirm prompts stay in `ui/cli/_preflight` (they own the
terminal) and are injected by the front-end."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
)
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.git_ops import is_git_repo
from agent6.models.pricing import lookup_price
from agent6.providers import TranscriptSink
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import ManifestError, read_manifest
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


def warn_if_headless_ask(cfg: Config, *, tui_enabled: bool) -> None:
    """Note when run_commands='ask' but no approver is reachable at start.

    A headless run (no TUI, no controlling TTY) has nothing here to answer the
    Allow/Deny prompt, so a run_command PAUSES for a front-end to attach. Say so
    up front instead of letting the agent look wedged. run_verify_command is
    unaffected.
    """
    if cfg.sandbox.run_commands == "ask" and not tui_enabled and not sys.stdin.isatty():
        print(
            "[agent6] NOTE: sandbox.run_commands='ask' with no terminal here, so a"
            " run_command will PAUSE until you attach a front-end to approve it"
            " (`agent6 attach <run>`, the TUI, or the web). Set"
            " sandbox.run_commands='yes'/'no' to auto-approve/deny for unattended runs.",
            file=sys.stderr,
        )


_RUN_BRANCH_PREFIX = "agent6/"


@dataclass(frozen=True, slots=True)
class BranchChoice:
    """Where a run's branch is cut from (``git.branch_from``). ``start_point`` is
    a branch/sha to cut from, or None to cut from the current HEAD (stack).
    ``abort`` is set when the operator declined at the ``ask`` prompt."""

    start_point: str | None
    abort: bool = False


def _manifest_base_branch(state_dir: Path, run_id: str) -> str | None:
    """The base branch a run recorded it was cut from (manifest.base_branch)."""
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    try:
        manifest = read_manifest(layout.run_dir)
    except ManifestError:
        return None
    return manifest.base_branch or None


def resolve_base_branch(state_dir: Path, current_branch: str) -> str:
    """Walk the run-branch chain down to the base line: the nearest ancestor
    branch that is NOT an ``agent6/*`` run branch. A run records the branch it
    was cut from, so we follow those manifests (guarding against a cycle) until a
    non-run branch or the chain breaks. Returns *current_branch* unchanged when
    it is already a base branch."""
    branch = current_branch
    seen: set[str] = set()
    while branch.startswith(_RUN_BRANCH_PREFIX) and branch not in seen:
        seen.add(branch)
        base = _manifest_base_branch(state_dir, branch[len(_RUN_BRANCH_PREFIX) :])
        if not base:
            break
        branch = base
    return branch


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
        inner = build_role_provider(cfg, "reviewer", transcript_sink=transcript_sink, budget=budget)
        rm = cfg.models.resolve("reviewer")
        provider = InstrumentedProvider(
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
        f" {' '.join(inferred.argv)}\n         (this run only; pin it with"
        " workflow.verify_command in your per-repo config)",
        file=sys.stderr,
    )
    return cfg.with_inferred_verify(inferred.argv)
