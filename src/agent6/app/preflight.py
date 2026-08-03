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
from agent6.app.reporter import Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.git_ops import is_git_repo
from agent6.models.pricing import lookup_price
from agent6.providers import TranscriptSink
from agent6.sessions.ipc import effective_run_commands
from agent6.sessions.manifest import manifest_for_branch
from agent6.verify_infer import VERIFY_INFER_SYSTEM_PROMPT, infer_verify_command


def budget_preflight(cfg: Config) -> str | None:
    """Budget refusals + notices over the RESOLVED role models, before any spend.

    ``max_tokens_fallback = 0`` refuses when a configured role model cannot be
    metered (zero unmetered tokens allowed); ``max_usd = 0`` refuses when one
    CAN be (a run-nothing-metered rig). Otherwise an unpriced model gets a
    one-line notice naming the fallback bound that covers it. Models chosen
    later (a `/parallel` lane spec) are caught by the tracker's runtime
    backstop instead."""
    models = {rm.model for rm in cfg.models.configured().values()}
    unpriced = sorted(m for m in models if lookup_price(m) is None)
    priced = sorted(m for m in models if lookup_price(m) is not None)
    if cfg.budget.max_tokens_fallback == 0 and unpriced:
        return (
            "[budget].max_tokens_fallback is 0 (unmetered calls refused), but "
            f"{', '.join(repr(m) for m in unpriced)} carr{'ies' if len(unpriced) == 1 else 'y'}"
            " no price data. Raise max_tokens_fallback or use priced models."
        )
    if cfg.budget.max_usd == 0.0 and priced:
        return (
            "[budget].max_usd is 0 (metered calls refused), but "
            f"{', '.join(repr(m) for m in priced)} {'is' if len(priced) == 1 else 'are'} priced."
            " Raise max_usd or use unpriced/local models."
        )
    if unpriced and cfg.budget.max_usd != 0.0:
        fb = cfg.budget.max_tokens_fallback
        bound = "unlimited tokens" if fb == -1 else f"{fb:,} fallback tokens"
        print(
            f"[agent6] NOTE: {', '.join(repr(m) for m in unpriced)} ha"
            f"{'s' if len(unpriced) == 1 else 've'} no price data: that spend is not"
            f" metered by max_usd and is bounded by {bound} instead.",
            file=sys.stderr,
        )
    return None


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


def git_repo_refusal(cwd: Path) -> str | None:
    """Refuse a workspace that is not a git repository, naming the fix.

    A clean early exit instead of the misleading "Git identity not configured"
    error (when there's no global identity) or an ugly failure deeper in the
    run. agent6 needs git to branch, commit per step, and let the user
    review/revert what the agent did.

    This is also the WALL on what becomes the model's workspace: whatever
    directory a run starts in is what the jail mounts writable. Every front-end
    that chooses one has to pass it through here -- `agent6 acp` takes that
    directory from the editor over the wire, and without this a client could
    point a run at any absolute path.

    Returns the message, or None when *cwd* is usable.
    """
    if not cwd.is_dir():
        # Asked git first, `subprocess` could not chdir into a missing
        # directory and the FileNotFoundError surfaced as an opaque
        # internal error. A stale workspace path is the ordinary editor
        # mistake, and it deserves the same named refusal as a wrong one.
        return f"{cwd} is not a directory."
    if is_git_repo(cwd):
        return None
    return (
        f"{cwd} is not a git repository.\n"
        "agent6 needs git here to create a run branch, commit each step, and let"
        " you review or revert what the agent did.\n"
        "  Fix: run `agent6 init` (it offers to set up git for you), or\n"
        '       `git init && git add -A && git commit -m "initial commit"`.'
    )


def require_git_repo(cwd: Path) -> bool:
    """:func:`git_repo_refusal` for a front-end that prints and branches."""
    refusal = git_repo_refusal(cwd)
    if refusal is None:
        return True
    print(f"ERROR: {refusal}", file=sys.stderr)
    return False


def headless_approval_refusal(
    cfg: Config, *, tui_enabled: bool, away: str, can_ask: bool
) -> str | None:
    """Refuse a run that would block forever waiting to be approved.

    `run_commands = "ask"` needs someone to answer. With no TUI, no way for the
    front-end to ask, and no away-mode telling us what an absent operator meant,
    the first command PAUSES indefinitely -- and the verify gate is a command
    too, so that is essentially every run, every `/parallel` lane included.
    Refuse with the fix rather than hang: a run that cannot ask should not start.

    *can_ask* is the front-end's own declaration. Testing the tty here instead
    made this the CLI's question rather than the surface's, so `agent6 acp` --
    whose stdin is the protocol pipe and which asks over
    `session/request_permission` -- had every run refused before it started.

    Returns the message, or None when approval is answerable.
    """
    if cfg.sandbox.run_commands != "ask" or tui_enabled or can_ask or away:
        return None
    return (
        "sandbox.run_commands = 'ask' needs someone to answer, and this run has no"
        " way to ask, no TUI and no away-mode. Every command -- the verify gate"
        " included -- would wait forever.\n"
        "  - unattended: sandbox.run_commands = 'yes' (or --auto-approve), or 'no'"
        " to withhold commands entirely\n"
        "  - attended: start it from a terminal, or set an away-mode"
        " (AGENT6_DETACHED_AWAY=wait|deny) so an absent operator's intent is known"
    )


_RUN_BRANCH_PREFIX = "agent6/"


@dataclass(frozen=True, slots=True)
class BranchChoice:
    """Where a run's branch is cut from (``git.branch_from``). ``start_point`` is
    a branch/sha to cut from, or None to cut from the current HEAD (stack).
    ``abort`` is set when the operator declined at the ``ask`` prompt."""

    start_point: str | None
    abort: bool = False


def _manifest_base_branch(state_dir: Path, branch: str) -> str | None:
    """The base branch the session that cut *branch* recorded it came from."""
    manifest = manifest_for_branch(state_dir, branch)
    return (manifest.base_branch or None) if manifest is not None else None


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
        base = _manifest_base_branch(state_dir, branch)
        if not base:
            break
        branch = base
    return branch


def drop_gate_if_unrunnable(cfg: Config, *, session_dir: Path, reporter: Reporter) -> Config:
    """Empty the verify command when this LEG cannot run one.

    Every command tool is withheld when the effective policy is ``no`` -- the
    operator's configured value, a session deny, or an away-mode of deny -- and
    the gate is a command. Keeping it made the leg unwinnable: nothing could go
    green, so nothing committed, and it finished red over work that may be fine.

    Decided ONCE per leg, by whichever lifecycle starts it, because the system
    prompt is frozen from the same config. A deny that lands MID-leg withdraws
    the tools (the dispatcher's own filter) but must not retroactively make a
    gate that already ran red look like a run that never had one.
    """
    if effective_run_commands(cfg.sandbox.run_commands, session_dir) != "no":
        return cfg
    if cfg.workflow.verify_command:
        reporter.err(
            "[agent6] commands are withheld, and the verify gate is a command:"
            " running gateless (per-step commits, no green gate)."
        )
    return cfg.with_verify_command(())


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

    ``drop_gate_if_unrunnable`` runs first and may have emptied the command, so
    a run that cannot execute one never infers one either.
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
                " gateless\n         (per-step commits, no green gate). If the run"
                " creates a recognizable project, a verify\n         command is"
                " adopted mid-run; pin one with workflow.verify_command.",
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
    return cfg.with_verify_command(inferred.argv)
