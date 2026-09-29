# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-loop guards shared by `agent6 run` and `resume`.

Refusals, startup warnings, branch-base resolution and per-run verify-command
resolution. The interactive confirm prompts live in `ui/cli/_preflight` and are
injected by the front-end.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import apply_git_ops_policy, check_provider_keys
from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, parse_seat_spec, plan_metered
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_ref_for,
    chain_tip,
    is_git_repo,
    run_branch_for,
    verify_git_identity,
    worktree_matches,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.kinds import RoleName
from agent6.models.pricing import lookup_price
from agent6.models.validate import (
    configured_model_refusal,
    flag_model_refusal,
    validate_configured_model,
    warning_message,
)
from agent6.providers import TranscriptSink
from agent6.sessions.ipc import AWAY_MODES, effective_run_commands
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.schema import UserQuestion
from agent6.verify_infer import VERIFY_INFER_SYSTEM_PROMPT, infer_verify_command, read_agents_md
from agent6.viewmodel.format import clip_cell
from agent6.viewmodel.listing import session_dirs


class SessionRefusedError(Exception):
    """A preflight refusal already reported; the caller exits with `rc`.

    Attributes:
        rc: The process exit code.
    """

    def __init__(self, rc: int) -> None:
        super().__init__(f"session refused (exit {rc})")
        self.rc = rc


def budget_preflight(
    cfg: Config,
    extra_routes: Iterable[tuple[str, str]] = (),
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> str | None:
    """Judge the budget config against every statically reachable model, before any spend.

    A zero `max_tokens_fallback` refuses an unpriced model, a zero `max_usd` refuses a
    priced one, a zero `max_percent` refuses a plan-metered one; otherwise each unpriced
    or plan-metered model gets a notice naming the bound that covers it. A model chosen
    later (a `/parallel` lane spec) is caught by the tracker's runtime backstop.

    Args:
        cfg: The run's config.
        extra_routes: Further `(provider, model)` pins, such as a machine's per-state
            routes; an unknown provider rides as "".
        reporter: Receives the notices.

    Returns:
        The refusal, or None when the budget admits every route.
    """
    routes = {(rm.provider, rm.model) for rm in cfg.models.configured().values()}
    for spec in cfg.review.seats:
        _persona, seat_provider, seat_model = parse_seat_spec(spec)
        if seat_model:
            routes.add((seat_provider, seat_model))
    routes.update((prov, m) for prov, m in extra_routes if m)

    # Plan-metered routes live in the percent ledger, never in the fallback spend.
    plan_models = sorted({m for prov, m in routes if plan_metered(cfg.providers.get(prov))})
    metered = {(prov, m) for prov, m in routes if not plan_metered(cfg.providers.get(prov))}
    if cfg.budget.max_percent == 0.0 and plan_models:
        return (
            "[budget].max_percent is 0 (plan-metered calls refused), but "
            f"{', '.join(repr(m) for m in plan_models)} route"
            f"{'s' if len(plan_models) == 1 else ''} through a subscription plan."
            " Raise max_percent or reroute those roles."
        )
    unpriced = sorted({m for prov, m in metered if lookup_price(m, prov) is None})
    priced = sorted({m for prov, m in metered if lookup_price(m, prov) is not None})
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
        reporter.note(
            f"{', '.join(repr(m) for m in unpriced)} ha"
            f"{'s' if len(unpriced) == 1 else 've'} no price data: that spend is not"
            f" metered by max_usd and is bounded by {bound} instead."
        )
    if plan_models:
        pct = cfg.budget.max_percent
        bound = "the plan itself" if pct == -1 else f"max_percent {pct:g} points per run"
        tick = (
            " The account reports whole percents, so a cap this small can end the run on"
            " its first tick."
            if 0 < pct < 3
            else ""
        )
        reporter.note(
            f"{', '.join(repr(m) for m in plan_models)} draw"
            f"{'s' if len(plan_models) == 1 else ''} on a subscription plan"
            f" (no dollars; bounded by {bound}).{tick}"
        )
    return None


def warn_if_prompt_override_incomplete(cfg: Config, *, reporter: Reporter = STDIO_REPORTER) -> None:
    """Warn when a custom `prompt.system_prompt_file` omits a core tool contract.

    `finish_session` is the only clean exit and an edit primitive is needed to do
    work; the override is operator-owned, so this warns rather than blocks.

    Args:
        cfg: The run's config.
        reporter: Receives the warning.
    """
    path = cfg.prompt.system_prompt_file
    if not path:
        return
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return  # config validation enforces existence
    missing = [t for t in ("finish_session",) if t not in text]
    if "apply_edit" not in text and "apply_patch" not in text:
        missing.append("apply_edit/apply_patch")
    if missing:
        # Name every absent capability, not the first one.
        actions = []
        if "finish_session" in missing:
            actions.append("terminate")
        if "apply_edit/apply_patch" in missing:
            actions.append("make edits")
        reporter.warn(
            f"custom system_prompt_file ({path}) does not mention "
            f"{', '.join(missing)}; the worker may not know how to "
            f"{' or '.join(actions)}. The override "
            "replaces the built-in run-mode base, so the tool contracts are yours "
            "to preserve. Inspect the assembled prompt with `agent6 prompt show`."
        )


@dataclass(frozen=True, slots=True)
class GitPreflight:
    """Where the run starts.

    Attributes:
        base_sha: HEAD at submission; empty for ask.
        base_branch: The checked-out branch at submission; empty for ask.
    """

    base_sha: str
    base_branch: str


def git_preflight(
    cwd: Path,
    cfg: Config,
    mode: str,
    *,
    confirm_run_on_run_branch: Callable[[str], bool],
    reporter: Reporter,
) -> GitPreflight:
    """Run the git checks a session needs before it creates anything.

    The git ops policy is applied from the lifecycle's own config, so a repo that
    opted into its own hooks gets them on every surface.

    Args:
        cwd: The workspace.
        cfg: The run's config.
        mode: The session mode; ask skips the commit-oriented checks.
        confirm_run_on_run_branch: Asked whether to start on another run's branch.
        reporter: Receives each refusal.

    Returns:
        The base commit and branch.

    Raises:
        SessionRefusedError: A check refused; the refusal is already reported.
    """
    apply_git_ops_policy(cfg)
    identity = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
    # ask is read-only and may run outside a git repo.
    if mode == "ask":
        return GitPreflight(base_sha="", base_branch="")
    try:
        verify_git_identity(cwd, identity)
        # Captured before a run branch exists: `sessions diff` needs the start point.
        pre_status = git_status(cwd)
    except GitError as exc:
        reporter.error(str(exc))
        raise SessionRefusedError(2) from exc
    # A run started on another run's branch piles onto unmerged work: confirm first.
    if (
        mode == "run"
        and pre_status.branch.startswith("agent6/")
        and not confirm_run_on_run_branch(pre_status.branch)
    ):
        reporter.note(
            "aborted. Merge (agent6 sessions merge) or switch branches first, then re-run."
        )
        raise SessionRefusedError(2)
    return GitPreflight(base_sha=pre_status.head_sha, base_branch=pre_status.branch)


# What a run does with uncommitted changes to tracked files; untracked files stay out.
DirtyTreeChoice = Literal["stash", "include", "cancel"]
DIRTY_TREE_OPTIONS: tuple[str, ...] = ("stash", "include", "cancel")


def unmerged_run_holding_the_tree(
    cwd: Path, state_dir: Path, *, except_id: str, modified: Sequence[str]
) -> str:
    """Find the unmerged run whose chain tip holds the working tree's modified files.

    A run's edits sit uncommitted on the checkout until its branch is merged; naming
    that run points the dirty-tree question at the merge. The match is per file, so a
    commit that landed on the base since does not hide it.

    Args:
        cwd: The workspace.
        state_dir: The repo's state directory.
        except_id: The session id to skip.
        modified: The modified tracked paths.

    Returns:
        The newest such run's id, or "".
    """
    if not modified:
        return ""
    for d in session_dirs(state_dir, buckets=("runs",))[:10]:
        if d.name == except_id:
            continue
        with contextlib.suppress(ManifestError):
            if read_manifest(d).merged is not None:
                continue
        tip = chain_tip(cwd, chain_ref_for(d.name))
        if tip is None:
            continue
        try:
            if worktree_matches(cwd, tip, modified):
                return d.name
        except GitError:
            return ""
    return ""


def _dirty_tree_listing(paths: Sequence[str], *, cap: int = 10, unmerged_run: str = "") -> str:
    # No leading indentation: a modal's text pane drops it.
    n = len(paths)
    head = f"{n} tracked {'file has' if n == 1 else 'files have'} uncommitted changes"
    head += (
        f" (the unmerged work of run {unmerged_run}, on {run_branch_for(unmerged_run)}):"
        if unmerged_run
        else ":"
    )
    lines = [f"- {p}" for p in paths[:cap]]
    if n > cap:
        lines.append(f"- ... {n - cap} more")
    return "\n".join([head, *lines])


def dirty_tree_question(paths: Sequence[str], *, unmerged_run: str = "") -> UserQuestion:
    """Build the start question a run with uncommitted tracked changes asks.

    Args:
        paths: The modified tracked paths.
        unmerged_run: The earlier run whose branch holds these changes, when one does.

    Returns:
        The question; the answer's first word is the choice, anything else cancels.
    """
    merge_hint = (
        f"cancel: park the run; `agent6 sessions merge {unmerged_run}` lands them, then resume it"
        if unmerged_run
        else "cancel: park the run; resume it once they are committed or stashed"
    )
    return UserQuestion(
        question=(
            f"{_dirty_tree_listing(paths, unmerged_run=unmerged_run)}\n"
            "How should this run treat them?\n"
            "stash: set them aside for the run (applied back at the end when the tree is"
            " clean, else the `git stash apply` line is printed)\n"
            "include: the run's first commit records them with its own work\n"
            f"{merge_hint}"
        ),
        options=DIRTY_TREE_OPTIONS,
    )


def dirty_tree_choice(answer: str) -> DirtyTreeChoice:
    """Read the choice from an answer's first word.

    Args:
        answer: The operator's answer.

    Returns:
        The choice; anything unrecognised cancels.
    """
    word = answer.strip().split(maxsplit=1)[0].rstrip(":").lower() if answer.strip() else ""
    if word == "stash":
        return "stash"
    if word == "include":
        return "include"
    return "cancel"


def dirty_tree_refusal(paths: Sequence[str], *, unmerged_run: str = "") -> str:
    """Build the refusal for when nobody can answer the dirty-tree question.

    Args:
        paths: The modified tracked paths.
        unmerged_run: The earlier run whose branch holds these changes, when one does.

    Returns:
        The refusal text.
    """
    settle = (
        f" `agent6 sessions merge {unmerged_run}` lands them; or"
        if unmerged_run
        else " Commit or stash them first, or"
    )
    return (
        f"{_dirty_tree_listing(paths, unmerged_run=unmerged_run)}\n"
        "This run has no terminal and no front-end to ask how to treat them."
        f'{settle} decide in config: [git].dirty_tree = "stash" stashes them for the run;'
        ' "include" lets the run\'s first commit record them.'
    )


def git_repo_refusal(cwd: Path) -> str | None:
    """Refuse a workspace that is not a git repository, naming the fix.

    This is also the wall on the model's workspace: the run's directory is what the
    jail mounts writable, so every front-end passes its choice through here.

    Args:
        cwd: The workspace.

    Returns:
        The refusal, or None when the workspace is usable.
    """
    if not cwd.is_dir():
        # git cannot chdir into a missing directory; the error would read as internal.
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


def require_git_repo(cwd: Path, *, reporter: Reporter = STDIO_REPORTER) -> bool:
    """Report the git-repository refusal, if any.

    Args:
        cwd: The workspace.
        reporter: Receives the refusal.

    Returns:
        Whether the workspace is usable.
    """
    refusal = git_repo_refusal(cwd)
    if refusal is None:
        return True
    reporter.refuse(refusal)
    return False


def headless_approval_refusal(
    cfg: Config, *, tui_enabled: bool, away: str, can_ask: bool, clamped: bool = False
) -> str | None:
    """Refuse a run whose first command approval would wait forever.

    `run_commands = "ask"` needs someone to answer, and the verify gate is a command
    too. An away value outside `AWAY_MODES` names no intent, so it refuses on every
    surface.

    Args:
        cfg: The run's config.
        tui_enabled: Whether a TUI can answer.
        away: The `AGENT6_DETACHED_AWAY` value.
        can_ask: The front-end's own declaration that it can ask; the tty is not tested
            here, since `agent6 acp` asks over its protocol pipe.
        clamped: Whether this session kind clamps a standing `yes` to `ask`, so the
            remedy names the flag rather than the config value.

    Returns:
        The refusal, or None when approval is answerable.
    """
    if away and away not in AWAY_MODES:
        return (
            f"AGENT6_DETACHED_AWAY={away!r} is not an away-mode, so an absent operator's"
            " intent is unknown and an approval would wait forever.\n"
            f"  - set AGENT6_DETACHED_AWAY={'|'.join(AWAY_MODES)}"
        )
    if cfg.sandbox.run_commands != "ask":
        return None
    if tui_enabled or can_ask or away:
        return None
    unattended = (
        "--auto-approve (this session kind clamps a standing sandbox.run_commands = 'yes' to 'ask')"
        if clamped
        else "sandbox.run_commands = 'yes' (or --auto-approve)"
    )
    gate = "" if clamped else ", the verify gate included,"
    return (
        "sandbox.run_commands = 'ask' needs someone to answer, and this run has no"
        f" TUI and no away-mode. Every command{gate} would wait forever.\n"
        f"  - unattended: {unattended}, or 'no' to withhold commands entirely\n"
        "  - attended: start it from a terminal, or set an away-mode"
        f" (AGENT6_DETACHED_AWAY={'|'.join(AWAY_MODES)}) so an absent operator's intent is known"
    )


def headless_parking_note(
    cfg: Config, *, tui_enabled: bool, away: str, can_ask: bool
) -> str | None:
    """Note that a fetch or MCP approval would park a run nobody can answer.

    Commands are settled by `yes` or `no`, but a fetch outside `sandbox.fetch_hosts`
    or an MCP call still asks, and that approval parks the run until a front-end
    attaches.

    Args:
        cfg: The run's config.
        tui_enabled: Whether a TUI can answer.
        away: The `AGENT6_DETACHED_AWAY` value.
        can_ask: The front-end's own declaration that it can ask.

    Returns:
        The note, or None when the run can be asked or an away-mode decides.
    """
    if cfg.sandbox.run_commands == "ask" or tui_enabled or can_ask or away:
        return None
    return (
        "no terminal, no front-end and no away-mode: a fetch outside sandbox.fetch_hosts or an"
        " MCP call parks the run at its approval until `agent6 attach` answers it"
        " (AGENT6_DETACHED_AWAY=deny auto-denies, =approve grants every scope)."
    )


def route_preflight(
    cfg: Config, role: RoleName, *, reporter: Reporter, model_flag: str = ""
) -> bool:
    """Check the run's model route before any state exists.

    Every reachable provider needs its key or sign-in, and the configured model is
    checked against the provider's listing so a typo refuses with a did-you-mean.
    A model the listing lacks whose live re-check failed warns and proceeds.

    Args:
        cfg: The run's config.
        role: The role whose route to check.
        reporter: Receives the refusal or warning.
        model_flag: The `--model` value that set the role, so the refusal names the
            flag rather than a config entry the operator never wrote.

    Returns:
        Whether the route can run.
    """
    missing = check_provider_keys(cfg)
    if missing is not None:
        reporter.err(missing)
        return False
    verdict = validate_configured_model(cfg, role)
    if verdict.refused:
        if model_flag:
            reporter.refuse(flag_model_refusal(verdict, cfg, role, model_flag))
            return False
        # Name the entry the operator wrote, not the role that fell back to it.
        reporter.refuse(configured_model_refusal(verdict, cfg.models.source_role(role)))
        return False
    if verdict.warned:
        reporter.warn(warning_message(verdict))
    return True


# A verify argv on one console line: an operator's gate can run to kilobytes.
GATE_TEXT_WIDTH = 120


def gate_text(argv: tuple[str, ...]) -> str:
    """Render a verify command for one console line.

    Args:
        argv: The command; empty for no gate.

    Returns:
        The clipped command line, or "none".
    """
    return clip_cell(" ".join(argv), GATE_TEXT_WIDTH) or "none"


def drop_gate_if_unrunnable(cfg: Config, *, session_dir: Path, reporter: Reporter) -> Config:
    """Empty the verify command when this execution cannot run one.

    Every command tool is withheld when the effective policy is `no`, and the gate is
    a command; keeping it would make the execution unwinnable. Decided once per
    execution, last at its start, because the system prompt is frozen from the same
    config; a deny that lands mid-execution withdraws the tools without unmaking a
    gate that already ran.

    Args:
        cfg: The execution's config.
        session_dir: The session directory holding any session deny.
        reporter: Receives the gateless note.

    Returns:
        The config, gateless when commands are withheld.
    """
    if effective_run_commands(cfg.sandbox.run_commands, session_dir) != "no":
        return cfg
    if cfg.harness.verify_command:
        reporter.note(
            "commands are withheld, and the verify gate is a command"
            f" ({gate_text(cfg.harness.verify_command)}): running gateless"
            " (per-step commits, no green gate)."
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
    reporter: Reporter = STDIO_REPORTER,
) -> Config:
    """Infer a verify command for a run or plan whose config sets none.

    The inferred command lives in memory only; runs never mutate config. Inference
    is layered cheapest first (AGENTS.md, repo signals, then a reviewer-role call
    over the manifests), see `agent6.verify_infer`; nothing inferred means gateless.
    `drop_gate_if_unrunnable` runs after this and has the last word.

    Args:
        cfg: The run's config.
        cwd: The workspace.
        mode: The session mode; only run and plan infer.
        events: Receives `loop.verify_inferred`.
        transcript_sink: The recorder the inference call goes to.
        budget: The tracker the inference call bills.
        reporter: Receives the note naming what was picked.

    Returns:
        The config with the verify command set, or unchanged.
    """
    if mode not in ("run", "plan") or cfg.harness.verify_command:
        return cfg
    if not cfg.harness.verify_infer:
        # Pinned gateless: mid-run adoption reads the same knob and stays off too.
        events.emit("loop.verify_inferred", command=[], source="disabled")
        if mode == "run":
            reporter.note(
                "verify_infer = false: running gateless"
                " (per-step commits, no green gate; nothing is inferred or adopted)."
            )
        return cfg
    agents_md = read_agents_md(cwd)

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
            reporter.note(
                "no verify_command set and none could be inferred; running"
                " gateless\n         (per-step commits, no green gate). If the run"
                " creates a recognizable project, a verify\n         command is"
                " adopted mid-run; pin one with harness.verify_command."
            )
        return cfg
    events.emit("loop.verify_inferred", command=list(inferred.argv), source=inferred.source)
    reporter.note(
        f"verify_command not set; inferred from {inferred.source}:"
        f" {' '.join(inferred.argv)}\n         (this run only; pin it with"
        " harness.verify_command in your per-repo config)"
    )
    return cfg.with_verify_command(inferred.argv)
