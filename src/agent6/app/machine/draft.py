# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Keep the attempts, checks and spend of one machine draft."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

from agent6 import budget, kinds, machine, path_safety
from agent6.app import reporter as app_reporter
from agent6.app.machine import check, scriptcheck
from agent6.config import Config
from agent6.sessions import event_log, journal

_CREATE_TIMEOUT_S = 900.0

_CREATE_STOP_REASONS = frozenset({"budget_exhausted", "timeout", "provider_error", "steer_abort"})


def machine_file(workspace: pathlib.Path) -> tuple[pathlib.Path | None, str]:
    """Return the one `.asm.toml` the agent wrote and "", or None and the problem."""
    found = sorted(workspace.glob("*.asm.toml"))
    if not found:
        return None, (
            "No .asm.toml in the workspace: write the machine file there"
            " (and any scripts/ it references) with apply_edit."
        )
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        return None, f"Write ONE machine file; the workspace holds {len(found)}: {names}."
    return found[0], ""


def read_scripts(workspace: pathlib.Path) -> dict[str, str]:
    """Return the bundle's whole `scripts/` tree as {bundle-relative path: source}.

    The whole tree is what the validators passed: a script may import a module or read a data
    file no `tool` command names.

    Args:
        workspace: The drafting workspace, the bundle's root.

    Returns:
        The sources; a file that is not plain readable text is left out, and so is one
        reached through a symlink.
    """
    out: dict[str, str] = {}
    for path in sorted((workspace / "scripts").rglob("*")):
        rel = str(path.relative_to(workspace))
        try:
            out[rel] = path_safety.read_contained(path_safety.contain(workspace, rel))
        except (OSError, UnicodeDecodeError, path_safety.ContainmentError):
            continue
    return out


def attempt_reason(problems: list[str]) -> str:
    """Return a one-line reason an attempt failed.

    Args:
        problems: The attempt's problems; one may span several lines.

    Returns:
        The first problem's first line, plus the block's last line when the first only
        introduces it (a traceback ends with the actual error), then a count of the rest.
    """
    first = problems[0] if problems and problems[0].strip() else ""
    lines = [ln.strip() for ln in first.splitlines() if ln.strip()]
    head = lines[0] if lines else "unknown"
    if head.endswith(":") and len(lines) > 1:
        # Clip the intro first: a long one would truncate away the appended error.
        if len(head) > 100:
            head = head[:97] + "..."
        head = f"{head} {lines[-1]}"
    if len(head) > 160:
        head = head[:157] + "..."
    extra = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{head}{extra}"


type Runner = Callable[[machine.AgentRequest, pathlib.Path | None], machine.AgentExecResult]


@dataclasses.dataclass(slots=True)
class Spend:
    """Total cost and tokens across attempts in one draft."""

    usd: float = 0.0
    own: float = 0.0
    partial: bool = False
    plan: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, result: machine.AgentExecResult) -> None:
        """Book one attempt before its draft is checked."""
        self.usd += result.usd
        self.own += result.usd_own_calls or 0.0
        self.partial |= result.usd_partial
        self.plan += result.plan_observed_points
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens


@dataclasses.dataclass(slots=True)
class Draft:
    """The workspace, journal and running totals for a create session."""

    workspace: pathlib.Path
    events: event_log.EventSink
    execution_id: str
    reporter: app_reporter.Reporter
    cwd: pathlib.Path
    output: pathlib.Path | None
    cfg: Config
    isolation: kinds.IsolationLevel
    config_paths: tuple[pathlib.Path, ...]
    attempt: int = 0
    spend: Spend = dataclasses.field(default_factory=Spend)

    def stop(self, reason: str, *, verification: str = "unverified") -> None:
        """Journal the outcome with the number of attempts spent on this draft."""
        self.events.emit(
            journal.SessionStop(
                execution_id=self.execution_id,
                reason=reason,
                turns=self.attempt,
                verification=verification,
            )
        )

    def report_spend(self) -> None:
        """Report the total and write it after the last subprocess budget update."""
        spend = self.spend
        points = f", {spend.plan:g} plan points" if spend.plan else ""
        self.reporter.err(
            f"machine create: spent {budget.format_usd(spend.usd, partial=spend.partial)}{points}"
            f" (in={spend.input_tokens} tok, out={spend.output_tokens} tok)"
        )
        self.events.emit(
            journal.BudgetUpdate(
                usd_total=spend.usd,
                usd_own_calls=spend.own,
                input_total=spend.input_tokens,
                output_total=spend.output_tokens,
                plan_observed_points=spend.plan,
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedDraft:
    """The machine file and the scripts that passed validation together."""

    path: pathlib.Path
    spec: machine.MachineSpec
    scripts: dict[str, str]


def validate_draft(
    draft: Draft,
    path: pathlib.Path,
) -> ValidatedDraft | list[str]:
    """Check the bundle and collect the diagnostics for the next attempt.

    Args:
        draft: The current draft.
        path: Its candidate machine file.

    Returns:
        The validated bundle or its problems.
    """

    def lint() -> list[str]:
        return scriptcheck.lint_and_typecheck(
            draft.workspace / "scripts",
            draft.isolation,
            fix=True,
            ruff_config_from=draft.output.parent if draft.output is not None else draft.cwd,
            reporter=draft.reporter,
        )

    scripts = read_scripts(draft.workspace)
    try:
        checked = check.check_bundle(path, draft.cwd, config_paths=draft.config_paths)
    except check.CheckError as exc:
        # Lint the scripts so one attempt reveals every problem class.
        return [*exc.problems, *lint()] if scripts else exc.problems
    for warning in checked.warnings:
        draft.reporter.err(f"[agent6] WARNING: {warning}")
    draft.reporter.err("machine create: linting + offline-testing scripts...")
    draft.events.emit(journal.LoopNote(text="linting + offline-testing the draft"))
    problems = lint()
    # ruff --fix rewrites the workspace copies. Publish what validated.
    scripts = read_scripts(draft.workspace)
    offline = scriptcheck.run_offline_tests(draft.workspace, draft.isolation)
    problems.extend(offline.problems)
    if offline.skipped:
        draft.reporter.err(
            f"machine create: {offline.skipped} offline script test"
            f"{'' if offline.skipped == 1 else 's'} not run ({offline.skip_reason})"
        )
    report = machine.dry_run(checked.spec, None)
    problems.extend(
        f"dry-run state {c.name!r}: {c.detail}"
        for c in (*report.states, *report.branches)
        if not c.ok
    )
    return problems or ValidatedDraft(path, checked.spec, scripts)


def try_draft(
    draft: Draft,
    runner: Runner,
    task: str,
    max_attempts: int,
) -> ValidatedDraft | None:
    """Run attempts until a bundle validates or a cap stops the draft.

    Args:
        draft: The create session.
        runner: The confined authoring execution.
        task: The requested machine.
        max_attempts: The attempt limit.

    Returns:
        The first valid bundle, or None after reporting why the draft failed.
    """
    diagnostics: list[str] | None = None
    valid: ValidatedDraft | None = None
    reporter = draft.reporter
    spend = draft.spend
    # One ledger across attempts: each subprocess is otherwise a fresh tracker. -1 = unlimited.
    cfg = draft.cfg
    create_cap = None if cfg.budget.max_usd == -1 else cfg.budget.max_usd
    plan_cap = None if cfg.budget.max_plan_points == -1 else cfg.budget.max_plan_points
    for next_attempt in range(1, max_attempts + 1):
        # A cap binds spend: a plan model's attempts spend no dollars and keep going.
        if spend.usd > 0 and create_cap is not None and spend.usd >= create_cap:
            reporter.err(
                f"machine create: budget max_usd (${create_cap}) exhausted after"
                f" {draft.attempt} attempt(s)"
                f" (spent {budget.format_usd(spend.usd, partial=spend.partial)}); stopping."
            )
            break
        if spend.plan > 0 and plan_cap is not None and spend.plan >= plan_cap:
            reporter.err(
                f"machine create: budget max_plan_points ({plan_cap:g}) exhausted after"
                f" {draft.attempt} attempt(s) ({spend.plan:g} plan points); stopping."
            )
            break
        draft.attempt = next_attempt
        prompt = machine.build_authoring_prompt(
            task, attempt=draft.attempt, diagnostics=diagnostics
        )
        reporter.err(f"machine create: attempt {draft.attempt}/{max_attempts}...")
        draft.events.emit(journal.LoopNote(text=f"attempt {draft.attempt}/{max_attempts}"))
        # effort="off": authoring transcribes a described design. Measured on kimi-k2.6: "low"
        # timed out on 3 of 3 attempts in 30-minute thinks; "off" drafted in 2.5 minutes for $0.02.
        remaining = None if create_cap is None else max(create_cap - spend.usd, 0.0)
        remaining_plan = None if plan_cap is None else max(plan_cap - spend.plan, 0.0)
        result = runner(
            machine.AgentRequest(
                prompt=prompt,
                operator_task=task,
                timeout_s=_CREATE_TIMEOUT_S,
                mode="run",
                effort="off",
                remaining_usd=remaining,
                max_plan_points=remaining_plan,
            ),
            draft.events.path,
        )
        spend.add(result)
        path, missing = machine_file(draft.workspace)
        if path is None:
            diagnostics = [f"{missing} (agent loop reason: {result.reason})"]
            failure = missing
        else:
            candidate = validate_draft(draft, path)
            if isinstance(candidate, ValidatedDraft):
                valid = candidate
                break
            diagnostics = candidate
            failure = attempt_reason(diagnostics)
        reporter.err(f"machine create: attempt {draft.attempt} failed: {failure}")
        if result.reason in _CREATE_STOP_REASONS:
            break
    draft.report_spend()
    if valid is None:
        draft.stop("no_valid_machine")
        reporter.err(f"FAILED: no valid machine after {draft.attempt} attempt(s).")
        if diagnostics:
            reporter.err("Last diagnostics:")
            for problem in diagnostics:
                reporter.err(f"  - {problem}")
        reporter.err(f"The last draft is in {draft.workspace}.")
    return valid
