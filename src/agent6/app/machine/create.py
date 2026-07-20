# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine create`: draft a `.asm.toml` + its scripts from a natural-
language task, validate each attempt (structural + bundle + lint + offline
tests + dry-run), and write the first fully-valid draft.

Authoring runs the same confined agent subprocess a running machine's `agent`
state uses (`build_machine_agent_runner`), in `mode="machine"` with a
finish_run-focused prompt. Output goes through the injected `MachineFrontend`
reporter; the watchable per-draft event log is a separate `EventSink`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent6.app._setup import check_provider_keys, detect_env
from agent6.app.egress import (
    check_network_profile,
    resolve_strict_egress_viability,
    warn_if_unsandboxed,
)
from agent6.app.machine._bundle import validate_bundle
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._scriptcheck import lint_and_typecheck, run_offline_tests
from agent6.app.machine_agent import build_machine_agent_runner
from agent6.config import ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.events import EventSink
from agent6.machine import (
    SCRIPTS_PAYLOAD_KEY,
    TOML_PAYLOAD_KEY,
    AgentRequest,
    MachineError,
    MachineSpec,
    build_authoring_prompt,
    dry_run,
    extract_scripts,
    extract_toml,
    load_machine,
)
from agent6.models.pricing import lookup_price
from agent6.runs.id import new_friendly_id
from agent6.sandbox.detect import ProfileUnavailableError, select_profile

_CREATE_TIMEOUT_S = 900.0


_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed"}
)


def _write_scripts(base_dir: Path, scripts: dict[str, str]) -> None:
    """Write the bundle's helper scripts (keys are bundle-relative, already
    validated by extract_scripts to live under scripts/ with no `..`).

    Defense-in-depth: unlink a pre-existing symlink at the target before writing
    so a planted `scripts/<name>` -> elsewhere link can't redirect the write out
    of the bundle. `validate_bundle` (run by check/run before any execution) is
    the comprehensive backstop for symlinks anywhere in the tree."""
    for rel, content in scripts.items():
        p = base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            p.unlink()
        p.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")


def _check_machine_text(
    text: str, scripts: dict[str, str], scratch: Path
) -> tuple[MachineSpec | None, list[str]]:
    """Validate a candidate `.asm.toml` + its scripts via `load_machine`.

    The scripts are written into the scratch bundle first so the missing-script
    check resolves against this attempt's files only (stale scripts from a prior
    attempt are cleared). Returns the parsed spec + empty problems on success, or
    `(None, problems)` when the source or its script bundle is invalid.
    """
    candidate_path = scratch / "candidate.asm.toml"
    candidate_path.write_text(text, encoding="utf-8")
    shutil.rmtree(scratch / "scripts", ignore_errors=True)
    _write_scripts(scratch, scripts)
    try:
        spec = load_machine(candidate_path)
    except MachineError as exc:
        return None, list(exc.problems)
    bundle_problems = validate_bundle(spec, candidate_path)
    if bundle_problems:
        return None, bundle_problems
    return spec, []


def _attempt_reason(problems: list[str]) -> str:
    """A one-line summary of why an attempt failed: the first problem's first
    line plus, when that line only introduces a block (ends with ':'), the
    block's last line (a test dump or traceback ends with the actual error;
    'offline test x failed (exit 1):' alone explains nothing). A count of
    the rest follows. Keeps the per-attempt log to a line while the full
    diagnostics still feed back into the next prompt."""
    first = problems[0] if problems and problems[0].strip() else ""
    lines = [ln.strip() for ln in first.splitlines() if ln.strip()]
    head = lines[0] if lines else "unknown"
    if head.endswith(":") and len(lines) > 1:
        # Clip the intro first: a long one truncates away the appended
        # error, the one part this line exists to surface.
        if len(head) > 100:
            head = head[:97] + "..."
        head = f"{head} {lines[-1]}"
    if len(head) > 160:
        head = head[:157] + "..."
    extra = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"{head}{extra}"


def create_machine(  # noqa: PLR0911, PLR0912, PLR0915
    task: str, frontend: MachineFrontend, *, output: Path | None, max_attempts: int
) -> int:
    reporter = frontend.reporter
    if max_attempts < 1:
        reporter.err("ERROR: --max-attempts must be >= 1.")
        return 2
    cwd = Path.cwd()
    try:
        cfg = load_effective(cwd, None).config
        cfg.require_runnable("worker")
    except ConfigError as exc:
        reporter.err(f"CONFIG ERROR:\n{exc}")
        return 2
    missing = check_provider_keys(cfg)
    if missing is not None:
        reporter.err(missing)
        return 2
    try:
        profile = select_profile(cfg.sandbox.profile, detect_env())
    except ProfileUnavailableError as exc:
        reporter.err(f"REFUSING: {exc}")
        return 2
    profile, egress_err = resolve_strict_egress_viability(cfg, profile, reporter=reporter)
    if egress_err is not None:
        reporter.err(egress_err)
        return 2
    net_err = check_network_profile(cfg, profile)
    if net_err is not None:
        reporter.err(f"REFUSING: {net_err}")
        return 2
    warn_if_unsandboxed(profile, reporter=reporter)

    scratch = resolved_state_dir(cwd) / "machine-drafts" / new_friendly_id()
    scratch.mkdir(parents=True, exist_ok=True)
    # Persist the natural-language task that drove this draft, so the draft dir is
    # self-describing (the agent_transcripts/ embed it inside the authoring prompt,
    # but a plain prompt.txt is what a human looks for).
    (scratch / "prompt.txt").write_text(task, encoding="utf-8")
    # A watchable event log for the draft: the TUI opens the dashboard on this dir
    # and follows the authoring agent live. The parent owns the run.start header
    # (the NL task) + the per-attempt markers + the final run.end; each attempt's
    # subprocess appends its own role.*_delta / tool.* events to the same file.
    events_log = scratch / "logs.jsonl"
    events = EventSink(events_log)
    events.emit("run.start", user_task=task, mode="machine")
    # Authoring can take minutes with nothing on this terminal; say where the
    # live reasoning streams so the operator can follow instead of wondering.
    reporter.err(
        f"machine create: drafting as {scratch.name} (follow live: agent6 attach {scratch.name})"
    )
    # Authoring drafts a machine; it has no machine [config] overlay of its own.
    runner = build_machine_agent_runner({}, cwd, profile, scratch / "agent_transcripts")

    # The drafted machine's agent states inherit this worker model. If it is
    # unpriced (anthropic-direct, local), steer the draft to best_effort_usd_limit
    # so the freshly-created machine actually runs -- a hard max_usd would refuse.
    # Checked after check_provider_keys refreshed the price cache.
    worker = cfg.models.resolve("worker")
    worker_unpriced = worker is None or lookup_price(worker.model) is None

    prior_toml: str | None = None
    prior_scripts: dict[str, str] = {}
    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_toml: str | None = None
    valid_scripts: dict[str, str] = {}
    total_usd = 0.0
    total_in = 0
    total_out = 0
    attempt = 0  # bound for the run.end below (the loop always runs: max_attempts >= 1)
    for attempt in range(1, max_attempts + 1):
        prompt = build_authoring_prompt(
            task,
            attempt=attempt,
            prior_toml=prior_toml,
            diagnostics=diagnostics,
            prior_scripts=prior_scripts,
            worker_unpriced=worker_unpriced,
        )
        reporter.err(f"machine create: attempt {attempt}/{max_attempts}...")
        events.emit("loop.note", text=f"attempt {attempt}/{max_attempts}")
        # model omitted (=None): inherit the operator's effective worker model.
        # mode="machine": authoring system prompt + read-only tools (see loop.py).
        # thinking="off": authoring is transcription of a described design, not
        # deep derivation. "low" is already the provider default and did not
        # help: kimi-k2.6 spiralled into 30-minute length-capped thinks and
        # timed out on every attempt (0/3 drafts across two spec sizes). With
        # the reasoning channel off it drafted in ~2.5 minutes for $0.02.
        result = runner(
            AgentRequest(
                prompt=prompt, timeout_s=_CREATE_TIMEOUT_S, mode="machine", thinking="off"
            ),
            events_log,
        )
        total_usd += result.usd
        total_in += result.input_tokens
        total_out += result.output_tokens
        candidate = extract_toml(result.payload)
        if candidate is None:
            diagnostics = [
                f"You did not return a draft: call finish_run with result.{TOML_PAYLOAD_KEY}"
                " set to the complete .asm.toml source as a single string."
                f" (agent loop reason: {result.reason})"
            ]
            prior_toml = None
            prior_scripts = {}
            reporter.err(
                f"machine create: attempt {attempt} failed:"
                f" returned no draft (agent stop reason: {result.reason})"
            )
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_scripts = extract_scripts(result.payload)
        candidate_spec, problems = _check_machine_text(candidate, candidate_scripts, scratch)
        if candidate_spec is None:
            # Structural / bundle failure. A missing-script problem (only produced
            # here, never by the lint/test pass below) gets an extra hint pointing
            # the agent at result.scripts.
            if any("not found in bundle" in p for p in problems):
                hint = (
                    f"Return each missing scripts/... file in finish_run"
                    f" result.{SCRIPTS_PAYLOAD_KEY} (a map of the path to its complete source)."
                )
                problems = [*problems, hint]
        else:
            # Structurally valid. Now make it production-ready: lint + type-check
            # the scripts, run their offline `*_test.py` mocks in a jail, and
            # dry-run the routing (synthesized facts through the real reducer;
            # catches e.g. a branch reading a field the schema doesn't declare).
            # Any failure becomes a retry diagnostic so the agent fixes it itself.
            reporter.err("machine create: linting + offline-testing scripts...")
            events.emit("loop.note", text="linting + offline-testing the draft")
            problems = lint_and_typecheck(scratch / "scripts")
            problems.extend(run_offline_tests(scratch, profile))
            report = dry_run(candidate_spec, None)
            problems.extend(
                f"dry-run state {c.name!r}: {c.detail}"
                for c in (*report.states, *report.branches)
                if not c.ok
            )
            if not problems:
                spec = candidate_spec
                valid_toml = candidate
                valid_scripts = candidate_scripts
                break
        # Reached only on a failed attempt (the success path broke above), so
        # surface a one-line reason instead of leaving "attempt N/M" unexplained.
        reporter.err(f"machine create: attempt {attempt} failed: {_attempt_reason(problems)}")
        prior_toml = candidate
        prior_scripts = candidate_scripts
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    reporter.err(f"machine create: spent ~${total_usd:.4f}")
    # Each attempt's subprocess logs its OWN budget.update (resetting to that
    # attempt's spend), so the fold's last one shows only the last attempt. Emit
    # the true cumulative total across attempts so the watchable draft's cost is
    # the real spend, not the last retry's slice.
    events.emit(
        "budget.update",
        usd_total=total_usd,
        input_total=total_in,
        output_total=total_out,
    )
    # End the watchable session (the file-write below is fast and event-less);
    # all_passed marks whether a valid machine was authored, for the TUI status.
    # `iterations` = authoring attempts made, so run.end keeps one shape.
    events.emit(
        "run.end",
        reason="machine create finished",
        iterations=attempt,
        all_passed=spec is not None and valid_toml is not None,
    )

    if spec is None or valid_toml is None:
        reporter.err(f"FAILED: no valid machine after {max_attempts} attempt(s).")
        if diagnostics:
            reporter.err("Last diagnostics:")
            for problem in diagnostics:
                reporter.err(f"  - {problem}")
        if prior_toml is not None:
            reporter.err("The last (invalid) draft is on stdout for reference.")
            # reporter.out re-adds one trailing newline, so strip one to match the
            # original `print(..., end="")` byte-for-byte.
            draft = prior_toml if prior_toml.endswith("\n") else prior_toml + "\n"
            reporter.out(draft.removesuffix("\n"))
        return 1

    payload = valid_toml if valid_toml.endswith("\n") else valid_toml + "\n"
    target = output if output is not None else cwd / f"{spec.machine}.asm.toml"
    if output is None:
        # The default path is documented as clobbering nothing; that covers the
        # WHOLE bundle, not just the machine file -- an LLM-chosen script name
        # colliding with an operator's existing scripts/<name> would otherwise
        # be silently replaced (unrecoverable if uncommitted). `-o` keeps its
        # documented overwrite-freely contract.
        clashes = [
            p for p in (target, *(target.parent / rel for rel in valid_scripts)) if p.exists()
        ]
        if clashes:
            reporter.err("REFUSING to overwrite existing file(s):")
            for clash in clashes:
                reporter.err(f"  {clash}")
            reporter.err("The validated draft is on stdout; redirect it or re-run with -o <file>.")
            reporter.out(payload.removesuffix("\n"))
            return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    _write_scripts(target.parent, valid_scripts)
    scripts_note = f" + {len(valid_scripts)} script(s)" if valid_scripts else ""
    reporter.err(
        f"OK: wrote draft to {target} ({spec.machine}, {len(spec.states)} states){scripts_note}."
    )
    # The scratch validation ran against a clean copy; re-run the STRUCTURAL
    # bundle check on the output dir, which can differ from scratch (e.g. a
    # pre-existing symlink under scripts/). Lint/types are NOT re-run: the
    # written files are byte-identical to the scratch copy that just passed.
    out_problems = validate_bundle(spec, target)
    if out_problems:
        reporter.err("WARNING: the written bundle has problems and won't run yet:")
        for problem in out_problems:
            reporter.err(f"  - {problem}")
    reporter.err("Review and commit it; `machine run` only accepts committed machines.")
    return 0
