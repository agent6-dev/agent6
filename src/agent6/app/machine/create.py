# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Draft a machine and its scripts from a task, for `agent6 machine create`.

Each attempt is validated (structure, bundle, lint, offline tests, dry run) and the first
valid draft is written. Authoring runs the confined agent subprocess a machine's `agent` state
uses, over a drafting workspace of its own; agent6 validates what is on disk between attempts.
"""

from __future__ import annotations

import contextlib
import pathlib
import shutil

from agent6 import events as agent6_events
from agent6 import git_ops, kinds, paths, portable
from agent6.app import _session, _setup, machine_agent, parallel, preflight
from agent6.app import reporter as app_reporter
from agent6.app.machine import _bundle, _frontend, _scriptcheck
from agent6.config import ConfigError, layer
from agent6.machine import (
    AgentRequest,
    MachineError,
    MachineSpec,
    build_authoring_prompt,
    dry_run,
    load_machine,
)
from agent6.sessions import id, ipc, layout

_CREATE_TIMEOUT_S = 900.0

_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed", "steer_abort"}
)


def _write_scripts(base_dir: pathlib.Path, scripts: dict[str, str]) -> None:
    """Write the bundle's scripts, keyed by bundle-relative path.

    A pre-existing symlink at a target is unlinked first, so a planted link cannot redirect the
    write out of the bundle; `validate_bundle` is the backstop for symlinks anywhere in the tree.
    """
    for rel, content in scripts.items():
        p = base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            p.unlink()
        portable.atomic_write(p, content if content.endswith("\n") else content + "\n")


def _machine_file(workspace: pathlib.Path) -> tuple[pathlib.Path | None, str]:
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


def _check_bundle(path: pathlib.Path) -> tuple[MachineSpec | None, list[str]]:
    """Return the parsed machine and no problems, or None and why the file or bundle is invalid."""
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return None, list(exc.problems)
    bundle_problems = _bundle.validate_bundle(spec, path)
    if bundle_problems:
        return None, bundle_problems
    return spec, []


def _read_scripts(workspace: pathlib.Path) -> dict[str, str]:
    """Return the bundle's whole `scripts/` tree as {bundle-relative path: source}.

    The whole tree is what the validators passed: a script may import a module or read a data
    file no `tool` command names. A file that is not plain readable text is left out;
    `validate_bundle` refuses a bundle that needed it.
    """
    out: dict[str, str] = {}
    for path in sorted((workspace / "scripts").rglob("*")):
        rel = str(path.relative_to(workspace))
        if path.is_symlink() or not path.is_file():
            continue
        try:
            out[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def _attempt_reason(problems: list[str]) -> str:
    """Return a one-line reason an attempt failed.

    The first problem's first line, plus the block's last line when the first only introduces
    it (a traceback ends with the actual error), then a count of the rest.
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


def _discard_workspace(workspace: pathlib.Path, reporter: app_reporter.Reporter) -> None:
    """Remove a drafting workspace whose bundle is published, with its per-repo state dir.

    The prune sweep knows fan-out clones and fork worktrees; a drafting workspace is neither.
    """
    errors: list[str] = []
    for path in (paths.state_dir(workspace), workspace):
        # A workspace that never ran has no state dir; rmtree reports a missing path through onexc.
        if path.exists():
            shutil.rmtree(path, onexc=lambda _fn, target, exc: errors.append(f"{target}: {exc}"))
    # The per-repo base every subordinate tree sits in goes only when empty.
    with contextlib.suppress(OSError):
        workspace.parent.rmdir()
    if errors:
        reporter.err(f"machine create: the drafting workspace stays ({errors[0]})")


def new_draft_dir(state: pathlib.Path) -> pathlib.Path:
    """Return a new draft's directory: an unused session id under the machine bucket."""
    bucket = kinds.session_bucket("machine")
    return layout.bucket_dir(state, bucket) / id.unused_session_id(state, bucket)


def create_machine(  # noqa: C901, PLR0911, PLR0912, PLR0915  # the create loop's attempts and refusals, in order
    task: str,
    frontend: _frontend.MachineFrontend,
    *,
    output: pathlib.Path | None,
    max_attempts: int,
    config_path: pathlib.Path | None = None,
) -> int:
    """Draft, validate and write a machine.

    Args:
        task: The natural-language task the machine should carry out.
        frontend: The reporter to write through.
        output: Where to write the `.asm.toml`; None writes `<machine>.asm.toml` in cwd and
            refuses to overwrite anything.
        max_attempts: How many authoring attempts to make.
        config_path: An explicit config file, else the effective one.

    Returns:
        The exit code: 0 on a written bundle, 1 on a failed draft, 2 on a refusal.
    """
    reporter = frontend.reporter
    if max_attempts < 1:
        reporter.error("--max-attempts must be >= 1.")
        return 2
    cwd = pathlib.Path.cwd()
    try:
        eff = layer.load_effective(cwd, config_path)
        cfg = eff.config
        cfg.require_runnable("worker")
    except ConfigError as exc:
        reporter.error(str(exc))
        return 2
    missing = _setup.check_provider_keys(cfg)
    if missing is not None:
        reporter.err(missing)
        return 2
    try:
        # A create session withholds the command tools: an unconfined autorun needs no confirmation.
        isolation = _session.select_isolation(
            cfg,
            cwd=cwd,
            confirm_unconfined=lambda _isolation, _cfg: True,
            reporter=reporter,
            explicit_leaves=eff.explicit_leaves,
        )
    except preflight.SessionRefusedError as refusal:
        return refusal.rc

    scratch = new_draft_dir(paths.state_dir(cwd))
    paths.mkdir_for_real_user(scratch.parent)
    scratch.mkdir(mode=0o700)
    (scratch / "prompt.txt").write_text(task, encoding="utf-8")
    # The draft's watchable log: this process owns session.start, the attempt markers and
    # session.end; each attempt's subprocess appends its own events to the same file.
    events_log = scratch / layout.LOGS_NAME
    events = agent6_events.EventSink(events_log)
    # A terminal draft has its own session.end, which the status reads first: no clearing.
    ipc.emit_session_start(events, scratch, "session.start", user_task=task, mode="machine")
    reporter.err(
        f"machine create: drafting as {scratch.name} (follow live: agent6 attach {scratch.name})"
    )
    # An empty repo of its own beside the other subordinate trees: the jail masks the state dir.
    workspace = parallel.subordinate_workdir_root(cfg, cwd, scratch.name)
    try:
        workspace.mkdir(parents=True)
        git_ops.init_repo(workspace)
    except (OSError, git_ops.GitError) as exc:
        reporter.error(f"could not prepare the drafting workspace {workspace}: {exc}")
        events.emit("session.end", reason="workspace_failed", iterations=0, all_passed=False)
        return 1
    # The complete effective config as the overlay: omitting values equal to built-in defaults
    # loses an explicit repo reset when the workspace reloads the operator's global layer.
    overlay = cfg.model_dump(mode="json")
    overlay.pop("preset", None)  # the overlay layer forbids a preset
    # The execution writes files and nothing else; agent6 runs the validators itself.
    overlay["sandbox"] = {**overlay.get("sandbox", {}), "run_commands": "no", "fetch_hosts": []}
    overlay["harness"] = {**overlay.get("harness", {}), "metric": None}
    # Resolved on the host: the confined execution cannot read ~/.gitconfig.
    try:
        name, email = git_ops.verify_git_identity(
            workspace, git_ops.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        )
    except git_ops.GitError as exc:
        reporter.error(str(exc))
        events.emit("session.end", reason="no_git_identity", iterations=0, all_passed=False)
        return 2
    runner = machine_agent.build_machine_agent_runner(
        overlay,
        workspace,
        isolation,
        scratch / "agent_transcripts",
        commit_identity=git_ops.CommitIdentity(name=name, email=email),
    )

    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_path: pathlib.Path | None = None
    valid_scripts: dict[str, str] = {}
    total_usd = 0.0
    total_in = 0
    total_out = 0
    # One ledger across attempts: each subprocess is otherwise a fresh tracker. -1 = unlimited.
    create_cap = None if cfg.budget.max_usd == -1 else cfg.budget.max_usd
    attempt = 0
    for next_attempt in range(1, max_attempts + 1):
        if create_cap is not None and total_usd >= create_cap:
            reporter.err(
                f"machine create: budget max_usd (${create_cap}) exhausted after"
                f" {attempt} attempt(s) (spent ~${total_usd:.4f}); stopping."
            )
            break
        attempt = next_attempt
        prompt = build_authoring_prompt(task, attempt=attempt, diagnostics=diagnostics)
        reporter.err(f"machine create: attempt {attempt}/{max_attempts}...")
        events.emit("loop.note", text=f"attempt {attempt}/{max_attempts}")
        # effort="off": authoring transcribes a described design. Measured on kimi-k2.6: "low"
        # timed out on 3 of 3 attempts in 30-minute thinks; "off" drafted in 2.5 minutes for $0.02.
        remaining = None if create_cap is None else max(create_cap - total_usd, 0.0)
        result = runner(
            AgentRequest(
                prompt=prompt,
                timeout_s=_CREATE_TIMEOUT_S,
                mode="run",
                effort="off",
                max_usd=remaining,
            ),
            events_log,
        )
        total_usd += result.usd
        total_in += result.input_tokens
        total_out += result.output_tokens
        candidate_path, missing = _machine_file(workspace)
        if candidate_path is None:
            diagnostics = [f"{missing} (agent loop reason: {result.reason})"]
            reporter.err(f"machine create: attempt {attempt} failed: {missing}")
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_scripts = _read_scripts(workspace)
        candidate_spec, problems = _check_bundle(candidate_path)
        if candidate_spec is None:
            if candidate_scripts:
                # Lint the scripts now, so one attempt reveals every problem class.
                problems = [
                    *problems,
                    *_scriptcheck.lint_and_typecheck(
                        workspace / "scripts",
                        fix=True,
                        ruff_config_from=output.parent if output is not None else cwd,
                    ),
                ]
        else:
            # Lint, type-check, offline-test and dry-run; a failure is the next attempt's input.
            reporter.err("machine create: linting + offline-testing scripts...")
            events.emit("loop.note", text="linting + offline-testing the draft")
            problems = _scriptcheck.lint_and_typecheck(
                workspace / "scripts",
                fix=True,
                ruff_config_from=output.parent if output is not None else cwd,
            )
            # ruff --fix rewrote the workspace copies: publish what validated.
            candidate_scripts = _read_scripts(workspace)
            offline = _scriptcheck.run_offline_tests(workspace, isolation)
            problems.extend(offline.problems)
            if offline.skipped:
                reporter.err(
                    f"machine create: {offline.skipped} offline script test"
                    f"{'' if offline.skipped == 1 else 's'} not run"
                    f" ({offline.skip_reason}); static checks still applied"
                )
            report = dry_run(candidate_spec, None)
            problems.extend(
                f"dry-run state {c.name!r}: {c.detail}"
                for c in (*report.states, *report.branches)
                if not c.ok
            )
            if not problems:
                spec = candidate_spec
                valid_path = candidate_path
                valid_scripts = candidate_scripts
                break
        reporter.err(f"machine create: attempt {attempt} failed: {_attempt_reason(problems)}")
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    reporter.err(f"machine create: spent ~${total_usd:.4f}")
    # Each attempt's subprocess logs its own budget.update; the fold's last one must be the total.
    events.emit(
        "budget.update",
        usd_total=total_usd,
        input_total=total_in,
        output_total=total_out,
    )
    # session.end reasons are tokens: the listing prints one beside "failed". iterations = attempts.
    if spec is None or valid_path is None:
        events.emit("session.end", reason="no_valid_machine", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: no valid machine after {attempt} attempt(s).")
        if diagnostics:
            reporter.err("Last diagnostics:")
            for problem in diagnostics:
                reporter.err(f"  - {problem}")
        reporter.err(f"The last draft is in {workspace}.")
        return 1

    draft_text = valid_path.read_text(encoding="utf-8")
    payload = draft_text if draft_text.endswith("\n") else draft_text + "\n"
    target = output if output is not None else cwd / f"{spec.machine}.asm.toml"
    if output is None:
        # The default path clobbers nothing, scripts included; `-o` overwrites freely.
        clashes = [
            p
            for p in (target, *(target.parent / rel for rel in valid_scripts))
            if p.exists() or p.is_symlink()
        ]
        if clashes:
            events.emit(
                "session.end", reason="output_collision", iterations=attempt, all_passed=False
            )
            reporter.err("REFUSING to overwrite existing file(s):")
            for clash in clashes:
                reporter.err(f"  {clash}")
            reporter.err(f"The validated bundle is in {workspace}; it is also on stdout.")
            reporter.out(payload.removesuffix("\n"))
            return 2
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Scripts first, the machine file last: a death mid-publish leaves inert scripts.
        _write_scripts(target.parent, valid_scripts)
        portable.atomic_write(target, payload)
    except OSError as exc:
        events.emit("session.end", reason="write_failed", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: could not write the bundle to {target.parent}: {exc}")
        reporter.err(f"The validated bundle is in {workspace}; it is also on stdout.")
        reporter.out(payload.removesuffix("\n"))
        return 1
    # A pre-existing symlink under scripts/ can make the published bundle differ: check it.
    out_problems = _bundle.validate_bundle(spec, target)
    if out_problems:
        events.emit("session.end", reason="bundle_invalid", iterations=attempt, all_passed=False)
        reporter.err(f"FAILED: the bundle written to {target.parent} does not validate:")
        for problem in out_problems:
            reporter.err(f"  - {problem}")
        return 1
    events.emit("session.end", reason="machine_created", iterations=attempt, all_passed=True)
    scripts_note = f" + {len(valid_scripts)} script(s)" if valid_scripts else ""
    reporter.err(
        f"OK: wrote draft to {target} ({spec.machine}, {len(spec.states)} states){scripts_note}."
    )
    reporter.err("Review and commit it; `machine run` only accepts committed machines.")
    # A failure keeps the workspace: the draft in it is what the operator reads.
    _discard_workspace(workspace, reporter)
    return 0
