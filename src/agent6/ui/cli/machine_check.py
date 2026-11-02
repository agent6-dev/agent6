# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine check/test/graph`: the offline authoring gate for a machine file.

Parse, semantics, bundle, script lint and types, mock tests, dry-run, and the diagram
render. No provider calls, no real network.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from typing import Any

from agent6 import errors
from agent6.app.machine import check, scriptcheck
from agent6.machine import (
    AgentState,
    DryRunReport,
    MachineError,
    MachineSpec,
    ToolState,
    dry_run,
    fixture_problems,
    load_machine,
    render_dot,
    render_mermaid,
)
from agent6.ui.cli import common


def _fail(path: pathlib.Path, problems: list[str], label: str = "") -> int:
    """Print a FAIL header and problem bullets to stderr.

    Args:
        path: The machine file.
        problems: The problems, one bullet each.
        label: The failing stage, for the header.

    Returns:
        1, the exit code.
    """
    suffix = f" ({label})" if label else ""
    print(f"FAIL: {path}{suffix}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def _leaves(table: dict[str, Any]) -> str:
    """Return an overlay table's leaves as `key = value` pairs on one line."""
    return ", ".join(f"{key} = {value!r}" for key, value in table.items())


def _print_notes(checked: check.CheckedBundle) -> None:
    """Print the shared warnings and sandbox and environment notes."""
    spec = checked.spec
    for warning in checked.warnings:
        common.warn(warning)
    if sandbox := spec.config.get("sandbox"):
        print(f"[config] sets the sandbox of every state: {_leaves(sandbox)}")
    for name, state in spec.states.items():
        if isinstance(state, (AgentState, ToolState)) and (sandbox := state.config.get("sandbox")):
            print(f"[states.{name}] sets its own sandbox: {_leaves(sandbox)}")
        if isinstance(state, ToolState) and state.pass_env:
            print(
                f"[states.{name}] receives the environment variable(s)"
                f" {', '.join(state.pass_env)} when [machine].pass_env allows them"
            )


def _cmd_machine_check(path: pathlib.Path, *, config_paths: tuple[pathlib.Path, ...] = ()) -> int:
    """Validate a machine file offline and print OK or FAIL.

    Args:
        path: The machine file.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The exit code, 1 on FAIL.
    """
    try:
        checked = check.check_bundle(path, pathlib.Path.cwd(), config_paths=config_paths)
    except check.CheckError as exc:
        return _fail(path, exc.problems, exc.label)
    spec = checked.spec
    script_problems = scriptcheck.lint_and_typecheck(path.parent / "scripts")
    if script_problems:
        return _fail(path, script_problems, "scripts")
    _print_notes(checked)
    print(f"OK: {path} ({spec.machine}, {len(spec.states)} states)")
    return 0


def _cmd_machine_test(
    path: pathlib.Path,
    *,
    blackboard: pathlib.Path | None,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> int:
    """Run the offline simulation: validation, script lint and types, mock tests, dry-run.

    Args:
        path: The machine file.
        blackboard: A TOML fixture overlaid on the blackboard defaults.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The exit code, 1 on FAIL.

    Raises:
        OperatorError: The fixture is not valid TOML.
    """
    try:
        checked = check.check_bundle(path, pathlib.Path.cwd(), config_paths=config_paths)
    except check.CheckError as exc:
        return _fail(path, exc.problems, exc.label)
    spec = checked.spec
    script_problems = scriptcheck.lint_and_typecheck(path.parent / "scripts")
    offline = scriptcheck.run_offline_tests(path.parent, checked.configs.machine.isolation)
    script_problems.extend(offline.problems)
    if script_problems:
        return _fail(path, script_problems, "scripts")
    _print_notes(checked)
    fixture: dict[str, Any] | None = None
    if blackboard is not None:
        try:
            fixture = tomllib.loads(errors.read_operator_file(blackboard))
        except tomllib.TOMLDecodeError as exc:
            raise errors.OperatorError(f"blackboard fixture is not valid TOML: {exc}") from exc
        fixture_errors = fixture_problems(spec, fixture)
        if fixture_errors:
            return _fail(path, fixture_errors, "blackboard")
    report = dry_run(spec, fixture)
    _print_dry_run_report(spec, report)
    if report.ok:
        # A skip rides the verdict line: OK with tests silently unrun reads as tests ran green.
        skipped = (
            f"; {common.plural(offline.skipped, 'offline script test')} not run "
            f"({offline.skip_reason})"
            if offline.skipped
            else ""
        )
        print(
            f"\nOK: {path} dry-run passed ({common.plural(len(spec.states), 'state')}, "
            f"including {common.plural(len(report.branches), 'branch', 'branches')}); "
            f"{common.plural(offline.ran, 'offline script test')} passed{skipped}"
        )
        return 0
    print(f"\nFAIL: {path} dry-run found problems", file=sys.stderr)
    return 1


def _print_dry_run_report(spec: MachineSpec, report: DryRunReport) -> None:
    """Render the per-state and per-branch dry-run tables."""
    mark = {True: "ok", False: "FAIL"}
    print(f"machine {spec.machine!r}: per-state dry-run")
    print(f"  {'STATE':<16} {'KIND':<9} {'->LABEL':<9} {'GOTO':<14} STATUS  DETAIL")
    for s in report.states:
        print(
            f"  {s.name:<16} {s.kind:<9} {(s.label or '-'):<9} {(s.goto or '-'):<14}"
            f" {mark[s.ok]:<6}  {s.detail}"
        )
    if report.branches:
        print("\nper-branch routing (fixture overlaid on defaults)")
        print(f"  {'STATE':<16} {'CLAUSE':<7} {'GOTO':<14} STATUS  PREDICATE")
        for b in report.branches:
            clause = "-" if b.clause_index is None else f"[{b.clause_index}]"
            pred = b.detail if not b.ok else (b.predicate or "")
            print(f"  {b.name:<16} {clause:<7} {(b.goto or '-'):<14} {mark[b.ok]:<6}  {pred}")


def _cmd_machine_graph(path: pathlib.Path, *, fmt: str) -> int:
    """Print the machine's diagram as DOT or Mermaid.

    Args:
        path: The machine file.
        fmt: "dot" or "mermaid".

    Returns:
        The exit code, 1 when the file does not parse.
    """
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return _fail(path, exc.problems)
    print((render_dot if fmt == "dot" else render_mermaid)(spec), end="")
    return 0
