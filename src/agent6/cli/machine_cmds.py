# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` subcommands + bundle validation + agent runner."""

from __future__ import annotations

import datetime as _dt
import sys
import threading
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from agent6.budget import BudgetTracker
from agent6.cli._common import _agent6_dir, _check_provider_keys, _machines_dir
from agent6.cli.egress import _warn_if_unsandboxed
from agent6.cli.providers import _build_role_provider
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config_layer import (
    load_effective,
    load_effective_with_overlay,
)
from agent6.detect import detect, select_profile
from agent6.machine import (
    TOML_PAYLOAD_KEY,
    AgentExecResult,
    AgentFact,
    AgentRequest,
    DryRunReport,
    EngineError,
    JournalError,
    LiveWorld,
    MachineError,
    MachineJournal,
    MachineSpec,
    StepEvent,
    ToolState,
    build_authoring_prompt,
    drive,
    dry_run,
    extract_toml,
    load_machine,
    machine_lock,
    render,
    write_source,
)
from agent6.providers import (
    TranscriptSink,
)
from agent6.run_id import new_friendly_id
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import SandboxProfile
from agent6.workflows.loop import RunResult, Workflow


def _is_inside(path: Path, root: Path) -> bool:
    """True iff *path* is *root* or lives beneath it (both already resolved)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _bundle_script_ref(element: str) -> str | None:
    """Return the relative script path a static command element names, else None.

    A bundle script reference is a relative path whose first component is
    ``scripts`` (e.g. ``scripts/fetch.sh`` or ``./scripts/fetch.sh``). Absolute
    paths (``/usr/bin/bash``) are interpreter/binary paths, not bundle refs.
    """
    cleaned = element[2:] if element.startswith("./") else element
    if not cleaned or cleaned.startswith("/"):
        return None
    parts = Path(cleaned).parts
    if parts and parts[0] == "scripts":
        return cleaned
    return None


def _check_scripts_dir(scripts_dir: Path, bundle: Path) -> list[str]:
    """Every entry under ``scripts/`` must resolve to a path inside the bundle."""
    if not scripts_dir.is_dir():
        return ["bundle 'scripts' exists but is not a directory"]
    problems: list[str] = []
    for entry in sorted(scripts_dir.rglob("*")):
        rel = entry.relative_to(scripts_dir)
        try:
            resolved = entry.resolve()
        except (OSError, RuntimeError) as exc:  # RuntimeError: circular symlink
            problems.append(f"scripts/{rel}: {exc}")
            continue
        if not _is_inside(resolved, bundle):
            problems.append(f"scripts/{rel} resolves outside the bundle ({resolved}) — refusing")
    return problems


def _check_command_scripts(name: str, state: ToolState, bundle: Path) -> list[str]:
    """Static tool-command script references must exist and stay in the bundle."""
    problems: list[str] = []
    for element in state.command:
        if "{{" in element:
            continue  # templated; cannot resolve statically
        ref = _bundle_script_ref(element)
        if ref is None:
            continue
        target = bundle / ref
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:  # RuntimeError: circular symlink
            problems.append(f"state {name!r}: script {element!r}: {exc}")
            continue
        if not _is_inside(resolved, bundle):
            problems.append(f"state {name!r}: script {element!r} escapes the bundle")
        elif not target.exists():
            problems.append(f"state {name!r}: script {element!r} not found in bundle")
    return problems


def _validate_bundle(spec: MachineSpec, machine_path: Path) -> list[str]:
    """Validate a machine's script bundle (the ``.asm.toml`` + a sibling ``scripts/``).

    Security-critical: every entry under ``scripts/`` must resolve to a path
    INSIDE the bundle (rejects symlinks that escape via ``..``/absolute), and
    every static tool-command element that references a bundled script must
    exist and stay inside the bundle. Dynamic (templated) command elements are
    skipped — they cannot be resolved without a blackboard.
    """
    try:
        bundle = machine_path.parent.resolve()
    except OSError as exc:
        return [f"cannot resolve bundle directory for {machine_path}: {exc}"]
    problems: list[str] = []
    scripts_dir = bundle / "scripts"
    if scripts_dir.exists():
        problems.extend(_check_scripts_dir(scripts_dir, bundle))
    for name, state in spec.states.items():
        if isinstance(state, ToolState):
            problems.extend(_check_command_scripts(name, state, bundle))
    return problems


def _cmd_machine_check(path: Path) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    bundle_problems = _validate_bundle(spec, path)
    if bundle_problems:
        print(f"FAIL: {path} (bundle)", file=sys.stderr)
        for problem in bundle_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {path} ({spec.machine}, {len(spec.states)} states)")
    return 0


def _cmd_machine_test(path: Path, *, blackboard: Path | None) -> int:
    # `machine test` is `machine check` plus a pure dry-run; reuse the same
    # load + bundle validation so a malformed machine fails the same way.
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    bundle_problems = _validate_bundle(spec, path)
    if bundle_problems:
        print(f"FAIL: {path} (bundle)", file=sys.stderr)
        for problem in bundle_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    fixture: dict[str, Any] | None = None
    if blackboard is not None:
        if not blackboard.is_file():
            print(f"ERROR: blackboard fixture not found: {blackboard}", file=sys.stderr)
            return 2
        try:
            fixture = tomllib.loads(blackboard.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            print(f"ERROR: blackboard fixture is not valid TOML: {exc}", file=sys.stderr)
            return 2
    report = dry_run(spec, fixture)
    _print_dry_run_report(spec, report)
    if report.ok:
        print(
            f"\nOK: {path} dry-run passed ({len(report.states)} states, "
            f"{len(report.branches)} branches)"
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


def _cmd_machine_graph(path: Path, *, fmt: str) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    render_fmt: Literal["mermaid", "dot"] = "dot" if fmt == "dot" else "mermaid"
    print(render(spec, render_fmt), end="")
    return 0


def _build_machine_agent_runner(
    cfg: Config, root: Path, profile: SandboxProfile, transcript_dir: Path
) -> Callable[[AgentRequest], AgentExecResult]:
    """Build the live runner an `agent` state uses to drive a normal agent6 loop.

    Each invocation gets a fresh budget slice, provider (with the state's model),
    dispatcher, and `Workflow`; it runs until the agent calls `finish_run` (or
    the loop stops for another reason) and surfaces the structured payload.

    The state's `timeout_secs` is enforced with a watchdog: the loop runs in a
    daemon thread joined for the timeout. On expiry we return the `timeout`
    outcome; the abandoned thread is bounded by its own one-shot budget slice
    (true mid-call cancellation needs out-of-process execution — Phase 4).
    """

    def run_agent(request: AgentRequest) -> AgentExecResult:
        # Apply this agent state's per-state overrides (model/provider/
        # thinking/temperature/budget) on top of the effective config.
        state_cfg = cfg.with_machine_agent_overrides(
            provider=request.provider,
            model=request.model,
            thinking=request.thinking,
            temperature=request.temperature,
            max_usd=request.max_usd,
            max_input_tokens=request.max_input_tokens,
            max_output_tokens=request.max_output_tokens,
        )
        budget = BudgetTracker(
            max_input_tokens=state_cfg.budget.max_input_tokens,
            max_output_tokens=state_cfg.budget.max_output_tokens,
        )
        transcript_sink = TranscriptSink(transcript_dir)
        provider = _build_role_provider(
            state_cfg,
            "worker",
            transcript_sink=transcript_sink,
            budget=budget,
        )
        dispatcher = ToolDispatcher(
            root=root,
            config=state_cfg,
            sandbox_profile=profile,
            approver=None,
            events=None,
            graph_client=None,
            run_root_node_id=None,
            mcp_manager=None,
        )
        wf = Workflow(
            root=root,
            config=state_cfg,
            provider=provider,
            dispatcher=dispatcher,
            logger=lambda msg: print(msg, file=sys.stderr),
            compact_drop_at_chars=state_cfg.workflow.compact_drop_at_chars,
            compact_summarise_at_chars=state_cfg.workflow.compact_summarise_at_chars,
            context_summary_max_tokens=state_cfg.workflow.context_summary_max_tokens,
        )

        box: dict[str, RunResult | BaseException] = {}

        def _target() -> None:
            try:
                box["result"] = wf.run(request.prompt)
            except Exception as exc:  # surfaced on the main thread
                box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(request.timeout_s)
        usd, _ = budget.estimate_usd()
        snap = budget.snapshot()
        input_total = snap["input_total"]
        output_total = snap["output_total"]
        assert isinstance(input_total, int)
        assert isinstance(output_total, int)
        input_tokens = input_total
        output_tokens = output_total
        if thread.is_alive():
            return AgentExecResult(
                reason="timeout",
                payload=None,
                usd=usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        error = box.get("error")
        if isinstance(error, BaseException):
            raise error
        result = box["result"]
        assert isinstance(result, RunResult)
        payload = result.finish_payload if result.reason == "finish_run" else None
        return AgentExecResult(
            reason=result.reason,
            payload=payload,
            usd=usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return run_agent


def _cmd_machine_run(path: Path, *, exit_on_wait: bool = False) -> int:  # noqa: PLR0911
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    cwd = Path.cwd()
    has_agent_state = any(getattr(state, "kind", None) == "agent" for state in spec.states.values())
    has_network_tool = any(
        isinstance(state, ToolState) and state.allow_network for state in spec.states.values()
    )
    agent_runner: Callable[[AgentRequest], AgentExecResult] | None = None
    # Default profile for tool-only machines (no agent6.toml required): resolve
    # from the host. On non-Linux this is `none` (unsandboxed); on Linux it is
    # strict/hardened per userns support.
    profile: SandboxProfile = detect().detected_profile
    tool_network_allowed = False
    # Load the effective config when an `agent` state needs it, OR when any
    # tool opts into the network (so we can read sandbox.network/profile to
    # gate that egress). Tool-only machines with no networked tool stay
    # config-free and fully isolated, exactly as before.
    if has_agent_state or has_network_tool:
        try:
            cfg = load_effective_with_overlay(cwd, spec.config).config
            if has_agent_state:
                cfg.require_runnable("worker", need_verify=False)
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2
        env = detect()
        try:
            profile = select_profile(cfg.sandbox.profile, env)
        except RuntimeError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        tool_network_allowed = cfg.sandbox.network == "allow"
        if has_network_tool and not tool_network_allowed:
            print(
                "[agent6] note: a tool state requests the network but"
                f" sandbox.network = {cfg.sandbox.network!r}; it will run"
                " network-isolated (set sandbox.network = 'allow' to permit it).",
                file=sys.stderr,
            )
        if has_agent_state:
            missing = _check_provider_keys(cfg)
            if missing is not None:
                print(missing, file=sys.stderr)
                return 2
            root = _machines_dir(cwd) / spec.machine
            agent_runner = _build_machine_agent_runner(
                cfg, cwd, profile, root / "agent_transcripts"
            )
    _warn_if_unsandboxed(profile)
    root = _machines_dir(cwd) / spec.machine
    journal = MachineJournal(root)
    try:
        with machine_lock(root):
            journal.ensure_dirs()
            if not journal.exists():
                write_source(root, path.read_text(encoding="utf-8"))
            world = LiveWorld(
                cwd=cwd,
                journal=journal,
                agent_runner=agent_runner,
                profile=profile,
                tool_network_allowed=tool_network_allowed,
            )
            result = drive(spec, journal, world, live=True, exit_on_wait=exit_on_wait)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.status == "waiting":
        print(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {result.transitions} transitions ({result.reason})"
        )
        return 0
    print(
        f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
        f" after {result.transitions} transitions ({result.reason})"
    )
    return 0 if result.status == "ok" else 1


def _cmd_machine_replay(machine_id: str) -> int:
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"{result.status.upper()}: {spec.machine} replayed to {result.state!r}"
        f" after {result.transitions} transitions ({result.reason})"
    )
    return 0 if result.status in ("ok", "incomplete") else 1


def _cmd_machine_status(machine_id: str) -> int:
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
        events = journal.read()
        snapshot = journal.latest_snapshot()
        pending = journal.read_pending_wait()
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    usd_total = 0.0
    input_total = 0
    output_total = 0
    for event in events:
        if isinstance(event, StepEvent) and isinstance(event.fact, AgentFact):
            usd_total += event.fact.usd
            input_total += event.fact.input_tokens
            output_total += event.fact.output_tokens

    print(f"machine: {spec.machine} (v{spec.version})")
    print(f"  status: {result.status}")
    print(f"  state: {result.state!r}")
    print(f"  transitions: {result.transitions}")
    print(f"  spend: ${usd_total:.4f} (in={input_total} tok, out={output_total} tok)")
    if pending is not None:
        wake = _dt.datetime.fromtimestamp(pending.wake_epoch, tz=_dt.UTC).isoformat()
        print(f"  next wake: {wake} (waiting in {pending.state!r})")
    if snapshot is not None and snapshot.blackboard:
        print("  blackboard:")
        for key, value in snapshot.blackboard.items():
            print(f"    {key} = {value!r}")
    step_events = [e for e in events if isinstance(e, StepEvent)]
    if step_events:
        print("  recent steps:")
        for event in step_events[-5:]:
            print(f"    [{event.seq}] {event.state!r} --{event.label}--> {event.goto!r}")
    return 0


def _cmd_machine_poke(machine_id: str) -> int:
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    try:
        MachineJournal(root).poke()
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"poked {machine_id}: it will wake on its next signal check")
    return 0


_CREATE_TIMEOUT_S = 900.0


_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed"}
)


def _check_machine_text(text: str, scratch: Path) -> tuple[MachineSpec | None, list[str]]:
    """Validate a candidate `.asm.toml` source by parsing it through `load_machine`.

    Returns the parsed spec and an empty problem list on success, or `(None,
    problems)` when the source is invalid.
    """
    candidate_path = scratch / "candidate.asm.toml"
    candidate_path.write_text(text, encoding="utf-8")
    try:
        spec = load_machine(candidate_path)
    except MachineError as exc:
        return None, list(exc.problems)
    return spec, []


def _cmd_machine_create(  # noqa: PLR0911, PLR0912, PLR0915
    task: str, *, output: Path | None, max_attempts: int
) -> int:
    if max_attempts < 1:
        print("ERROR: --max-attempts must be >= 1.", file=sys.stderr)
        return 2
    cwd = Path.cwd()
    try:
        cfg = load_effective(cwd, None).config
        cfg.require_runnable("worker", need_verify=False)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    missing = _check_provider_keys(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2
    env = detect()
    try:
        profile = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    _warn_if_unsandboxed(profile)

    scratch = _agent6_dir(cwd) / "machine-drafts" / new_friendly_id()
    scratch.mkdir(parents=True, exist_ok=True)
    runner = _build_machine_agent_runner(cfg, cwd, profile, scratch / "agent_transcripts")

    prior_toml: str | None = None
    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_toml: str | None = None
    total_usd = 0.0
    for attempt in range(1, max_attempts + 1):
        prompt = build_authoring_prompt(
            task, attempt=attempt, prior_toml=prior_toml, diagnostics=diagnostics
        )
        print(f"machine create: attempt {attempt}/{max_attempts}...", file=sys.stderr)
        result = runner(AgentRequest(model="", prompt=prompt, timeout_s=_CREATE_TIMEOUT_S))
        total_usd += result.usd
        candidate = extract_toml(result.payload)
        if candidate is None:
            diagnostics = [
                f"You did not return a draft: call finish_run with result.{TOML_PAYLOAD_KEY}"
                " set to the complete .asm.toml source as a single string."
                f" (agent loop reason: {result.reason})"
            ]
            prior_toml = None
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_spec, problems = _check_machine_text(candidate, scratch)
        if candidate_spec is not None:
            spec = candidate_spec
            valid_toml = candidate
            break
        prior_toml = candidate
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    print(f"machine create: spent ~${total_usd:.4f}", file=sys.stderr)

    if spec is None or valid_toml is None:
        print(f"FAILED: no valid machine after {max_attempts} attempt(s).", file=sys.stderr)
        if diagnostics:
            print("Last diagnostics:", file=sys.stderr)
            for problem in diagnostics:
                print(f"  - {problem}", file=sys.stderr)
        if prior_toml is not None:
            print("The last (invalid) draft is on stdout for reference.", file=sys.stderr)
            print(prior_toml if prior_toml.endswith("\n") else prior_toml + "\n", end="")
        return 1

    payload = valid_toml if valid_toml.endswith("\n") else valid_toml + "\n"
    if output is not None:
        output.write_text(payload, encoding="utf-8")
        print(
            f"OK: wrote draft to {output} ({spec.machine}, {len(spec.states)} states).",
            file=sys.stderr,
        )
        print(
            "Review and commit it; `machine run` only accepts committed machines.",
            file=sys.stderr,
        )
        return 0

    default_path = cwd / f"{spec.machine}.asm.toml"
    if default_path.exists():
        print(f"REFUSING to overwrite existing {default_path}.", file=sys.stderr)
        print(
            "The validated draft is on stdout; redirect it or re-run with -o <file>.",
            file=sys.stderr,
        )
        print(payload, end="")
        return 1
    default_path.write_text(payload, encoding="utf-8")
    print(
        f"OK: wrote draft to {default_path} ({spec.machine}, {len(spec.states)} states).",
        file=sys.stderr,
    )
    print(
        "Review and commit it; `machine run` only accepts committed machines.",
        file=sys.stderr,
    )
    return 0
