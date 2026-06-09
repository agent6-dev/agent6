# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` subcommands + bundle validation + agent runner."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from agent6.cli._common import _agent6_dir, _check_provider_keys, _machines_dir, detect_env
from agent6.cli.egress import _check_network_profile, _warn_if_unsandboxed
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config_layer import (
    load_effective,
    load_effective_with_overlay,
)
from agent6.detect import select_profile
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
from agent6.run_id import new_friendly_id
from agent6.types import SandboxProfile


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
    overlay: dict[str, Any],
    cwd: Path,
    profile: SandboxProfile,
    transcript_dir: Path,
    protect_paths: tuple[Path, ...] = (),
) -> Callable[[AgentRequest], AgentExecResult]:
    """Build the runner an `agent` state uses to drive a confined agent6 loop.

    The machine engine is a host-netns supervisor; each `agent` state runs in
    its OWN subprocess (`agent6.cli.machine_agent`) which confines its egress
    per `sandbox.agent_network` before running the loop — independently of the
    engine and of sibling `tool` states. The subprocess is spawned with a fixed
    argv (no LLM-derived content) and handed the request via a temp file; the
    operator-authored prompt travels in that file, never on the command line.
    ``timeout_secs`` is enforced by killing the subprocess's whole process group
    (true mid-call cancellation, and the per-agent broker is torn down with it).
    """

    def run_agent(request: AgentRequest) -> AgentExecResult:
        payload = {
            "cwd": str(cwd),
            "root": str(cwd),
            "overlay": overlay,
            "profile": profile,
            "transcript_dir": str(transcript_dir),
            "protect_paths": [str(p) for p in protect_paths],
            "request": {
                "model": request.model,
                "prompt": request.prompt,
                "timeout_s": request.timeout_s,
                "provider": request.provider,
                "thinking": request.thinking,
                "temperature": request.temperature,
                "max_usd": request.max_usd,
                "max_input_tokens": request.max_input_tokens,
                "max_output_tokens": request.max_output_tokens,
                "mode": request.mode,
            },
        }
        with tempfile.TemporaryDirectory(prefix="agent6-machine-agent-") as td:
            req_file = Path(td) / "request.json"
            out_file = Path(td) / "result.json"
            req_file.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                sys.executable,
                "-m",
                "agent6.cli.machine_agent",
                str(req_file),
                str(out_file),
            ]
            # Own session/process group so the timeout kill takes the agent
            # subprocess AND its broker/jail children with it.
            proc = subprocess.Popen(argv, start_new_session=True)
            try:
                proc.wait(timeout=request.timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                return AgentExecResult(reason="timeout", payload=None)
            if proc.returncode != 0 or not out_file.is_file():
                return AgentExecResult(reason="error", payload=None)
            try:
                out = json.loads(out_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return AgentExecResult(reason="error", payload=None)
            result_payload = out.get("payload")
            return AgentExecResult(
                reason=str(out.get("reason", "error")),
                payload=result_payload if isinstance(result_payload, dict) else None,
                usd=float(out.get("usd", 0.0)),
                input_tokens=int(out.get("input_tokens", 0)),
                output_tokens=int(out.get("output_tokens", 0)),
            )

    return run_agent


def _machine_protect_paths(machine_path: Path, cwd: Path) -> tuple[Path, ...]:
    """The machine's own ``.asm.toml`` + ``scripts/`` bundle, to mark read-only
    in run jails. Only paths under the jail-mounted cwd are enforceable (a path
    outside cwd isn't in the child's view, so it can't edit it anyway)."""
    cwd_r = cwd.resolve()
    out: list[Path] = []
    for p in (machine_path, machine_path.parent / "scripts"):
        rp = p.resolve()
        if rp.exists() and _is_inside(rp, cwd_r):
            out.append(rp)
    return tuple(out)


def _machine_network_refusal(
    cfg: Config, profile: SandboxProfile, tool_states: list[ToolState]
) -> str | None:
    """A refusal message if this machine's tool-network needs can't be honored.

    Layers machine-specific rules on top of `_check_network_profile` (which
    handles agent_network=local / tool_network=only_explicit_states on
    `hardened`). On `hardened` per-tool isolation is impossible, so we refuse —
    rather than silently mis-confine — whenever isolation is *required*: by the
    operator (`tool_network = "block"`) or by a state (`allow_network = "block"`).
    A networked state under `tool_network = "block"` is a config conflict and is
    refused on any profile. Returns None when fine.
    """
    net_err = _check_network_profile(cfg, profile)
    if net_err is not None:
        return net_err
    tn = cfg.sandbox.tool_network
    has_allow = any(s.allow_network == "allow" for s in tool_states)
    has_block = any(s.allow_network == "block" for s in tool_states)
    if has_allow and tn == "block":
        return (
            'a tool state sets allow_network = "allow" but sandbox.tool_network ='
            " 'block'. Set sandbox.tool_network = 'only_explicit_states' for"
            " audited per-tool egress."
        )
    if tool_states and tn == "block" and profile == "hardened":
        return (
            "isolating a machine's tool-state network requires the strict profile"
            " (a per-tool network namespace); this host supports only 'hardened'."
            " Run on strict, or set sandbox.tool_network = 'allow' (tools share"
            " the host network)."
        )
    if has_block and profile == "hardened":
        return (
            'a tool state sets allow_network = "block" (network must be denied),'
            " but the hardened profile can't isolate one tool's network. Run on"
            ' strict, or use allow_network = "auto" to tolerate the host network.'
        )
    return None


def _cmd_machine_run(path: Path, *, exit_on_wait: bool = False) -> int:  # noqa: PLR0911
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    cwd = Path.cwd()
    states = list(spec.states.values())
    has_agent_state = any(getattr(s, "kind", None) == "agent" for s in states)
    tool_states = [s for s in states if isinstance(s, ToolState)]
    agent_runner: Callable[[AgentRequest], AgentExecResult] | None = None
    # Default profile for confinement-free machines: resolve from the host.
    profile: SandboxProfile = detect_env().detected_profile
    # The running machine's own file + scripts bundle are read-only in every
    # run jail, so a tool/agent can't rewrite its own logic or audited scripts.
    protect_paths = _machine_protect_paths(path, cwd)
    # Load the effective config when an `agent` state needs it, or when there
    # are any tool states (we need sandbox.tool_network/profile to gate egress).
    if has_agent_state or tool_states:
        try:
            cfg = load_effective_with_overlay(cwd, spec.config).config
            if has_agent_state:
                cfg.require_runnable("worker", need_verify=False)
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2
        try:
            profile = select_profile(cfg.sandbox.profile, detect_env())
        except RuntimeError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        refusal = _machine_network_refusal(cfg, profile, tool_states)
        if refusal is not None:
            print(f"REFUSING: {refusal}", file=sys.stderr)
            return 2
        if has_agent_state:
            missing = _check_provider_keys(cfg)
            if missing is not None:
                print(missing, file=sys.stderr)
                return 2
            root = _machines_dir(cwd) / spec.machine
            # The engine is a host-netns supervisor; each agent state confines
            # itself in its own subprocess per sandbox.agent_network.
            agent_runner = _build_machine_agent_runner(
                spec.config, cwd, profile, root / "agent_transcripts", protect_paths
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
                protect_paths=protect_paths,
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
    try:
        profile = select_profile(cfg.sandbox.profile, detect_env())
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    net_err = _check_network_profile(cfg, profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return 2
    _warn_if_unsandboxed(profile)

    scratch = _agent6_dir(cwd) / "machine-drafts" / new_friendly_id()
    scratch.mkdir(parents=True, exist_ok=True)
    # Authoring drafts a machine; it has no machine [config] overlay of its own.
    runner = _build_machine_agent_runner({}, cwd, profile, scratch / "agent_transcripts")

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
        # model omitted (=None): inherit the operator's effective worker model.
        # mode="machine": authoring system prompt + read-only tools (see loop.py).
        result = runner(AgentRequest(prompt=prompt, timeout_s=_CREATE_TIMEOUT_S, mode="machine"))
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
