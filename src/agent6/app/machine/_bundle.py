# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validate a machine's script bundle: the `.asm.toml` plus its sibling `scripts/`.

Security boundary: every entry under `scripts/` resolves inside the bundle, and every static
tool-command element naming a bundled script exists there. `machine check` and `test` run it
offline; `machine run` and `create` run it again before any execution, so a symlink escaping
the bundle is never read by a tool on an isolation level that cannot read-only bind the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent6.machine import MachineError, MachineSpec, ToolState, load_machine, validate_semantics


def _bundle_script_ref(element: str) -> str | None:
    """Return the relative script path a static command element names, else None.

    A reference is a relative path whose first component is `scripts` (`scripts/fetch.sh`,
    `./scripts/fetch.sh`); an absolute path names an interpreter or binary.
    """
    cleaned = element[2:] if element.startswith("./") else element
    if not cleaned or cleaned.startswith("/"):
        return None
    parts = Path(cleaned).parts
    if parts and parts[0] == "scripts":
        return cleaned
    return None


def _check_scripts_dir(scripts_dir: Path, bundle: Path) -> list[str]:
    """Return a problem per entry under `scripts/` that does not resolve inside the bundle."""
    if not scripts_dir.is_dir():
        return ["bundle 'scripts' exists but is not a directory"]
    problems: list[str] = []
    for entry in sorted(scripts_dir.rglob("*")):
        rel = entry.relative_to(scripts_dir)
        try:
            resolved = entry.resolve()
            # Python 3.14's resolve() returns a symlink loop; stat() raises ELOOP on it.
            entry.stat()
        except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop before 3.14
            problems.append(f"scripts/{rel}: {exc}")
            continue
        if not resolved.is_relative_to(bundle):
            problems.append(f"scripts/{rel} resolves outside the bundle ({resolved}); refused")
    return problems


def _check_command_scripts(name: str, state: ToolState, bundle: Path) -> list[str]:
    """Return a problem per static command element whose script is missing or escapes."""
    problems: list[str] = []
    for element in state.command:
        if "{{" in element:
            continue  # templated: resolves only against a blackboard
        ref = _bundle_script_ref(element)
        if ref is None:
            continue
        target = bundle / ref
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:  # RuntimeError: circular symlink
            problems.append(f"state {name!r}: script {element!r}: {exc}")
            continue
        if not resolved.is_relative_to(bundle):
            problems.append(f"state {name!r}: script {element!r} escapes the bundle")
        elif not target.exists():
            problems.append(f"state {name!r}: script {element!r} not found in bundle")
    return problems


def validate_bundle(spec: MachineSpec, machine_path: Path) -> list[str]:
    """Validate the script bundle beside a machine file.

    Args:
        spec: The parsed machine.
        machine_path: The `.asm.toml`; the bundle is its directory.

    Returns:
        The problems found, empty when the bundle checks out.
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


@dataclass(frozen=True, slots=True)
class MachineFileSummary:
    """Hold one authored file's columns for a machines listing.

    Attributes:
        name: The declared `machine` name; "-" when the file does not parse.
        states: The state count; "-" when the file does not parse.
        spec: "valid", "N issue(s)", or "invalid" (does not parse).
    """

    name: str
    states: str
    spec: str


def summarize_machine_file(path: Path) -> MachineFileSummary:
    """Summarize whether a machine file checks out, as `machine check` and `run` judge it.

    The verdict word is "valid", never "ok" (a machine run's terminal status). A file that does
    not parse has no name to show.

    Args:
        path: The `.asm.toml`.

    Returns:
        The file's listing columns.
    """
    try:
        spec = load_machine(path)
    except (MachineError, OSError):
        return MachineFileSummary("-", "-", "invalid")
    problems = validate_semantics(spec) + validate_bundle(spec, path)
    verdict = "valid" if not problems else f"{len(problems)} issue(s)"
    return MachineFileSummary(spec.machine, str(len(spec.states)), verdict)
