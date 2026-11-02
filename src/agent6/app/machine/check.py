"""Check a machine bundle and its config before check, test or create accepts it."""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import shutil

from agent6 import machine
from agent6.app.machine import bundle, state_configs
from agent6.config import ConfigError
from agent6.sandbox import detect, tool_paths


class CheckError(Exception):
    """A machine file, bundle or config failed the offline gate."""

    def __init__(self, problems: list[str], label: str = "") -> None:
        self.problems = problems
        self.label = label
        super().__init__("\n".join(problems))


@dataclasses.dataclass(frozen=True, slots=True)
class CheckedBundle:
    """The validated machine, its resolved configs and advisory jail-PATH warnings."""

    spec: machine.MachineSpec
    configs: state_configs.MachineConfigs
    warnings: tuple[str, ...]


def check_bundle(
    path: pathlib.Path,
    cwd: pathlib.Path,
    *,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> CheckedBundle:
    """Load a bundle and resolve the configs it will run under.

    Args:
        path: The machine file.
        cwd: The working directory whose operator config applies.
        config_paths: The explicit config layers, in command-line order.

    Returns:
        The validated bundle and its warnings.

    Raises:
        CheckError: The file, bundle or config is invalid.
    """
    try:
        spec = machine.load_machine(path)
    except machine.MachineError as exc:
        raise CheckError(list(exc.problems)) from exc
    if problems := bundle.validate_bundle(spec, path):
        raise CheckError(problems, "bundle")
    try:
        configs = state_configs.resolve_configs(
            spec, cwd, detect.probed_environment(), config_paths=config_paths
        )
    except state_configs.SandboxWidenedError as exc:
        raise CheckError(exc.problems, "config") from exc
    except (ConfigError, detect.IsolationUnavailableError) as exc:
        raise CheckError([str(exc)], "config") from exc
    return CheckedBundle(spec, configs, tuple(tool_reachability_warnings(spec, path)))


_SUBPROCESS_CALLS = frozenset({"run", "Popen", "call", "check_call", "check_output"})


def script_binaries(scripts_dir: pathlib.Path) -> dict[str, str]:
    """Return the literal first-argv string of each subprocess call in the bundle's scripts.

    Best effort: dynamic argv is invisible to this scan, so absence proves nothing; only a
    hit feeds the reachability warning.

    Args:
        scripts_dir: The bundle's scripts directory.

    Returns:
        Each binary mapped to one script that calls it.
    """
    out: dict[str, str] = {}
    for py in sorted(scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue  # the script linter owns reporting unparseable scripts
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _SUBPROCESS_CALLS or not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.List) and first_arg.elts:
                head = first_arg.elts[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    out.setdefault(head.value, py.name)
    return out


def tool_reachability_warnings(spec: machine.MachineSpec, path: pathlib.Path) -> list[str]:
    """Return a warning for each binary the machine execs that the jail PATH does not resolve.

    Tool-state `command[0]` plus literal subprocess argv in bundle scripts. Offline
    validation mocks subprocess, so without this probe a machine passes check and test
    and dies on its first real state. Advisory: the operator may install the tool later.

    Args:
        spec: The machine.
        path: The machine file.

    Returns:
        The warnings, sorted by binary name.
    """
    search = tool_paths.jail_search_path()
    sources: dict[str, str] = {}
    for name, state in spec.states.items():
        if isinstance(state, machine.ToolState):
            sources.setdefault(state.command[0], f"[states.{name}] command")
    for binary, script in script_binaries(path.parent / "scripts").items():
        sources.setdefault(binary, f"scripts/{script}")
    return [
        f"`{binary}` ({src}) does not resolve on the jail PATH; that state"
        " will fail at run time. Install it into a standard bin dir"
        " (~/.local/bin, /usr/local/bin), or use an absolute path in the"
        " state's command."
        for binary, src in sorted(sources.items())
        if "/" not in binary and shutil.which(binary, path=search) is None
    ]
