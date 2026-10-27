# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the sandbox report's path rows from each scope's resolved grants."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

from agent6 import kinds
from agent6 import paths as agent6_paths
from agent6.app import confine
from agent6.config import Config
from agent6.sandbox import tool_paths
from agent6.tools import _path_safety
from agent6.tools import policy as tools_policy

# Mirrors the launcher's system binds and WriteFile device rules, pinned in tests/security.
FULL_SYSTEM_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives")
DEVICE_WRITES = ("/dev/null", "/dev/zero", "/dev/full")
_TMP = pathlib.Path("/tmp")  # noqa: S108 - a policy path, never opened
_DEV_SHM = pathlib.Path("/dev/shm")  # noqa: S108 - a policy path, never opened


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """A report path with effective access, purpose tags and provenance."""

    mode: str
    path: pathlib.Path
    note: str = ""
    tag: str = ""
    warning: bool = False


def _merge(rows: list[Row], access: Callable[[pathlib.Path], str]) -> list[Row]:
    grouped: dict[pathlib.Path, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.path, []).append(row)
    return [
        Row(
            access(path),
            path,
            "; ".join(dict.fromkeys(r.note for r in group if r.note)),
            ",".join(dict.fromkeys(r.tag for r in group if r.tag)),
            any(r.warning for r in group),
        )
        for path, group in sorted(grouped.items(), key=lambda item: item[0].parts)
    ]


def with_defaults(rows: list[Row], *defaults: Row) -> list[Row]:
    """Return the rows plus boundary entries not already covered by a policy path."""
    return [
        *rows,
        *(d for d in defaults if not any(d.path.is_relative_to(r.path) for r in rows)),
    ]


def _grant_rows(grants: tools_policy.Grants) -> list[Row]:
    rows = [
        Row(mode, real, f"sandbox.{key}" + (f", configured as {path}" if path != real else ""))
        for mode, key, entries in (
            ("ro", "extra_read_paths", grants.read),
            ("rw", "extra_write_paths", grants.write),
        )
        for path, real in entries
    ]
    rows += [
        Row(
            "--",
            d.path,
            f"{d.key}: not granted on its own because"
            f" {'a link on its way sits' if d.via_link else 'it sits'} inside {d.region},"
            " which jailed commands write",
        )
        for d in grants.dropped
    ]
    return rows


def _home_row(
    cfg: Config,
    selected: kinds.IsolationLevel,
    root: pathlib.Path,
    git_grant: pathlib.Path | None,
) -> Row:
    persistent = tools_policy.persistent_jail_home(cfg, selected, root, git_grant)
    if persistent is None:
        return Row("rw", tools_policy.JAIL_TMP_HOME, "HOME, inside the private /tmp")
    why = (
        "sandbox.home = 'cache'"
        if selected == "full"
        else f"sandbox.isolation = '{selected}' has no private /tmp"
    )
    return Row("rw", persistent, f"HOME, persists across runs: {why}", warning=True)


def _tool_rows(mounts: tuple[tool_paths.ToolMount, ...] | None = None) -> list[Row]:
    rows: list[Row] = []
    for mount in tool_paths.operator_tool_mounts() if mounts is None else mounts:
        notes = ["on the jail's PATH"] if mount.on_path else []
        if mount.links:
            first, *others = sorted(mount.links, key=lambda link: link.through is None)
            more = f", {len(others)} more" if others else ""
            through = f" through {first.through}" if first.through is not None else ""
            notes.append(f"{first.source} links into {mount.path}{through}{more}")
        if mount.root_refusal:
            notes.append(f"install root not widened: {mount.root_refusal}")
        rows.append(Row("ro", mount.path, "; ".join(notes), "tool"))
    return rows


def _group_install_roots(
    rows: list[Row],
    mounts: tuple[tool_paths.ToolMount, ...],
    access: Callable[[pathlib.Path], str],
) -> list[Row]:
    roots: dict[pathlib.Path, tool_paths.ToolMount] = {}
    for mount in mounts:
        if (
            mount.on_path
            or not mount.links
            or mount.root_refusal
            or any(link.through is not None for link in mount.links)
        ):
            continue
        try:
            if any(link.source.resolve().parent == mount.path / "bin" for link in mount.links):
                roots[mount.path] = mount
        except OSError:
            # A link that disappears during the report leaves its grant in a separate row.
            continue
    merged = _merge(rows, access)
    groups: dict[tuple[pathlib.Path, str, str], list[Row]] = {}
    for row in merged:
        if row.tag == "tool" and row.path in roots:
            groups.setdefault((row.path.parent, row.mode, row.tag), []).append(row)
    folded: dict[pathlib.Path, Row | None] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        first = group[0]
        links = [roots[row.path].links for row in group]
        shown = [entries[0].source for entries in links]
        names = [str(shown[0])] + [
            link.name if link.parent == shown[0].parent else str(link) for link in shown[1:]
        ]
        more = sum(len(entries) for entries in links) - len(shown)
        note = f"{len(group)} install roots; {', '.join(names[:-1])} and {names[-1]} link into them"
        if more:
            note += f", {more} more"
        members = {row.path for row in group}
        notes = [note] + [
            f"{row.path.name}: {row.note}"
            for row in rows
            if row.path in members and row.tag != "tool" and row.note
        ]
        folded.update(dict.fromkeys(members))
        folded[first.path] = Row(
            first.mode,
            first.path.parent / ("{" + ",".join(row.path.name for row in group) + "}"),
            "; ".join(dict.fromkeys(notes)),
            first.tag,
            any(row.warning for row in group),
        )
    return [row for entry in merged if (row := folded.get(entry.path, entry)) is not None]


def file_tool_rows(ws: _path_safety.Workspace, grants: tools_policy.Grants) -> list[Row]:
    """Return the file tools' rows, including the memory exception to the private state dir."""

    def access(path: pathlib.Path) -> str:
        path = path.resolve()
        if ws.is_denied(path):
            return "--"
        for mode, roots in (("rw", ws.write_roots), ("ro", ws.read_roots)):
            if any(_path_safety.path_within(path, p) for p in (ws.root, *roots)):
                return mode
        return "--"

    rows = [Row("rw", ws.root, "workspace"), *_grant_rows(grants)]
    rows += [Row("rw", p, "repo memory, inside the hidden state dir") for p in ws.exempt]
    rows += [Row("--", p, "hidden") for p in ws.denied]
    rows = with_defaults(
        rows,
        Row("--", pathlib.Path("/"), "refused except the paths below"),
        Row(
            "--",
            agent6_paths.effective_user().home.resolve(),
            "refused, except the rows under it below",
        ),
    )
    return _merge(rows, access)


def _git_rows(
    cfg: Config,
    root: pathlib.Path,
    selected: kinds.IsolationLevel,
    git_grant: pathlib.Path | None,
    access: Callable[[pathlib.Path], str],
) -> list[Row]:
    paths = [(root / ".git").resolve()] if (root / ".git").exists() else []
    if git_grant is not None:
        paths.append(git_grant)
    rows: list[Row] = []
    for path in paths:
        mode = access(path)
        note = f"sandbox.protect_git: configured {cfg.sandbox.protect_git}, "
        if mode == "--":
            note += "inaccessible"
        elif path == git_grant and mode == "ro":
            note += "read-only: repository metadata grant for this linked worktree"
        elif mode == "ro":
            note += "read-only under full"
        else:
            note += "writable"
            if cfg.sandbox.protect_git != "off":
                note += (
                    ": protection requires full isolation"
                    if selected != "full"
                    else ": auto is off under git.control = 'model'"
                )
        rows.append(Row(mode, path, note, warning=mode == "rw" and cfg.protects_git))
    return rows


def unconfined_rows(cfg: Config, root: pathlib.Path) -> list[Row]:
    """Return the deliberate boundary entries for commands running without a jail."""
    user = agent6_paths.effective_user()
    rows = [
        Row(
            "rw",
            pathlib.Path("/"),
            f"sandbox disabled: the whole filesystem and network, as {user.name}"
            f" (uid {user.uid}), no syscall filter",
            warning=True,
        ),
        Row(
            "rw",
            user.home.resolve(),
            "your home: ~/.ssh, ~/.aws and every credential in it",
            warning=True,
        ),
        Row("rw", root, "workspace"),
        _home_row(cfg, "none", root, None),
        Row(
            "rw",
            agent6_paths.global_config_dir(),
            "not hidden: secrets.toml is readable",
            warning=True,
        ),
        Row(
            "rw",
            agent6_paths.state_base(),
            "not hidden: every session's records are writable",
            warning=True,
        ),
        *_git_rows(cfg, root, "none", None, lambda _p: "rw"),
    ]
    return _merge(rows, lambda _p: "rw")


def _system_rows(selected: kinds.IsolationLevel, net: kinds.NetworkMode, abi: int) -> list[Row]:
    if selected == "full":
        rows = [
            Row(
                "ro",
                pathlib.Path(p),
                "the rest of /etc is empty" if p == "/etc/alternatives" and net != "host" else "",
                "sys",
            )
            for p in FULL_SYSTEM_BINDS
        ]
        rows += [
            Row("rw", _TMP, "a private tmpfs the run's commands share, gone at teardown"),
            Row("rw", _DEV_SHM, "private shared memory, gone at teardown"),
            Row(
                "ro",
                pathlib.Path("/proc"),
                "private process filesystem; no execute"
                if abi > 0
                else "private process filesystem",
            ),
        ]
    else:
        rows = [Row("ro", pathlib.Path(p), tag="sys") for p in confine.PARTIAL_SYSTEM_RO]
        rows += [
            Row(
                "rw",
                pathlib.Path(p),
                f"the host's {p}, shared with every process of yours",
                warning=True,
            )
            for p in confine.PARTIAL_SYSTEM_RW
        ]
    rows += [
        Row(
            "rw" if p in DEVICE_WRITES or (selected == "full" and abi == 0) else "ro",
            pathlib.Path(p),
            tag="sys",
        )
        for p in confine.PARTIAL_DEVICE_NODES
    ]
    return rows


def command_rows(
    cfg: Config,
    ws: _path_safety.Workspace,
    grants: tools_policy.Grants,
    selected: kinds.IsolationLevel,
    git_grant: pathlib.Path | None,
    abi: int,
) -> list[Row]:
    """Return command-policy rows, folding sibling linked install roots with equal access."""
    net = tools_policy.resolve_network(cfg, selected)
    configured = _grant_rows(grants)
    mounts = tool_paths.operator_tool_mounts()
    rows = [*_system_rows(selected, net, abi), *_tool_rows(mounts)]
    rows += [r for r in configured if r.mode != "rw"]
    if net == "host":
        rows += [Row("ro", p, tag="net") for p in tools_policy.host_network_reads()]
    if git_grant is not None:
        rows.append(Row("ro", git_grant))
    # The launcher binds read grants before write grants, then the workspace and its protections.
    rows += [r for r in configured if r.mode == "rw"]
    rows += [Row("rw", ws.root, "workspace"), _home_row(cfg, selected, ws.root, git_grant)]
    rows += [
        Row("rw", pathlib.Path(p), "sandbox.extra_device_paths", "sys")
        for p in cfg.sandbox.extra_device_paths
    ]
    rules = [r for r in rows if r.mode != "--"]
    git = ws.root / ".git"
    if selected == "full" and cfg.protects_git and git.resolve().is_relative_to(ws.root):
        rules.append(Row("ro", git.resolve()))
    masked = {
        p.resolve()
        for p in agent6_paths.hidden_paths(pathlib.Path(p) for p in cfg.sandbox.hide_paths)
    }
    if selected == "full":
        rules += [Row("--", p) for p in masked]

    def access(path: pathlib.Path) -> str:
        matching = [r.mode for r in rules if path.is_relative_to(r.path)]
        if selected == "full":
            return matching[-1] if matching else "--"
        return "rw" if "rw" in matching else "ro" if "ro" in matching else "--"

    if selected == "partial":
        for path, resource in (
            (pathlib.Path("/proc"), "process filesystem"),
            (_DEV_SHM, "host shared memory"),
        ):
            unavailable = access(path) == "--"
            note = resource + (" unavailable" if unavailable else " available")
            rows.append(Row(access(path), path, note, warning=unavailable))
    rows += _git_rows(cfg, ws.root, selected, git_grant, access)
    rows += [
        Row(
            "--",
            p,
            "hidden" if access(p) == "--" else "not hidden: partial cannot mask this grant",
            warning=access(p) != "--",
        )
        for p in sorted(masked)
    ]
    rows = with_defaults(
        rows,
        Row(
            "--",
            pathlib.Path("/"),
            "empty except /proc, /dev and the paths below"
            if selected == "full"
            else "denied except the paths below",
        ),
        Row(
            "--",
            agent6_paths.effective_user().home.resolve(),
            "not mounted, except the rows under it below"
            if selected == "full"
            else "denied, except the rows under it below",
        ),
    )
    return _group_install_roots(rows, mounts, access)
