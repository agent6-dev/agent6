# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Discover skills in the SKILL.md format from the operator's directories.

A skill is a directory holding a `SKILL.md` whose frontmatter carries `name`
and `description`. A leaf: it scans, parses the two fields and applies the
operator's state map; where the content goes is the consumers' business. The
frontmatter parser covers the scalar, quoted, folded and literal forms the two
fields use, not YAML; anything unparseable is a warning, never a crash.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import stat
from collections.abc import Mapping, Sequence

# The agentskills.io name rule.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def is_valid_skill_name(name: str) -> bool:
    """Return whether the name is valid: alphanumeric plus hyphens, so one safe path component.

    Discovery and install both gate on this; an unvalidated name from untrusted
    frontmatter could escape the skills dir.
    """
    return bool(_NAME_RE.match(name))


@dataclasses.dataclass(frozen=True, slots=True)
class Skill:
    """One discovered skill.

    Attributes:
        name: The frontmatter name.
        description: The frontmatter description.
        dir: The skill's directory.
        text: The whole SKILL.md.
    """

    name: str
    description: str
    dir: pathlib.Path
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedSkills:
    """Discovery output after the operator's state map is applied.

    Attributes:
        enabled: The skills the system-prompt index offers on demand.
        always: The skills whose full text is injected instead.
        warnings: What discovery could not load or resolve.
    """

    enabled: tuple[Skill, ...]
    always: tuple[Skill, ...]
    warnings: tuple[str, ...]


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse a SKILL.md's leading frontmatter block.

    Returns:
        The fields and the warnings; a missing or unclosed block yields no fields and
        a warning.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["no frontmatter block (file must start with ---)"]
    try:
        end = next(
            i
            for i, line in enumerate(lines[1:], start=1)
            if line == line.lstrip() and line.strip() == "---"
        )
    except StopIteration:
        return {}, ["unclosed frontmatter block (no closing ---)"]

    fields: dict[str, str] = {}
    warnings: list[str] = []
    i = 1
    while i < end:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = _KEY_RE.match(line)
        if m is None:
            warnings.append(f"unparseable frontmatter line {i + 1}: {line.strip()!r}")
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if value in (">", ">-", "|", "|-"):
            block: list[str] = []
            i += 1
            while i < end and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                block.append(lines[i].strip())
                i += 1
            joiner = " " if value.startswith(">") else "\n"
            fields[key] = joiner.join(b for b in block if b).strip()
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        fields[key] = value
        i += 1
    return fields, warnings


def _load_skill(skill_dir: pathlib.Path) -> tuple[Skill | None, list[str]]:
    """Load one skill directory.

    Returns:
        The skill, or None when it does not load, and the warnings.
    """
    path = skill_dir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # Discovery runs at startup; one unreadable file must not take down every run.
        return None, [f"{path}: unreadable ({exc})"]
    fields, warnings = parse_frontmatter(text)
    name = fields.get("name", "")
    description = fields.get("description", "")
    if not name or not description:
        return None, [f"{path}: missing required frontmatter name/description", *warnings]
    if not is_valid_skill_name(name):
        return None, [f"{path}: invalid skill name {name!r}", *warnings]
    prefixed = [f"{path}: {w}" for w in warnings]
    if name != skill_dir.name:
        prefixed.append(f"{path}: frontmatter name {name!r} != directory {skill_dir.name!r}")
    return Skill(name=name, description=description, dir=skill_dir, text=text), prefixed


def _mode(path: pathlib.Path) -> int | None:
    """Return the path's mode, or None when it does not exist.

    A path the process may not reach raises, where `Path.is_dir` reports it absent
    from Python 3.13 on and discovery would skip it silently.
    """
    try:
        return path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None


def _is_dir(path: pathlib.Path) -> bool:
    """Return whether the path is a directory, raising when it cannot be reached."""
    return (mode := _mode(path)) is not None and stat.S_ISDIR(mode)


def _is_file(path: pathlib.Path) -> bool:
    """Return whether the path is a regular file, raising when it cannot be reached."""
    return (mode := _mode(path)) is not None and stat.S_ISREG(mode)


def discover_skills(dirs: Sequence[pathlib.Path]) -> tuple[tuple[Skill, ...], tuple[str, ...]]:
    """Scan directories for skills, the first directory winning a duplicate name.

    A directory holds skill subdirectories or is a single skill itself; dotted
    entries are ignored and a missing directory is fine.

    Args:
        dirs: The directories, in precedence order.

    Returns:
        The skills and the warnings.
    """
    found: dict[str, Skill] = {}
    warnings: list[str] = []
    for base in dirs:
        try:
            if not _is_dir(base):
                continue
            if _is_file(base / "SKILL.md"):
                candidates = [base]
            else:
                candidates = sorted(
                    p
                    for p in base.iterdir()
                    if not p.name.startswith(".") and _is_dir(p) and _is_file(p / "SKILL.md")
                )
        except OSError as exc:
            # Discovery runs at startup; one unlistable dir must not take down every run.
            warnings.append(f"{base}: unreadable ({exc})")
            continue
        for skill_dir in candidates:
            skill, warns = _load_skill(skill_dir)
            warnings.extend(warns)
            if skill is None:
                continue
            if skill.name in found:
                warnings.append(
                    f"{skill_dir}: duplicate skill name {skill.name!r}"
                    f" (keeping {found[skill.name].dir})"
                )
                continue
            found[skill.name] = skill
    return tuple(found.values()), tuple(warnings)


def skill_search_dirs(
    extra_dirs: Sequence[str], installed_dir: pathlib.Path
) -> tuple[pathlib.Path, ...]:
    """Return the search order: the extra dirs first, so a local checkout wins over an install."""
    return (*(pathlib.Path(d).expanduser() for d in extra_dirs), installed_dir)


def resolve_states(skills: Sequence[Skill], state: Mapping[str, str]) -> ResolvedSkills:
    """Apply the operator's `[skills.state]` map; an absent name is enabled.

    Returns:
        The resolved skills, with a warning per name the map has and the skills lack.
    """
    warnings = [
        f"[skills.state] names an unknown skill: {name!r}"
        for name in state
        if name not in {s.name for s in skills}
    ]
    enabled = tuple(s for s in skills if state.get(s.name, "enabled") == "enabled")
    always = tuple(s for s in skills if state.get(s.name) == "always")
    return ResolvedSkills(enabled=enabled, always=always, warnings=tuple(warnings))


def operator_skills(
    enabled: bool, extra_dirs: Sequence[str], state: Mapping[str, str], installed_dir: pathlib.Path
) -> ResolvedSkills:
    """Return the skills a run has, from `[skills]`.

    The one owner of the master switch: off means no skills anywhere.

    Args:
        enabled: The `[skills].enabled` switch.
        extra_dirs: The `[skills].extra_dirs` search paths.
        state: The `[skills.state]` map.
        installed_dir: Where `agent6 skills install` puts skills.

    Returns:
        The resolved skills.
    """
    if not enabled:
        return ResolvedSkills(enabled=(), always=(), warnings=())
    found, warns = discover_skills(skill_search_dirs(extra_dirs, installed_dir))
    resolved = resolve_states(found, state)
    return ResolvedSkills(
        enabled=resolved.enabled,
        always=resolved.always,
        warnings=(*warns, *resolved.warnings),
    )


def skill_steer_payload(name: str, text: str, args: str) -> str:
    """Return the instruction a `/<skill> [args]` steer injects, the skill's text inline."""
    args_line = f"\nSkill arguments: {args}" if args else ""
    return (
        f"Apply the operator-installed skill {name!r} for the rest of this run."
        f"{args_line}\n\n"
        f'<skill name="{name}">\n{text.rstrip()}\n</skill>'
    )


def skill_command(text: str, skills: ResolvedSkills | None) -> tuple[Skill, str] | None:
    """Return the skill a `/<name> [args]` steer names and its arguments, or None."""
    stripped = text.strip()
    if not stripped.startswith("/") or skills is None:
        return None
    word, _, args = stripped[1:].partition(" ")
    for skill in (*skills.enabled, *skills.always):
        if skill.name == word:
            return skill, args.strip()
    return None
