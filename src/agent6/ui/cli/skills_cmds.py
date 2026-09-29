# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 skills`: install, update, list, enable, disable, remove.

Install fetches operator-chosen skill content (a direct SKILL.md URL, a git repository,
or a local path) into the user data dir. Nothing fetched is ever executed: skills are
prompt text, trusted like config because the operator chose them. Each installed skill
carries a `.origin.toml` provenance file, ignored by discovery, so `skills update` can
re-fetch from the same source.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx2

from agent6.config import ConfigError
from agent6.config.io import remove_toml_leaf, upsert_toml_leaf
from agent6.config.layer import load_effective
from agent6.errors import OperatorError, read_operator_file
from agent6.paths import (
    chown_to_real_user,
    data_dir,
    global_config_path,
    mkdir_for_real_user,
    repo_config_path,
)
from agent6.skills import (
    Skill,
    discover_skills,
    is_valid_skill_name,
    parse_frontmatter,
    resolve_states,
    skill_search_dirs,
)
from agent6.tools.http_body import BodyRefusedError, read_capped
from agent6.ui.cli._common import home_contracted, sgr, warn
from agent6.ui.cli._steer_menu import MENU_COMMANDS

_ORIGIN_FILE = ".origin.toml"
_FETCH_TIMEOUT_S = 30.0
_FETCH_MAX_BYTES = 1_048_576  # a SKILL.md is prose; 1 MiB is already generous


def _term_width() -> int:
    """Return the terminal width, or 80 without one."""
    return shutil.get_terminal_size((100, 24)).columns


def _one_line(text: str, width: int) -> str:
    """Return the text collapsed to one line and truncated to the width."""
    text = " ".join(text.split())
    if width < 12 or len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _short_source(src: str) -> str:
    """Return a compact provenance label: no URL scheme or `.git`, $HOME as `~`."""
    src = src.removesuffix(".git")
    for scheme in ("https://", "http://", "ssh://", "git://"):
        if src.startswith(scheme):
            src = src[len(scheme) :]
            break
    return home_contracted(src)


def _installed_dir() -> Path:
    """Return the managed skills dir under the user data dir."""
    return data_dir() / "skills"


def _search_dirs(repo_root: Path, config_path: Path | None = None) -> tuple[Path, ...]:
    """Return the discovery search path: the effective config's dirs plus the managed one."""
    cfg = load_effective(repo_root, config_path).config
    return skill_search_dirs(cfg.skills.extra_dirs, _installed_dir())


def _toml_str(value: str) -> str:
    """Return the value as a TOML basic string with backslashes and quotes escaped.

    A quote in a source path would otherwise make the hand-built origin unparseable, and
    `skills update` would lose its source.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_origin(skill_dir: Path, *, url: str, kind: str, source_sha: str) -> None:
    """Write the skill's `.origin.toml` provenance file."""
    digest = hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).hexdigest()
    fetched = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = (
        f"url = {_toml_str(url)}\nkind = {_toml_str(kind)}\n"
        f"source_sha = {_toml_str(source_sha)}\n"
        f"fetched_at = {_toml_str(fetched)}\nsha256 = {_toml_str(digest)}\n"
    )
    (skill_dir / _ORIGIN_FILE).write_text(body, encoding="utf-8")


def _read_origin(skill_dir: Path) -> dict[str, str] | None:
    """Return the skill's recorded origin, or None when it has none or it does not parse."""
    p = skill_dir / _ORIGIN_FILE
    if not p.is_file():
        return None
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return {k: str(v) for k, v in raw.items()}


def _fetch_url(url: str) -> str:
    """Return the body of the URL as UTF-8 text through the one capped streaming read.

    Refused past `_FETCH_MAX_BYTES` while it arrives, past `_FETCH_TIMEOUT_S` from its
    first byte, or under a content-encoding other than identity.

    Raises:
        OperatorError: The fetch was refused, failed, or the body is not UTF-8.
    """
    try:
        with httpx2.stream(
            "GET",
            url,
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
            headers={"Accept-Encoding": "identity"},
        ) as resp:
            resp.raise_for_status()
            deadline = time.monotonic() + _FETCH_TIMEOUT_S
            body = read_capped(
                resp, cap=_FETCH_MAX_BYTES, deadline=deadline, timeout_s=_FETCH_TIMEOUT_S
            )
    except BodyRefusedError as exc:
        raise OperatorError(f"{url}: {exc}") from exc
    except httpx2.HTTPError as exc:
        raise OperatorError(f"could not fetch {url}: {exc}") from exc
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorError(f"{url}: not UTF-8 text: {exc}") from exc


def _skill_name_from_text(text: str, source: str) -> str:
    """Return the skill's name from its SKILL.md frontmatter, gated as discovery gates it.

    Raises:
        OperatorError: The frontmatter lacks a name or description, or the name is not one
            safe path component.
    """
    fields, _warnings = parse_frontmatter(text)
    name, description = fields.get("name", ""), fields.get("description", "")
    if not name or not description:
        raise OperatorError(f"{source}: SKILL.md lacks required frontmatter name/description")
    # The name becomes a path component and, under --force, an rmtree target: discovery's gate.
    if not is_valid_skill_name(name):
        raise OperatorError(
            f"{source}: invalid skill name {name!r} "
            "(letters, digits, and hyphens only, starting alphanumeric)"
        )
    return name


def _refuse_existing(name: str, *, force: bool) -> Path:
    """Return the target dir for the name, refusing when it exists without `--force`.

    Never clears: the old install survives until the staged replacement is fully built
    (`_publish_staged`), so a copy or write fault cannot destroy a good skill.

    Raises:
        OperatorError: The skill is installed and `--force` was not given.
    """
    target = _installed_dir() / name
    if target.exists() and not force:
        origin = _read_origin(target)
        src = f" (installed from {origin['url']})" if origin and origin.get("url") else ""
        raise OperatorError(f"skill {name!r} is already installed{src}; use --force to replace")
    return target


def _publish_staged(staging: Path, target: Path) -> None:
    """Swap the fully built staging dir into place; the old install goes only now.

    The dot-prefixed staging name fails the skill-name gate, so a crash's leftover is
    invisible to discovery.
    """
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def _install_skill_dir(src: Path, *, url: str, kind: str, source_sha: str, force: bool) -> str:
    """Copy one skill directory (SKILL.md plus supplementary files) into place.

    `symlinks=True`: the skill comes from an untrusted source, and copying a link's content
    would turn `reference.md -> secrets.toml` into a real file `use_skill` serves; preserved,
    the link stays subject to `use_skill`'s containment check.

    Args:
        src: The directory to copy.
        url: The source, for the origin file.
        kind: The origin kind.
        source_sha: The source's commit, or "".
        force: Replace an existing install.

    Returns:
        The installed name.

    Raises:
        OperatorError: The copy failed or the name is already installed without `--force`.
    """
    name = _skill_name_from_text(read_operator_file(src / "SKILL.md"), str(src))
    target = _refuse_existing(name, force=force)
    mkdir_for_real_user(target.parent)
    staging = target.parent / f".staging-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(src, staging, symlinks=True)
        (staging / _ORIGIN_FILE).unlink(missing_ok=True)  # never inherit a copied origin
        _write_origin(staging, url=url, kind=kind, source_sha=source_sha)
        chown_to_real_user(staging)
        _publish_staged(staging, target)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise OperatorError(f"could not install the skill from {src}: {exc}") from exc
    return name


def _install_skill_text(text: str, *, url: str, force: bool) -> str:
    """Install a single-file skill from raw SKILL.md text.

    Returns:
        The installed name.

    Raises:
        OperatorError: The write failed or the name is already installed without `--force`.
    """
    name = _skill_name_from_text(text, url)
    target = _refuse_existing(name, force=force)
    mkdir_for_real_user(target.parent)
    staging = target.parent / f".staging-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir()
        (staging / "SKILL.md").write_text(text, encoding="utf-8")
        _write_origin(staging, url=url, kind="skillmd", source_sha="")
        chown_to_real_user(staging)
        _publish_staged(staging, target)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise OperatorError(f"could not install the skill from {url}: {exc}") from exc
    return name


def _git_clone(url: str, dest: Path) -> str:
    """Shallow-clone an operator-chosen URL; nothing in it is executed.

    Returns:
        The clone's HEAD sha.

    Raises:
        OperatorError: git is missing or the clone failed.
    """
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", "--", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise OperatorError("git not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise OperatorError(f"git clone of {url} failed: {exc.stderr.strip()}") from exc
    return head.stdout.strip()


def _source_declaring(candidates: list[Path], name: str) -> Path | None:
    """Return the skill directory whose SKILL.md declares the name, or None.

    A skill installs under its frontmatter name, which need not be its directory's.
    """
    for d in candidates:
        try:
            declared = _skill_name_from_text(read_operator_file(d / "SKILL.md"), str(d))
        except OperatorError:
            continue
        if declared == name:
            return d
    return None


def _repo_skill_dirs(root: Path) -> list[Path]:
    """Return the skill directories in a fetched repository: `skills/*/SKILL.md`, or the root."""
    out = [
        p
        for p in sorted((root / "skills").glob("*"))
        if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").is_file()
    ]
    if not out and (root / "SKILL.md").is_file():
        out = [root]
    return out


def _refuse_any_existing(dirs: list[Path], *, force: bool) -> None:
    """Pre-check every skill name of a multi-skill install, so a conflict refuses the whole install.

    Raises:
        OperatorError: A skill is already installed and `--force` was not given.
    """
    if force:
        return
    conflicts = [
        name
        for d in dirs
        if (name := _skill_name_from_text(read_operator_file(d / "SKILL.md"), str(d)))
        and (_installed_dir() / name).exists()
    ]
    if conflicts:
        raise OperatorError(
            f"already installed: {', '.join(conflicts)}; use --force to replace"
            " (nothing was installed)"
        )


def _install_from_local(local: Path, *, force: bool) -> list[str]:
    """Install from a local SKILL.md file, one skill dir, or a repo checkout.

    Returns:
        The installed names.

    Raises:
        OperatorError: The path holds no skill, or an install refused.
    """
    src_url = str(local.resolve())
    if local.is_file():
        return [_install_skill_text(read_operator_file(local), url=src_url, force=force)]
    if (local / "SKILL.md").is_file():
        return [_install_skill_dir(local, url=src_url, kind="dir", source_sha="", force=force)]
    dirs = _repo_skill_dirs(local)
    _refuse_any_existing(dirs, force=force)
    return [
        _install_skill_dir(d, url=src_url, kind="dir", source_sha="", force=force) for d in dirs
    ]


def _install_from_git(url: str, *, force: bool) -> list[str]:
    """Install every skill a git repository ships.

    Returns:
        The installed names.

    Raises:
        OperatorError: The clone failed, the repository holds no skill, or an install refused.
    """
    with tempfile.TemporaryDirectory(prefix="agent6-skill-") as tmp:
        clone = Path(tmp) / "repo"
        sha = _git_clone(url, clone)
        dirs = _repo_skill_dirs(clone)
        if not dirs:
            raise OperatorError(f"no skills found in {url} (expected skills/*/SKILL.md)")
        _refuse_any_existing(dirs, force=force)
        return [
            _install_skill_dir(d, url=url, kind="git", source_sha=sha, force=force) for d in dirs
        ]


def _cmd_skills_install(url: str, *, force: bool, config_path: Path | None = None) -> int:
    """Install the skills at a URL or path and print each name.

    Args:
        url: A SKILL.md URL, a git URL, or a local path.
        force: Replace existing installs.
        config_path: The `--config` file, if any.

    Returns:
        The exit code, 0.

    Raises:
        OperatorError: The fetch, clone or install refused.
    """
    local = Path(url).expanduser()
    if local.exists():
        installed = _install_from_local(local, force=force)
    elif url.endswith(".md"):
        installed = [_install_skill_text(_fetch_url(url), url=url, force=force)]
    else:
        installed = _install_from_git(url, force=force)
    if not installed:
        raise OperatorError(f"no skills found in {url} (expected SKILL.md or skills/*/SKILL.md)")
    skills, _ = discover_skills([_installed_dir()])
    by_name = {s.name: s for s in skills}
    width = _term_width()
    if len(installed) == 1:
        name = installed[0]
        print(f"Installed {sgr(name, '1')}")
        if desc := (by_name[name].description if name in by_name else ""):
            print(f"  {sgr(_one_line(desc, width - 2), '2')}")
    else:
        print(sgr(f"Installed {len(installed)} skills from {_short_source(url)}:", "1"))
        name_w = min(32, max(len(n) for n in installed))
        for name in sorted(installed):
            desc = by_name[name].description if name in by_name else ""
            prefix = f"  {name:<{name_w}}  "
            print(f"{prefix}{sgr(_one_line(desc, max(20, width - len(prefix))), '2')}")
    for name in installed:
        if f"/{name}" in MENU_COMMANDS:
            print(
                f"note: /{name} is a built-in pause-menu command and keeps its meaning;"
                " the skill stays reachable via the <skills> index, use_skill, and --skill"
            )
    if _print_disabled_notes(installed, config_path):
        print(sgr("Installed; `agent6 skills list` shows the effective state.", "2"))
    else:
        print(sgr("Enabled and active now; `agent6 skills list` to review.", "2"))
    return 0


def _print_disabled_notes(installed: list[str], config_path: Path | None) -> bool:
    """Name every installed skill a surviving `skills.state = "disabled"` leaf covers.

    Returns:
        Whether any was named; the closing line must not then read "active".
    """
    disabled = sorted(n for n in installed if _state_map(config_path).get(n) == "disabled")
    for name in disabled:
        print(
            f'note: skills.state.{name} = "disabled" applies to this name;'
            f" `agent6 skills enable {name}` clears it"
        )
    return bool(disabled)


def _state_map(config_path: Path | None) -> dict[str, str]:
    """Return the effective `[skills.state]` map, or {} when config is unreadable.

    The notes built on it then just do not print; the command's own work is already done.
    """
    try:
        return dict(load_effective(Path.cwd(), config_path).config.skills.state)
    except ConfigError:
        return {}


def _refetch_skill(name: str, origin: dict[str, str]) -> tuple[str, str]:
    """Re-install one skill in place from its recorded origin.

    Dispatches on the recorded kind, not on whether a path happens to exist: a local file or
    dir that was moved or deleted is a clean skip, not a failed HTTP fetch of the path. A
    skillmd origin whose frontmatter declares a different name is a rename: the skill
    installs under the new name and the old directory goes.

    Args:
        name: The installed skill.
        origin: Its recorded origin.

    Returns:
        The installed name and "" on success, or the name and a short skip note when the
        source no longer exists.

    Raises:
        OperatorError: The fetch or install refused.
    """
    url, kind = origin["url"], origin.get("kind", "skillmd")
    if kind == "git":
        with tempfile.TemporaryDirectory(prefix="agent6-skill-") as tmp:
            clone = Path(tmp) / "repo"
            sha = _git_clone(url, clone)
            src = _source_declaring(_repo_skill_dirs(clone), name)
            if src is None:
                return name, "(gone from origin)"
            _install_skill_dir(src, url=url, kind="git", source_sha=sha, force=True)
        return name, ""
    if kind == "dir":
        root = Path(url)
        candidates = _repo_skill_dirs(root)
        # A repo install records the repo root and finds the skill by name; a dir install, the dir.
        src = root if candidates == [root] else _source_declaring(candidates, name)
        if src is None:
            return name, "(gone from origin)"
        _install_skill_dir(src, url=url, kind="dir", source_sha="", force=True)
        return name, ""
    # skillmd: a single SKILL.md, either a remote URL or a local file.
    if url.startswith(("http://", "https://")):
        text = _fetch_url(url)
    elif Path(url).is_file():
        text = read_operator_file(Path(url))
    else:
        return name, "(gone from origin)"
    installed = _install_skill_text(text, url=url, force=True)
    if installed != name:
        shutil.rmtree(_installed_dir() / name, ignore_errors=True)
    return installed, ""


def _cmd_skills_update(name: str) -> int:
    """Re-fetch one skill, or every skill with an origin, and print a row per skill.

    Returns:
        The exit code, 0.

    Raises:
        OperatorError: The named skill is not installed, or a refetch refused.
    """
    base = _installed_dir()
    if name and not (base / name).is_dir():
        raise OperatorError(f"{name!r} is not installed")
    targets = [base / name] if name else sorted(p for p in base.glob("*") if p.is_dir())
    if not targets:
        print("no skills installed. Install one with `agent6 skills install <url>`.")
        return 0
    name_w = min(32, max(len(p.name) for p in targets))

    def _row(skill: str, status: str, *, dim: bool, note: str = "") -> None:
        """Print one update row."""
        line = f"  {skill:<{name_w}}  {f'{status}  {note}'.rstrip()}"
        print(sgr(line, "2") if dim else line)

    counts = {"updated": 0, "unchanged": 0, "skipped": 0}
    for skill_dir in targets:
        origin = _read_origin(skill_dir)
        if origin is None or not origin.get("url"):
            _row(skill_dir.name, "skipped", dim=True, note="(no origin recorded)")
            counts["skipped"] += 1
            continue
        before = origin.get("sha256", "")
        try:
            installed, note = _refetch_skill(skill_dir.name, origin)
        except OperatorError as exc:
            # With the skill's name: an all-skills sweep otherwise names only the failing origin.
            raise OperatorError(f"{skill_dir.name}: {exc}") from exc
        if note:
            _row(skill_dir.name, "skipped", dim=True, note=note)
            counts["skipped"] += 1
            continue
        if installed != skill_dir.name:
            _row(skill_dir.name, "updated", dim=False, note=f"(renamed to {installed})")
            counts["updated"] += 1
            continue
        after = _read_origin(base / skill_dir.name) or {}
        if after.get("sha256", "") != before:
            _row(skill_dir.name, "updated", dim=False)
            counts["updated"] += 1
        else:
            _row(skill_dir.name, "unchanged", dim=True)
            counts["unchanged"] += 1
    parts = [f"{counts[k]} {k}" for k in ("updated", "unchanged", "skipped") if counts[k]]
    print(sgr(", ".join(parts), "1"))
    return 0


def _cmd_skills_list(config_path: Path | None = None) -> int:
    """List the installed skills grouped by origin, with their state when any is not enabled.

    Returns:
        The exit code, 0.
    """
    repo_root = Path.cwd()
    try:
        cfg = load_effective(repo_root, config_path).config
    except ConfigError as exc:
        print(f"(config unreadable, showing installed dir only: {exc})", file=sys.stderr)
        cfg = None
    dirs = (
        skill_search_dirs(cfg.skills.extra_dirs, _installed_dir())
        if cfg is not None
        else (_installed_dir(),)
    )
    skills, warnings = discover_skills(dirs)
    state = dict(cfg.skills.state) if cfg is not None else {}
    if not skills:
        print("no skills installed. Install one with `agent6 skills install <url>`.")
        return 0

    if cfg is not None and not cfg.skills.enabled:
        # With the master switch off a run has none of this: the index is empty, use_skill absent.
        print(sgr("skills are DISABLED (agent6 config set skills.enabled true)", "1"))
        print("installed, but no run loads any of them:\n")

    states = [state.get(s.name, "enabled") for s in skills]
    counts = Counter(states)
    detail = [f"{counts[k]} {k}" for k in ("disabled", "always") if counts[k]]
    summary = f"{len(skills)} skill{'s' if len(skills) != 1 else ''}"
    if detail:
        summary += f"  ({', '.join(detail)})"
    print(sgr(summary, "1"))

    # Grouped by origin, so a repo shipping 20 skills prints its URL once; the state column
    # appears only when some skill is not plain-enabled.
    groups: dict[str, list[Skill]] = {}
    for s in skills:
        origin = _read_origin(s.dir)
        src = origin["url"] if origin and origin.get("url") else str(s.dir.parent)
        groups.setdefault(src, []).append(s)
    show_state = any(st != "enabled" for st in states)
    name_w = min(32, max(len(s.name) for s in skills))
    tag_w = len("[disabled]")
    for src, items in groups.items():
        print(f"\n{sgr(_short_source(src), '2')}")
        for s in sorted(items, key=lambda k: k.name):
            st = state.get(s.name, "enabled")
            name = s.name if len(s.name) <= name_w else s.name[: name_w - 1] + "…"
            prefix = f"  {name:<{name_w}}  "
            if show_state:
                prefix += f"{('' if st == 'enabled' else f'[{st}]'):<{tag_w}}  "
            print(f"{prefix}{_one_line(s.description, max(20, _term_width() - len(prefix)))}")
    for w in warnings:
        warn(f"{w}")
    return 0


def _known_skill_names(repo_root: Path, config_path: Path | None = None) -> tuple[str, ...]:
    """Return the installed skill names; refuses when discovery itself fails.

    An empty tuple would make `skills enable` and `disable` answer "unknown skill", sending
    the operator after a skill that is only unreadable.

    Raises:
        OperatorError: The installed skills could not be read.
    """
    try:
        skills, _ = discover_skills(_search_dirs(repo_root, config_path))
    except OSError as exc:
        raise OperatorError(f"could not read the installed skills: {exc}") from exc
    return tuple(s.name for s in skills)


def _state_target(repo: bool) -> Path:
    """Return the config file a state write goes to: the repo's or the global one."""
    return repo_config_path(Path.cwd()) if repo else global_config_path()


def _require_known(name: str, repo_root: Path, config_path: Path | None = None) -> None:
    """Refuse a name no installed skill has.

    Raises:
        OperatorError: The skill is unknown.
    """
    known = _known_skill_names(repo_root, config_path)
    if name not in known:
        raise OperatorError(f"unknown skill {name!r}; installed: {', '.join(known) or '(none)'}")


def _cmd_skills_enable(
    name: str, *, always: bool, repo: bool, config_path: Path | None = None
) -> int:
    """Set a skill's state to enabled or always, clearing a plain enable.

    Args:
        name: The skill.
        always: Include the skill in every session's prompt.
        repo: Write to the repo config.
        config_path: The `--config` file, if any.

    Returns:
        The exit code, 0.

    Raises:
        OperatorError: The skill is unknown.
    """
    # A state leaf can outlive its skill, so clearing it must not need the skill to exist.
    if not (not always and _state_map(config_path).get(name)):
        _require_known(name, Path.cwd(), config_path)
    target = _state_target(repo)
    mkdir_for_real_user(target.parent)
    try:
        if always:
            upsert_toml_leaf(target, f"skills.state.{name}", "always")
            print(f'Set skills.state.{name} = "always" in {target}')
        # Absent means enabled; removing the key keeps the config free of no-op entries.
        elif remove_toml_leaf(target, f"skills.state.{name}") if target.is_file() else False:
            print(f"Unset skills.state.{name} in {target} (enabled is the default)")
        else:
            print(f"{name} is already enabled (no state entry in {target})")
    finally:
        chown_to_real_user(target)
    return 0


def _cmd_skills_disable(name: str, *, repo: bool, config_path: Path | None = None) -> int:
    """Set a skill's state to disabled.

    Returns:
        The exit code, 0.

    Raises:
        OperatorError: The skill is unknown.
    """
    _require_known(name, Path.cwd(), config_path)
    target = _state_target(repo)
    mkdir_for_real_user(target.parent)
    try:
        upsert_toml_leaf(target, f"skills.state.{name}", "disabled")
    finally:
        chown_to_real_user(target)
    print(f'Set skills.state.{name} = "disabled" in {target}')
    return 0


def _cmd_skills_remove(name: str, config_path: Path | None = None) -> int:
    """Delete the installed skill from the managed skills dir.

    The name becomes a path component under that dir and an rmtree target, so it must be
    one safe component, the same gate install applies.

    Returns:
        The exit code, 0.

    Raises:
        OperatorError: The name is unsafe, managed elsewhere, or not installed.
    """
    if not is_valid_skill_name(name):
        raise OperatorError(
            f"invalid skill name {name!r} "
            "(letters, digits, and hyphens only, starting alphanumeric)"
        )
    target = _installed_dir() / name
    if not target.is_dir():
        # Distinguish "managed elsewhere" from "unknown" for a useful error.
        skills, _ = discover_skills(_search_dirs(Path.cwd(), config_path))
        match = next((s for s in skills if s.name == name), None)
        if match is not None:
            raise OperatorError(
                f"{name!r} lives in an extra_dirs location ({match.dir});"
                " remove it there or drop the dir from [skills].extra_dirs"
            )
        raise OperatorError(f"{name!r} is not installed")
    shutil.rmtree(target)
    print(f"removed {name}")
    if state := _state_map(config_path).get(name, ""):
        print(
            f'note: skills.state.{name} = "{state}" remains in config and applies if'
            f" {name!r} is installed again; `agent6 skills enable {name}` clears it"
        )
    return 0


def resolved_skill_names_for_completion(repo_root: Path) -> list[str]:
    """Return the skill names for argcomplete: cheap discovery, never raises.

    A shell completion has nowhere to show an error and must not raise into the shell, so a
    discovery failure is nothing at all.
    """
    try:
        return list(_known_skill_names(repo_root))
    except OperatorError:
        return []


__all__ = [
    "Skill",
    "resolve_states",
    "resolved_skill_names_for_completion",
]
