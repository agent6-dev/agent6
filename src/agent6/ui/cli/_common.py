# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-cutting CLI helpers: parser builders, session resolution, messages and styling."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import sys

from agent6 import paths
from agent6.app import reporter
from agent6.sessions import id
from agent6.sessions import layout as sessions_layout
from agent6.viewmodel import format


def _sub(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help: str,
) -> argparse.ArgumentParser:
    """Add a subparser whose help is also its description; the parent lists its first sentence.

    Args:
        subparsers: The parent's subparsers.
        name: The command.
        help: The command's help.

    Returns:
        The subparser.
    """
    summary, _, rest = help.partition(". ")
    return subparsers.add_parser(name, help=summary + "." if rest else summary, description=help)


SESSION_ID = "Session id or unambiguous prefix"
SESSION_ID_HELP = f"{SESSION_ID}; omit for the newest."


def _add_session_id(
    parser: argparse.ArgumentParser, completer: object, *, help_text: str = SESSION_ID_HELP
) -> None:
    """Add the session positional: an id or prefix, omitted for the newest.

    Args:
        parser: The verb's parser.
        completer: Offers the ids the verb accepts; passed in, since `completers` imports this.
        help_text: The argument's help.
    """
    arg = parser.add_argument("session_id", nargs="?", default="", help=help_text)
    arg.completer = completer  # type: ignore[attr-defined]


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    """Add a subcommand's `--config FILE`.

    Its default is SUPPRESS, so the flag works before or after the subcommand and the
    top-level flag supplies the default.
    """
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Load FILE after the global and per-repository config files.",
    )


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    """Add the per-run `[budget]` override flags."""
    group = parser.add_argument_group("budget")
    group.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Set this command's metered spending limit in US dollars. -1 allows unlimited"
            " spending; 0 refuses metered calls. Default: [budget].max_usd."
        ),
    )
    group.add_argument(
        "--max-percent",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "The plan percentage points this run may use on a subscription provider. The"
            " account reports whole percents and a tick counts as a full point, so the cap"
            " ends the run early, never late. -1: unlimited; 0: refuse plan-metered calls."
            " Default: [budget].max_percent."
        ),
    )
    group.add_argument(
        "--max-tokens-fallback",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Set the combined input and output token limit for calls with no price data."
            " -1 allows unlimited tokens; 0 refuses calls that cannot be priced. Default:"
            " [budget].max_tokens_fallback."
        ),
    )


def _add_sandbox_flags(parser: argparse.ArgumentParser) -> None:
    """Add the per-invocation sandbox and approval override flags every paid command carries.

    `--dangerously-disable-sandbox` is a one-off `sandbox.isolation = "none"`;
    `--auto-approve` upgrades `sandbox.run_commands` from ask to yes and never
    resurrects a withheld no.
    """
    group = parser.add_argument_group("sandbox")
    group.add_argument(
        "--dangerously-disable-sandbox",
        action="store_true",
        help=(
            "Run the agent's commands directly on the host with no sandbox. This disables"
            " file limits, system-call filtering, and namespace isolation. Use only on a"
            " disposable or already-isolated machine; the host becomes the only boundary."
        ),
    )
    approval = group.add_mutually_exclusive_group()
    approval.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Approve every sandboxed command for this session without prompting. Changes"
            " sandbox.run_commands from `ask` to `yes`, but never overrides `no` or changes"
            " sandbox.isolation. With --dangerously-disable-sandbox, this gives the agent"
            " unprompted host access."
        ),
    )
    approval.add_argument(
        "--no-commands",
        action="store_true",
        help=(
            "Do not let this session run commands: no run_command, verify gate, or background"
            " commands. Sets sandbox.run_commands to `no`. This is also how /btw runs its side"
            " question."
        ),
    )


def safe_input(prompt: str) -> str | None:
    """Return the stripped input, or None on EOF or an unreadable stdin; an interrupt propagates."""
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return None


def editor_argv() -> list[str] | None:
    """Return `$EDITOR` as argv (default vi), or None after naming its unbalanced quoting."""
    editor = os.environ.get("EDITOR", "vi")
    try:
        return shlex.split(editor) or ["vi"]
    except ValueError as exc:
        error(f"$EDITOR {editor!r} is not a valid command: {exc}")
        return None


def sgr(text: str, code: str) -> str:
    """Return the text in an ANSI style on a tty, plain otherwise."""
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


def _runs_dir(repo_root: pathlib.Path) -> pathlib.Path:
    """Return the `runs/` directory under the per-repo state dir."""
    return sessions_layout.bucket_dir(paths.state_dir(repo_root), "runs")


def _plans_dir(repo_root: pathlib.Path) -> pathlib.Path:
    """Return the `plans/` directory under the per-repo state dir."""
    return sessions_layout.bucket_dir(paths.state_dir(repo_root), "plans")


def nothing_yet(what: str = "sessions") -> str:
    """Return the one first-contact line for a fresh install, naming the way out."""
    return f'no {what} yet. Start one with `agent6 run "<task>"`.'


# The stderr conventions belong to app.reporter; every CLI message goes through these.
error = reporter.STDIO_REPORTER.error
note = reporter.STDIO_REPORTER.note
refuse = reporter.STDIO_REPORTER.refuse
warn = reporter.STDIO_REPORTER.warn

# The one sentence every command's id argument prints.
MACHINE_ID_HELP = "Machine id (a directory under the per-repo state dir's machines/)."
REPO_FLAG_HELP = "Write to the per-repo config instead of the global config."


def print_nothing_yet(what: str = "sessions") -> None:
    """Print that there is nothing yet and how to change that; an empty state dir is no fault."""
    print(nothing_yet(what), file=sys.stderr)


def print_no_session_match(query: str, state: pathlib.Path) -> None:
    """Print the one missing-session error: the query and where it looked, or the first contact."""
    if query:
        print(f"ERROR: no session matches {query!r} (looked under {state})", file=sys.stderr)
    else:
        print_nothing_yet()


def session_bucket_dirs(repo_root: pathlib.Path) -> list[pathlib.Path]:
    """Return every session bucket dir, present or not; iterators skip the missing ones."""
    state = paths.state_dir(repo_root)
    return [sessions_layout.bucket_dir(state, subdir) for subdir in sessions_layout.SESSION_BUCKETS]


def all_session_dirs(repo_root: pathlib.Path) -> list[pathlib.Path]:
    """Return every session directory across all buckets, so a bare `attach` finds an ask too."""
    dirs: list[pathlib.Path] = []
    for bucket in session_bucket_dirs(repo_root):
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir())
    return dirs


def resolve_session_layout(
    repo_root: pathlib.Path, query: str, *, allow_husk: bool = False
) -> sessions_layout.SessionLayout:
    """Resolve a session id or unique prefix across every bucket.

    Args:
        repo_root: The repo.
        query: The id or prefix.
        allow_husk: Accept a session with no manifest and no log, for `sessions rm`.

    Returns:
        The session's layout.

    Raises:
        SessionIdError: No session matches, several do, or the match is a husk.
    """
    layout = id.resolve_session(paths.state_dir(repo_root), query)
    from agent6.viewmodel import is_session_husk  # noqa: PLC0415

    if not allow_husk and is_session_husk(layout.session_dir):
        raise id.SessionIdError(
            f"session {layout.session_id} crashed before it ever started (no log, nothing"
            f" to resume); `agent6 sessions rm {layout.session_id}` removes it"
        )
    return layout


def resolve_target(target: str) -> sessions_layout.SessionLayout | None:
    """Return the named session, or the newest when none was named, printing why when neither."""
    try:
        layout = resolve_or_newest_layout(pathlib.Path.cwd(), target)
    except id.SessionIdError as exc:
        error(f"{exc}")
        return None
    if layout is None:
        print_no_session_match(target, paths.state_dir(pathlib.Path.cwd()))
    return layout


def newest_layout_holding(
    repo_root: pathlib.Path, child: str
) -> sessions_layout.SessionLayout | None:
    """Return the newest session across every bucket whose dir holds the child, or None."""
    candidates = [d for d in all_session_dirs(repo_root) if (d / child).is_dir()]
    if not candidates:
        return None
    from agent6.viewmodel import session_mtime  # noqa: PLC0415

    return sessions_layout.layout_of(max(candidates, key=session_mtime))


def resolve_or_newest_layout(
    repo_root: pathlib.Path, session_id: str, *, allow_husk: bool = False
) -> sessions_layout.SessionLayout | None:
    """Resolve an explicit session id, or the newest session when the id is empty.

    Args:
        repo_root: The repo.
        session_id: The id or prefix, or "".
        allow_husk: Accept a session with no manifest and no log.

    Returns:
        The layout; None only when the id is empty and no session exists.

    Raises:
        SessionIdError: An explicit id has no match (`.no_match` set) or several.
    """
    if session_id:
        return resolve_session_layout(repo_root, session_id, allow_husk=allow_husk)
    from agent6.viewmodel import newest_session_dir  # noqa: PLC0415

    newest = newest_session_dir(session_bucket_dirs(repo_root))
    if newest is None:
        return None
    return sessions_layout.layout_of(newest)


def _enforce_root_policy(allow_root: bool) -> int | None:
    """Refuse to run as root without `--allow-root` or `AGENT6_ALLOW_ROOT=1`.

    Privileges are never dropped: under sudo the jailed commands run as root, and the
    jail is the boundary.

    Args:
        allow_root: The opt-in.

    Returns:
        The exit code to refuse with, or None to proceed (with a loud banner as root).
    """
    if not paths.is_root():
        return None
    if not paths.root_optin_enabled(allow_root):
        print(
            "REFUSING: running as root. An LLM-driven agent as root is dangerous;"
            " if a task genuinely needs it, re-run as `agent6 --allow-root <command> ...`"
            " (the flag goes before the command), or set AGENT6_ALLOW_ROOT=1.",
            file=sys.stderr,
        )
        return 2
    user = paths.effective_user()
    who = f" on behalf of {user.name} (uid {user.uid})" if user.via_sudo else ""
    print(
        f"[agent6] WARNING: running as root{who}. The LLM's commands execute as"
        " root inside the jail; files agent6 writes under the repo are chowned"
        " back to you when invoked via sudo. Proceed with care.",
        file=sys.stderr,
    )
    return None


# The ANSI SGR per `viewmodel.format.status_level`; the TUI's Rich map is the sibling.
_LEVEL_SGR: dict[format.StatusLevel, str] = {
    "ok": "32",
    "info": "35",  # magenta (mauve on the TUI/web)
    "active": "1;36",
    "warn": "33",
    "error": "1;31",
    "neutral": "",
}


def styled_status(
    status: str, reason: str, *, color: bool, label: str | None = None
) -> tuple[str, str]:
    """Return a listing row's status as (styled label, plain label); the plain one drives widths.

    Args:
        status: The status word, which picks the colour.
        reason: The status's reason, folded into the plain label.
        color: Style the label.
        label: Overrides the text.
    """
    text = format.status_label(status, reason) if label is None else label
    sgr_code = _LEVEL_SGR[format.status_level(status)]
    if color and sgr_code:
        return f"\x1b[{sgr_code}m{text}\x1b[0m", text
    return text, text


def plural(n: int, singular: str, plural: str | None = None) -> str:
    """Return the count with the right noun form."""
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def home_contracted(path: str) -> str:
    """Return the path with `$HOME` shortened to `~`, only at a path boundary."""
    home = str(pathlib.Path.home())
    return "~" + path[len(home) :] if path == home or path.startswith(home + "/") else path
