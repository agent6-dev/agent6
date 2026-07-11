# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""``agent6 completions``: install (or print) shell tab-completion.

The argcomplete docs' ``eval "$(register-python-argcomplete agent6)"`` needs
that register script on PATH, which a uv-tool install does not guarantee. This
command is self-contained: ``argcomplete.shellcode()`` emits the equivalent
registration, and the default action INSTALLS it. For bash/zsh the script is
written under the agent6 config dir and one marker-guarded ``source`` line is
appended to the shell's rc file (idempotent: rerunning refreshes the script and
never duplicates the block). Fish gets a file in its native completions dir --
no rc edit, and fish picks it up automatically. The shell is detected by
walking up the process tree to the nearest known shell (a fish started from
bash leaves ``$SHELL=bash``), falling back to ``$SHELL``; pass ``bash``,
``zsh``, or ``fish`` explicitly when both are wrong.

A child process cannot restart the shell that launched it, so instead of
pretending to, the bash/zsh install ends by printing exactly what to run
(``source <rc>`` or ``exec $SHELL``) to activate the completions in the
current session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from argcomplete.shell_integration import shellcode

from agent6.paths import global_config_dir

SHELLS = ("bash", "zsh", "fish")
_MARK_BEGIN = "# >>> agent6 completions >>>"
_MARK_END = "# <<< agent6 completions <<<"
_PROC = Path("/proc")  # patched in tests


def detect_shell() -> str:
    """The interactive shell this command was launched from.

    Walks up the process tree (Linux ``/proc``) to the nearest ancestor that
    is a known shell: ``$SHELL`` is the login shell, not the running one (a
    fish started from bash keeps ``$SHELL=bash``), and the direct parent can
    be a wrapper like ``uv``. Falls back to ``$SHELL``'s basename when the
    walk finds nothing (macOS, no ``/proc``)."""
    pid = os.getppid()
    for _ in range(10):
        try:
            comm = (_PROC / str(pid) / "comm").read_text(encoding="utf-8").strip()
            stat = (_PROC / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            break
        if comm in SHELLS:
            return comm
        # stat: "pid (comm) state ppid ..."; comm may contain spaces/parens,
        # so split at the LAST ")". ppid is the second field after it.
        try:
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            break
        if pid <= 1:
            break
    return Path(os.environ.get("SHELL", "")).name


def _rc_path(shell: str) -> Path:
    if shell == "bash":
        return Path.home() / ".bashrc"
    # zsh reads .zshrc from $ZDOTDIR when set, else $HOME.
    return Path(os.environ.get("ZDOTDIR") or Path.home()) / ".zshrc"


def _fish_completions_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "fish" / "completions" / "agent6.fish"


def _install_bash_zsh(shell: str, code: str) -> int:
    script = global_config_dir() / f"completions.{shell}"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(code, encoding="utf-8")
    rc = _rc_path(shell)
    block = (
        f"\n{_MARK_BEGIN}\n"
        f'[ -f "{script}" ] && source "{script}"  # agent6 tab-completion\n'
        f"{_MARK_END}\n"
    )
    existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
    if _MARK_BEGIN in existing:
        print(f"[agent6] refreshed {script} (already sourced from {rc})")
    else:
        with rc.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"[agent6] wrote {script} and added a source line to {rc}")
    print(f"[agent6] activate now with: source {rc}   (or start a new shell: exec $SHELL)")
    return 0


def _install_fish(code: str) -> int:
    target = _fish_completions_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    print(f"[agent6] wrote {target} (fish loads it automatically)")
    return 0


def cmd_completions(shell_arg: str, *, print_only: bool) -> int:
    shell = shell_arg or detect_shell()
    if shell not in SHELLS:
        detected = f" (detected {shell!r})" if shell else ""
        print(
            f"ERROR: unsupported or unknown shell{detected}."
            f" Pass one of: agent6 completions {'|'.join(SHELLS)}",
            file=sys.stderr,
        )
        return 2
    code = shellcode(["agent6"], shell=shell)
    if print_only:
        print(code)
        return 0
    if shell == "fish":
        return _install_fish(code)
    return _install_bash_zsh(shell, code)
