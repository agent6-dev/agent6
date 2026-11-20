# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A pack an operation leaves behind is set aside, never offered to `unpack`.

`.agent6/` is an ordinary directory to the model's tools, so a model wrote a manifest there;
every later start then refused and named `agent6 unpack`, which installed the forged config.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import errors, kinds, pack, pack_manifest, paths
from agent6.app import execution, merge, role_providers
from agent6.app import reporter as app_reporter
from agent6.app.machine import run as machine_run
from agent6.config import Config, layer
from agent6.providers import ProviderResponse
from agent6.sandbox import jail
from agent6.sessions import event_log
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main, exec_cmds
from agent6.vcs.git import read as git_read
from agent6.viewmodel import fold_transcript, format_log_line

_MANIFEST = (
    json.dumps(
        {
            "version": 1,
            "packed_at": "2026-11-17T00:00:00Z",
            "origin": {"host": "laptop", "home": "/home/you", "repo_id": "x", "workspace": "/x"},
            "files": {},
        }
    )
    + "\n"
)
# The start check's instruction, which no operator may be given for a forged pack.
_UNPACK_IT = "Run agent6 unpack"


def _forge(root: pathlib.Path) -> pathlib.Path:
    """Write the manifest a model would; the start check then names `agent6 unpack`."""
    forged = root / ".agent6" / "pack.json"
    forged.parent.mkdir(exist_ok=True)
    forged.write_text(_MANIFEST, encoding="utf-8")
    with pytest.raises(errors.PackError, match=_UNPACK_IT):
        pack_manifest.check_origin(root)
    return forged


def _names(directory: pathlib.Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_a_pack_json_left_behind_moves_to_the_next_free_name(tmp_path: pathlib.Path) -> None:
    """A forged manifest refused every later start with "Run agent6 unpack in that workspace"."""
    said: list[str] = []
    forged = _forge(tmp_path)
    pack.quarantine(tmp_path, warn=said.append)
    assert _names(tmp_path / ".agent6") == ["pack.invalid"]
    assert _names(tmp_path / ".agent6/pack.invalid") == ["pack.json.invalid"]
    moved = tmp_path / ".agent6/pack.invalid/pack.json.invalid"
    assert moved.read_text(encoding="utf-8") == _MANIFEST
    pack_manifest.check_origin(tmp_path)  # the next start is not refused
    with pytest.raises(errors.PackError, match="expected a regular file"):
        pack.unpack(tmp_path)
    (warning,) = said
    assert str(forged) in warning and "`pack.invalid/pack.json.invalid`." in warning
    assert "No agent6 command starts beside a pack, so this operation created it" in warning
    assert "Do not run `agent6 unpack` on it." in warning

    for taken in ("pack.invalid.1", "pack.invalid.2"):
        _forge(tmp_path)
        pack.quarantine(tmp_path, warn=said.append)
        moved = tmp_path / ".agent6" / taken / "pack.json.invalid"
        assert moved.read_text(encoding="utf-8") == _MANIFEST
        assert f"`{taken}/pack.json.invalid`" in said[-1]
    assert _names(tmp_path / ".agent6") == ["pack.invalid", "pack.invalid.1", "pack.invalid.2"]


def test_the_forged_files_move_with_the_manifest_and_a_real_pack_follows(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename left `config/` and `repo/` under the pack's names: `agent6 pack` then refused.

    It read them as a pack without a manifest ("pack.json: expected a regular file"), and a
    second forged `pack.json` would have made them a pack again.
    """
    repo = _repo(tmp_path, monkeypatch)
    forged = _forge(repo)
    planted = '[notify]\non_event = ["sh", "-c", "echo forged"]\n'
    for name in ("config/config.toml", "repo/config.toml", "skills/s/SKILL.md"):
        (repo / ".agent6" / name).parent.mkdir(parents=True)
        (repo / ".agent6" / name).write_text(planted, encoding="utf-8")
    (repo / ".agent6/worktrees").mkdir()  # not a pack's name: it stays
    said: list[str] = []

    pack.quarantine(repo, warn=said.append)

    assert _names(repo / ".agent6") == ["pack.invalid", "worktrees"]
    aside = repo / ".agent6/pack.invalid"
    assert _names(aside) == ["config", "pack.json.invalid", "repo", "skills"]
    assert (aside / "repo/config.toml").read_text(encoding="utf-8") == planted
    assert said == [
        f"found {forged}. No agent6 command starts beside a pack, so this operation created it;"
        " `agent6 pack` did not. agent6 moved it to `pack.invalid/pack.json.invalid`, with"
        " `config/`, `repo/` and `skills/`. Do not run `agent6 unpack` on it."
    ]

    manifest = pack.pack(repo)  # a real pack starts from none of the forged files
    assert "repo/config.toml" not in manifest.files and "skills/s/SKILL.md" not in manifest.files
    assert not (repo / ".agent6/skills/s").exists()
    assert _names(aside) == ["config", "pack.json.invalid", "repo", "skills"]


def test_an_area_that_will_not_move_is_named_for_deletion(tmp_path: pathlib.Path) -> None:
    """A directory moves only when it is writable, so a model can pin its `repo/` in place."""
    forged = _forge(tmp_path)
    pinned = tmp_path / ".agent6/repo"
    pinned.mkdir()
    pinned.chmod(0o500)
    said: list[str] = []
    try:
        pack.quarantine(tmp_path, warn=said.append)
    finally:
        pinned.chmod(0o700)
    if os.geteuid() == 0:
        pytest.skip("root moves a read-only directory")
    assert _names(tmp_path / ".agent6") == ["pack.invalid", "repo"]
    (warning,) = said
    assert f"found {forged}." in warning
    assert "agent6 moved it to `pack.invalid/pack.json.invalid`." in warning
    assert "It could not move `repo/` ([Errno 13] Permission denied" in warning
    assert warning.endswith("delete that. Do not run `agent6 unpack` on it.")


def test_a_linked_pack_json_moves_as_the_link(tmp_path: pathlib.Path) -> None:
    """A link the model plants moves as a link; nothing it points at is written or entered."""
    target = tmp_path / "outside.json"
    target.write_text(_MANIFEST, encoding="utf-8")
    bait = tmp_path / "bait"
    bait.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "kept").write_text("kept", encoding="utf-8")
    root = tmp_path / "repo"
    (root / ".agent6").mkdir(parents=True)
    (root / ".agent6/pack.json").symlink_to(target)
    (root / ".agent6/repo").symlink_to(elsewhere, target_is_directory=True)
    # A link planted on the first quarantine name is not followed and not replaced.
    (root / ".agent6/pack.invalid").symlink_to(bait, target_is_directory=True)
    said: list[str] = []
    pack.quarantine(root, warn=said.append)
    aside = root / ".agent6/pack.invalid.1"
    moved = aside / "pack.json.invalid"
    assert moved.is_symlink() and moved.readlink() == target
    assert (aside / "repo").is_symlink() and (aside / "repo").readlink() == elsewhere
    assert not os.path.lexists(root / ".agent6/pack.json")
    assert target.read_text(encoding="utf-8") == _MANIFEST and not target.is_symlink()
    assert (root / ".agent6/pack.invalid").readlink() == bait
    assert _names(bait) == [] and _names(elsewhere) == ["kept"]
    assert _names(tmp_path) == ["bait", "elsewhere", "outside.json", "repo"]
    assert len(said) == 1 and "`pack.invalid.1/pack.json.invalid`, with `repo/`." in said[0]
    pack_manifest.check_origin(root)


def test_a_linked_agent6_directory_is_reported_and_left_alone(tmp_path: pathlib.Path) -> None:
    """A link planted as `.agent6` is never written through, and never unpacked."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "pack.json").write_text(_MANIFEST, encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".agent6").symlink_to(elsewhere, target_is_directory=True)
    said: list[str] = []
    pack.quarantine(root, warn=said.append)
    assert _names(elsewhere) == ["pack.json"]
    assert (elsewhere / "pack.json").read_text(encoding="utf-8") == _MANIFEST
    assert (root / ".agent6").is_symlink() and _names(root) == [".agent6"]
    (warning,) = said
    assert f"{root / '.agent6'} is a symlink" in warning
    assert "Do not run `agent6 unpack` there." in warning
    # The start check and `unpack` both refuse the link by name, so nothing offers the pack.
    with pytest.raises(errors.PackError, match="symlink paths cannot be packed") as refused:
        pack_manifest.check_origin(root)
    assert _UNPACK_IT not in str(refused.value)
    with pytest.raises(errors.PackError, match="symlink paths cannot be packed"):
        pack.unpack(root)


def test_a_checkout_without_a_pack_json_gets_no_warning(tmp_path: pathlib.Path) -> None:
    """Every operation ends through the check, so it says nothing where nothing was forged."""
    said: list[str] = []
    pack.quarantine(tmp_path, warn=said.append)  # no `.agent6`
    kept = tmp_path / ".agent6/worktrees/run-1/file.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("a session's tree", encoding="utf-8")
    pack.quarantine(tmp_path, warn=said.append)  # other names only
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / ".agent6").write_text("a file", encoding="utf-8")
    pack.quarantine(plain, warn=said.append)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".agent6").symlink_to(tmp_path / ".agent6", target_is_directory=True)
    pack.quarantine(linked, warn=said.append)  # a link that leads to no pack.json
    assert said == []
    assert kept.read_text(encoding="utf-8") == "a session's tree"
    assert _names(tmp_path / ".agent6") == ["worktrees"]


def test_a_pack_json_that_cannot_be_moved_is_named_for_deletion(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused move still warns: silent, the next start would name `agent6 unpack`."""
    forged = _forge(tmp_path)

    def refused(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "rename", refused)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    pack.quarantine(tmp_path, warn=lambda _text: None, events=sink)
    assert forged.read_text(encoding="utf-8") == _MANIFEST
    assert _names(tmp_path / ".agent6") == ["pack.json"]  # no empty directory is left
    (row,) = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    assert row["renamed"] is None
    assert "agent6 could not move it ([Errno 13] Permission denied): delete it." in row["warning"]
    assert row["warning"].endswith("Do not run `agent6 unpack` on it.")


def test_a_session_says_it_once_through_its_journal(tmp_path: pathlib.Path) -> None:
    """The TUI and the web fold the journal, so a warning only on stderr never reached them.

    A console view renders the row as a marker; a headless run prints it; `warn` stays silent
    unless the journal cannot take the row.
    """
    said: list[str] = []
    printed: list[str] = []
    root = tmp_path / "repo"
    root.mkdir()
    forged = _forge(root)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.subscribe(functools.partial(execution.headless_marker, log=printed.append))
    pack.quarantine(root, warn=said.append, events=sink)
    (row,) = [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]
    assert row["type"] == "pack.quarantined"
    assert row["path"] == str(forged)
    assert row["renamed"] == str(root / ".agent6/pack.invalid/pack.json.invalid")
    assert row["warning"].startswith(f"found {forged}.")
    assert said == []
    assert printed == [f"[agent6] WARNING: {row['warning']}"]
    (item,) = fold_transcript([row])
    assert (item.kind, item.body) == ("marker", f"WARNING: {row['warning']}")
    assert f"pack.quarantined   found {forged}." in format_log_line(row)

    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the session directory would be", encoding="utf-8")
    _forge(root)
    pack.quarantine(root, warn=said.append, events=event_log.EventSink(blocker / "logs.jsonl"))
    assert len(said) == 1 and "`pack.invalid.1/pack.json.invalid`" in said[0]


# --- a session's end --------------------------------------------------------------------------

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "deny"

[models.default]
model = "p/m"

[models.worker]
model = "p/m"

[providers.p]
backend = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
when = "never"
"""

_Turn = tuple[dict[str, Any], ...]


class _Scripted:
    """One canned response per call, the last repeating; an exception in the script is raised."""

    def __init__(self, script: list[_Turn | BaseException]) -> None:
        self.script = script
        self.calls = 0

    def call(self, **_kw: Any) -> ProviderResponse:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        return ProviderResponse(
            text="",
            tool_uses=step,
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw={"content": list(step)},
        )


def _create(path: str, content: str) -> _Turn:
    return (
        {
            "type": "tool_use",
            "id": "e1",
            "name": "apply_edit",
            "input": {"path": path, "old_string": "", "new_string": content, "kind": "create"},
        },
    )


_FORGE = _create(".agent6/pack.json", _MANIFEST)
_FINISH: _Turn = (
    {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "done"}},
)


def _script(monkeypatch: pytest.MonkeyPatch, *steps: _Turn | BaseException) -> None:
    provider = _Scripted(list(steps))

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return provider

    monkeypatch.setattr(role_providers, "build_role_provider", _factory)


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, config: str = _CONFIG
) -> pathlib.Path:
    """A clean committed repo, the process cwd, under the scripted provider's config."""
    target = paths.global_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config, encoding="utf-8")
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "seed")
    monkeypatch.chdir(repo)
    return repo


def _run_dir(repo: pathlib.Path, session_id: str) -> pathlib.Path:
    return paths.state_dir(repo) / "sessions" / "runs" / session_id


def _quarantined(session_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = (session_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    return [row for row in map(json.loads, rows) if row["type"] == "pack.quarantined"]


def _next_start_is_not_refused(capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli_main(["config", "show", "sandbox.isolation"]) == 0
    out, err = capsys.readouterr()
    assert _UNPACK_IT not in out + err


@pytest.mark.parametrize(
    ("end", "rc"), [("finish", 0), ("turn limit", 1), ("crash", 1), ("interrupt", 130)]
)
def test_a_session_whose_model_forges_a_pack_ends_with_it_renamed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    end: str,
    rc: int,
) -> None:
    """The model wrote `.agent6/pack.json` with `apply_edit`; the next start said to unpack it.

    Every end runs the check: the model's finish, a spent turn allowance, a crash and Ctrl-C.
    """
    limit = "\n[harness]\nmax_turns = 1\n" if end == "turn limit" else ""
    repo = _repo(tmp_path, monkeypatch, config=_CONFIG + limit)
    last: _Turn | BaseException = {
        "finish": _FINISH,
        "turn limit": _FORGE,
        "crash": RuntimeError("the provider broke"),
        "interrupt": KeyboardInterrupt(),
    }[end]
    _script(monkeypatch, _FORGE, last)

    assert cli_main(["run", "--session-id", "pk-ENDS01", "tidy the repo"]) == rc
    out, err = capsys.readouterr()

    assert _names(repo / ".agent6") == ["pack.invalid"]
    moved = repo / ".agent6/pack.invalid/pack.json.invalid"
    assert moved.read_text(encoding="utf-8") == _MANIFEST
    said = [line for line in (out + err).splitlines() if "pack.json" in line]
    assert said == [
        f"[agent6] WARNING: found {repo / '.agent6/pack.json'}. No agent6 command starts beside"
        " a pack, so this operation created it; `agent6 pack` did not. agent6 moved it to"
        " `pack.invalid/pack.json.invalid`. Do not run `agent6 unpack` on it."
    ]
    (row,) = _quarantined(_run_dir(repo, "pk-ENDS01"))
    assert row["renamed"] == str(moved)
    _next_start_is_not_refused(capsys)


def test_a_merge_that_lands_a_tracked_pack_json_leaves_it_renamed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run's commits hold the forged file, so `sessions merge` wrote it back into the checkout.

    The session's own end took the first name; the merge's copy gets the next free one.
    """
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _FORGE, _FINISH)
    assert cli_main(["run", "--session-id", "pk-MERG01", "tidy the repo"]) == 0
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("pk-MERG01"))
    assert tip is not None and ".agent6/pack.json" in git_read.tree_paths(repo, tip)
    assert _names(repo / ".agent6") == ["pack.invalid"]
    capsys.readouterr()

    assert cli_main(["sessions", "merge", "pk-MERG01"]) == 0
    out, err = capsys.readouterr()

    assert ".agent6/pack.json" in git_read.tree_paths(repo, "main")
    assert _names(repo / ".agent6") == ["pack.invalid", "pack.invalid.1"]
    landed = repo / ".agent6/pack.invalid.1/pack.json.invalid"
    assert landed.read_text(encoding="utf-8") == _MANIFEST
    assert f"[agent6] WARNING: found {repo / '.agent6/pack.json'}." in err
    assert "agent6 moved it to `pack.invalid.1/pack.json.invalid`." in err
    assert _UNPACK_IT not in out + err
    _next_start_is_not_refused(capsys)


def test_an_automatic_merge_from_a_worktree_run_renames_what_it_lands(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--worktree` run under `git.auto_merge` put the forged file in the checkout unseen.

    The run's own checkout is its worktree, so only the landing reaches the operator's.
    """
    repo = _repo(tmp_path, monkeypatch, config=_CONFIG + "\n[git]\nauto_merge = true\n")
    _script(monkeypatch, _FORGE, _FINISH)

    assert cli_main(["run", "--worktree", "--session-id", "pk-AUTO01", "tidy the repo"]) == 0
    out, err = capsys.readouterr()

    worktree = sessions_manifest.read_manifest(_run_dir(repo, "pk-AUTO01")).worktree
    assert worktree is not None
    assert ".agent6/pack.json" in git_read.tree_paths(repo, "main")
    assert _names(repo / ".agent6") == ["pack.invalid"]
    assert _names(worktree / ".agent6") == ["pack.invalid"]
    assert {row["path"] for row in _quarantined(_run_dir(repo, "pk-AUTO01"))} == {
        str(repo / ".agent6/pack.json"),
        str(worktree / ".agent6/pack.json"),
    }
    assert _UNPACK_IT not in out + err
    _next_start_is_not_refused(capsys)


def test_a_landing_leaves_a_pack_that_was_there_before_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree run does not hold the checkout's lease, so `agent6 pack` can write beside it.

    The landing renames only what appears during it: a pack made first is `agent6 pack`'s own.
    """
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _create("new.txt", "made by the run\n"), _FINISH)
    assert cli_main(["run", "--worktree", "--session-id", "pk-KEEP01", "add new.txt"]) == 0
    run_dir = _run_dir(repo, "pk-KEEP01")
    manifest = sessions_manifest.read_manifest(run_dir)
    cfg = layer.load_effective(repo).config
    pack.pack(repo)
    packed = _names(repo / ".agent6"), (repo / ".agent6/pack.json").read_bytes()
    said: list[str] = []

    outcome = merge.execute_merge(
        repo,
        layout=sessions_layout.layout_of(run_dir),
        manifest=manifest,
        run_branch=git_read.chain_ref_for("pk-KEEP01"),
        target="main",
        base_sha=manifest.base_sha,
        method="squash",
        message="land the run",
        cfg=cfg,
        identity=merge.commit_identity(cfg, run_dir, manifest),
        warn=said.append,
    )

    assert outcome.status == "merged", outcome
    assert (repo / "new.txt").read_text(encoding="utf-8") == "made by the run\n"
    assert said == []
    assert (_names(repo / ".agent6"), (repo / ".agent6/pack.json").read_bytes()) == packed
    assert pack_manifest.read(repo).origin.workspace == str(repo)


# --- the other commands that run in the checkout ------------------------------------------------

_TOOL_MACHINE = """
machine = "forger"
version = 1
initial = "scan"

[budget]
max_transitions = 10

[states.scan]
kind = "tool"
command = ["true"]
timeout_s = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"

[states.fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def test_a_machine_whose_tool_state_forges_a_pack_ends_with_it_renamed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with no run state runs its tool commands in the operator's checkout."""
    monkeypatch.chdir(tmp_path)

    def forging(policy: kinds.JailPolicy) -> kinds.CommandResult:
        (policy.cwd / ".agent6").mkdir(exist_ok=True)
        (policy.cwd / ".agent6/pack.json").write_text(_MANIFEST, encoding="utf-8")
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(jail, "run_in_jail", forging)
    monkeypatch.setattr(jail, "namespaces_work", lambda: jail.NamespaceProbe(True))
    mfile = tmp_path / "m.asm.toml"
    mfile.write_text(_TOOL_MACHINE, encoding="utf-8")
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)

    assert machine_run.run_machine(mfile, reporter) == 0, said

    assert _names(tmp_path / ".agent6") == ["pack.invalid"]
    (warning,) = [line for line in said if "pack.json" in line]
    assert warning.startswith(f"[agent6] WARNING: found {tmp_path / '.agent6/pack.json'}.")
    pack_manifest.check_origin(tmp_path)


def test_an_exec_whose_command_forges_a_pack_ends_with_it_renamed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 exec --new` runs a script the model wrote; the script wrote the manifest."""

    def forging(policy: kinds.JailPolicy, session_net: Any = None) -> int:
        del session_net
        (policy.cwd / ".agent6").mkdir(exist_ok=True)
        (policy.cwd / ".agent6/pack.json").write_text(_MANIFEST, encoding="utf-8")
        return 7

    monkeypatch.setattr(exec_cmds, "run_attached", forging)
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})

    assert exec_cmds.exec_new(cfg, tmp_path, ("sh", "build.sh")) == 7

    assert _names(tmp_path / ".agent6") == ["pack.invalid"]
    assert f"[agent6] WARNING: found {tmp_path / '.agent6/pack.json'}." in capsys.readouterr().err
    pack_manifest.check_origin(tmp_path)


# --- `agent6 pack` itself -----------------------------------------------------------------------


def test_a_real_pack_is_never_renamed_and_unpack_accepts_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 pack` writes the one `pack.json` that stays: no command's end may take it.

    A start refused beside it ran nothing in the checkout, so it checks nothing on its way out.
    """
    repo = _repo(tmp_path, monkeypatch)
    note = paths.state_dir(repo) / "note"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("carried", encoding="utf-8")
    assert cli_main(["pack"]) == 0
    packed = (repo / ".agent6/pack.json").read_text(encoding="utf-8")
    capsys.readouterr()

    _script(monkeypatch, _FINISH)
    assert cli_main(["run", "--session-id", "pk-REAL01", "tidy the repo"]) == 2
    assert _UNPACK_IT in capsys.readouterr().err  # the start check's own text, unchanged
    assert (repo / ".agent6/pack.json").read_text(encoding="utf-8") == packed
    assert not [name for name in _names(repo / ".agent6") if "invalid" in name]

    note.unlink()
    assert cli_main(["unpack"]) == 0
    assert note.read_text(encoding="utf-8") == "carried"
    assert "WARNING" not in capsys.readouterr().err
