# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An `@path` the operator typed is inlined in an opening and a steer, and completes."""

from __future__ import annotations

import json
import pathlib
import subprocess
import warnings

import pytest

from agent6 import paths, task_text
from agent6.config import Config, layer_data
from agent6.sessions import ipc, manifest
from agent6.ui import directive_actions, task_refs


def test_inlines_existing_file(tmp_path: pathlib.Path) -> None:
    (tmp_path / "notes.md").write_text("hello world\n", encoding="utf-8")
    out = task_refs.expand_task_file_refs("read @notes.md please", tmp_path)
    assert '<file path="notes.md">' in out
    assert "hello world" in out
    assert "</file>" in out
    # The surrounding sentence words are still there.
    assert out.startswith("read @notes.md\n")
    assert out.rstrip().endswith("please")
    # The operator's reference stays once the block is stripped, so every task line keeps it.
    assert task_text.operator_task_text(out) == "read @notes.md\nplease"


def test_leaves_missing_path_untouched(tmp_path: pathlib.Path) -> None:
    out = task_refs.expand_task_file_refs("see @does/not/exist for context", tmp_path)
    assert out == "see @does/not/exist for context"


def test_does_not_touch_email_addresses(tmp_path: pathlib.Path) -> None:
    out = task_refs.expand_task_file_refs("ping user@example.com about it", tmp_path)
    assert out == "ping user@example.com about it"


def test_rejects_escape_via_parent(tmp_path: pathlib.Path) -> None:
    # The bait sits where `@../escape` RESOLVES (one up from root), so it is a
    # real, readable file outside root -- the relative_to guard is the only
    # reason the read is refused, not an is_file() miss.
    sub = tmp_path / "work"
    sub.mkdir()
    outside = tmp_path / "agent6_test_escape.txt"
    outside.write_text("secret\n", encoding="utf-8")
    out = task_refs.expand_task_file_refs("look at @../agent6_test_escape.txt", sub)
    assert "secret" not in out
    assert "@../agent6_test_escape.txt" in out


def test_ignores_directories(tmp_path: pathlib.Path) -> None:
    (tmp_path / "subdir").mkdir()
    out = task_refs.expand_task_file_refs("see @subdir for context", tmp_path)
    assert out == "see @subdir for context"


def test_truncates_large_files(tmp_path: pathlib.Path) -> None:
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (task_refs.TASK_FILE_REF_MAX_BYTES + 5000))
    out = task_refs.expand_task_file_refs("@big.txt", tmp_path)
    assert "truncated" in out
    assert "5000 bytes omitted" in out


def test_handles_binary_with_replacement(tmp_path: pathlib.Path) -> None:
    binf = tmp_path / "bin.dat"
    binf.write_bytes(b"\xff\xfe\x00garbage")
    out = task_refs.expand_task_file_refs("@bin.dat", tmp_path)
    assert '<file path="bin.dat">' in out
    # Replacement char appears, no crash.
    assert "\ufffd" in out or "garbage" in out


def test_multiple_refs_in_one_string(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    out = task_refs.expand_task_file_refs("compare @a.txt with @b.txt", tmp_path)
    assert "AAA" in out
    assert "BBB" in out


def test_empty_task_is_passthrough(tmp_path: pathlib.Path) -> None:
    assert task_refs.expand_task_file_refs("", tmp_path) == ""


def test_no_refs_is_passthrough(tmp_path: pathlib.Path) -> None:
    assert task_refs.expand_task_file_refs("just a task", tmp_path) == "just a task"


def _task_the_run_received(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], cfg: Config | None = None
) -> str:
    """Run the CLI with the lifecycle stubbed and return the task `run_task` was handed."""
    from agent6.app import preflight
    from agent6.app import run as app_run
    from agent6.app.session import setup as session_setup
    from agent6.ui.cli import entry

    seen: list[str] = []

    def loaded(*_a: object, **_k: object) -> layer_data.EffectiveConfig:
        return layer_data.EffectiveConfig(config=cfg or Config(), sources={}, layers=())

    def run_task(_cfg: object, task: str, **_kw: object) -> int:
        seen.append(task)
        return 0

    def in_a_repo(_cwd: object) -> bool:
        return True

    def no_prompt(_args: object, rc: int, _session_id: str) -> int:
        return rc

    monkeypatch.setattr(session_setup, "load_session_config", loaded)
    monkeypatch.setattr(app_run, "run_task", run_task)
    monkeypatch.setattr(preflight, "require_git_repo", in_a_repo)
    monkeypatch.setattr(entry, "_prompt_for_the_next_input", no_prompt)
    assert entry.main(argv) == 0
    return seen[0]


def _seed_plan(root: pathlib.Path, session_id: str, plan_md: str) -> None:
    plan_dir = paths.state_dir(root) / "sessions" / "plans" / session_id
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.md").write_text(plan_md, encoding="utf-8")
    (plan_dir / "manifest.json").write_text(
        json.dumps({"version": 0, "mode": "plan", "user_task": "plan the notes"}),
        encoding="utf-8",
    )
    (plan_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.stop", "reason": "planned", "turns": 1}) + "\n",
        encoding="utf-8",
    )


def test_a_plan_run_as_the_task_is_not_searched_for_refs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run --from PLAN` inlined every `@path` the plan's model wrote into the first message."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: tidy\n\n1. See @notes.md first.\n")

    task = _task_the_run_received(monkeypatch, ["run", "--from", "happy-tree-abcd"])

    assert "See @notes.md first." in task
    assert "NOTE BODY" not in task


def test_a_seeded_session_is_not_searched_for_refs_and_the_typed_task_is(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run --from ID TASK` inlined the `@path` in the seeded session's plan with the typed ones."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    (tmp_path / "typed.md").write_text("TYPED BODY\n", encoding="utf-8")
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: tidy\n\n1. See @notes.md first.\n")

    task = _task_the_run_received(
        monkeypatch, ["run", "--from", "happy-tree-abcd", "start from @typed.md"]
    )

    assert "TYPED BODY" in task and task.rstrip().endswith("</file>")
    assert "See @notes.md first." in task
    assert "NOTE BODY" not in task


def test_an_ask_file_seed_is_not_searched_for_refs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ask --file F` inlined the `@path` references the file's own text held."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("Read @notes.md too.\n", encoding="utf-8")

    task = _task_the_run_received(
        monkeypatch, ["ask", "query", "--file", "readme.md", "what does it say"]
    )

    assert "Read @notes.md too." in task and task.endswith("what does it say")
    assert "NOTE BODY" not in task


def test_a_typed_ref_to_a_hidden_path_is_left_as_typed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed `@path` under `sandbox.hide_paths` was inlined: only the root was checked."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    cfg = Config.model_validate({"sandbox": {"hide_paths": [str(tmp_path / ".env")]}})

    task = _task_the_run_received(monkeypatch, ["run", "use @.env and @notes.md"], cfg)

    assert "hunter2" not in task and "use @.env and" in task
    assert "NOTE BODY" in task


def test_a_ref_into_the_git_dir_is_left_as_typed(tmp_path: pathlib.Path) -> None:
    """`@.git/config` is outside what a host-side read takes, like a hidden path."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[remote]\n", encoding="utf-8")
    assert task_refs.expand_task_file_refs("see @.git/config", tmp_path) == "see @.git/config"


def _session(tmp_path: pathlib.Path) -> pathlib.Path:
    session_dir = tmp_path / "state" / "sessions" / "runs" / "one"
    session_dir.mkdir(parents=True)
    return session_dir


def test_an_ordinary_steer_inlines_the_file_a_typed_ref_names(tmp_path: pathlib.Path) -> None:
    """A steer's `@path` reached the model as typed: only `agent6 run` inlined the file."""
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    session_dir = _session(tmp_path)

    did, _said = directive_actions.submit_composer_line(
        session_dir, "follow @notes.md", repo=tmp_path
    )

    assert did
    assert ipc.take_steer_answer(session_dir) == (
        'follow @notes.md\n<file path="notes.md">\nNOTE BODY\n\n</file>\n'
    )


def test_an_urgent_steer_inlines_and_a_loop_directive_does_not(tmp_path: pathlib.Path) -> None:
    """`/now` carries an ordinary steer; a `/pin` line is text the loop keeps whole."""
    (tmp_path / "notes.md").write_text("NOTE BODY\n", encoding="utf-8")
    session_dir = _session(tmp_path)

    assert directive_actions.submit_composer_line(
        session_dir, "/now follow @notes.md", repo=tmp_path
    )[0]
    assert "NOTE BODY" in (ipc.take_steer_answer(session_dir) or "")
    ipc.clear_steer_request(session_dir)
    assert directive_actions.submit_composer_line(
        session_dir, "/pin keep to @notes.md", repo=tmp_path
    )[0]
    assert ipc.take_steer_answer(session_dir) == "/pin keep to @notes.md"


def test_a_steer_leaves_a_ref_to_a_hidden_path_as_typed(tmp_path: pathlib.Path) -> None:
    """The steer reads under the same `sandbox.hide_paths` boundary the opening does."""
    (tmp_path / ".env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    overlay = tmp_path / "overlay.toml"
    overlay.write_text(f"[sandbox]\nhide_paths = [{str(tmp_path / '.env')!r}]\n", encoding="utf-8")
    session_dir = _session(tmp_path)

    assert directive_actions.submit_composer_line(
        session_dir, "use @.env", repo=tmp_path, config_paths=(overlay,)
    )[0]

    assert ipc.take_steer_answer(session_dir) == "use @.env"


def test_a_steer_to_a_worktree_session_reads_the_worktree(tmp_path: pathlib.Path) -> None:
    """A worktree run's files are its own: the repository's copy may differ."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (tmp_path / "notes.md").write_text("REPO COPY\n", encoding="utf-8")
    (worktree / "notes.md").write_text("WORKTREE COPY\n", encoding="utf-8")
    session_dir = _session(tmp_path)
    (session_dir / manifest.MANIFEST_NAME).write_text(
        manifest.SessionManifest(mode="run", worktree=worktree).model_dump_json(),
        encoding="utf-8",
    )

    assert directive_actions.submit_composer_line(session_dir, "follow @notes.md", repo=tmp_path)[0]

    sent = ipc.take_steer_answer(session_dir) or ""
    assert "WORKTREE COPY" in sent and "REPO COPY" not in sent


def _repo(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_ref_fragment_is_the_reference_the_text_ends_in() -> None:
    assert task_refs.ref_fragment("read @src/ma") == "src/ma"
    assert task_refs.ref_fragment("read @") == ""
    assert task_refs.ref_fragment("@") == ""
    assert task_refs.ref_fragment("mail user@exam") is None  # an address, as the expander reads it
    assert task_refs.ref_fragment("read @src/main.py then") is None
    assert task_refs.ref_fragment("plain words") is None


def test_suggestions_put_prefix_matches_first_and_stop_at_the_cap(tmp_path: pathlib.Path) -> None:
    """The completion source returns at most its cap and counts what it left out."""
    names = {f"src/mod{i:02d}.py": "" for i in range(12)}
    repo = _repo(tmp_path, {**names, "docs/src-notes.md": "", ".gitignore": "ignored.txt\n"})
    (repo / "ignored.txt").write_text("", encoding="utf-8")

    found = task_refs.suggest_refs(repo, (), "src")

    assert len(found.paths) == task_refs.REF_SUGGESTIONS_MAX == 8
    assert found.paths == tuple(f"src/mod{i:02d}.py" for i in range(8))
    assert found.more == 5  # four more modules, then the path that only contains the fragment
    assert found.common == "src/mod"
    assert task_refs.suggest_refs(repo, (), "notes").paths == ("docs/src-notes.md",)
    assert task_refs.suggest_refs(repo, (), "ignored").paths == ()
    assert task_refs.suggest_refs(repo, (), "", limit=100).more == 0


def test_a_completions_common_prefix_is_counted_in_characters(tmp_path: pathlib.Path) -> None:
    """`os.path.commonprefix` did the count and warns as deprecated on Python 3.15."""
    repo = _repo(tmp_path, {"src/main.py": "", "src/map.py": ""})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert task_refs.suggest_refs(repo, (), "src").common == "src/ma"
    assert task_refs.common_prefix(["/parallel", "/pin", "/plan"]) == "/p"
    assert task_refs.common_prefix(["ab", "abc"]) == "ab"
    assert task_refs.common_prefix(["x", "y"]) == task_refs.common_prefix([]) == ""


def test_suggestions_leave_out_what_the_expander_cannot_read(tmp_path: pathlib.Path) -> None:
    """A hidden path, a symlink and a name outside the reference characters are never offered."""
    repo = _repo(
        tmp_path,
        {"secrets/key.pem": "k", "open/key.txt": "t", "open/key with space.txt": "s"},
    )
    (repo / "open" / "key-link").symlink_to("key.txt")

    found = task_refs.suggest_refs(repo, (str(repo / "secrets"),), "key")

    assert found.paths == ("open/key.txt",)
    assert task_refs.suggest_refs(repo, (), "key").paths == ("open/key.txt", "secrets/key.pem")


def test_suggestions_are_empty_outside_a_repository(tmp_path: pathlib.Path) -> None:
    (tmp_path / "notes.md").write_text("", encoding="utf-8")
    assert task_refs.suggest_refs(tmp_path, (), "no") == task_refs.RefSuggestions((), 0, "no")
