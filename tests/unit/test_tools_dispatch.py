# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.tools.dispatch — path safety, edit semantics, no-net I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher, ToolError

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
profile = "auto"
agent_network = "open"
run_commands = "no"
protect_git = true
protect_agent6 = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _config(tmp_path: Path) -> Config:
    from agent6.config import load_config

    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _config_with_run_commands(tmp_path: Path, mode: str) -> Config:
    from agent6.config import load_config

    p = tmp_path / f"agent6-{mode}.toml"
    p.write_text(
        _VALID_TOML.replace('run_commands = "no"', f'run_commands = "{mode}"'),
        encoding="utf-8",
    )
    return load_config(p)


def test_read_file_ok(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("read_file", {"path": "hello.txt"})
    assert out["content"] == "hi"


def test_absolute_path_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch("read_file", {"path": "/etc/passwd"})


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.\."):
        d.dispatch("read_file", {"path": "../outside.txt"})


def test_apply_edit_refuses_agent6_dir(tmp_path: Path) -> None:
    # Writes under .agent6/ would corrupt the harness's own
    # observability + resume state (graph.jsonl, log.jsonl, transcripts/).
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.agent6"):
        d.dispatch(
            "apply_edit",
            {
                "path": ".agent6/runs/foo/graph.jsonl",
                "edits": [{"kind": "create", "old_string": "", "new_string": "hax"}],
            },
        )


def test_apply_patch_refuses_agent6_dir(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.agent6"):
        d.dispatch(
            "apply_patch",
            {
                "path": ".agent6/runs/foo/log.jsonl",
                "patch": "--- /dev/null\n+++ .agent6/runs/foo/log.jsonl\n@@ -0,0 +1 @@\n+hax\n",
            },
        )


def test_read_file_allowed_under_agent6(tmp_path: Path) -> None:
    # Reads under .agent6/ are still allowed (resume may want to inspect
    # the prior transcript).
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    target = tmp_path / ".agent6" / "runs" / "foo"
    target.mkdir(parents=True)
    (target / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    out = d.dispatch("read_file", {"path": ".agent6/runs/foo/transcript.jsonl"})
    assert out["content"] == "{}\n"


def test_apply_edit_refuses_agent6_via_symlink(tmp_path: Path) -> None:
    # A symlink under the repo pointing at .agent6/ must not
    # launder a write through the prefix check; the post-resolution
    # check catches it.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / ".agent6").mkdir()
    (tmp_path / "decoy").symlink_to(".agent6", target_is_directory=True)
    with pytest.raises(ToolError, match=r"\.agent6.*symlink"):
        d.dispatch(
            "apply_edit",
            {
                "path": "decoy/graph.jsonl",
                "edits": [{"kind": "create", "old_string": "", "new_string": "hax"}],
            },
        )


def test_apply_edit_refuses_git_dir(tmp_path: Path) -> None:
    # apply_edit writes in-process (outside the jail), so without a guard the
    # LLM could plant a .git/hooks/pre-commit or rewrite .git/config and get
    # code run outside the sandbox on the next commit -- bypassing protect_git.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_edit",
            {
                "path": ".git/hooks/pre-commit",
                "edits": [{"kind": "create", "old_string": "", "new_string": "#!/bin/sh\nid\n"}],
            },
        )


def test_apply_patch_refuses_git_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_patch",
            {
                "path": ".git/config",
                "patch": "--- /dev/null\n+++ .git/config\n@@ -0,0 +1 @@\n+[core]\n",
            },
        )


def test_apply_edit_refuses_git_via_symlink(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / ".git").mkdir()
    (tmp_path / "decoy").symlink_to(".git", target_is_directory=True)
    with pytest.raises(ToolError, match=r"\.git.*symlink"):
        d.dispatch(
            "apply_edit",
            {
                "path": "decoy/hooks/pre-commit",
                "edits": [{"kind": "create", "old_string": "", "new_string": "x"}],
            },
        )


def test_apply_edit_allows_git_write_when_protect_git_false(tmp_path: Path) -> None:
    # Opting out of protect_git lifts the guard (consistent with the jail,
    # which also stops RO-binding .git when protect_git is false).
    from agent6.config import load_config

    p = tmp_path / "agent6-nogit.toml"
    p.write_text(_VALID_TOML.replace("protect_git = true", "protect_git = false"), encoding="utf-8")
    cfg = load_config(p)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": ".git/description",
            "edits": [{"kind": "create", "old_string": "", "new_string": "ok\n"}],
        },
    )
    assert (tmp_path / ".git" / "description").read_text(encoding="utf-8") == "ok\n"


def test_apply_edit_rejects_create_combined_with_other_edits(tmp_path: Path) -> None:
    # A `create` after a `replace` used to skip the file-exists guard (which
    # only fired for the first edit) and silently overwrite the whole file.
    # The schema now requires create to be the sole edit.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("keep me\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="create"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [
                    {"kind": "replace", "old_string": "keep me", "new_string": "edited"},
                    {"kind": "create", "old_string": "", "new_string": "OVERWRITE\n"},
                ],
            },
        )
    # File untouched (the call was rejected before any write).
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "keep me\n"


def test_apply_edit_create_and_replace(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "x = 1\n"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 2"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2\n"


def test_apply_edit_non_unique_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\na\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not unique"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "a", "new_string": "b"}],
            },
        )


def test_apply_edit_missing_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not found"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "bye", "new_string": "x"}],
            },
        )


def test_apply_edit_not_found_error_format(tmp_path: Path) -> None:
    # Finding C: the "old_string not found" error must NOT wrap the
    # file body in `---BEGIN <path>---` / `---END <path>---` markers, and
    # must NOT dump the entire body. Models that degenerate on repetition
    # were observed copying the marker scaffolding verbatim into the next
    # old_string. The new format gives "shape" (size, line count, head,
    # tail) and a tells-the-worker-what-to-do recovery hint.
    cfg = _config(tmp_path)
    body = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"  # 20 lines
    (tmp_path / "f.py").write_text(body, encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc_info:
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [
                    {"kind": "replace", "old_string": "no such text here", "new_string": "x"}
                ],
            },
        )
    msg = str(exc_info.value)
    assert "old_string not found" in msg
    # No scaffolding markers — these were the leak-back vector.
    assert "---BEGIN" not in msg
    assert "---END" not in msg
    # Shape and tail markers are present.
    assert f"{len(body)} bytes" in msg
    assert "20 lines" in msg
    assert "first 5 lines:" in msg
    assert "last 5 lines:" in msg
    # The full file body MUST NOT be in the error (otherwise the model
    # might still echo it back wholesale).
    assert "line 10" not in msg
    # Recovery hint must tell the model what to do next.
    assert "read_file" in msg


def test_apply_edit_not_found_short_file_omits_tail(tmp_path: Path) -> None:
    # Files of <=10 lines don't need the "...last 5 lines" duplication.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc_info:
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "no match", "new_string": "x"}],
            },
        )
    msg = str(exc_info.value)
    assert "first 5 lines:" in msg
    assert "last 5 lines:" not in msg
    assert "3 lines" in msg


def test_apply_edit_replace_requires_new_string(tmp_path: Path) -> None:
    # Kimi was emitting {kind:"replace", old_string:"..."}
    # WITHOUT a new_string. The old default `new_string: str = ""` silently
    # turned a malformed replace into a deletion, which corrupted the file
    # and then put the agent into an unrecoverable hallucination loop. The
    # boundary now rejects the malformed input loud per AGENTS.md.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="new_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "x = 1"}],
            },
        )
    # File untouched.
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_edit_replace_rejects_empty_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="old_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "", "new_string": "x"}],
            },
        )


def test_apply_edit_create_rejects_nonempty_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="old_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "create", "old_string": "junk", "new_string": "x"}],
            },
        )


def test_apply_patch_ok(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "path": "f.py",
            "patch": ("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"),
        },
    )
    assert out["path"] == "f.py"
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "a\nB\nc\n"


def test_apply_patch_create_new_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_patch",
        {
            "path": "new.py",
            "patch": "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+x = 1\n",
        },
    )
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_patch_context_mismatch_raises_tool_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nWRONG\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Context mismatch"):
        d.dispatch(
            "apply_patch",
            {
                "path": "f.py",
                "patch": ("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"),
            },
        )


def test_apply_patch_path_header_must_match_arg(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="disagrees"):
        d.dispatch(
            "apply_patch",
            {
                "path": "f.py",
                "patch": "--- a/g.py\n+++ b/g.py\n@@ -1 +1 @@\n-a\n+A\n",
            },
        )


def test_apply_patch_absolute_path_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch(
            "apply_patch",
            {
                "path": "/etc/passwd",
                "patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+A\n",
            },
        )


def test_apply_edit_preview_does_not_write(tmp_path: Path) -> None:
    """preview=true returns a diff but leaves disk untouched."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    res = d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 99"}],
            "preview": True,
        },
    )
    # File on disk is unchanged.
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    # Preview payload has expected shape.
    assert res["preview"] is True
    assert res["path"] == "f.py"
    assert res["hunks"] == 1
    assert "-x = 1" in res["diff"]
    assert "+x = 99" in res["diff"]
    assert res["bytes_before"] == len("x = 1\n")
    assert res["bytes_after"] == len("x = 99\n")
    assert res["would_apply"] == ["replace"]
    assert res["truncated"] is False


def test_apply_edit_preview_for_new_file_shows_dev_null(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    res = d.dispatch(
        "apply_edit",
        {
            "path": "new.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "hello\n"}],
            "preview": True,
        },
    )
    assert not (tmp_path / "new.py").exists()
    assert "/dev/null" in res["diff"]
    assert res["hunks"] == 1


def test_apply_patch_preview_does_not_write(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / "f.py").write_text("a\n", encoding="utf-8")
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+A\n"
    res = d.dispatch(
        "apply_patch",
        {"path": "f.py", "patch": patch, "preview": True},
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "a\n"
    assert res["preview"] is True
    assert res["hunks"] == 1
    assert "+A" in res["diff"]


def test_apply_edit_preview_truncates_giant_diff(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    big_old = "line\n" * 5000
    big_new = "LINE\n" * 5000
    (tmp_path / "big.txt").write_text(big_old, encoding="utf-8")
    res = d.dispatch(
        "apply_edit",
        {
            "path": "big.txt",
            "edits": [{"kind": "replace", "old_string": big_old, "new_string": big_new}],
            "preview": True,
        },
    )
    assert res["truncated"] is True
    assert "<truncated" in res["diff"]
    # Disk untouched.
    assert (tmp_path / "big.txt").read_text(encoding="utf-8") == big_old


def test_run_command_disabled_when_no(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="disabled"):
        d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert "run_command" not in d.available_tool_names()


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "checkout", "perf_takehome.py"],
        ["/usr/bin/git", "reset", "--hard"],
        ["git", "-C", ".", "restore", "src/foo.py"],
        ["git", "--no-pager", "stash", "push"],
        # `env` wrapper must not slip a mutating git command past the refusal.
        ["env", "git", "clean", "-fdx"],
        ["env", "FOO=bar", "git", "reset", "--hard"],
        ["/usr/bin/env", "-u", "GIT_DIR", "git", "checkout", "x.py"],
    ],
)
def test_run_command_refuses_mutating_git_commands(tmp_path: Path, argv: list[str]) -> None:
    cfg = _config_with_run_commands(tmp_path, "yes")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc_info:
        d.dispatch("run_command", {"argv": argv})
    msg = str(exc_info.value)
    assert "mutating git" in msg
    assert "git show HEAD:path/to/file" in msg
    assert "apply_patch" in msg


def test_run_command_refuses_mutating_git_before_approval(tmp_path: Path) -> None:
    cfg = _config_with_run_commands(tmp_path, "ask")

    def fail_if_called(prompt: str) -> bool:
        raise AssertionError(f"approval should not be requested: {prompt}")

    d = ToolDispatcher(root=tmp_path, config=cfg, approver=fail_if_called)
    with pytest.raises(ToolError, match="git checkout"):
        d.dispatch("run_command", {"argv": ["git", "checkout", "bad.py"]})


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Unknown"):
        d.dispatch("nope", {})


def test_grep_finds_match(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("hello world\nfoo\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("grep", {"pattern": "hello", "path": "."})
    assert len(out["hits"]) == 1
    assert out["hits"][0]["text"] == "hello world"


def test_grep_skips_dotdirs_by_default_but_searches_explicit_ones(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text("needle: here\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("needle\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    # Default recursive search from root skips the dot-dir.
    root_hits = d.dispatch("grep", {"pattern": "needle", "path": "."})["hits"]
    assert [h["path"] for h in root_hits] == ["plain.txt"]
    # Explicitly targeting the dot-dir searches inside it.
    dot_hits = d.dispatch("grep", {"pattern": "needle", "path": ".github"})["hits"]
    assert any(h["path"].endswith("ci.yml") for h in dot_hits)


def test_list_dir(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "y.txt").write_text("y", encoding="utf-8")
    (tmp_path / ".hidden").write_text("h", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("list_dir", {"path": "."})
    assert "x/" in out["entries"]
    assert "y.txt" in out["entries"]
    assert ".hidden" in out["entries"]  # hidden entries are included (per the description)


def test_passthrough_env_is_fixed_allowlist() -> None:
    """Regression: dispatch must never forward LD_*/PYTHON*/DYLD_* to the jail.

    The Rust launcher does `env_clear()` before applying `policy.env`, so this
    is defense-in-depth — but if someone ever widens the allowlist without
    auditing, this test fails loudly.
    """
    from agent6.tools import dispatch as _disp

    passthrough_keys: tuple[str, ...] = _disp._PASSTHROUGH_ENV_KEYS  # pyright: ignore[reportPrivateUsage]

    forbidden_prefixes = ("LD_", "DYLD_", "PYTHON")
    for key in passthrough_keys:
        assert not any(key.startswith(p) for p in forbidden_prefixes), (
            f"dangerous env key in allowlist: {key}"
        )
    # And at runtime, even if such vars are set in the parent, they must not
    # appear in the dict the dispatcher builds for the jail policy.
    import os

    saved = {k: os.environ.get(k) for k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH")}
    try:
        os.environ["LD_PRELOAD"] = "/tmp/evil.so"
        os.environ["LD_LIBRARY_PATH"] = "/tmp/evil"
        os.environ["PYTHONPATH"] = "/tmp/evil"
        env = _disp._passthrough_env()  # pyright: ignore[reportPrivateUsage]
        assert "LD_PRELOAD" not in env
        assert "LD_LIBRARY_PATH" not in env
        assert "PYTHONPATH" not in env
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_outline_returns_symbols(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\nclass Bar:\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("outline", {"path": "a.py"})
    names = {(s["name"], s["kind"]) for s in out["symbols"]}
    assert ("foo", "function") in names
    assert ("Bar", "class") in names
    assert out["truncated"] is False


def test_outline_rejects_directory(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "sub").mkdir()
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Not a file"):
        d.dispatch("outline", {"path": "sub"})


def test_outline_rejects_escape(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch("outline", {"path": "/etc/hosts"})


def test_find_definition_returns_relative_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def target():\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("find_definition", {"name": "target"})
    assert len(out["definitions"]) == 1
    assert out["definitions"][0]["path"] == "a.py"
    assert out["definitions"][0]["kind"] == "function"


def test_find_references_returns_relative_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\nfoo()\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("find_references", {"name": "foo"})
    # Definition + call
    assert len(out["references"]) == 2
    assert all(r["path"] == "a.py" for r in out["references"])


def test_apply_edit_invalidates_index(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    # Prime the index
    assert d.dispatch("find_definition", {"name": "foo"})["definitions"]
    assert d.dispatch("find_definition", {"name": "bar"})["definitions"] == []
    # Edit the file via the tool layer.
    d.dispatch(
        "apply_edit",
        {
            "path": "a.py",
            "edits": [
                {
                    "kind": "replace",
                    "old_string": "def foo():\n    pass\n",
                    "new_string": "def bar():\n    pass\n",
                }
            ],
        },
    )
    assert d.dispatch("find_definition", {"name": "bar"})["definitions"]
    assert d.dispatch("find_definition", {"name": "foo"})["definitions"] == []


def test_new_index_tools_listed_in_available(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = set(d.available_tool_names())
    assert {"outline", "find_definition", "find_references"} <= names


def test_run_metric_command_no_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"no \[workflow.metric\]"):
        d.dispatch("run_metric_command", {})
    # Not in the LLM-visible tool surface either.
    assert "run_metric_command" not in d.available_tool_names()


def test_run_metric_command_invokes_jail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _VALID_TOML + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"CYCLES: 42\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    cfg = load_config(p)

    captured: dict[str, object] = {}

    def fake_run_in_jail(policy):  # type: ignore[no-untyped-def]
        captured["argv"] = tuple(policy.argv)
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=0,
            stdout="CYCLES: 42\n",
            stderr="",
            duration_s=0.01,
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("run_metric_command", {})
    assert out["returncode"] == 0
    assert "CYCLES: 42" in out["stdout"]
    assert captured["argv"] == ("/usr/bin/python3", "-c", 'print("CYCLES: 42")')
    # audit: the handler now parses the pattern's first capture
    # group to a float and surfaces it as `score`.
    assert out["score"] == 42.0


def test_run_metric_command_score_null_on_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pattern compiles fine but doesn't match the output -> score is null,
    rest of the result is unchanged."""
    body = _VALID_TOML + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"no number here\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    cfg = load_config(p)

    def fake_run_in_jail(policy):  # type: ignore[no-untyped-def]
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=0,
            stdout="no number here\n",
            stderr="",
            duration_s=0.01,
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("run_metric_command", {})
    assert out["score"] is None
    assert out["returncode"] == 0
    assert "no number here" in out["stdout"]


def test_disable_apply_edit_env_hides_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # AGENT6_DISABLE_APPLY_EDIT=1 removes apply_edit
    # from the surface advertised to the LLM and refuses to dispatch
    # any straggler calls. apply_patch stays available — it's the
    # whole point of the experiment.
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = d.available_tool_names()
    assert "apply_edit" not in names
    assert "apply_patch" in names


def test_disable_apply_edit_env_blocks_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="AGENT6_DISABLE_APPLY_EDIT"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}],
            },
        )


def test_disable_apply_edit_unset_leaves_tool_available(tmp_path: Path) -> None:
    # Default behaviour: env var unset, both tools available.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = d.available_tool_names()
    assert "apply_edit" in names
    assert "apply_patch" in names


def test_dispatcher_refuses_mutations_in_plan_mode(tmp_path: Path) -> None:
    # Defense-in-depth: even if a mutation tool reaches dispatch() in plan mode
    # (the LLM's tool list already omits them), the dispatcher must refuse.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
    with pytest.raises(ToolError, match="plan mode"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}]},
        )
    with pytest.raises(ToolError, match="plan mode"):
        d.dispatch("apply_patch", {"patch": "--- a\n+++ b\n"})


def test_agent6_docs_tool_lists_and_reads(tmp_path: Path) -> None:
    # agent6_docs reads agent6's own bundled docs (for "how do I use agent6").
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    listing = d.dispatch("agent6_docs", {})
    assert "CONFIG" in listing["available"]
    assert "README" in listing["available"]
    doc = d.dispatch("agent6_docs", {"name": "CONFIG"})
    assert "content" in doc and len(doc["content"]) > 100
    with pytest.raises(ToolError, match="unknown agent6 doc"):
        d.dispatch("agent6_docs", {"name": "NOPE"})
