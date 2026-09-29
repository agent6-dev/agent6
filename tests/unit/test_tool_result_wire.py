# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""FROZEN model-facing wire: the bytes each tool handler serializes to the LLM.

The loop JSON-dumps a dispatched tool's result verbatim into the tool_result
the model reads (harness/loop.py). That JSON -- keys, key ORDER (dicts
preserve insertion order), and value formats -- is frozen LLM I/O: a drift
silently changes every model's tool feedback. This pins a representative
handler from each family, including the optional-field, score-append, preview,
and error shapes, so the 8b typed-result reshape must reproduce the bytes.

``_wire`` bridges the reshape: ``dispatch`` returns a bare dict today and a
typed result carrying ``to_wire()`` after. Either way this compares the exact
model-facing bytes, so the file stays green across the change unedited.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6.config import Config, load_config
from agent6.graph import curator, models
from agent6.sessions import layout
from agent6.tools import dispatch, errors, operator_prompts

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
isolation = "auto"
run_commands = "yes"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[harness]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _config(tmp_path: pathlib.Path, *, extra: str = "") -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML + extra, encoding="utf-8")
    return load_config(p)


def _wire(result: object) -> dict[str, Any]:
    """Model-facing bytes of a dispatch result, before or after the typed reshape.

    Post-reshape ``dispatch`` returns a result with ``to_wire()``; today it returns the dict itself.
    """
    to_wire = getattr(result, "to_wire", None)
    return to_wire() if callable(to_wire) else result  # type: ignore[return-value]


def _dumps(result: object) -> str:
    return json.dumps(_wire(result), ensure_ascii=False)


# --- content access family ---------------------------------------------------


def test_wire_read_file_full(tmp_path: pathlib.Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("read_file", {"path": "hello.txt"})) == (
        '{"content": "hi", "size": 2, "lines_total": 1}'
    )


def test_wire_read_file_slice(tmp_path: pathlib.Path) -> None:
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("read_file", {"path": "abc.txt", "start_line": 2, "limit": 1})
    assert _dumps(out) == (
        '{"content": "b\\n", "size": 2, "lines_total": 3, "start_line": 2, "lines_returned": 1}'
    )


def test_wire_read_file_full_agrees_with_slice_on_lines_total(tmp_path: pathlib.Path) -> None:
    """Full and partial reads of one file report the same lines_total, the splitlines count.

    The newline-count-plus-one heuristic overshot every newline-terminated file.
    """
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("read_file", {"path": "abc.txt"})) == (
        '{"content": "a\\nb\\nc\\n", "size": 6, "lines_total": 3}'
    )


def test_wire_read_file_start_past_eof(tmp_path: pathlib.Path) -> None:
    """A paging overshoot returns an empty slice with lines_returned=0, never a negative count."""
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("read_file", {"path": "abc.txt", "start_line": 10, "limit": 5})
    assert _dumps(out) == (
        '{"content": "", "size": 0, "lines_total": 3, "start_line": 10, "lines_returned": 0}'
    )


def test_wire_list_dir(tmp_path: pathlib.Path) -> None:
    sub = tmp_path / "d"
    sub.mkdir()
    (sub / "a.txt").write_text("", encoding="utf-8")
    (sub / "b.txt").write_text("", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("list_dir", {"path": "d"})) == '{"entries": ["a.txt", "b.txt"]}'


# --- filesystem-write family (applied + preview) -----------------------------


def test_wire_apply_edit_applied(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch(
        "apply_edit",
        {"path": "new.txt", "edits": [{"kind": "create", "new_string": "x\n"}]},
    )
    assert _dumps(out) == '{"applied": ["create"], "path": "new.txt"}'


def test_wire_apply_edit_preview_carries_would_apply(tmp_path: pathlib.Path) -> None:
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "preview": True,
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 99"}],
        },
    )
    w = _wire(out)
    # Preview shape: fixed key order, and would_apply present ONLY for apply_edit.
    assert list(w) == [
        "preview",
        "path",
        "diff",
        "hunks",
        "bytes_before",
        "bytes_after",
        "truncated",
        "would_apply",
    ]
    assert w["preview"] is True
    assert w["would_apply"] == ["replace"]
    assert w["hunks"] == 1


def test_wire_apply_patch_preview_omits_would_apply(tmp_path: pathlib.Path) -> None:
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("apply_patch", {"path": "f.py", "patch": patch, "preview": True})
    w = _wire(out)
    assert list(w) == [
        "preview",
        "path",
        "diff",
        "hunks",
        "bytes_before",
        "bytes_after",
        "truncated",
    ]


# --- run-control family ------------------------------------------------------


def test_wire_finish_session(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("finish_session", {"summary": "done", "result": {"k": 1}})
    assert _dumps(out) == '{"acknowledged": true, "summary": "done", "result": {"k": 1}}'


def test_wire_finish_session_null_result(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("finish_session", {"summary": "done"})
    assert _dumps(out) == '{"acknowledged": true, "summary": "done", "result": null}'


def test_wire_finish_planning(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), mode="plan")
    out = d.dispatch("finish_planning", {"summary": "s", "plan_markdown": "# Plan\n"})
    assert _dumps(out) == '{"acknowledged": true, "summary": "s", "plan_bytes": 7}'


def test_wire_ask_user(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(
        root=tmp_path,
        config=_config(tmp_path),
        prompts=operator_prompts.OperatorPrompts(
            questioner=lambda request: operator_prompts.QuestionAnswer(
                tuple("ans" for _ in request.questions), "stdin"
            )
        ),
    )
    out = d.dispatch("ask_user", {"questions": [{"question": "q?", "options": ["a", "b"]}]})
    assert _dumps(out) == '{"answers": ["ans"]}'


# --- DAG family (dynamic ULID ids -> pin order + shape) ----------------------


def test_wire_add_task_order(tmp_path: pathlib.Path) -> None:
    cur = curator.GraphCurator(layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="r"))
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(
        root=tmp_path, config=_config(tmp_path), curator=cur, run_root_node_id=root.id
    )
    w = _wire(d.dispatch("add_task", {"title": "sub"}))
    assert list(w) == ["id", "parent_id", "title", "status"]
    assert w["id"].isdigit() and w["id"] != root.id  # the run's next number, not the root's
    assert (w["parent_id"], w["title"], w["status"]) == (root.id, "sub", "pending")


# --- execution family (jail-backed; mock run_in_jail) ------------------------


def _cmd_result(**kw: Any):
    from agent6 import kinds

    base = dict(argv=("x",), returncode=0, stdout="", stderr="", duration_s=0.5, exec_failed=False)
    base.update(kw)
    return kinds.CommandResult(**base)  # type: ignore[arg-type]


def test_wire_run_verify(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with mock.patch("agent6.sandbox.jail.run_in_jail", return_value=_cmd_result(stdout="ok")):
        out = d.dispatch("run_verify_command", {})
    # The gate names itself: the worker never chose this command, so without
    # it a real failure and a stale gate look identical from the result.
    assert _dumps(out) == (
        '{"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.5,'
        ' "exec_failed": false, "command": "true"}'
    )


def test_wire_run_verify_timeout_names_the_cap(tmp_path: pathlib.Path) -> None:
    """A verify killed at verify_timeout_s reaches the model as timed_out with the cap.

    A bare returncode 124 with empty output was indistinguishable from a failing suite, and
    SWE-bench runs re-called the 240s gate back to back.
    """
    toml = _VALID_TOML.replace(
        'verify_command = ["true"]', 'verify_command = ["true"]\nverify_timeout_s = 240'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(toml, encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=load_config(p))
    with mock.patch(
        "agent6.sandbox.jail.run_in_jail",
        return_value=_cmd_result(returncode=124, duration_s=240.1),
    ):
        out = d.dispatch("run_verify_command", {})
    assert _dumps(out) == (
        '{"returncode": 124, "stdout": "", "stderr": "", "duration_s": 240.1,'
        ' "exec_failed": false, "command": "true", "timed_out": true, "timeout_s": 240.0}'
    )


def test_wire_run_command(tmp_path: pathlib.Path) -> None:
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with mock.patch(
        "agent6.sandbox.jail.run_in_jail",
        return_value=_cmd_result(returncode=3, stdout="o", stderr="e"),
    ):
        out = d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert _dumps(out) == (
        '{"returncode": 3, "stdout": "o", "stderr": "e", "duration_s": 0.5, "exec_failed": false}'
    )


def test_wire_run_command_clip_names_dropped_chars(tmp_path: pathlib.Path) -> None:
    """Output over the 20k cap leads with a marker naming the dropped char count.

    A bare tail read as the complete output.
    """
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    big = "x" * 25_000
    with mock.patch("agent6.sandbox.jail.run_in_jail", return_value=_cmd_result(stdout=big)):
        out = d.dispatch("run_command", {"argv": ["echo", "hi"]})
    stdout = _wire(out)["stdout"]
    assert stdout.startswith("... 5000 earlier chars clipped ...\n")
    assert stdout.endswith("x" * 100) and len(stdout) < 20_100
    extra = (
        "\n[harness.metric]\n"
        'command = ["/usr/bin/true"]\n'
        'pattern = "CYCLES: (\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path, extra=extra))
    with mock.patch(
        "agent6.sandbox.jail.run_in_jail", return_value=_cmd_result(stdout="CYCLES: 42")
    ):
        out = d.dispatch("run_metric_command", {})
    # score is APPENDED after the exec fields, in that order.
    assert _dumps(out) == (
        '{"returncode": 0, "stdout": "CYCLES: 42", "stderr": "", "duration_s": 0.5,'
        ' "exec_failed": false, "score": 42.0}'
    )


def test_metric_score_survives_the_display_clip(tmp_path: pathlib.Path) -> None:
    """The metric score comes from the command's unclipped output.

    Parsed from the display-clipped stdout, a real score before the tail window was lost, or a
    permissive pattern matched the clip marker's own dropped-byte count.
    """
    toml = _VALID_TOML.replace('isolation = "auto"', 'isolation = "auto"\nnetwork = "host"') + (
        '\n[harness.metric]\ncommand = ["/usr/bin/true"]\npattern = "(\\\\d+)"\ngoal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(toml, encoding="utf-8")
    d = dispatch.ToolDispatcher(root=tmp_path, config=load_config(p))
    stdout = "CYCLES: 42\n" + "y" * 25_000
    with mock.patch("agent6.sandbox.jail.run_in_jail", return_value=_cmd_result(stdout=stdout)):
        out = d.dispatch("run_metric_command", {})
    assert _wire(out)["score"] == 42.0


# --- error shape (loop wraps a raised ToolError) -----------------------------


def test_wire_tool_error_shape(tmp_path: pathlib.Path) -> None:
    """The model-facing error bytes are driven through the loop's error path, not a rebuilt dict.

    Rebuilding it here pinned the test's own literal and left the producer unpinned.
    """
    from agent6.harness import _loop_state, loop

    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with pytest.raises(errors.ToolError) as exc:
        d.dispatch("no_such_tool", {})

    wf = mock.MagicMock()
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    content = loop.Harness._note_tool_error(  # pyright: ignore[reportPrivateUsage]
        wf, state, "no_such_tool", {}, exc.value
    )
    assert content == '{"error": "Unknown tool: no_such_tool"}'


def test_the_tool_error_log_line_names_the_tool_once() -> None:
    """A tool raises the bare message; the logger prefixes the tool's name."""
    from agent6 import skills
    from agent6.harness import _loop_state, loop
    from agent6.tools import _skill_tools  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(errors.ToolError) as exc:
        _skill_tools.use_skill(
            lambda: skills.ResolvedSkills(enabled=(), always=(), warnings=()), {"name": "x"}
        )
    assert str(exc.value).startswith("unknown or disabled skill 'x'")
    wf = mock.MagicMock()
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    loop.Harness._note_tool_error(  # pyright: ignore[reportPrivateUsage]
        wf, state, "use_skill", {}, exc.value
    )
    wf._log.assert_called_with(f"  tool_error: use_skill: {exc.value}")
