# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child reads an origin journal by full id without listing the origin."""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from agent6 import paths
from agent6.app.parallel import launch as child_launch
from agent6.config import model as config_model
from agent6.tools import dispatch, tool_errors
from tests import child_env, launch_env


def test_child_reads_full_origin_ids_outside_its_clone_state(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"
    for root in (origin, clone):
        child_env.init_repo(root)
    origin_sessions = paths.state_dir(origin) / "sessions" / "runs"
    parent = origin_sessions / "calm-heron-parent"
    child = origin_sessions / "calm-heron-parent-p1-l1"
    sibling = origin_sessions / "calm-heron-parent-p2-l1"
    for directory, body in (
        (parent, "parent instruction"),
        (child, "child note"),
        (sibling, "sibling note"),
    ):
        directory.mkdir(parents=True)
        entry = {"type": "model.result", "text": body}
        (directory / "logs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    reader = dispatch.ToolDispatcher(
        root=clone,
        state_dir=paths.state_dir(clone),
        session_dir=child,
        config=config_model.Config(),
    )
    page = reader.dispatch("read_session", {"id": child.name, "query": "child note"}).to_wire()
    assert "child note" in page["conversation"]
    assert (
        "sibling note"
        in reader.dispatch("read_session", {"id": sibling.name}).to_wire()["conversation"]
    )
    assert (
        "parent instruction"
        in reader.dispatch("read_session", {"id": parent.name}).to_wire()["conversation"]
    )
    for other in (child.name[:-1], sibling.name[:-1], "../" + parent.name):
        with pytest.raises(tool_errors.ToolError, match="no session"):
            reader.dispatch("read_session", {"id": other})
    assert not reader.dispatch("read_session", {}).to_wire()["sessions"]


def test_a_continued_child_looks_up_sessions_in_its_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume appends to the bound journal without granting its containing roster."""
    repo, clone = tmp_path / "repo", tmp_path / "clone"
    child_env.init_repo(repo)
    child_env.create_record(repo, "parent")
    probes = [
        ("read_session", args)
        for args in (
            {"id": "parent-p1-l1"},
            dict[str, str](),
            {"id": "parent-p1-l"},
            {"id": "parent"},
            {"id": "local"},
        )
    ]
    replies = iter((launch_env.FINISH, probes))
    with launch_env.scripted_provider(reply_for=lambda _: next(replies, launch_env.FINISH)) as (
        url,
        calls,
    ):
        path, launch = child_env.launch_worker(repo, clone, child_env.config(url), monkeypatch)
        local = child_env.create_record(clone, "local-run")
        (local / "logs.jsonl").write_text('{"type":"model.result","text":"clone journal"}\n')
        for steer in ("", "check session scope"):
            child_launch.write(
                path.parent,
                launch.model_copy(
                    update={
                        "continuation": steer,
                        "sibling_notice": "[harness siblings]\n[parent-p1-l2] ended: check parsing",
                    }
                ),
            )
            proc = subprocess.run(
                child_launch.bootstrap_argv(),
                cwd=clone,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr

    assert any(
        "[harness siblings]\n[parent-p1-l2] ended: check parsing" in message["content"]
        for message in calls[0]["messages"]
        if message["role"] == "user"
    )
    log = paths.state_dir(repo) / "sessions" / "runs" / launch.session_id / "logs.jsonl"
    assert [json.loads(row)["type"] for row in log.read_text().splitlines()].count(
        "session.resume"
    ) == 1
    own, roster, partial, parent, local_page = [
        json.loads(m["content"]) for m in calls[-1]["messages"] if m["role"] == "tool"
    ][-len(probes) :]
    assert "worker" in own["conversation"]
    assert roster["sessions"] == ["[local-run] run: t"]
    assert "no session" in partial["error"] and "conversation" in parent
    assert "clone journal" in local_page["conversation"]
