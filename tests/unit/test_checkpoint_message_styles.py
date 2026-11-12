# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[git.commit].checkpoint_message` styles at the loop's auto-commit."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6.config import Config
from agent6.harness import loop, loop_state, run_chain
from agent6.tools import dispatch
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def _wf(
    tmp_path: pathlib.Path, style: str, provider: Any = None, logger: Any = print
) -> loop.Harness:
    cfg = Config.model_validate({"git": {"commit": {"checkpoint_message": style}}})
    git_ops.init_repo(tmp_path)
    return loop.Harness(
        chain=run_chain.RunChain(
            tmp_path,
            ref="refs/agent6/test/head",
            identity=git_read.CommitIdentity(name="Test", email="test@example.com"),
        ),
        config=cfg,
        provider=provider or mock.MagicMock(),
        dispatcher=dispatch.ToolDispatcher(root=tmp_path, config=cfg),
        logger=logger,
    )


def _turn(text: str) -> loop_state.TurnState:
    return loop_state.TurnState(
        number=3, resp=mock.MagicMock(text=text), assistant=mock.MagicMock()
    )


def test_agent6_style_is_the_default_and_unchanged(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path, "agent6")
    got = wf.checkpoints.subject(
        _turn("Add the unified write path.\nmore prose"), fallback="verify passed"
    )
    assert got == "agent6 turn 3: Add the unified write path."


def test_conventional_style_derives_from_the_chain(tmp_path: pathlib.Path) -> None:
    """The conventional subject's type and scope come from the chain's own delta."""
    wf = _wf(tmp_path, "conventional")
    target = tmp_path / "src/agent6/config/write.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n")
    got = wf.checkpoints.subject(_turn("Add the unified write path."), fallback="verify passed")
    assert got == "feat(config): add the unified write path"


def test_model_style_uses_the_provider_text(tmp_path: pathlib.Path) -> None:
    provider = mock.MagicMock()
    provider.call.return_value = mock.MagicMock(text=" fix: tighten the resolver \n")
    wf = _wf(tmp_path, "model", provider=provider)
    got = wf.checkpoints.subject(_turn("prose"), fallback="verify passed")
    assert got == "fix: tighten the resolver"


def test_model_style_degrades_to_agent6_with_a_warning(tmp_path: pathlib.Path) -> None:
    provider = mock.MagicMock()
    provider.call.side_effect = RuntimeError("no endpoint")
    logged: list[str] = []
    wf = _wf(tmp_path, "model", provider=provider, logger=logged.append)
    got = wf.checkpoints.subject(_turn("Fix the thing."), fallback="verify passed")
    assert got == "agent6 turn 3: Fix the thing."
    assert any("model commit message failed (RuntimeError: no endpoint)" in m for m in logged)


@pytest.mark.parametrize("style", ["agent6", "conventional"])
def test_each_checkpoint_subject_names_only_its_delta(tmp_path: pathlib.Path, style: str) -> None:
    """The second checkpoint names its one new file, not the run's cumulative changes."""
    wf = _wf(tmp_path, style)
    (tmp_path / "first.py").write_text("first = 1\n")
    first_subject = wf.checkpoints.subject(_turn(""), fallback="checkpoint")
    first = wf.checkpoints.commit(first_subject, turn=3)
    assert first
    (tmp_path / "second.py").write_text("second = 2\n")
    second_subject = wf.checkpoints.subject(
        _turn("<thinking>done</thinking>"), fallback="checkpoint"
    )
    second = wf.checkpoints.commit(second_subject, turn=4)
    assert second
    row = git_read.list_run_commits(tmp_path, first, second)[0]
    assert "second.py" in row.subject
    assert "first.py" not in row.subject and "2 files" not in row.subject


@pytest.mark.parametrize("style", ["agent6", "conventional"])
def test_empty_prose_names_the_number_of_changed_files(tmp_path: pathlib.Path, style: str) -> None:
    """A tool-only checkpoint still says how much it changes."""
    wf = _wf(tmp_path, style)
    (tmp_path / "one.py").write_text("one = 1\n")
    (tmp_path / "two.py").write_text("two = 2\n")
    assert "2 files" in wf.checkpoints.subject(_turn(""), fallback="checkpoint")


def test_a_pure_rename_is_a_refactor(tmp_path: pathlib.Path) -> None:
    """An identical blob at a new path is a rename, not a feature or fix."""
    wf = _wf(tmp_path, "conventional")
    (tmp_path / "old.py").write_text("value = 1\n")
    assert wf.chain.commit("base")
    (tmp_path / "old.py").rename(tmp_path / "new.py")
    assert wf.chain.name_status() == (("R100", "new.py"),)
    assert wf.checkpoints.subject(_turn("Rename the module."), fallback="checkpoint") == (
        "refactor: rename the module"
    )


@pytest.mark.parametrize("name", ["with\ttab.py", "with\nnewline.py"])
def test_checkpoint_paths_keep_filename_separators(tmp_path: pathlib.Path, name: str) -> None:
    """Tabs and newlines inside a filename do not split or quote the chain's path."""
    wf = _wf(tmp_path, "conventional")
    (tmp_path / name).write_text("value = 1\n")
    assert wf.chain.name_status() == (("A", name),)
