# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[git.commit.checkpoint].message` styles at the loop's auto-commit."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6.config import Config
from agent6.harness import _chain as chain_mod
from agent6.harness import _loop_state, loop
from agent6.tools import dispatch


def _wf(
    tmp_path: pathlib.Path, style: str, provider: Any = None, logger: Any = print
) -> loop.Harness:
    cfg = Config.model_validate({"git": {"commit": {"checkpoint": {"message": style}}}})
    return loop.Harness(
        chain=chain_mod.RunChain(tmp_path),
        config=cfg,
        provider=provider or mock.MagicMock(),
        dispatcher=dispatch.ToolDispatcher(root=tmp_path, config=cfg),
        logger=logger,
    )


def _turn(text: str) -> _loop_state.TurnState:
    return _loop_state.TurnState(
        iteration=3, resp=mock.MagicMock(text=text), assistant=mock.MagicMock()
    )


def test_agent6_style_is_the_default_and_unchanged(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path, "agent6")
    got = wf.checkpoints.subject(
        _turn("Add the unified write path.\nmore prose"), fallback="verify passed"
    )
    assert got == "agent6 iter 3: Add the unified write path."


def test_conventional_style_derives_from_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_added(_p: pathlib.Path, *, exclude: object = ()) -> tuple[tuple[str, str], ...]:
        return (("A", "src/agent6/config/write.py"),)

    monkeypatch.setattr(chain_mod, "worktree_name_status", _one_added)
    wf = _wf(tmp_path, "conventional")
    got = wf.checkpoints.subject(_turn("Add the unified write path."), fallback="verify passed")
    assert got == "feat(config): add the unified write path"


def test_model_style_uses_the_provider_text(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_modified(_p: pathlib.Path, *, exclude: object = ()) -> tuple[tuple[str, str], ...]:
        return (("M", "a.py"),)

    monkeypatch.setattr(chain_mod, "worktree_name_status", _one_modified)
    provider = mock.MagicMock()
    provider.call.return_value = mock.MagicMock(text=" fix: tighten the resolver \n")
    wf = _wf(tmp_path, "model", provider=provider)
    got = wf.checkpoints.subject(_turn("prose"), fallback="verify passed")
    assert got == "fix: tighten the resolver"


def test_model_style_degrades_to_agent6_with_a_warning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one_modified(_p: pathlib.Path, *, exclude: object = ()) -> tuple[tuple[str, str], ...]:
        return (("M", "a.py"),)

    monkeypatch.setattr(chain_mod, "worktree_name_status", _one_modified)
    provider = mock.MagicMock()
    provider.call.side_effect = RuntimeError("no endpoint")
    logged: list[str] = []
    wf = _wf(tmp_path, "model", provider=provider, logger=logged.append)
    got = wf.checkpoints.subject(_turn("Fix the thing."), fallback="verify passed")
    assert got == "agent6 iter 3: Fix the thing."
    assert any("model commit message failed" in m for m in logged)
