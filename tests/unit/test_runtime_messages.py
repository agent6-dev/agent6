from __future__ import annotations

import io
import pathlib
import subprocess
import sys

import pytest

from agent6.ui.cli import connect
from agent6.ui.web import git_model


def test_worktree_commit_refusal_names_file_and_tree_reads(tmp_path: pathlib.Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(git_model.GitRequestError) as caught:
        git_model.commit_payload(tmp_path, "WORKTREE")
    assert "a file or tree read, or a diff or compare head, never a commit or a base" in str(
        caught.value
    )


def test_api_key_prompt_says_masked_without_echoing_the_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("sample-secret\n"))
    assert connect._prompt_api_key("local") == "sample-secret"  # pyright: ignore[reportPrivateUsage]
    output = capsys.readouterr()
    assert "input masked, blank for none" in output.err
    assert "Captured key: 13 chars." in output.out
    assert "sample-secret" not in output.out + output.err
