from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from tests import launch_env


@pytest.mark.parametrize("mode", ["ask", "run"])
@pytest.mark.parametrize("reported", [False, True])
def test_cli_compaction_cost_matches_the_receipt(
    tmp_path: pathlib.Path, mode: str, reported: bool
) -> None:
    """Real CLI calls compact once and print the journal-derived cost on both surfaces."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(os.environ)
    for key in ("CONFIG", "STATE", "DATA", "CACHE"):
        env[f"XDG_{key}_HOME"] = str(tmp_path / key.lower())
    for key in ("AGENT6_LAUNCH", "AGENT6_CHILD"):
        env.pop(key, None)
    env["AGENT6_ALLOW_ROOT"] = "1"
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.invalid"
    binary = pathlib.Path(sys.executable).with_name("agent6")

    def run(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, cwd=repo, env=env, text=True, capture_output=True, check=False, timeout=90
        )

    assert run("git", "init", "-q").returncode == 0
    replies: list[tuple[str, dict[str, Any]] | str] = []
    for ch in "abcdefgh":
        (repo / f"{ch}.txt").write_text(ch * 1000)
        replies.append(("read_file", {"path": f"{ch}.txt"}))
    assert run("git", "add", "--", *(f"{ch}.txt" for ch in "abcdefgh")).returncode == 0
    assert run("git", "commit", "-qm", "seed").returncode == 0
    replies.append("Read all files." if mode == "ask" else launch_env.FINISH)
    usage: dict[str, Any] = {"prompt_tokens": 1000, "completion_tokens": 10}
    if reported:
        usage["prompt_tokens_details"] = {"cached_tokens": 700}
    with launch_env.scripted_provider(replies, usage=usage) as (url, calls):
        cfg = tmp_path / "config" / "agent6" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            f'[providers.test]\nbackend = "openai"\nbase_url = "{url}"\nauth_style = "none"\n'
            '[models.default]\nmodel = "test/smoke"\n'
            '[sandbox]\nisolation = "none"\nprotect_git = "off"\nrun_commands = "deny"\n'
            '[git]\ncontrol = "model"\n[review]\nwhen = "never"\n'
            '[harness.verify]\nwhen = "never"\nat_start = false\n'
            '[harness.prompt]\ndecompose = "off"\n'
            "[compaction]\ndrop_at_chars = 6000\nsummarise_at_chars = 1000000\n"
            "elision_gists = false\n"
        )
        proc = run(str(binary), mode, "Read a.txt through h.txt, then finish.")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(calls) == 9
    (log,) = (tmp_path / "state").rglob("logs.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(row["type"] == "compact.dropped" for row in rows) == 1
    shown = run(str(binary), "sessions", "receipt", log.parent.name)
    assert shown.returncode == 0, shown.stderr
    end = proc.stdout + proc.stderr
    if reported:
        for line in (
            "Worker input cache (session): 70.0% (6,300/9,000 tokens)",
            "After compaction (session): 1 worker call, 300 uncached input tokens",
        ):
            assert line in end
            assert line in shown.stdout
    else:
        assert "Worker input cache" not in end + shown.stdout
        assert "After compaction" not in end + shown.stdout
