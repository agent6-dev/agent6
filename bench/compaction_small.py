"""Record the common opening of the six-module task using only loopback replies."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from bench import compaction_messages, compaction_slot, compaction_worker
from bench.tools import grading, journals

from agent6 import paths
from agent6.config import io
from agent6.config import model as config_model
from agent6.sessions import ipc
from agent6.vcs.git import ops, read

TASK = (
    "Extend the six modules in spec.md in the given order. Add one unittest per rule, "
    "including the error case. Run the tests after each module before starting the next."
)
LABEL = "field-amber-741"
STEER = f"In your finish summary include this exact label on a line by itself: {LABEL}"
GATE = ("python3", "-m", "unittest", "discover")
MODULES = {
    "pairs": "def first(pair): return pair[0]",
    "limits": "def positive(n): return n > 0",
    "labels": "def trimmed(s): return s.strip()",
    "batches": "def count(items): return len(items)",
    "clockface": "def minutes(hours): return 60 * hours",
    "rangeset": "def ordered(lo, hi): return lo <= hi",
}
FIXTURE = pathlib.Path(__file__).with_name("six_modules")
CLI = pathlib.Path(sys.executable).with_name("agent6")


def policy() -> config_model.Config:
    """Require explicit command approval and no external access for the six-module stage."""
    path = paths.global_config_path()
    raw = io.read_toml_file(path) if path.is_file() else {}
    # The slot has fresh project state and no repo layer. Only these global tables apply.
    for key in raw:
        if key not in ("providers", "budget", "agent6", "sandbox"):
            raise ValueError(f"unsupported small benchmark profile setting: {key}")
    for table, allowed in [
        ("agent6", {"config_version"}),
        ("sandbox", {"run_commands", "isolation"}),
    ]:
        for key in raw.get(table, {}):
            if key not in allowed:
                raise ValueError(f"unsupported small benchmark profile setting: {table}.{key}")
    if raw.get("providers", {}) not in ({}, {"chatgpt": {"backend": "chatgpt"}}):
        raise ValueError("small profile permits only the standard chatgpt provider")
    cfg = config_model.Config.model_validate(raw)
    if cfg.sandbox.run_commands == "deny" or cfg.sandbox.isolation not in ("auto", "full"):
        raise ValueError(
            "the small task requires sandbox.run_commands = ask or allow and auto or full isolation"
        )
    return cfg


def prepare(dest: pathlib.Path) -> compaction_slot.Opening:
    """Run three reads, deliver a steer and freeze the next worker checkpoint."""
    policy()
    dest.mkdir(mode=0o700)
    slot = dest / "active"
    repo = slot / "repo"
    repo.mkdir(parents=True)
    env = compaction_slot.environment(slot)
    for module, source in MODULES.items():
        (repo / f"{module}.py").write_text(source + "\n")
    (repo / "spec.md").write_bytes((FIXTURE / "spec.md").read_bytes())
    (repo / ".gitignore").write_text("__pycache__/\n")
    files_to_commit = tuple(p.name for p in repo.iterdir())
    ops.init_repo(repo)
    ops.commit_paths(
        repo, "seed", files_to_commit, identity=read.CommitIdentity("bench", "bench@localhost")
    )
    session = compaction_slot.session_in(slot)
    files = iter(("spec.md", "pairs.py", "limits.py"))

    def reply(_: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        file = next(files)
        if file == "spec.md" and not ipc.submit_steer(session, STEER):
            raise RuntimeError("could not queue the small task's steer")
        return "read_file", {"path": file}

    with compaction_messages.scripted(reply) as url:
        config = dest / "opening.toml"
        config.write_text(
            f'[providers.opening]\nbackend = "anthropic"\nbase_url = "{url}"\nauth_style = "none"\nserver_compaction = false\n'
            + "".join(
                f'[models.{role}]\nmodel = "opening/claude-sonnet-4-5"\n'
                for role in ("default", "worker", "helper")
            )
            + '[review]\nwhen = "never"\n[harness]\nmax_turns = 3\nparallel.tool = false\n'
            + f"verify.command = {json.dumps(GATE)}\nverify.at_start = false\n"
        )
        with (dest / "opening.log").open("xb") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    CLI,
                    "--config",
                    config,
                    "run",
                    "--session-id",
                    compaction_worker.SID,
                    TASK,
                ],
                cwd=repo,
                env=env | {"AGENT6_DETACHED_AWAY": "deny"},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                timeout=120,
                check=False,
            )
    return compaction_slot.seal(dest, turns=3, recall=LABEL)


def grade(
    repo: pathlib.Path, policy: config_model.Config, trial: pathlib.Path, *, scripted: bool
) -> int:
    """Run the external function and error checks under the trial's sandbox policy."""
    if policy.sandbox.run_commands == "ask" and (
        not sys.stdin.isatty()
        or input(f"Run the six-module Python scorer on {repo} in its jail? [y/N] ").lower() != "y"
    ):
        raise ValueError("the operator did not approve the six-module scorer")
    report = grading.bounded_report(
        compaction_worker.command("grade", str(repo.resolve()), "scripted" if scripted else "live"),
        FIXTURE.parent,
        compaction_slot.environment(repo.parent),
        policy.model_dump_json(by_alias=True).encode(),
        trial / "grader.stderr",
    )
    try:
        values = json.loads(report, object_pairs_hook=grading.unique_fields)
    except (ValueError, RecursionError) as exc:
        raise journals.MeasurementError("expected one JSON grader report") from exc
    if (
        not isinstance(values, dict)
        or values.keys() != MODULES.keys()
        or not all(type(v) is bool for v in values.values())
    ):
        raise journals.MeasurementError("expected six boolean group verdicts")
    return sum(values.values())


def pilot_reply() -> Callable[[dict[str, Any]], str | tuple[str, dict[str, Any]]]:
    """Return scripted edits for a one-group pilot with a green worker gate."""
    calls = 0

    def reply(body: dict[str, Any]) -> str | tuple[str, dict[str, Any]]:
        nonlocal calls
        calls += 1
        if "Write the summary now." in str(body["messages"][-1]):
            return "A handoff."
        if calls == 2:
            return "apply_edit", dict(
                path="pairs.py",
                kind="overwrite",
                old_string="",
                new_string=(
                    "def first(pair): return pair[0]\n"
                    "def swap(p):\n a,b=p\n return b,a\n"
                    "def span(p):\n a,b=p\n return abs(a-b)\n"
                ),
            )
        if calls == 3:
            return "apply_edit", dict(
                path="test_pairs.py",
                kind="create",
                old_string="",
                new_string=(
                    "import unittest\nimport pairs\nclass PairTest(unittest.TestCase):\n"
                    " def test_swap(self): self.assertEqual(pairs.swap((2, 3)), (3, 2))\n"
                ),
            )
        return "finish_session", {"summary": "Done.\nfield-amber-741"}

    return reply
