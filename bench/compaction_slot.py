"""Run each treatment at the same pathname as its stopped benchmark opening."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess
import time
from collections.abc import Generator
from typing import Any

from bench import compaction_arms as arms
from bench import compaction_counts, compaction_worker

from agent6 import kinds, paths
from agent6.app import conversation_state
from agent6.harness import loop_snapshot
from agent6.harness.compaction import cold_start
from agent6.sessions import layout

COLD_SECONDS = 3600
CONTINUE = "Continue the original task. In your finish summary put the label requested earlier on a line by itself."


def environment(slot: pathlib.Path) -> dict[str, str]:
    # No child launch, sandbox switches, model route, proxy, Git or Python overrides.
    return {
        "HOME": str(paths.effective_user().home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "XDG_CONFIG_HOME": str(paths.global_config_dir().parent.resolve()),
        **{f"XDG_{key}_HOME": str(slot / key.lower()) for key in ("STATE", "DATA", "CACHE")},
        "GIT_AUTHOR_NAME": "bench",
        "GIT_AUTHOR_EMAIL": "bench@localhost",
        "GIT_COMMITTER_NAME": "bench",
        "GIT_COMMITTER_EMAIL": "bench@localhost",
    }


def session_in(slot: pathlib.Path) -> pathlib.Path:
    return layout.SessionLayout(
        slot / "state/agent6" / paths.repo_id(slot / "repo"), compaction_worker.SID
    ).session_dir


@dataclasses.dataclass(frozen=True, slots=True)
class Opening:
    root: pathlib.Path
    head: str
    turn: int
    digest: str
    recall: str | None

    @property
    def slot(self) -> pathlib.Path:
        return self.root / "active"

    @property
    def session(self) -> pathlib.Path:
        return self.root / "opening" / session_in(self.slot).relative_to(self.slot)


def seal(root: pathlib.Path, *, turns: int, recall: str | None) -> Opening:
    """Save the entire stopped prefix, including uncommitted files, at the turn cap."""
    slot = root / "active"
    session = session_in(slot)
    journal = (session / "logs.jsonl").read_bytes()
    rows = [json.loads(line) for line in journal.splitlines()]
    stops = [r for r in rows if r["type"] == "session.stop"]
    if len(stops) != 1 or (stops[0].get("reason"), stops[0].get("turns")) != ("max_turns", turns):
        raise ValueError("opening did not stop at the selected turn cap; see its worker log")
    if recall and not any(r["type"] == "steer.injected" and recall in r["text"] for r in rows):
        raise ValueError("the recall label was not delivered before the turn cap")
    path = session / "loop_state.json"
    snap = loop_snapshot.load_session_snapshot(path)
    if snap.next_turn != turns + 1 or not snap.head_sha:
        raise ValueError("opening lacks the committed head and next turn at the stop")
    saved = Opening(
        root, snap.head_sha, snap.next_turn, hashlib.sha256(path.read_bytes()).hexdigest(), recall
    )
    slot.rename(root / "opening")
    return saved


@contextlib.contextmanager
def restored(opening: Opening, trial: pathlib.Path) -> Generator[pathlib.Path]:
    """Use the fixed, vacant active directory, never a caller's existing checkout."""
    slot = opening.slot
    trial.mkdir(parents=True, mode=0o700)
    # mkdir claims the slot. A concurrent or interrupted attempt is never overwritten.
    slot.mkdir(mode=0o700)
    try:
        for name in ("repo", "state"):
            shutil.copytree(opening.root / "opening" / name, slot / name, symlinks=True)
        session = session_in(slot)
        if hashlib.sha256((session / "loop_state.json").read_bytes()).hexdigest() != opening.digest:
            raise ValueError("opening checkpoint has changed")
        (session / conversation_state.FILE_NAME).unlink(missing_ok=True)
        yield session
    finally:
        slot.rename(trial / "artifacts")


def overlay(
    arm: arms.Arm,
    model: str,
    *,
    cold: bool,
    small: bool,
    gate: tuple[str, ...],
    scripted_url: str = "",
) -> str:
    provider, _ = model.split("/", 1)
    return (
        f"[providers.{json.dumps(provider)}]\nidle_compact_s = {COLD_SECONDS if cold else -1}\n"
        + (
            f'backend = "anthropic"\nbase_url = "{scripted_url}"\nauth_style = "none"\nserver_compaction = false\n'
            if scripted_url
            else ""
        )
        + "".join(
            f'[models.{role}]\nmodel = {json.dumps(model)}\neffort = "medium"\n'
            for role in ("default", "worker", "helper")
        )
        + '[git.commit]\ncheckpoint_message = "agent6"\nsquash_message = "agent6"\n'
        '[review]\nwhen = "never"\n[harness]\nmax_turns = 100\nparallel.tool = false\n'
        'verify.infer = false\nverify.at_start = false\nverify.when = "finish"\n'
        f"verify.command = {json.dumps(gate)}\n"
        '[compaction]\nsummariser = "worker"\nelision_gists = false\n'
        + (
            "drop_at_chars = 8000\nsummarise_at_chars = 100000\n"
            if small
            else "drop_at_chars = 489600\nsummarise_at_chars = 1000000\n"
        )
        + f"keep_operator_messages = {arm.operator}\nkeep_recent_turns = {arm.turns}\n"
        f'keep_recent_chars = {arm.chars}\nkeep_thinking_turns = {arm.thinking}\ncold_start = "{arm.cold}"\n'
    )


def resume(
    opening: Opening,
    session: pathlib.Path,
    row: dict[str, Any],
    *,
    gate: tuple[str, ...],
    scripted_url: str = "",
) -> None:
    arm, model = row["arm"], row["model"]
    cold = arm in arms.COLD
    slot = opening.slot
    if cold and not cold_start.detect(
        session / "logs.jsonl",
        actor="worker",
        route=kinds.ModelRoute(*model.split("/", 1)),
        limit=COLD_SECONDS,
        now=time.time(),
    ):
        raise ValueError("checkpoint is not cold: wait for the idle limit or use a changed route")
    small = row["task"] == "small"
    config = slot / "arm.toml"
    config.write_text(
        overlay(
            (arms.COLD if cold else arms.KEEP)[arm],
            model,
            cold=cold,
            small=small,
            gate=gate,
            scripted_url=scripted_url,
        )
    )
    mode = "cold" if cold else "pointers" if arm == "pointers" else "keep"
    command = compaction_worker.command(
        "resume",
        mode,
        str(config),
        model,
        CONTINUE if opening.recall else "Continue the original task.",
        "scripted" if scripted_url else "live",
    )
    prefix = (session / "logs.jsonl").read_bytes()
    trial = pathlib.Path(row["trial"])
    env = environment(slot)
    if scripted_url:
        env["AGENT6_DETACHED_AWAY"] = "deny"
        with (trial / "agent.log").open("xb") as log:
            proc = subprocess.run(
                command,
                cwd=slot / "repo",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                check=False,
            )
    else:
        proc = subprocess.run(command, cwd=slot / "repo", env=env, check=False)
    logs = session / "logs.jsonl"
    row.update(
        returncode=proc.returncode, journal=str(trial / "artifacts" / logs.relative_to(slot))
    )
    with logs.open("rb") as stream:
        if stream.read(len(prefix)) != prefix:
            raise ValueError("resumed journal differs from the opening")
        row.update(
            compaction_counts.measure(
                [json.loads(line) for line in prefix.splitlines()],
                [json.loads(line) for line in stream],
                arm,
                opening.recall,
            )
        )
