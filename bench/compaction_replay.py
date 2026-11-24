from __future__ import annotations

# Both arms use bare placeholders and recorded restart boundaries.
# Dedup, gist text, edited-path priority and thinking sizes are outside this size-only replay.
import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
from typing import Any

from agent6.harness.compaction import elision
from agent6.sessions import objects


class ReplayError(ValueError):
    """The recording lacks an input the replay needs."""


def journal_prefix(journal: pathlib.Path, turn: int) -> bytes:
    """Cut the journal before the selected worker turn, not after its reply."""
    with journal.open("rb") as stream:
        cut = None
        for line in stream:
            if b'"budget.turn"' not in line:
                continue
            row = json.loads(line)
            if row.get("type") == "budget.turn" and row.get("turn") == turn:
                if cut is not None:
                    raise ReplayError(f"{journal}: ambiguous journal boundary for turn {turn}")
                cut = stream.tell() - len(line)
        if cut is None:
            raise ReplayError(f"{journal}: no journal boundary for turn {turn}")
        stream.seek(0)
        return stream.read(cut)


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """One tool result's original size and bare replacement size."""

    call_id: str
    chars: int
    marker_chars: int
    operator_answer: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Measurement:
    """Request-weighted retained characters and requests with at least one drop."""

    drops: int
    mean_chars: float


def read_frames(session: pathlib.Path) -> list[tuple[Result, ...]]:
    """Read worker request frames, retaining each result's first full size.

    Args:
        session: A directory containing transcripts and their objects.

    Returns:
        The tool results present at each worker request, oldest first.

    Raises:
        ReplayError: A result first appears elided or the wire is unsupported.
    """
    originals: dict[str, Result] = {}
    frames: list[tuple[Result, ...]] = []
    for path in sorted((session / "transcripts").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("actor") != "worker":
            continue
        # Session objects live beside transcripts, not inside them.
        body = objects.materialize(record["request"]["body"], session / "objects", record_path=path)
        if not isinstance(body.get("input"), list):
            raise ReplayError(f"{path}: expected a Responses input list")
        calls: dict[str, dict[str, Any]] = {}
        frame: list[Result] = []
        for item in body["input"]:
            if item.get("type") == "function_call":
                calls[item["call_id"]] = item
            elif item.get("type") == "function_call_output":
                call_id = item["call_id"]
                if call_id not in originals:
                    output = item["output"]
                    if not isinstance(output, str) or output.startswith(elision.ELISION_PREFIX):
                        raise ReplayError(f"{path}: {call_id} has no original text result")
                    call = calls.get(call_id)
                    if call is None:
                        raise ReplayError(f"{path}: {call_id} has no function call")
                    marker = elision.elision_placeholder(
                        call["name"], json.loads(call["arguments"])
                    )
                    originals[call_id] = Result(
                        call_id, len(output), len(marker), call["name"] == "ask_user"
                    )
                frame.append(originals[call_id])
        frames.append(tuple(frame))
    if not frames:
        raise ReplayError(f"{session}: no worker requests")
    return frames


def replay(frames: list[tuple[Result, ...]], *, bound: int, stepped: bool) -> Measurement:
    """Replay the size-only rule with the recorded worker request cadence.

    New results and the newest two results stay whole in both arms.
    A result absent from a recorded request leaves both arms at that boundary.

    Args:
        frames: Original result sizes at each request.
        bound: The tier-1 trigger in characters.
        stepped: Aim for half the bound rather than the bound.

    Returns:
        The number of drop requests and mean retained characters after compaction.
    """
    held: dict[str, int] = {}
    totals: list[int] = []
    drops = 0
    for frame in frames:
        fresh = {r.call_id for r in frame} - held.keys()
        held = {r.call_id: held.get(r.call_id, r.chars) for r in frame}
        total = sum(held.values())
        changed = False
        if total > bound:
            target = bound // 2 if stepped else bound
            for result in frame[:-2]:
                if total <= target:
                    break
                if result.call_id in fresh or result.operator_answer:
                    continue
                size = held[result.call_id]
                if size > result.marker_chars:
                    held[result.call_id] = result.marker_chars
                    total -= size - result.marker_chars
                    changed = True
        drops += changed
        totals.append(total)
    return Measurement(drops, statistics.mean(totals))


def main() -> int:
    """Print a comparison table and fail when any session lacks complete sizes.

    Returns:
        Zero for a complete measurement, one for unavailable input.
    """
    parser = argparse.ArgumentParser(
        description="Replay Responses tool-result sizes without a model."
    )
    parser.add_argument("recordings", type=pathlib.Path)
    parser.add_argument("--drop-at-chars", type=int, required=True)
    args = parser.parse_args()
    if args.drop_at_chars <= 0:
        parser.error("--drop-at-chars must be positive")
    print(
        "| session | worker requests | old drops | stepped drops | old mean chars | stepped mean chars |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    failed = False
    sessions = sorted(p for p in args.recordings.iterdir() if (p / "transcripts").is_dir())
    if not sessions:
        parser.error("no session directories containing transcripts")
    for session in sessions:
        try:
            frames = read_frames(session)
        except (ReplayError, objects.ObjectStoreError, OSError, ValueError, KeyError) as exc:
            print(f"{session.name}: {exc}", file=sys.stderr)
            print(
                f"| {session.name} | unavailable | unavailable | unavailable | unavailable | unavailable |"
            )
            failed = True
            continue
        old = replay(frames, bound=args.drop_at_chars, stepped=False)
        new = replay(frames, bound=args.drop_at_chars, stepped=True)
        print(
            f"| {session.name} | {len(frames)} | {old.drops} | {new.drops} | {old.mean_chars:.1f} | {new.mean_chars:.1f} |"
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
