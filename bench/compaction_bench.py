"""Compare keep settings on the six-module task, or print trial spreads."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

from bench import compaction_arms, compaction_messages, compaction_slot, compaction_small
from bench.tools import journals

from agent6 import errors

SOL = "chatgpt/gpt-6.2-sol"
ASTRA = "chatgpt/gpt-6.2-astra"
ARMS = {"keep": compaction_arms.KEEP, "cold": compaction_arms.COLD}
type Row = dict[str, Any]
OVERLAP = ("turns", "rereads_6_calls", "second_restart", "uncached_input_tokens", "exact_recall")


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    opening: compaction_slot.Opening
    attempt: Callable[[Row], None]


def small_case(root: pathlib.Path, *, scripted: bool = False) -> Case:
    opening = compaction_small.prepare(root)

    def attempt(row: Row) -> None:
        trial = pathlib.Path(row["trial"])
        policy = compaction_small.policy()
        with compaction_slot.restored(opening, trial) as session:
            if scripted:
                with compaction_messages.scripted(compaction_small.pilot_reply()) as url:
                    compaction_slot.resume(
                        opening, session, row, gate=compaction_small.GATE, scripted_url=url
                    )
            else:
                compaction_slot.resume(opening, session, row, gate=compaction_small.GATE)
            gate = row["passed"]
            row["passed"] = None
            row["groups"] = compaction_small.grade(opening.slot / "repo", policy, trial)
            row["passed"] = gate if row["groups"] == 6 else False

    return Case(opening, attempt)


def overlap(a: Sequence[Row], b: Sequence[Row]) -> bool:
    """Stop only on complete small passes with intersecting ranges for every measure."""
    if (
        len(a) != 5
        or len(b) != 5
        or not all(r.get("passed") is True and not r.get("error") for r in (*a, *b))
    ):
        return False
    for key in OVERLAP:
        if any(r.get(key) is None for r in (*a, *b)):
            return False
        x, y = [r[key] for r in a], [r[key] for r in b]
        if max(min(x), min(y)) > min(max(x), max(y)):
            return False
    return True


def run_set(
    family: str,
    cases: dict[str, Case],
    out: pathlib.Path,
    *,
    finalists: Sequence[str] = (),
    stop_on_overlap: bool = False,
    pilot: str | None = None,
    pilot_model: str | None = None,
) -> int:
    """Write one flushed row per attempt. Never reuse an output directory."""
    arms = list(ARMS[family])
    model, reps, stage = SOL, 5, "screen"
    if finalists:
        if len(finalists) != 2 or len(set(finalists)) != 2:
            raise ValueError("choose two distinct finalists")
        arms, model, reps, stage = list(finalists), ASTRA, 3, "final"
    if pilot:
        arms, model, reps, stage = [pilot], pilot_model or SOL, 1, "pilot"
    elif pilot_model:
        raise ValueError("--pilot-model requires --pilot")
    if stop_on_overlap and (
        family != "keep" or set(cases) != {"small", "replay"} or stage != "screen"
    ):
        raise ValueError("--stop-on-overlap requires the Sol keep set with both tasks")
    out = out.resolve()
    tasks = list(cases)
    first_two: dict[str, list[Row]] = collections.defaultdict(list)
    failures = 0
    with (out / "runs.jsonl").open("x") as stream:
        for block in (arms[:2], arms[2:]):
            for rep in range(1, reps + 1):
                for arm in block:
                    for task in tasks:
                        seed = cases[task].opening
                        row: Row = dict(
                            task=task,
                            family=family,
                            stage=stage,
                            arm=arm,
                            replicate=rep,
                            model=model,
                            source=seed.root.name,
                            turn=seed.turn,
                            head=seed.head,
                            digest=seed.digest,
                            trial=str(out / f"{arm}-{task}-{rep}"),
                        )
                        try:
                            cases[task].attempt(row)
                        except (
                            OSError,
                            ValueError,
                            KeyError,
                            subprocess.SubprocessError,
                            journals.MeasurementError,
                            errors.OperatorError,
                            errors.GitError,
                        ) as exc:
                            row["error"] = str(exc)
                            failures += 1
                        except KeyboardInterrupt:
                            row["error"] = "interrupted"
                            raise
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            raise
                        finally:
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                        if task == "small":
                            first_two[arm].append(row)
            if (
                block == arms[:2]
                and stop_on_overlap
                and overlap(first_two[arms[0]], first_two[arms[1]])
            ):
                tasks = ["replay"]
                print("Small ranges overlap. Remaining settings use replay alone.", file=sys.stderr)
    return int(bool(failures))


def table(files: Sequence[pathlib.Path]) -> None:
    """Print independent spreads and the observed denominator of each outcome."""
    groups: dict[tuple[str, ...], list[Row]] = collections.defaultdict(list)
    for path in files:
        with path.open() as stream:
            for line in stream:
                r = json.loads(line)
                groups[
                    tuple(
                        str(r[k])
                        for k in (
                            "family",
                            "stage",
                            "model",
                            "task",
                            "source",
                            "turn",
                            "head",
                            "digest",
                            "arm",
                        )
                    )
                ].append(r)
    print(
        "| set / model / task / checkpoint / arm | n | errors | pass | recall | second restart | groups | turns | rereads (6 calls) | rereads (10 turns) | first input | first cache read | first worker input | first worker cache read | uncached input | total tokens |"
    )
    print("|" + " --- |" * 16)
    for key, rows in groups.items():
        family, stage, model, task, source, turn, head, digest, arm = key
        values = [
            f"{family}:{stage} / {model} / {task} / {source}:{turn}:{head[:8]}:{digest[:8]} / {arm}",
            str(len(rows)),
            str(sum("error" in r for r in rows)),
        ]
        for metric in (
            "passed",
            "exact_recall",
            "second_restart",
            "groups",
            "turns",
            "rereads_6_calls",
            "rereads_10_turns",
            "first_request_input_tokens",
            "first_request_cache_read_tokens",
            "first_worker_input_tokens",
            "first_worker_cache_read_tokens",
            "uncached_input_tokens",
            "total_tokens",
        ):
            known = [r[metric] for r in rows if r.get(metric) is not None]
            values.append(
                "unavailable"
                if not known
                else f"{sum(known)}/{len(known)}"
                if isinstance(known[0], bool)
                else f"{min(known)}..{max(known)} ({len(known)})"
            )
        print("| " + " | ".join(values) + " |")


def set_options(parser: argparse.ArgumentParser, family: str) -> None:
    parser.add_argument(
        "--out", required=True, type=pathlib.Path, help="new private run-set directory"
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--finalists",
        nargs=2,
        choices=ARMS[family],
        default=[],
        help="three Astra replicates per arm",
    )
    phase.add_argument(
        "--pilot", choices=ARMS[family], help="one replicate, outside the measured set"
    )
    parser.add_argument("--pilot-model", choices=(SOL, ASTRA), help="model for a live pilot only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    small = subs.add_parser("small", help="generate and grade the six-module task")
    set_options(small, "keep")
    small.add_argument("--scripted", action="store_true", help="use loopback replies for a pilot")
    subs.add_parser("table").add_argument("files", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "table":
            table(args.files)
        else:
            if args.scripted and (not args.pilot or args.pilot_model):
                raise ValueError("--scripted needs --pilot and no --pilot-model")
            out = args.out.resolve()
            out.mkdir(parents=True, mode=0o700)
            return run_set(
                "keep",
                {"small": small_case(out / "six-modules", scripted=args.scripted)},
                out,
                finalists=args.finalists,
                pilot=args.pilot,
                pilot_model="scripted/claude-sonnet-4-5" if args.scripted else args.pilot_model,
            )
    except (ValueError, OSError, errors.OperatorError, errors.GitError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
