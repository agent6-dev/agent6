"""Print replicate spread for a handoff run set."""

from __future__ import annotations

import argparse
import json
import pathlib

from bench.tools import table as tool_table


def print_set(path: pathlib.Path) -> None:
    """Show acceptance over known outcomes and median [min,max] per run."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"no runs in {path}")
    first = rows[0]
    print(f"{first['split']} | {first['model']} | effort {first['effort']} | {first['revision']}")
    print(
        "arm | n | all accepted/known | with issues | stuck | turns | fresh input | cache read | cache write | output | started twice | never started | under two (s) | off-arm calls"
    )
    print(" | ".join(["---"] * 14))
    for arm in ("text", "id"):
        samples = [r for r in rows if r["arm"] == arm]
        accepted = [r["all_accepted"] for r in samples if r["all_accepted"] is not None]
        print(
            arm,
            len(samples),
            f"{sum(accepted)}/{len(accepted)}" if accepted else "?",
            sum(bool(r["issues"]) for r in samples),
            *(
                tool_table.spread(samples, key)
                for key in (
                    "stuck_notices",
                    "turns",
                    "input_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                    "started_twice",
                    "never_started",
                    "under_two_s",
                    "off_arm_calls",
                )
            ),
            sep=" | ",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path, metavar="RESULTS.jsonl")
    print_set(parser.parse_args().results)
