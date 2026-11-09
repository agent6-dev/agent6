from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
from typing import Any

MEASURES = (
    "grounded_true_blocks",
    "false_blocks",
    "abstentions",
    "missing_risk",
    "risk_length_violations",
    "tokens_in",
    "tokens_out",
    "usd",
    "plan_points",
    "elapsed_s",
)


def adjudicate(rows: list[dict[str, Any]], judgments: dict[str, dict[str, bool]]) -> None:
    """Fill true-block counts only when every grounded block has an oracle judgment."""
    known_runs = {row["run_id"] for row in rows}
    if judgments.keys() - known_runs:
        raise ValueError("adjudications name unknown runs")
    for row in rows:
        if row["run_id"] not in judgments:
            continue
        decisions = judgments[row["run_id"]]
        ids = {f["finding_id"] for f in row.get("block_findings", [])}
        if decisions.keys() - ids or any(type(v) is not bool for v in decisions.values()):
            raise ValueError("adjudications must map recorded block finding IDs to booleans")
        if row["case"].startswith("clean-"):
            if any(decisions.values()):
                raise ValueError("a registered clean case cannot have a true planted block")
        elif ids == decisions.keys():
            row["grounded_true_blocks"] = sum(decisions.values())


def describe(values: list[float], total: int) -> str:
    if not values:
        return f"not recorded (0/{total})"
    return (
        f"{statistics.mean(values):.3f} [{min(values):.3f}..{max(values):.3f}] "
        f"({len(values)}/{total})"
    )


def summarize(rows: list[dict[str, Any]]) -> None:
    """Print split and case counts, repetition ranges and matched B-minus-A differences."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        route = tuple(row[k] for k in ("label", "model", "effort", "temperature"))
        key = (*route, row["case"], row["rep"], row["arm"])
        if key in seen:
            raise ValueError(f"duplicate cell: {key}")
        seen.add(key)
        groups[(*route, row["split"])].append(row)
    for group, cells in sorted(groups.items()):
        split = group[-1]
        cases = sorted({r["case"] for r in cells})
        reps = [r["rep"] for r in cells]
        print(
            f"{split}: cases={len(cases)} panels={len(cells)} reps={min(reps)}..{max(reps)} {group[:-1]}"
        )
        for case in ["ALL", *cases]:
            selected = [r for r in cells if case == "ALL" or r["case"] == case]
            print(f"  {case}: errors={sum('error' in r for r in selected)}")
            pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
            for row in selected:
                pairs[(row["case"], row["rep"])][row["arm"]] = row
            for metric in MEASURES:
                arms = []
                for arm in ("A", "B"):
                    arm_rows = [r for r in selected if r["arm"] == arm]
                    values = [
                        float(r[metric])
                        for r in arm_rows
                        if isinstance(r.get(metric), (int, float))
                    ]
                    arms.append(f"{arm} {describe(values, len(arm_rows))}")
                differences = [
                    float(p["B"][metric]) - float(p["A"][metric])
                    for p in pairs.values()
                    if all(
                        arm in p and isinstance(p[arm].get(metric), (int, float))
                        for arm in ("A", "B")
                    )
                ]
                print(f"    {metric}: {'; '.join(arms)}; B-A {describe(differences, len(pairs))}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="B14 means [repetition ranges] and paired B-A differences"
    )
    ap.add_argument("paths", type=pathlib.Path, nargs="+")
    ap.add_argument("--adjudications", type=pathlib.Path)
    args = ap.parse_args()
    rows = [
        json.loads(line)
        for path in args.paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.adjudications:
        adjudicate(rows, json.loads(args.adjudications.read_text(encoding="utf-8")))
    summarize(rows)


if __name__ == "__main__":
    main()
