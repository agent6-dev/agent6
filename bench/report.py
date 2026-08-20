# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-arm bench report: N result jsonls -> one static HTML.

The experiment view the SaaS eval platforms sell, as a file: per-arm
aggregates with confidence intervals, then a per-instance grid (rows =
task/rep, columns = arms) so a regression is visible as a column of red
cells, with each cell naming its session id for transcript drill-down.
Stdlib only; the output works over file://.

    uv run python bench/report.py --out report.html results/a.jsonl results/b.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def _load(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ci95(values: list[float]) -> float:
    """Half-width of a normal-approx 95% interval; 0 for n < 2."""
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _agg(rows: list[Row]) -> dict[str, Any]:
    scores = [float(r.get("score") or 0) for r in rows]
    return {
        "n": len(rows),
        "score": statistics.mean(scores) if rows else 0.0,
        "ci": _ci95(scores),
        "solved": sum(1 for s in scores if s >= 0.999),
        "wall_s": statistics.mean([float(r.get("wall_s") or 0) for r in rows]) if rows else 0.0,
        "iters": statistics.mean([float(r.get("iterations") or 0) for r in rows]) if rows else 0.0,
        "tokens_out": statistics.mean([float(r.get("tokens_out") or 0) for r in rows])
        if rows
        else 0.0,
        "usd": sum(float(r.get("usd") or 0) for r in rows),
        "tampered": sum(1 for r in rows if r.get("tampered")),
    }


def _cell_style(score: float | None, best: float) -> str:
    if score is None:
        return "background:#eee;color:#888"
    if score >= 0.999:
        return "background:#d7f4d7"
    if score >= best - 1e-9:
        return "background:#fff6d6"
    return "background:#f8d7d7"


def render(arms: dict[str, list[Row]]) -> str:
    """The report HTML for {arm label: rows}."""
    head = (
        "<meta charset='utf-8'><title>bench report</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;margin:1rem 0}"
        "td,th{border:1px solid #ccc;padding:.35rem .6rem;text-align:right}"
        "th{background:#f4f4f4}td.key,th.key{text-align:left}"
        "small{color:#666}</style>"
    )
    out = [head, "<h1>bench report</h1>"]

    out.append("<h2>arms</h2><table><tr><th class='key'>arm</th><th>n</th><th>score</th>")
    out.append("<th>solved</th><th>wall s</th><th>iters</th><th>out tok</th>")
    out.append("<th>usd</th><th>tampered</th></tr>")
    for label, rows in arms.items():
        a = _agg(rows)
        out.append(
            f"<tr><td class='key'>{html.escape(label)}</td><td>{a['n']}</td>"
            f"<td>{a['score']:.3f} &plusmn; {a['ci']:.3f}</td>"
            f"<td>{a['solved']}/{a['n']}</td><td>{a['wall_s']:.0f}</td>"
            f"<td>{a['iters']:.1f}</td><td>{a['tokens_out']:.0f}</td>"
            f"<td>{a['usd']:.2f}</td><td>{a['tampered']}</td></tr>"
        )
    out.append("</table>")

    # Per-instance grid keyed on (task, rep); a run's cell names its session
    # id so the transcript is one `agent6 sessions show <id>` away.
    keys = sorted(
        {(str(r.get("task")), int(r.get("rep") or 0)) for rows in arms.values() for r in rows}
    )
    out.append("<h2>per instance</h2><table><tr><th class='key'>task / rep</th>")
    out.extend(f"<th>{html.escape(label)}</th>" for label in arms)
    out.append("</tr>")
    for task, rep in keys:
        cells = []
        by_arm: dict[str, Row | None] = {}
        for label, rows in arms.items():
            match = [
                r for r in rows if str(r.get("task")) == task and int(r.get("rep") or 0) == rep
            ]
            by_arm[label] = match[0] if match else None
        best = max(
            (float(r.get("score") or 0) for r in by_arm.values() if r is not None), default=0.0
        )
        for label in arms:
            r = by_arm[label]
            if r is None:
                cells.append(f"<td style='{_cell_style(None, best)}'>&mdash;</td>")
                continue
            score = float(r.get("score") or 0)
            sid = html.escape(str(r.get("session_id") or ""))
            cells.append(
                f"<td style='{_cell_style(score, best)}'>{score:.2f}<br><small>{sid}</small></td>"
            )
        out.append(f"<tr><td class='key'>{html.escape(task)} r{rep}</td>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path, help="result jsonl files, one per arm")
    ap.add_argument("--out", type=Path, default=Path("bench-report.html"))
    args = ap.parse_args()
    arms = {p.stem: _load(p) for p in args.results}
    args.out.write_text(render(arms), encoding="utf-8")
    print(f"wrote {args.out} ({sum(len(r) for r in arms.values())} rows, {len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
