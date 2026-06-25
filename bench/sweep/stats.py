#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rigorous summary statistics for the agent6 cross-model sweep.

Reads per-run sample JSON files (one object per agent6 run) and emits a
scientific-style markdown report: per-model success rates with Wilson score
intervals, median cost/tokens/wall-clock with inter-quartile ranges and
percentile-bootstrap 95% confidence intervals, cost-per-successful-task, and
pairwise significance tests (Fisher's exact for success counts; Mann-Whitney U
with a tie-corrected normal approximation, plus Cliff's delta effect size, for
cost-on-success).

All randomness (the bootstrap) is seeded so the report is reproducible. No third
-party numerical dependency: every statistic is implemented here against the
standard library, so the method is fully auditable.

Sample schema (one JSON object per file, see run_sweep.py):
    {"model": str, "task": str, "rep": int, "success": bool,
     "cost_usd": float, "input_tokens": int, "output_tokens": int,
     "wall_seconds": float, "steps": int|null, "agent_exit": int|null}

Usage:
    python3 bench/sweep/stats.py <samples_dir> [--ref <model>] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_BOOTSTRAP_B = 10000
_BOOTSTRAP_SEED = 20260623
_Z95 = 1.959963984540054  # standard normal 0.975 quantile

# Public list prices (USD per 1M tokens: input, output) used ONLY to derive a
# cost for models whose API does not report a per-call cost (Anthropic direct;
# OpenRouter reports usage.cost so its models need no entry). Marked "derived
# (list price)" in the report -- a transparent estimate from measured token
# counts, not actual billing (no caching/volume discount). opus-4-8 confirmed
# via the OpenRouter listing; the Sonnet line is its long-standing list price.
_DERIVED_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def effective_cost(sample: dict) -> float:
    """Measured cost if the provider reported one; otherwise a list-price
    estimate from token counts for the known unpriced (Anthropic) models."""
    measured = float(sample.get("cost_usd") or 0.0)
    if measured > 0:
        return measured
    slug = sample.get("model_slug", "")
    price = _DERIVED_PRICES.get(slug)
    if price is None:
        return measured
    pin, pout = price
    in_tok = float(sample.get("input_tokens") or 0.0)
    out_tok = float(sample.get("output_tokens") or 0.0)
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


def _any_derived(samples: list[dict]) -> bool:
    return any(
        float(s.get("cost_usd") or 0.0) <= 0 and s.get("model_slug") in _DERIVED_PRICES
        for s in samples
    )


# --- core statistics (stdlib only) -------------------------------------------


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Preferred over the normal (Wald) interval at small n / extreme p: it never
    leaves [0, 1] and stays sensible at p = 0 or 1.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def quartiles(xs: list[float]) -> tuple[float, float, float]:
    """(Q1, median, Q3) via linear interpolation; safe for tiny samples."""
    s = sorted(xs)
    if not s:
        return (math.nan, math.nan, math.nan)
    if len(s) == 1:
        return (s[0], s[0], s[0])
    med = statistics.median(s)
    # statistics.quantiles needs n >= 2; n==1 handled above.
    q = statistics.quantiles(s, n=4, method="inclusive")
    return (q[0], med, q[2])


def _lcg(seed: int):
    """Tiny deterministic PRNG (no global random state, fully reproducible)."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (6364136223846793005 * state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield state >> 11  # 53 high bits


def bootstrap_median_ci(
    xs: list[float], b: int = _BOOTSTRAP_B, seed: int = _BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for the median. Deterministic (seeded)."""
    n = len(xs)
    if n < 3:
        return (math.nan, math.nan)
    rng = _lcg(seed)
    mods = 2**53
    meds: list[float] = []
    for _ in range(b):
        resample = [xs[next(rng) % n] for _ in range(n)]
        meds.append(statistics.median(resample))
    meds.sort()
    lo = meds[int(0.025 * b)]
    hi = meds[min(b - 1, int(0.975 * b))]
    _ = mods
    return (lo, hi)


def gmean(xs: list[float]) -> float:
    pos = [x for x in xs if x > 0]
    if not pos:
        return math.nan
    return math.exp(sum(math.log(x) for x in pos) / len(pos))


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def _hypergeom_logp(a: int, b: int, c: int, d: int) -> float:
    """log P of one 2x2 table with fixed margins (Fisher kernel)."""
    n = a + b + c + d
    return (
        _log_factorial(a + b)
        + _log_factorial(c + d)
        + _log_factorial(a + c)
        + _log_factorial(b + d)
        - _log_factorial(a)
        - _log_factorial(b)
        - _log_factorial(c)
        - _log_factorial(d)
        - _log_factorial(n)
    )


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for the 2x2 table [[a,b],[c,d]].

    Sums the probabilities of all tables (with the same margins) that are no
    more probable than the observed one. Exact; good at the small counts a
    repeated-run agent benchmark produces.
    """
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    p_obs = math.exp(_hypergeom_logp(a, b, c, d))
    total = 0.0
    a_min = max(0, col1 - row2)
    a_max = min(row1, col1)
    for ai in range(a_min, a_max + 1):
        bi = row1 - ai
        ci = col1 - ai
        di = row2 - ci
        p = math.exp(_hypergeom_logp(ai, bi, ci, di))
        if p <= p_obs * (1 + 1e-9):
            total += p
    _ = n
    return min(1.0, total)


def _normal_sf(z: float) -> float:
    """Upper-tail standard-normal survival function via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class MWResult:
    u: float
    p_two_sided: float
    cliffs_delta: float


def mann_whitney_u(a: list[float], b: list[float]) -> MWResult:
    """Mann-Whitney U (two-sided) with tie correction + normal approximation,
    plus Cliff's delta effect size. Distribution-free: compares whether cost in
    group A tends to exceed group B without assuming normality.

    The normal approximation is adequate for the n we report and noted as such;
    Cliff's delta (-1..1) gives a magnitude that does not depend on n.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return MWResult(math.nan, math.nan, math.nan)
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    # average ranks (1-based) with tie handling
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u_a = r_a - na * (na + 1) / 2.0
    u_b = na * nb - u_a
    u = min(u_a, u_b)
    # Cliff's delta from U_a: delta = 2*U_a/(na*nb) - 1, oriented A vs B.
    cliffs = (2.0 * u_a) / (na * nb) - 1.0
    mu = na * nb / 2.0
    # tie correction term
    counts: dict[float, int] = defaultdict(int)
    for v, _g in combined:
        counts[v] += 1
    tie = sum(t**3 - t for t in counts.values())
    n = na + nb
    sigma2 = (na * nb / 12.0) * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if sigma2 <= 0:
        return MWResult(u, math.nan, cliffs)
    z = (abs(u - mu) - 0.5) / math.sqrt(sigma2)  # continuity-corrected
    p = 2.0 * _normal_sf(max(0.0, z))
    return MWResult(u, min(1.0, p), cliffs)


# --- aggregation --------------------------------------------------------------


def load_samples(samples_dir: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(samples_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(d, dict) and "model" in d and "task" in d:
            out.append(d)
    return out


@dataclass
class ModelAgg:
    model: str
    n: int
    successes: int
    success_rate: float
    success_ci: tuple[float, float]
    cost_med_iqr_ci: tuple[float, float, float, float, float]  # med,q1,q3,lo,hi
    cost_on_success: list[float]
    wall_med: float
    in_med: float
    out_med: float
    cost_per_success: float


def aggregate_by_model(samples: list[dict]) -> dict[str, ModelAgg]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_model[s["model"]].append(s)
    aggs: dict[str, ModelAgg] = {}
    for model, rows in by_model.items():
        n = len(rows)
        succ_rows = [r for r in rows if r.get("success")]
        k = len(succ_rows)
        costs = [effective_cost(r) for r in rows]
        cost_succ = [effective_cost(r) for r in succ_rows]
        q1, med, q3 = quartiles(costs)
        lo, hi = bootstrap_median_ci(costs)
        total_cost = sum(costs)
        aggs[model] = ModelAgg(
            model=model,
            n=n,
            successes=k,
            success_rate=k / n if n else math.nan,
            success_ci=wilson_ci(k, n),
            cost_med_iqr_ci=(med, q1, q3, lo, hi),
            cost_on_success=cost_succ,
            wall_med=quartiles([float(r.get("wall_seconds") or 0.0) for r in rows])[1],
            in_med=quartiles([float(r.get("input_tokens") or 0.0) for r in rows])[1],
            out_med=quartiles([float(r.get("output_tokens") or 0.0) for r in rows])[1],
            cost_per_success=(total_cost / k) if k else math.nan,
        )
    return aggs


def _fmt_usd(x: float) -> str:
    return "n/a" if math.isnan(x) else f"${x:.4f}"


def build_report(samples: list[dict], ref: str | None = None) -> str:
    if not samples:
        return "# agent6 cross-model sweep\n\nNo samples found.\n"
    aggs = aggregate_by_model(samples)
    models = sorted(aggs, key=lambda m: (-aggs[m].success_rate, aggs[m].cost_per_success))
    tasks = sorted({s["task"] for s in samples})
    ref = ref if ref in aggs else models[0]
    total_runs = len(samples)
    total_cost = sum(float(s.get("cost_usd") or 0.0) for s in samples)

    L: list[str] = []
    L.append("# agent6 cross-model benchmark")
    L.append("")
    L.append(
        f"{total_runs} runs across {len(models)} models and {len(tasks)} tasks "
        f"(${total_cost:.2f} measured provider-reported spend)."
    )
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(
        "Each task scopes the agent to a single function in a pinned open-source\n"
        "repository plus that project's own verify command (a subset of its test\n"
        "suite). Success is scored out-of-band by re-running the verify command after\n"
        "the run -- the agent's own claim is not trusted. (Tasks differ in whether the\n"
        "verify starts failing or passing; see the suite notes.) Each (model, task)\n"
        "cell is repeated independently; cost and token counts come from the run's\n"
        "usage accounting and wall-clock from the harness."
    )
    if _any_derived(samples):
        L.append("")
        L.append(
            "**Cost note:** OpenRouter reports a per-call USD cost, used directly. The\n"
            "Anthropic API does not, so cost for those models is *derived* from measured\n"
            "token counts at public list price (opus-4-8 $5/$25, sonnet-4-6 $3/$15 per\n"
            "1M input/output tokens) — an estimate, not billed spend, and it excludes\n"
            "prompt-caching discounts the other models' reported costs already include."
        )
    L.append("")
    L.append(
        "Reported statistics: success rate with a 95% Wilson score interval; cost,\n"
        "tokens and wall-clock as median [Q1, Q3] with a seeded percentile-bootstrap\n"
        "95% CI for the median; cost-per-successful-task = total spend / successes.\n"
        f"Pairwise tests are versus `{ref}` (the reference): Fisher's exact test for\n"
        "success counts and the Mann-Whitney U test (tie-corrected normal\n"
        "approximation) with Cliff's delta for cost-on-success. All intervals are 95%.\n"
        "n is small by construction; treat single comparisons as indicative and read\n"
        "the intervals, not the point estimates."
    )
    L.append("")

    # --- per-model aggregate table ---
    L.append("## Per-model results")
    L.append("")
    L.append(
        "| model | n | success | 95% CI | median cost | cost 95% CI | "
        "cost/success | med in tok | med out tok | med wall |"
    )
    L.append("|---|--:|--:|---|--:|---|--:|--:|--:|--:|")
    for m in models:
        a = aggs[m]
        med, _q1, _q3, lo, hi = a.cost_med_iqr_ci
        sr = f"{a.success_rate*100:.0f}% ({a.successes}/{a.n})"
        ci = f"[{a.success_ci[0]*100:.0f}, {a.success_ci[1]*100:.0f}]%"
        cost = _fmt_usd(med)
        cost_ci = "n/a" if math.isnan(hi) else f"[{lo:.4f}, {hi:.4f}]"
        L.append(
            f"| `{m}` | {a.n} | {sr} | {ci} | {cost} | {cost_ci} | "
            f"{_fmt_usd(a.cost_per_success)} | {a.in_med:.0f} | {a.out_med:.0f} | "
            f"{a.wall_med:.0f}s |"
        )
    L.append("")

    # --- per-(model,task) success matrix ---
    L.append("## Success by task (passes / reps)")
    L.append("")
    L.append("| model | " + " | ".join(tasks) + " |")
    L.append("|---|" + "|".join(["--:"] * len(tasks)) + "|")
    for m in models:
        cells = []
        for t in tasks:
            rows = [s for s in samples if s["model"] == m and s["task"] == t]
            k = sum(1 for r in rows if r.get("success"))
            cells.append(f"{k}/{len(rows)}" if rows else "-")
        L.append(f"| `{m}` | " + " | ".join(cells) + " |")
    L.append("")

    # --- pairwise vs reference ---
    L.append(f"## Pairwise comparison vs `{ref}`")
    L.append("")
    L.append(
        "Cost-on-success compares only runs that passed in each model (a fair cost\n"
        "comparison conditions on success). Fisher's p tests the success-count "
        "difference over all reps."
    )
    L.append("")
    L.append(
        "| model | Δ success rate | Fisher p | median cost Δ (succ) | "
        "Mann-Whitney p | Cliff's δ |"
    )
    L.append("|---|--:|--:|--:|--:|--:|")
    ra = aggs[ref]
    for m in models:
        if m == ref:
            continue
        a = aggs[m]
        # Fisher on success counts
        fp = fisher_exact_two_sided(
            a.successes, a.n - a.successes, ra.successes, ra.n - ra.successes
        )
        d_sr = a.success_rate - ra.success_rate
        # Mann-Whitney on cost-on-success
        if a.cost_on_success and ra.cost_on_success:
            mw = mann_whitney_u(a.cost_on_success, ra.cost_on_success)
            dcost = statistics.median(a.cost_on_success) - statistics.median(ra.cost_on_success)
            mwp = f"{mw.p_two_sided:.3f}" if not math.isnan(mw.p_two_sided) else "n/a"
            cd = f"{mw.cliffs_delta:+.2f}" if not math.isnan(mw.cliffs_delta) else "n/a"
            dcost_s = f"{dcost:+.4f}"
        else:
            mwp, cd, dcost_s = "n/a", "n/a", "n/a"
        L.append(
            f"| `{m}` | {d_sr*100:+.0f}pp | {fp:.3f} | {dcost_s} | {mwp} | {cd} |"
        )
    L.append("")
    L.append(
        "_pp = percentage points. δ > 0 means this model spent more than the reference\n"
        "on successful runs; |δ| ≈ 0.15/0.33/0.47 ≈ small/medium/large. p-values are\n"
        "uncorrected; with this many pairwise tests, apply a Holm-Bonferroni correction\n"
        "before claiming any single difference._"
    )
    L.append("")
    L.append("## Reading the success rate")
    L.append("")
    L.append(
        "Read the per-task matrix, not just the aggregate. A suite can mix *construction*\n"
        "tasks (a function is stubbed so the verify FAILS until it is restored) with\n"
        "*restraint* tasks (the code already passes its verify; success means resisting\n"
        "unnecessary edits that would break it). A do-nothing or crashed run still 'passes'\n"
        "every restraint task, so an aggregate success rate has a non-zero floor and a\n"
        "mid-range score can mean 'did nothing' rather than 'half-capable' — the matrix\n"
        "disambiguates. (This is how a crashed run was caught here: see the report notes.)"
    )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples_dir", type=Path)
    ap.add_argument("--ref", default=None, help="reference model slug for pairwise tests")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here (else stdout)")
    args = ap.parse_args()
    samples = load_samples(args.samples_dir)
    report = build_report(samples, ref=args.ref)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out} ({len(samples)} samples)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
