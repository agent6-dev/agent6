"""Top-N report."""

import config


def top_users(totals):
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    lines = []
    for user, cents in ranked[: config.get("top_n")]:
        lines.append(f"{user} {cents // 100}.{cents % 100:02d}")
    return lines
