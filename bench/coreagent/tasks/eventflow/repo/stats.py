"""Per-user aggregates."""

_memo = {}


def user_totals(sessions):
    totals = {}
    for s in sessions:
        for ev in s.events:
            totals[ev.user] = totals.get(ev.user, 0) + ev.value_cents
    return totals


def avg_session_value(sessions, user):
    if user in _memo:
        return _memo[user]
    total = 0
    count = 0
    for s in sessions:
        if s.user == user:
            count += 1
            for ev in s.events:
                total += ev.value_cents
    if count == 0:
        raise ValueError(f"no sessions for {user!r}")
    avg = round(total / count)
    _memo[user] = avg
    return avg
