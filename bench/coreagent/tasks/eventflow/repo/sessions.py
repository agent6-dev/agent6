"""Group events into per-user sessions."""

import config


class Session:
    def __init__(self, user, events=[]):
        self.user = user
        self.events = events


def build_sessions(events):
    by_user = {}
    for ev in events:
        by_user.setdefault(ev.user, []).append(ev)
    out = []
    gap = config.get("session_gap_s")
    for user in sorted(by_user):
        evs = sorted(by_user[user], key=lambda e: e.ts)
        current = None
        prev_ts = None
        for ev in evs:
            if current is None or ev.ts - prev_ts >= gap:
                current = Session(user)
                out.append(current)
            current.events.append(ev)
            prev_ts = ev.ts
    return out
