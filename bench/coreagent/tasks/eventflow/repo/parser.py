"""Event-line parsing."""

from datetime import datetime, timezone


class Event:
    def __init__(self, ts, user, action, value_cents):
        self.ts = ts
        self.user = user
        self.action = action
        self.value_cents = value_cents

    def __repr__(self):
        return f"Event({self.ts}, {self.user!r}, {self.action!r}, {self.value_cents})"


def parse_line(line):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 4:
        raise ValueError(f"expected 4 fields: {line!r}")
    raw_ts, user, action, raw_value = parts
    if not user or not action:
        raise ValueError(f"empty field: {line!r}")
    try:
        dt = datetime.strptime(raw_ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"bad timestamp: {raw_ts!r}") from None
    value = float(raw_value)
    if value < 0:
        raise ValueError(f"negative value: {raw_value!r}")
    cents = int(value * 100)
    return Event(int(dt.timestamp()), user, action, cents)


def parse_lines(lines):
    for line in lines:
        if not line.strip():
            continue
        yield parse_line(line)
