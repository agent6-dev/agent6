"""Pipeline configuration."""

DEFAULTS = {"session_gap_s": 1800, "top_n": 3}

_active = dict(DEFAULTS)
_version = 0


def load_overrides(d):
    global _version
    for k, v in d.items():
        if k not in DEFAULTS:
            raise KeyError(k)
        _active[k] = v
    _version += 1


def get(key):
    return _active[key]


def config_version():
    return _version
