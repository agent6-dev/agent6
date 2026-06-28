#!/usr/bin/env python3
"""Install the committed seed fixtures into an isolated agent6 home for screenshots.

Reads docs/screenshots/seed/runs/* and copies each run under
``$AGENT6_STATE_HOME/<repo-id>/runs/`` where ``<repo-id>`` is computed for the
demo repo (``$AGENT6_DEMO_REPO`` or cwd). Also writes a demo ``config.toml`` and
``ui.toml`` (theme = agent6-dark) into ``$AGENT6_CONFIG_HOME`` so the config and
theme are deterministic. No secrets are written; this never touches the real
``~/.config/agent6`` when both env vars point at temp dirs (which generate.sh does).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from agent6.paths import repo_id

SEED = Path(__file__).resolve().parent / "seed" / "runs"

# Newest first: the order the hub lists runs (we space mtimes so the sort is
# stable regardless of copy order). The featured run sits on top.
ORDER = [
    "willing-glen-9ZYWWB",
    "friendly-crane-1X3ER0",
    "tidy-river-165YS6",
    "thoughtful-comet-1TQASW",
    "ready-rowan-A5P972",
]

DEMO_CONFIG = """\
# Demo config used only to render the docs screenshots; no secrets here.
profile = "standard"

[sandbox]
profile = "auto"
agent_network = "providers"
tool_network = "block"
run_commands = "ask"
protect_git = true

[git]
require_clean_worktree = true
branch_per_run = true
commit_strategy = "per_step"
allow_push = false

[budget]
max_input_tokens = 2000000
max_output_tokens = 200000
best_effort_usd_limit = 10.0

[workflow]
verify_command = ["uv", "run", "pytest", "-x"]

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[providers.openrouter]
api_format = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[models.worker]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"

[models.reviewer]
provider = "anthropic"
model = "claude-sonnet-4-6"

[models.planner]
provider = "anthropic"
model = "claude-sonnet-4-6"
"""

UI_TOML = '[ui]\ntheme = "agent6-dark"\n'


def main() -> None:
    state_base = os.environ.get("AGENT6_STATE_HOME")
    config_home = os.environ.get("AGENT6_CONFIG_HOME")
    if not state_base or not config_home:
        sys.exit("set AGENT6_STATE_HOME and AGENT6_CONFIG_HOME (generate.sh does this)")

    demo = Path(os.environ.get("AGENT6_DEMO_REPO") or Path.cwd()).resolve()
    runs_dir = Path(state_base) / repo_id(demo) / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True)

    # Copy in ORDER, oldest first, spacing mtimes 5 min apart so the hub's
    # newest-first sort puts the featured run on top.
    base = time.time() - len(ORDER) * 300
    found = sorted(p.name for p in SEED.iterdir() if p.is_dir())
    ordered = [r for r in ORDER if r in found] + [r for r in found if r not in ORDER]
    for i, run_id in enumerate(reversed(ordered)):
        dst = runs_dir / run_id
        shutil.copytree(SEED / run_id, dst)
        mtime = base + i * 300
        for p in [*dst.rglob("*"), dst]:
            os.utime(p, (mtime, mtime))
    print(f"seeded {len(ordered)} runs -> {runs_dir}")

    cfg = Path(config_home)
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.toml").write_text(DEMO_CONFIG, encoding="utf-8")
    (cfg / "ui.toml").write_text(UI_TOML, encoding="utf-8")
    print(f"wrote demo config + theme -> {cfg}")


if __name__ == "__main__":
    main()
