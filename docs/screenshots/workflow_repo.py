#!/usr/bin/env python3
"""Build the workflow tour's repository, sessions and config in an isolated agent6 home.

Clones the stats demo bundle into ``$AGENT6_DEMO_REPO`` and commits the temperature module
with its failing test on top, so the children the tour starts have a test to fix from the
committed temperature cassette. Installs the seeded sessions (seed.py), then points the
default model at the replay proxy. Reads ``$XDG_STATE_HOME`` and ``$XDG_CONFIG_HOME`` as
seed.py does.

  python3 -m docs.screenshots.workflow_repo PROXY_PORT   # from the repository root
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

from docs.screenshots import seed

from agent6 import paths

SEED = pathlib.Path(__file__).resolve().parent / "seed"

# judges = []: a judge is a model call, and the cassette holds none, so the attempts rank
# by verify result, then cost.
CONFIG = """\
# Config for the workflow tour: the model's answers come from llm_proxy.py; no secrets here.
[sandbox]
network = "session"
run_commands = "allow"
protect_git = "auto"

[git]
dirty_tree = "ask"
branch_per_run = true

[git.commit]
name = "agent6 demo"
email = "demo@agent6.dev"

[budget]
max_usd = 2.0
max_tokens_fallback = 2000000

[harness]
verify.command = ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."]

[harness.parallel]
judges = []

[providers.openrouter]
backend = "openai"
base_url = "http://127.0.0.1:{port}/api/v1"

[models.default]
model = "openrouter/moonshotai/kimi-k2.6"
"""


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def main(proxy_port: int) -> None:
    """Create the repository, seed the sessions and write the replay config.

    Args:
        proxy_port: The loopback port llm_proxy.py serves the cassette on.
    """
    repo = pathlib.Path(os.environ["AGENT6_DEMO_REPO"])
    bundle = SEED / "stats-demo.bundle"
    subprocess.run(["git", "clone", "--quiet", str(bundle), str(repo)], check=True)
    _git(repo, "fetch", "--quiet", str(bundle), "+refs/agent6/*:refs/agent6/*")
    _git(repo, "config", "user.name", "agent6 demo")
    _git(repo, "config", "user.email", "demo@agent6.dev")
    shutil.copy(SEED / "temps-repo" / "temps.py", repo / "temps.py")
    shutil.copy(SEED / "temps-repo" / "tests" / "test_temps.py", repo / "tests" / "test_temps.py")
    _git(repo, "add", "temps.py", "tests/test_temps.py")
    _git(repo, "commit", "--quiet", "-m", "Add the temperature helpers and their tests")

    seed.main()
    config = paths.global_config_dir()
    (config / "config.toml").write_text(CONFIG.format(port=proxy_port), encoding="utf-8")
    secrets = config / "secrets.toml"
    secrets.write_text('[providers.openrouter]\napi_key = "unused-in-replay"\n', encoding="utf-8")
    secrets.chmod(0o600)


if __name__ == "__main__":
    main(int(sys.argv[1]))
