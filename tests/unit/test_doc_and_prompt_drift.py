# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drift pins: prose that names code must track the code.

Each pin failed in the wild before it existed: a tool renamed out from under a prompt mention, a
bench config key that did not match the Config field, a dependency count written as a word.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tomllib

from agent6.prompts import loop as prompts
from agent6.tools import patch_apply
from agent6.tools import schema as tool_schema

REPO = pathlib.Path(__file__).resolve().parents[2]

_PROMPT_CONSTANTS = (
    prompts.SYSTEM_PROMPT_BASE,
    prompts.PLAN_SYSTEM_PROMPT_BASE,
    prompts.ASK_SYSTEM_PROMPT_BASE,
    prompts.AGENT_SYSTEM_PROMPT_BASE,
    prompts.GIT_PROTECT_RULE,
    prompts.NO_AUTO_COMMIT_RULE,
    prompts.HARDENED_FS_RULE,
    prompts.DAG_RULES_OPTIONAL,
    prompts.DAG_RULES_DECOMPOSE,
    prompts.APPLY_EDIT_RULE,
    prompts.PLAN_VERIFY_RULE,
    prompts.READONLY_COMMAND_RULE,
    prompts.AUTO_COMMIT_RULE,
    prompts.AUTO_COMMIT_RULE_GATELESS,
    prompts.MODEL_GIT_RULE,
    prompts.MODEL_GIT_RULE_NO_COMMANDS,
    prompts.CREATE_HINT,
    prompts.CREATE_HINT_PATCH_ONLY,
)

# Backticked identifiers in the prompts that are deliberately not tool names.
_NON_TOOL_SPANS = {
    "acceptance",
    "agent",
    "build_system_prompt",
    "depends_on",
    "obsolete",
    "parent_id",
    "result",
    "skipped",
    "summary",
    "stale_gate",
    "title",
    "toml",
    "old_string",
    "new_string",
    "kind",
    "preview",
    "in_progress",
    "passed",
    "standing",
}


def _registered_tool_names() -> set[str]:
    names = {cls.TOOL_NAME for cls in tool_schema.ALL_TOOLS}
    for mode in ("run", "plan", "ask", "machine", "agent"):
        mt = tool_schema.mode_tools(mode)
        names |= {cls.TOOL_NAME for cls in (*mt.base, *mt.extras)}
    return names


def test_apply_edit_prompt_contract_mentions_its_indent_heal() -> None:
    """The system prompt does not claim byte-exact matching is apply_edit's only accepted shape."""
    assert "indent" in prompts.SYSTEM_PROMPT_BASE
    assert "heal" in prompts.SYSTEM_PROMPT_BASE


def test_apply_patch_prompt_contract_mentions_multi_file_patches() -> None:
    patch = (
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+A\n"
        "diff --git a/b b/b\n--- a/b\n+++ b/b\n@@ -1 +1 @@\n-b\n+B\n"
    )
    assert len(patch_apply.split_patch_files(patch)) == 2
    assert "multi-file" in prompts.SYSTEM_PROMPT_BASE


def test_every_tool_mention_in_the_prompts_is_a_registered_tool() -> None:
    registered = _registered_tool_names()
    for text in _PROMPT_CONSTANTS:
        for span in re.findall(r"`([a-z][a-z0-9_]+)`", text):
            assert span in registered or span in _NON_TOOL_SPANS, (
                f"prompt mentions `{span}`, which is neither a registered tool"
                " nor in the non-tool allowlist: a rename left stale text, or"
                " the allowlist needs the new parameter"
            )


def test_readme_dependency_count_matches_pyproject() -> None:
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    readme = (REPO / "README.md").read_text()
    m = re.search(r"(\w+) runtime dependencies", readme)
    assert m is not None, "README no longer states the dependency count"
    assert m.group(1).lower() == words.get(len(deps), str(len(deps))), (
        f"README says '{m.group(1)} runtime dependencies' but pyproject has {len(deps)}"
    )


def test_readme_quoted_security_defaults_match_config() -> None:
    from agent6.config import Config

    cfg = Config()
    readme = (REPO / "README.md").read_text()
    for quoted, actual in (
        ('network = "auto"', cfg.sandbox.network),
        ('run_commands = "ask"', cfg.sandbox.run_commands),
        ("protect_git = true", cfg.sandbox.protect_git),
    ):
        assert quoted in readme, f"README no longer quotes {quoted!r}"
        want = quoted.split("= ")[1].strip('"')
        assert str(actual).lower() == want, (
            f"README quotes {quoted!r} but the Config default is {actual!r}"
        )


def test_bench_container_config_template_validates() -> None:
    """in_container.sh's config heredoc validates through Config, so a stray key is caught."""
    from agent6.config import Config

    script = (REPO / "bench" / "swebench" / "in_container.sh").read_text()
    m = re.search(r"cat > /root/agent6\.toml <<EOF\n(.*?)\nEOF\n", script, re.S)
    assert m is not None, "config heredoc not found in in_container.sh"
    # The two shell-computed blocks run ahead of the heredoc; a stub validates the stub.
    review = re.search(r'^REVIEW_LINES=""\n.*?^fi\n', script, re.S | re.M)
    verify = re.search(
        r'^if \[ "\$\{AGENT6_SB_VERIFY:-\}" = "none" \].*?^  VERIFY_TOML="verify_command.*?\nfi\n',
        script,
        re.S | re.M,
    )
    assert review is not None and verify is not None, "the config blocks moved"
    stub = {
        "PROVIDER_BLOCK": (
            '[providers.openrouter]\napi_format = "openai"\napi_key_env = "K"\n'
            'base_url = "https://openrouter.ai/api/v1"'
        ),
        "PROVIDER": "openrouter",
        "MODEL": "moonshotai/stub",
        "MAX_USD": "1.0",
        "CONDA_PY": "python3",
        "PROMPT_FILE_LINE": "",
    }

    def _render(env: dict[str, str]) -> str:
        # bash renders the heredoc as the container does; a hand-rolled `${VAR:+...}` drifted.
        script = "set -uo pipefail\n" + review.group(0) + verify.group(0)
        script += "cat <<EOF\n" + m.group(1) + "\nEOF\n"
        return subprocess.run(
            ["bash", "-c", script],
            env={"PATH": os.environ.get("PATH", ""), **env},
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    cases = (
        stub,
        {**stub, "AGENT6_SB_EFFORT": "medium"},
        {**stub, "AGENT6_SB_VERIFY_WHEN": "step"},
        {**stub, "AGENT6_SB_MAX_PERCENT": "40"},
        {**stub, "AGENT6_SB_REVIEW_SEATS": "security@openrouter/a;tests@anthropic/b"},
        {**stub, "AGENT6_SB_VERIFY": "none"},
    )
    for env in cases:
        rendered = _render(env)
        data = tomllib.loads(rendered)
        data.pop("agent6", None)
        Config.model_validate(data)
        if env.get("AGENT6_SB_EFFORT"):
            assert data["models"]["worker"]["effort"] == "medium"
        if env.get("AGENT6_SB_VERIFY_WHEN"):
            assert data["harness"]["verify_when"] == "step"
        if env.get("AGENT6_SB_MAX_PERCENT"):
            assert data["budget"]["max_percent"] == 40
        if env.get("AGENT6_SB_REVIEW_SEATS"):
            assert data["review"]["seats"] == ["security@openrouter/a", "tests@anthropic/b"]
        if env.get("AGENT6_SB_VERIFY") == "none":
            assert data["harness"]["verify_infer"] is False
            assert "verify_command" not in data["harness"]
        else:
            assert data["harness"]["verify_command"][0] == "python3"
