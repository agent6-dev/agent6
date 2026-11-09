from __future__ import annotations

import pathlib

import pydantic
import pytest

from agent6.config import git_config, harness_config, model, model_roles, provider_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("section", "field", "phrases"),
    [
        (
            harness_config.RoleModel,
            "effort",
            (
                'no thinking on Anthropic, none sent with `backend = "chatgpt"`',
                "(the model's default), as printed",
            ),
        ),
        (
            harness_config.HarnessConfig,
            "verify_timeout_s",
            ("`run_command` handed back by `harness.command_checkin_s`",),
        ),
        (
            harness_config.HarnessConfig,
            "command_checkin_s",
            ("hand-back disabled, command killed at 600 s", "without background tools"),
        ),
        (
            harness_config.RoleModel,
            "temperature",
            (
                "sent where the wire accepts one",
                "Anthropic thinking",
                "`chatgpt` or `claude_code`",
                "after a 400 naming it",
                "`extra_body.temperature` wins",
            ),
        ),
        (provider_config.OpenAIProviderEntry, "base_url", ('and for `backend = "responses"`',)),
        (
            provider_config.OpenAIProviderEntry,
            "base_url",
            (
                "a ChatGPT sign-in also dials its fixed OAuth authority",
                "its issuer and exchange endpoints",
            ),
        ),
        (
            git_config.GitConfig,
            "commit_per_turn",
            (
                "degraded resume-from-git and `sessions diff`/`merge`",
                "`/parallel` dispatch from a changed tree refused",
            ),
        ),
        (
            git_config.GitCommitConfig,
            "name",
            ("`GIT_COMMITTER_NAME`, then the repo's own `git config` identity, when unset",),
        ),
        (
            model_roles.ModelsConfig,
            "worker",
            (
                "`agent6 ask`",
                "their resumes",
                "machine agent states",
                "model-written squash message",
            ),
        ),
        (
            model_roles.ModelsConfig,
            "reviewer",
            (
                "`sessions review`",
                "`sessions compare`",
                "`--parallel` judge",
                "in-loop review panel seats without a route",
            ),
        ),
        (model_roles.ModelsConfig, "planner", ("plan resumes",)),
        (
            model.Config,
            "preset",
            ("`resume --preset`: from that resume on, recorded for later resumes",),
        ),
    ],
    ids=[
        "effort-default",
        "verify-command-checkin",
        "command-timeout",
        "temperature",
        "responses-url",
        "oauth-hosts",
        "parallel-refusal",
        "git-name",
        "worker-consumers",
        "reviewer-consumers",
        "planner-resumes",
        "resume-preset",
    ],
)
def test_field_description_matches_its_consumers(
    section: type[pydantic.BaseModel], field: str, phrases: tuple[str, ...]
) -> None:
    description = section.model_fields[field].description or ""
    for phrase in phrases:
        assert phrase in description


def test_commit_identity_docstring_names_environment_precedence() -> None:
    text = " ".join((git_config.GitCommitConfig.__doc__ or "").split())
    assert "`GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL`, then" in text


def test_settled_descriptions_arm_on_editing_progress() -> None:
    description = harness_config.HarnessConfig.model_fields["settled_stop_after"].description
    assert description is not None
    assert "after a green verify or an editing step" in description
    for path in ("guards.py", "nudges.py"):
        text = " ".join((_ROOT / "src/agent6/harness" / path).read_text().split())
        assert "a green verify or an editing step" in text
        assert "gateless run took an editing step" not in text.lower()
        assert "never re-runs verify leaves the settled detector unable to engage" not in text


def test_timeout_description_keeps_model_scoping_independent_of_gate_policy() -> None:
    """The scoped retry belongs to the model's timed-out call even under `never`."""
    description = harness_config.HarnessConfig.model_fields["verify_timeout_s"].description or ""
    assert "harness call or model's `run_verify_command`" in description
    assert "verify_when" not in description


def test_finish_description_uses_any_verdict_not_only_a_green_one() -> None:
    description = harness_config.HarnessConfig.model_fields["verify_when"].description or ""
    assert harness_config.HarnessConfig().verify_when == "never"
    assert "uncovered final tree" in description
    extra = harness_config.HarnessConfig.model_fields["verify_when"].json_schema_extra
    assert isinstance(extra, dict)
    meanings = extra["choice_meanings"]
    assert isinstance(meanings, dict)
    assert meanings["finish"] == "Verify uncovered final tree at finish"
    assert "last green" not in description
