# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Model facts for the run path.

Context windows (they size adaptive compaction), decompose-first families and the effort
each role sends. Entries change only with bench evidence; the live cache covers windows the
table omits. Lookups never raise and never touch the network.
"""

from __future__ import annotations

import re

from agent6 import kinds
from agent6.config import Config
from agent6.models import cache
from agent6.providers import openai

__all__ = [
    "BUNDLED_CONTEXT_WINDOWS",
    "DECOMPOSE_WIN_MODEL_FAMILIES",
    "compaction_thresholds",
    "context_window",
    "decompose_default",
    "resolved_adaptive_values",
    "role_effort",
]

# Context windows in tokens, canonical ids only (`normalize_model_id` strips a date or `:tag`).
# The table wins over the live cache; Anthropic's listing reports no window.
BUNDLED_CONTEXT_WINDOWS: dict[str, int] = {
    # The 4.x window is 200k with a 1M beta opt-in: pin [context] when enabling that beta.
    "claude-fable-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    # The open-weights models the bench runs, cross-checked against the live listing.
    "moonshotai/kimi-k2.6": 262_144,
    "moonshotai/kimi-k2": 131_072,
    "qwen/qwen3-coder": 1_048_576,
    "qwen/qwen3-coder-30b-a3b-instruct": 160_000,
    "z-ai/glm-4.6": 202_752,
    "z-ai/glm-5.2": 1_048_576,
    "deepseek/deepseek-v3.2-exp": 163_840,
}

# Adaptive sizing: tokens ~= chars/4 (the loop's `context_chars` approximation); tier 1 elides
# old tool results past 45% of the window, tier 2 summarises within a reserve of the window
# that leaves room for the next turn's output and the summary call.
_CHARS_PER_TOKEN = 4
_DROP_FRACTION = 0.45
_RESERVE_TOKENS = 16_384
# For an unknown window; mirrors harness._compaction DROP_BLOCKS_AT_CHARS / SUMMARISE_AT_CHARS.
_FALLBACK_DROP_CHARS = 256_000
_FALLBACK_SUMMARISE_CHARS = 768_000


def normalize_model_id(model_id: str) -> str:
    """Return the id without a trailing `-YYYYMMDD` date or `:tag`, the bundled key's form."""
    base = model_id.split(":", 1)[0]
    return re.sub(r"-\d{8}$", "", base)


def _bundled_context_window(model_id: str) -> int | None:
    """Return the bundled window for the id or its normalized form."""
    if model_id in BUNDLED_CONTEXT_WINDOWS:
        return BUNDLED_CONTEXT_WINDOWS[model_id]
    return BUNDLED_CONTEXT_WINDOWS.get(normalize_model_id(model_id))


def context_window(provider_name: str, model_id: str) -> int | None:
    """Return the context window in tokens for a configured model.

    The bundled table first, then the live model cache; reads only, never a fetch.

    Args:
        provider_name: The provider's config name.
        model_id: The model id.

    Returns:
        The window, or None when neither source knows it.
    """
    return _bundled_context_window(model_id) or cache.cached_context_window(
        provider_name, (model_id, normalize_model_id(model_id))
    )


def compaction_thresholds(
    provider_name: str,
    model_id: str,
    *,
    drop_override: int | None,
    summarise_override: int | None,
) -> tuple[int, int]:
    """Return the effective (drop_at_chars, summarise_at_chars) thresholds.

    Explicit config wins (the validator requires both or neither); otherwise the model's
    window sizes them, and an unknown window takes the fixed defaults.

    Args:
        provider_name: The provider's config name.
        model_id: The model id.
        drop_override: The configured `context.drop_at_chars`, or None.
        summarise_override: The configured `context.summarise_at_chars`, or None.

    Returns:
        The two thresholds in chars.
    """
    if drop_override is not None and summarise_override is not None:
        return drop_override, summarise_override
    ctx = context_window(provider_name, model_id)
    if ctx is None or ctx <= 0:
        return _FALLBACK_DROP_CHARS, _FALLBACK_SUMMARISE_CHARS
    drop = int(ctx * _CHARS_PER_TOKEN * _DROP_FRACTION)
    summarise = max(drop + 1, (ctx - _RESERVE_TOKENS) * _CHARS_PER_TOKEN)
    return drop, summarise


# Families with a measured decompose-first win (bench/coreagent/FINDINGS.md: mistral-small-3.2-24b
# textkit +0.53, rpn +0.13, ledger +0.18); the other benched models paid a 2-4x iteration tax.
DECOMPOSE_WIN_MODEL_FAMILIES: tuple[str, ...] = ("mistral-small-3.2",)


def decompose_default(model_id: str) -> bool:
    """Return whether `prompt.decompose = "auto"` resolves to on for a model.

    Family matching ignores the org prefix and any date or `:tag` suffix.

    Args:
        model_id: The model id.

    Returns:
        True when the model's family has a measured decompose-first win.
    """
    family = normalize_model_id(model_id).rsplit("/", 1)[-1].lower()
    return family.startswith(DECOMPOSE_WIN_MODEL_FAMILIES)


def role_effort(cfg: Config, role: kinds.RoleName) -> str | None:
    """Return the reasoning effort a role's calls carry.

    The configured `[models.<role>].effort` when set, else each wire's default: `low` for an
    OpenAI-compatible reasoning model, `off` (no thinking) for Anthropic.

    Args:
        cfg: The effective config.
        role: The role.

    Returns:
        The effort, or None when agent6 sends none and the provider's default decides.
    """
    rm = cfg.models.resolve(role)
    if rm is None:
        return None
    entry = cfg.providers.get(rm.provider)
    if entry is None:
        return None
    match entry.api_format:
        case "anthropic":
            return rm.effort or "off"
        case "chatgpt" | "claude_code":
            return rm.effort
        case _:
            return openai.sent_reasoning_effort(
                rm.model,
                rm.effort,
                direct_openai=openai.is_openai_direct_host(entry.base_url, entry.deployment),
            )


def resolved_adaptive_values(cfg: Config) -> dict[str, object]:
    """Return the config leaves whose effective value resolves at runtime.

    `config show` and the config pages print these in place of the adaptive placeholder.

    Args:
        cfg: The effective config.

    Returns:
        The compaction thresholds, the auto decompose decision and each role's unset effort,
        by leaf; empty when nothing resolves.
    """
    out: dict[str, object] = {}
    for role in ("worker", "reviewer", "planner"):
        role_model = cfg.models.resolve(role)
        if role_model is None or role_model.effort is not None:
            continue
        if (effort := role_effort(cfg, role)) is not None:
            out[f"models.{role}.effort"] = effort
    rm = cfg.models.resolve("worker")
    if rm is None:
        return out
    drop, summarise = compaction_thresholds(
        rm.provider,
        rm.model,
        drop_override=cfg.context.drop_at_chars,
        summarise_override=cfg.context.summarise_at_chars,
    )
    out["context.drop_at_chars"] = drop
    out["context.summarise_at_chars"] = summarise
    if cfg.prompt.decompose == "auto":
        out["prompt.decompose"] = "on" if decompose_default(rm.model) else "off"
    return out
