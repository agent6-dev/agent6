# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Model-capability registry: what agent6 knows about specific models.

Curated, evidence-backed model facts consumed on the run path: context
windows (to size adaptive compaction) and which models measurably benefit
from decompose-first prompting. The live model cache (`models.cache`)
covers context windows for models the bundled table omits; capability
entries here change only with bench evidence (bench/coreagent/FINDINGS.md).
Everything is read-only and best-effort: lookups never raise and never
touch the network.
"""

from __future__ import annotations

import re

from agent6.config import Config
from agent6.models.cache import cached_context_window

__all__ = [
    "BUNDLED_CONTEXT_WINDOWS",
    "compaction_thresholds",
    "context_window",
    "resolved_adaptive_values",
]

# Curated context windows (TOKENS) for models we have tested or that are
# popular, used to size adaptive compaction without a network round-trip and to
# cover providers whose listing omits the window (Anthropic's /models does not
# report it). The live cache (``context_length`` from the provider listing)
# covers everything else; this table just guarantees good behaviour offline and
# on the first run before any listing has been fetched, and wins over the live
# cache when both know a model. Keep ids canonical (no date/`:tag` suffix --
# ``_normalize_model_id`` strips those before matching).
BUNDLED_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic (standard window; the 1M-context beta is opt-in, so pin it
    # explicitly in [workflow] if you enable it).
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    # OpenRouter open-weights we bench against (cross-checked against the live
    # listing).
    "moonshotai/kimi-k2.6": 262_144,
    "moonshotai/kimi-k2": 131_072,
    "qwen/qwen3-coder": 1_048_576,
    "qwen/qwen3-coder-30b-a3b-instruct": 160_000,
    "z-ai/glm-4.6": 202_752,
    "z-ai/glm-5.2": 1_048_576,
    "deepseek/deepseek-v3.2-exp": 163_840,
}

# Adaptive sizing. tokens ~= chars/4 (matches the loop's ``context_chars``
# approximation). Tier-1 elides old tool_results once they pass ~45% of the
# window; tier-2 summarises+restarts at ~80%, leaving headroom for the next
# turn's output and the summary call itself.
_CHARS_PER_TOKEN = 4
_DROP_FRACTION = 0.45
_SUMMARISE_FRACTION = 0.80
# Used when the window is unknown: the historical fixed defaults, so behaviour
# is unchanged for unsizable models. Mirrors workflows._compaction
# DROP_BLOCKS_AT_CHARS / SUMMARISE_AT_CHARS.
_FALLBACK_DROP_CHARS = 256_000
_FALLBACK_SUMMARISE_CHARS = 768_000


def _normalize_model_id(model_id: str) -> str:
    """Strip a trailing ``-YYYYMMDD`` snapshot date or ``:tag`` so dated/tagged
    ids (``claude-haiku-4-5-20251001``, ``qwen/qwen3-coder:free``) match the
    canonical bundled key."""
    base = model_id.split(":", 1)[0]
    return re.sub(r"-\d{8}$", "", base)


def _bundled_context_window(model_id: str) -> int | None:
    if model_id in BUNDLED_CONTEXT_WINDOWS:
        return BUNDLED_CONTEXT_WINDOWS[model_id]
    return BUNDLED_CONTEXT_WINDOWS.get(_normalize_model_id(model_id))


def context_window(provider_name: str, model_id: str) -> int | None:
    """Best-effort context window (tokens) for a configured model. Never raises.

    Bundled table (curated; tested models + Anthropic) first, then the live
    model cache (``context_length`` from the provider listing, populated by
    completion / ``agent6 model``), then None. Reads only -- never triggers a
    network fetch -- so it is safe and fast on the run path.
    """
    return _bundled_context_window(model_id) or cached_context_window(
        provider_name, (model_id, _normalize_model_id(model_id))
    )


def compaction_thresholds(
    provider_name: str,
    model_id: str,
    *,
    drop_override: int | None,
    summarise_override: int | None,
) -> tuple[int, int]:
    """Effective ``(compact_drop_at_chars, compact_summarise_at_chars)``.

    Explicit config wins (both set, by construction -- the config validator
    requires both-or-neither). Otherwise size from the model's context window
    (tier-1 ~45%, tier-2 ~80%); if the window is unknown, the historical fixed
    defaults. Never raises.
    """
    if drop_override is not None and summarise_override is not None:
        return drop_override, summarise_override
    ctx = context_window(provider_name, model_id)
    if ctx is None or ctx <= 0:
        return _FALLBACK_DROP_CHARS, _FALLBACK_SUMMARISE_CHARS
    drop = int(ctx * _CHARS_PER_TOKEN * _DROP_FRACTION)
    summarise = int(ctx * _CHARS_PER_TOKEN * _SUMMARISE_FRACTION)
    return drop, summarise


def resolved_adaptive_values(cfg: Config) -> dict[str, object]:
    """Config settings whose effective value is resolved at runtime, so a UI
    (`config show`, the TUI/web config page) can display the real number rather
    than the unset/adaptive placeholder. Currently the adaptive compaction
    thresholds, sized from the worker model's context window; empty when no
    worker model is configured."""
    rm = cfg.models.resolve("worker")
    if rm is None:
        return {}
    drop, summarise = compaction_thresholds(
        rm.provider,
        rm.model,
        drop_override=cfg.context.drop_at_chars,
        summarise_override=cfg.context.summarise_at_chars,
    )
    return {
        "context.drop_at_chars": drop,
        "context.summarise_at_chars": summarise,
    }
