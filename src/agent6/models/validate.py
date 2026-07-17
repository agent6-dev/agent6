# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn validation for `/parallel` model specs: catch a bogus model id
before any lane is cloned or spawned, with a did-you-mean.

A `/parallel` spec (and `run --parallel`) names one model per lane. A typo
(`opsu`, `moonshotai/kimi-k2.7`) would otherwise spawn a lane that dies at its
first provider call -- late and wasteful. Every lane runs on the WORKER
provider by construction (the lane config overrides only the model; see
`_write_lane_config` in ui/cli/parallel.py), so the universe a spec model is
checked against is worker-scoped: the worker's configured model unioned with
the worker provider's on-disk model-list cache snapshot. A sibling provider's
catalog is unrunnable in a lane, so it can neither accept a model nor vouch
that validation is possible. On a miss, either refuse with the closest matches
(the worker provider has a cache to check against) or proceed with a warning
(no cache: a fresh/offline machine must never be blocked on a regenerable
cache).

Cache-only, never a network fetch, never raises. Lives in the models layer so
all three front-ends and the coordinator's group dispatcher share one policy
without a UI or workflows dependency.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from agent6.config import Config
from agent6.models.cache import cached_models

__all__ = [
    "ModelValidation",
    "known_models",
    "refusal_message",
    "validate_spec_models",
    "warning_message",
]

_MAX_SUGGESTIONS = 3


def known_models(cfg: Config) -> set[str]:
    """Every model id a `/parallel` lane can actually run, without touching the
    network: the worker's configured model unioned with the worker provider's
    on-disk model-list cache snapshot (lanes inherit the worker provider; only
    the model is overridden per lane). Empty when no worker role is set."""
    worker = cfg.models.worker
    if worker is None:
        return set()
    return {worker.model} | set(cached_models(worker.provider))


@dataclass(frozen=True, slots=True)
class ModelValidation:
    """Outcome of checking a spec's per-lane models against `known_models`.

    `unknown` lists the named models not found (deduped, in spec order);
    `suggestions` maps each to its closest known ids; `can_validate` is True when
    the worker provider has a cache to check against. `refused` (unknown +
    can_validate) is a hard stop; `warned` (unknown + no cache) proceeds -- an
    offline machine is never blocked on a regenerable cache."""

    unknown: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]]
    can_validate: bool

    @property
    def refused(self) -> bool:
        return bool(self.unknown) and self.can_validate

    @property
    def warned(self) -> bool:
        return bool(self.unknown) and not self.can_validate


def _close_ids(typo: str, pool: list[str], bare_to_full: dict[str, list[str]]) -> tuple[str, ...]:
    """Closest known ids to *typo*: matched against the full provider-prefixed ids
    AND against the un-prefixed model segment (the part after the last `/`). The
    bare match catches a short nickname near-miss (`glm`, `kimi-typo`) that scores
    below difflib's cutoff against a full id, because the provider prefix dominates
    the ratio (`glm` vs `z-ai/glm-4.6`). Bare hits map back to full ids (what the
    operator must actually pass); full-id hits keep priority, capped overall."""
    close = list(difflib.get_close_matches(typo, pool, n=_MAX_SUGGESTIONS))
    bare_typo = typo.rsplit("/", 1)[-1]
    for bare in difflib.get_close_matches(bare_typo, sorted(bare_to_full), n=_MAX_SUGGESTIONS):
        close.extend(full for full in bare_to_full[bare] if full not in close)
    return tuple(close[:_MAX_SUGGESTIONS])


def validate_spec_models(models: list[str | None], cfg: Config) -> ModelValidation:
    """Check per-lane *models* (a `parse_spec` result; `None` = the worker model,
    skipped) against `known_models`. Cache-only, never raises."""
    known = known_models(cfg)
    pool = sorted(known)
    bare_to_full: dict[str, list[str]] = {}
    for full in pool:
        bare_to_full.setdefault(full.rsplit("/", 1)[-1], []).append(full)
    unknown: list[str] = []
    suggestions: dict[str, tuple[str, ...]] = {}
    for model in models:
        if model is None or model in known or model in suggestions:
            continue
        suggestions[model] = _close_ids(model, pool, bare_to_full)
        unknown.append(model)
    worker = cfg.models.worker
    can_validate = worker is not None and bool(cached_models(worker.provider))
    return ModelValidation(
        unknown=tuple(unknown), suggestions=suggestions, can_validate=can_validate
    )


def refusal_message(v: ModelValidation, *, directive: bool) -> str:
    """The refusal text for an `unknown + can_validate` result: one line per
    unknown model with its closest matches. On a directive surface (the composers
    and the coordinator, where the same token could be task text) add the backtick
    hint."""
    lines = [
        f"unknown model {model!r} in /parallel spec"
        + (f"; closest: {', '.join(close)}" if (close := v.suggestions.get(model, ())) else "")
        for model in v.unknown
    ]
    if directive:
        lines.append("backtick the word if you meant it as task text.")
    return "\n".join(lines)


def warning_message(v: ModelValidation) -> str:
    """The single warning line for an `unknown + not can_validate` result: no
    cache to check against, so proceed but name the unvalidated model(s)."""
    return (
        f"unvalidated model(s) {', '.join(v.unknown)}: no cached model list for the"
        " worker provider to check against; proceeding (run `agent6 model` to refresh)."
    )
