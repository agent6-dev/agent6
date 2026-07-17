# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run identity and on-disk state: friendly run ids + prefix resolution
(`agent6.runs.id`), the filesystem layout of one run's state directory
(`agent6.runs.layout`), the single-writer flock (`agent6.runs.lock`), and the
front-end<->workflow answer-file contract (`agent6.runs.bridge`). All leaves;
import the submodules directly."""

from __future__ import annotations
