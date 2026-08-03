# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Session identity and on-disk state: friendly session ids + prefix resolution
(`agent6.sessions.id`), the filesystem layout of one session's state directory
(`agent6.sessions.layout`), the single-writer flock (`agent6.sessions.lock`), and the
front-end<->workflow answer-file contract (`agent6.sessions.ipc`). All leaves;
import the submodules directly."""

from __future__ import annotations
