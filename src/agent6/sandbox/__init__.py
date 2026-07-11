# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox subsystem: Landlock + jail launcher."""

from __future__ import annotations

from agent6.sandbox.broker import (
    BrokerHandle,
    EgressBrokerError,
    Endpoint,
    enter_network_isolation,
    start_egress_broker,
)
from agent6.sandbox.host_spawn import HostSpawner, fork_host_spawner
from agent6.sandbox.jail import JailUnavailableError, run_in_jail, strict_namespaces_work
from agent6.sandbox.landlock import (
    LandlockError,
    LandlockNotSupportedError,
    apply_agent_landlock,
    landlock_abi,
)

__all__ = [
    "BrokerHandle",
    "EgressBrokerError",
    "Endpoint",
    "HostSpawner",
    "JailUnavailableError",
    "LandlockError",
    "LandlockNotSupportedError",
    "apply_agent_landlock",
    "enter_network_isolation",
    "fork_host_spawner",
    "landlock_abi",
    "run_in_jail",
    "start_egress_broker",
    "strict_namespaces_work",
]
