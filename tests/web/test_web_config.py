# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[web]` config section: secure by default (loopback), non-loopback opt-in."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent6.config import Config, WebConfig


def test_web_defaults_are_loopback() -> None:
    w = WebConfig()
    assert w.host == "127.0.0.1"
    assert w.port == 8901
    assert w.allow_non_loopback is False


def test_config_carries_web_section() -> None:
    assert Config().web.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "LOCALHOST"])
def test_loopback_hosts_need_no_optin(host: str) -> None:
    assert WebConfig(host=host).host == host


def test_non_loopback_rejected_without_optin() -> None:
    with pytest.raises(ValidationError, match="allow_non_loopback"):
        WebConfig(host="0.0.0.0")


def test_non_loopback_allowed_with_optin() -> None:
    w = WebConfig(host="0.0.0.0", allow_non_loopback=True)
    assert w.host == "0.0.0.0"
