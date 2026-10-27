# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The telemetry exporter: a reader of session journals beside the four front-ends.

`_otlp` encodes the projection's records as OTLP/HTTP JSON bodies and `_prometheus` as the
text exposition; neither opens a socket.
"""
