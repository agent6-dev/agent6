# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The telemetry exporter: a reader of session journals beside the four front-ends.

It imports `viewmodel`, `sessions` and `config`, and no layer below `ui`.
`otlp` and `prometheus` encode the projection's records, and neither opens a socket.
Main modules: `export`, `otlp`, `prometheus`.
"""
