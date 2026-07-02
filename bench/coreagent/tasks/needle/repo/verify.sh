#!/usr/bin/env bash
# Verify command for the agent. Stdlib unittest only (no pytest dependency).
set -e
cd "$(dirname "$0")"
exec python3 -m unittest test_report -v
