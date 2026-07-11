#!/usr/bin/env bash
# Verify: rebuild the data feed, then run the whole suite (stdlib unittest).
set -e
cd "$(dirname "$0")"
python3 tools/gen_catalog.py
exec python3 -m unittest discover -s . -p 'test_*.py' -v
