#!/usr/bin/env bash
# Verify: rebuild every data feed, then run the whole suite (stdlib unittest).
set -e
cd "$(dirname "$0")"
for gen in tools/gen_*.py; do python3 "$gen"; done
exec python3 -m unittest discover -s . -p 'test_*.py' -v
