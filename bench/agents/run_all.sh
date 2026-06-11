#!/usr/bin/env bash
# Sequential so the OpenRouter usage delta cleanly attributes cost per run.
set -u
for task in go-logwindow rust-ratelimit go-kvstore-debug; do
  for agent in agent6 aider opencode claude; do
    echo "=== $(date +%H:%M:%S) $task / $agent ==="
    bash "$(dirname "${BASH_SOURCE[0]}")/run_one.sh" "$task" "$agent"
  done
done
echo "ALL DONE"
