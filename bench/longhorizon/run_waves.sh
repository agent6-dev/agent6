#!/usr/bin/env bash
# Full first-findings matrix for the longhorizon bench. Run from this dir.
# Wave 1: compaction A/B (does window32k hurt, and where does the damage show).
# Wave 2: memory A/B (does the <memories> block make leg 2 cheaper/better).
# Roughly $15-25 of OpenRouter spend at reps=3; scale with --reps/--budget-scale.
set -euo pipefail
cd "$(dirname "$0")"

QWEN=qwen/qwen3-coder-30b-a3b-instruct
KIMI=moonshotai/kimi-k2.6

python3 run.py --model "$QWEN" --tasks stylebook,relay \
    --conditions baseline,window32k --reps 3 --parallel 4 --label wave1-qwen
python3 run.py --model "$KIMI" --tasks stylebook,relay \
    --conditions baseline,window32k --reps 3 --parallel 4 --label wave1-kimi

python3 run.py --model "$QWEN" --tasks orchard \
    --conditions baseline,fresh_state --reps 4 --parallel 4 --label mem1-qwen
python3 run.py --model "$KIMI" --tasks orchard \
    --conditions baseline,fresh_state --reps 4 --parallel 4 --label mem1-kimi

python3 stats.py results/wave1-*.jsonl --components
python3 stats.py results/mem1-*.jsonl
