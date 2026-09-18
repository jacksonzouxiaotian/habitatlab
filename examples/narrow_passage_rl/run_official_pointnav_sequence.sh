#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xiaotian/navigation/habitat-lab"
PYTHON_BIN="/home/xiaotian/miniconda3/envs/habitat/bin/python"
EVALUATOR="$REPO_ROOT/examples/narrow_passage_rl/eval_official_pointnav_baselines.py"
OUTPUT_ROOT="$REPO_ROOT/examples/narrow_passage_rl/results/official_pointnav_mp3d_v1_20260904"
WAIT_PID="${WAIT_PID:-}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/preflight"

if [[ -n "$WAIT_PID" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        printf '%s waiting for GPU owner PID %s\n' "$(date --iso-8601=seconds)" "$WAIT_PID"
        sleep 30
    done
fi

run_eval() {
    local method="$1"
    shift
    mkdir -p "$OUTPUT_ROOT/$method"
    "$PYTHON_BIN" "$EVALUATOR" \
        --method "$method" \
        --output-dir "$OUTPUT_ROOT" \
        "$@" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${method}.log"
}

method_complete() {
    local method="$1"
    local expected_seeds="$2"
    "$PYTHON_BIN" - "$OUTPUT_ROOT/$method/summary.csv" \
        "$OUTPUT_ROOT/$method/episodes.csv" "$expected_seeds" <<'PY'
import csv
import sys
from pathlib import Path

summary_path, episode_path = map(Path, sys.argv[1:3])
expected_seeds = int(sys.argv[3])
if not summary_path.is_file() or not episode_path.is_file():
    raise SystemExit(1)
with summary_path.open(newline="", encoding="utf-8") as handle:
    summaries = list(csv.DictReader(handle))
with episode_path.open(newline="", encoding="utf-8") as handle:
    episodes = list(csv.DictReader(handle))
valid = (
    len(summaries) == expected_seeds
    and all(int(row["episodes"]) == 495 for row in summaries)
    and len(episodes) == 495 * expected_seeds
)
raise SystemExit(0 if valid else 1)
PY
}

run_if_missing() {
    local method="$1"
    local expected_seeds="$2"
    shift 2
    if method_complete "$method" "$expected_seeds"; then
        printf '%s skip completed method %s\n' "$(date --iso-8601=seconds)" "$method"
    else
        run_eval "$method" "$@"
    fi
}

# Learned policies are intentionally first.  A three-episode integration gate
# catches sensor/checkpoint incompatibilities before the 495-episode run.
if [[ ! -s "$OUTPUT_ROOT/preflight/pointnav_ppo/summary.csv" ]]; then
    "$PYTHON_BIN" "$EVALUATOR" \
        --method pointnav_ppo \
        --num-episodes 3 \
        --seed 1701 \
        --output-dir "$OUTPUT_ROOT/preflight" \
        2>&1 | tee "$OUTPUT_ROOT/logs/pointnav_ppo_preflight.log"
fi
run_if_missing pointnav_ppo 3 --seed 1701 --seed 1702 --seed 1703

if [[ ! -s "$OUTPUT_ROOT/preflight/ddppo/summary.csv" ]]; then
    "$PYTHON_BIN" "$EVALUATOR" \
        --method ddppo \
        --num-episodes 3 \
        --seed 1701 \
        --output-dir "$OUTPUT_ROOT/preflight" \
        2>&1 | tee "$OUTPUT_ROOT/logs/ddppo_preflight.log"
fi
run_if_missing ddppo 3 --seed 1701 --seed 1702 --seed 1703

run_if_missing goal_follower 1 --seed 1701
run_if_missing random 3 --seed 1701 --seed 1702 --seed 1703
run_if_missing forward_only 1 --seed 1701
run_if_missing random_forward 3 --seed 1701 --seed 1702 --seed 1703
run_if_missing shortest_path_follower 1 --seed 1701

"$PYTHON_BIN" \
    "$REPO_ROOT/examples/narrow_passage_rl/summarize_official_pointnav_baselines.py" \
    --input-dir "$OUTPUT_ROOT"

printf '%s official PointNav sequence complete\n' "$(date --iso-8601=seconds)"
