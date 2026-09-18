#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xiaotian/navigation/habitat-lab"
PYTHON_BIN="/home/xiaotian/miniconda3/envs/habitat/bin/python"
DATA_ROOT="/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/mp3d_narrow_v1"
TEACHER_ROOT="$DATA_ROOT/e2e_teacher"
OUTPUT_ROOT="$REPO_ROOT/examples/narrow_passage_rl/results/degnav_e2e_mp3d_narrow_v1_20260904"
WAIT_PID="${WAIT_PID:-}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"

if [[ -n "$WAIT_PID" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        printf '%s waiting for official-baseline queue PID %s\n' "$(date --iso-8601=seconds)" "$WAIT_PID"
        sleep 30
    done
fi

run_19d_eval() {
    local name="$1"
    shift
    local output_csv="$OUTPUT_ROOT/${name}_episodes.csv"
    if [[ -f "$output_csv" ]] && [[ "$(wc -l < "$output_csv")" -eq 81 ]]; then
        printf '%s skip completed 19-D evaluation %s\n' "$(date --iso-8601=seconds)" "$name"
        return
    fi
    "$PYTHON_BIN" examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
        --data-path "$DATA_ROOT/{split}/{split}.json.gz" \
        --split val \
        --num-episodes -1 \
        --max-steps 500 \
        --allow-sliding 0 \
        --agent-radius 0.18 \
        --output-csv "$output_csv" \
        "$@" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${name}.log"
}

# Same MP3D-derived held-out episodes used later by E2E.  These runs provide
# the required 19-D reference and the two memory comparators.
run_19d_eval geometry_19d --use-memory 0
run_19d_eval knn_memory_19d --use-memory 1 --memory-backend knn
run_19d_eval degnav_memory_19d --use-memory 1 --memory-backend degnav

"$PYTHON_BIN" examples/narrow_passage_rl/summarize_formal_mp3d.py \
    --dataset "$DATA_ROOT/val/val.json.gz" \
    --output-dir "$OUTPUT_ROOT/19d_comparison" \
    --input "Geometry-19D=$OUTPUT_ROOT/geometry_19d_episodes.csv" \
    --input "kNN-memory-19D=$OUTPUT_ROOT/knn_memory_19d_episodes.csv" \
    --input "DEGNAV-memory-19D=$OUTPUT_ROOT/degnav_memory_19d_episodes.csv"

teacher_split_complete() {
    local split="$1"
    local expected="$2"
    "$PYTHON_BIN" - "$TEACHER_ROOT/depth/$split" "$expected" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = int(sys.argv[2])
metadata_path = root / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(1)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
valid = (
    int(metadata.get("episodes", -1)) == expected
    and len(list(root.glob("*.npz"))) == expected
    and all(int(metadata.get("mode_counts", {}).get(name, 0)) > 0
            for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT"))
    and metadata.get("teacher_rule", {}).get(
        "explore_if_body_margin_below_m"
    ) == 0.3
    and metadata.get("teacher_rule", {}).get(
        "reject_if_infeasible_and_observed_failure"
    ) is True
    and int(metadata.get("history_storage_version", -1)) == 2
)
if valid:
    for path in root.glob("*.npz"):
        with np.load(path, allow_pickle=False) as episode:
            modes = np.asarray(episode["mode"])
            outcomes = np.asarray(episode["action_outcome"])
        if not np.allclose(outcomes[0], 0.0):
            valid = False
            break
        if len(modes) > 1 and (
            np.any(outcomes[1:, :4].argmax(axis=1) != modes[:-1])
            or np.any(~np.isclose(outcomes[1:, :4].sum(axis=1), 1.0))
        ):
            valid = False
            break
raise SystemExit(0 if valid else 1)
PY
}

training_complete() {
    local directory="$1"
    "$PYTHON_BIN" - "$directory" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (root / "best.pt", root / "last.pt", root / "resolved_config.json")
history_path = root / "training_history.csv"
if not all(path.is_file() for path in required) or not history_path.is_file():
    raise SystemExit(1)
with history_path.open(newline="", encoding="utf-8") as handle:
    history = list(csv.DictReader(handle))
config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
valid = (
    len(history) == 20
    and int(history[-1]["epoch"]) == 20
    and int(config.get("epochs", -1)) == 20
    and config.get("actor_uses_19d") is False
    and config.get("actor_uses_handdesigned_task_memory") is False
)
raise SystemExit(0 if valid else 1)
PY
}

evaluation_complete() {
    local directory="$1"
    local expected="$2"
    "$PYTHON_BIN" - "$directory" "$expected" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
summary_path = root / "summary.json"
episodes_path = root / "episodes.csv"
if not summary_path.is_file() or not episodes_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
with episodes_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
valid = (
    int(summary.get("episodes", -1)) == expected
    and len(rows) == expected
    and summary.get("actor_uses_19d") is False
    and summary.get("actor_uses_handdesigned_task_memory") is False
    and summary.get("evaluator_uses_19d_for_task_measures") is True
)
raise SystemExit(0 if valid else 1)
PY
}

for split in train val; do
    expected=400
    if [[ "$split" == "val" ]]; then expected=80; fi
    if teacher_split_complete "$split" "$expected"; then
        printf '%s skip completed E2E teacher split %s\n' "$(date --iso-8601=seconds)" "$split"
    else
        "$PYTHON_BIN" examples/narrow_passage_rl/collect_habitat_e2e_teacher.py \
            --split "$split" \
            --dataset "$DATA_ROOT/$split/$split.json.gz" \
            --output-dir "$TEACHER_ROOT" \
            --input-type depth \
            --image-size 128 \
            --agent-radius 0.18 \
            --seed 1701 \
            --resume \
            2>&1 | tee "$OUTPUT_ROOT/logs/collect_${split}.log"
    fi
done

for seed in 1701 1702 1703; do
    seed_dir="$OUTPUT_ROOT/seed${seed}"
    mkdir -p "$seed_dir"
    if training_complete "$seed_dir/train"; then
        printf '%s skip completed E2E training seed %s\n' "$(date --iso-8601=seconds)" "$seed"
    else
        "$PYTHON_BIN" examples/narrow_passage_rl/train_degnav_e2e_bc.py \
            --data-root "$TEACHER_ROOT" \
            --input-type depth \
            --output-dir "$seed_dir/train" \
            --epochs 20 \
            --batch-size 2 \
            --sequence-length 16 \
            --stride 8 \
            --hidden-size 256 \
            --class-balance-power 1.0 \
            --seed "$seed" \
            2>&1 | tee "$OUTPUT_ROOT/logs/train_seed${seed}.log"
    fi

    if evaluation_complete "$seed_dir/preflight" 3; then
        printf '%s skip completed E2E preflight seed %s\n' "$(date --iso-8601=seconds)" "$seed"
    else
        "$PYTHON_BIN" examples/narrow_passage_rl/eval_degnav_e2e.py \
            --checkpoint "$seed_dir/train/best.pt" \
            --dataset "$DATA_ROOT/val/val.json.gz" \
            --output-dir "$seed_dir/preflight" \
            --num-episodes 3 \
            --seed "$seed" \
            2>&1 | tee "$OUTPUT_ROOT/logs/eval_preflight_seed${seed}.log"
    fi

    if evaluation_complete "$seed_dir/eval" 80; then
        printf '%s skip completed E2E evaluation seed %s\n' "$(date --iso-8601=seconds)" "$seed"
    else
        "$PYTHON_BIN" examples/narrow_passage_rl/eval_degnav_e2e.py \
            --checkpoint "$seed_dir/train/best.pt" \
            --dataset "$DATA_ROOT/val/val.json.gz" \
            --output-dir "$seed_dir/eval" \
            --seed "$seed" \
            2>&1 | tee "$OUTPUT_ROOT/logs/eval_seed${seed}.log"
    fi
done

"$PYTHON_BIN" examples/narrow_passage_rl/summarize_degnav_e2e.py \
    --input-dir "$OUTPUT_ROOT" \
    --teacher-root "$TEACHER_ROOT"
printf '%s DEGNAV-E2E sequence complete\n' "$(date --iso-8601=seconds)"
