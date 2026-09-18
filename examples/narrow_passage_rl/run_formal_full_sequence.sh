#!/usr/bin/env bash
set -Eeuo pipefail

HABITAT_ROOT="${HABITAT_ROOT:-/home/xiaotian/navigation/habitat-lab}"
NAVILA_EVAL_ROOT="${NAVILA_EVAL_ROOT:-/home/xiaotian/vla/NaVILA/evaluation}"
VLN_DATA_ROOT="${VLN_DATA_ROOT:-/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data}"
MODEL_PATH="${MODEL_PATH:-/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f}"
FORMAL_ROOT="${FORMAL_ROOT:-/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/formal_runs}"
UNSEEN_ROOT="${UNSEEN_ROOT:-${FORMAL_ROOT}/navila_r2r_val_unseen_full_4bit_20260831}"
SEEN_ROOT="${SEEN_ROOT:-${FORMAL_ROOT}/navila_r2r_val_seen_full_4bit_20260901}"
DEGNAV_UNSEEN_ROOT="${DEGNAV_UNSEEN_ROOT:-${FORMAL_ROOT}/navila_degnav_r2r_val_unseen_full_4bit_20260901}"
DEGNAV_SEEN_ROOT="${DEGNAV_SEEN_ROOT:-${FORMAL_ROOT}/navila_degnav_r2r_val_seen_full_4bit_20260901}"
DEGNAV_SMOKE_ROOT="${DEGNAV_SMOKE_ROOT:-${FORMAL_ROOT}/navila_degnav_r2r_val_unseen_online_smoke_20260901}"
MP3D_ROOT="${MP3D_ROOT:-${VLN_DATA_ROOT}/scene_datasets/mp3d}"
MP3D_RUN_ROOT="${MP3D_RUN_ROOT:-${FORMAL_ROOT}/mp3d_narrow_full_20260901}"
DIRECT_MODEL="${DIRECT_MODEL:-${HABITAT_ROOT}/examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/unified_direct_ppo_seed0_3m_current/ppo_narrow_passage_v2.zip}"
RECURRENT_MODEL="${RECURRENT_MODEL:-${HABITAT_ROOT}/examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_ppo_seed0_lstm64_direct_init_finetune100k/recurrent_ppo_narrow_passage_v2.zip}"
POINTNAV_MODEL="${POINTNAV_MODEL:-${HABITAT_ROOT}/data_old/new_checkpoints/ckpt.49.pth}"
MP3D_FINETUNE_STEPS="${MP3D_FINETUNE_STEPS:-500000}"
MP3D_DIRECT_DIR="${MP3D_RUN_ROOT}/checkpoints/direct_control_ppo_seed0_mp3d_finetune"
MP3D_RECURRENT_DIR="${MP3D_RUN_ROOT}/checkpoints/recurrent_ppo_seed0_mp3d_finetune"
MP3D_DIRECT_MODEL="${MP3D_DIRECT_DIR}/ppo_habitat_narrow_passage.zip"
MP3D_RECURRENT_MODEL="${MP3D_RECURRENT_DIR}/recurrent_ppo_habitat_narrow_passage.zip"

QUEUE_LOG="${FORMAL_ROOT}/formal_full_sequence_20260901.log"
mkdir -p "${FORMAL_ROOT}" "${SEEN_ROOT}/logs" "${MP3D_RUN_ROOT}/logs"

stamp() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

run_logged() {
    local sentinel="$1"
    local name="$2"
    local log_file="$3"
    shift 3
    if [[ -s "${sentinel}" ]]; then
        stamp "SKIP ${name}: ${sentinel}"
        return 0
    fi
    stamp "START ${name}"
    "$@" >"${log_file}" 2>&1
    if [[ ! -s "${sentinel}" ]]; then
        stamp "FAIL ${name}: expected ${sentinel}"
        return 1
    fi
    stamp "DONE ${name}: ${sentinel}"
}

wait_for_unseen() {
    local sentinel="${UNSEEN_ROOT}/aggregate_metrics.json"
    while [[ ! -s "${sentinel}" ]]; do
        if ! pgrep -f "RESULTS_DIR ${UNSEEN_ROOT}" >/dev/null; then
            stamp "FAIL prerequisite: val_unseen process ended without ${sentinel}"
            return 1
        fi
        stamp "WAIT val_unseen full evaluation"
        sleep 30
    done
    stamp "READY val_unseen: ${sentinel}"
}

run_navila_seen() {
    local chunk_idx result_file log_file
    cd "${NAVILA_EVAL_ROOT}"
    for chunk_idx in {0..9}; do
        result_file="${SEEN_ROOT}/navila-llama3-8b-8f/VLN-CE-v1/val_seen/val_seen_10-${chunk_idx}.json"
        log_file="${SEEN_ROOT}/logs/chunk_${chunk_idx}.log"
        if [[ -s "${result_file}" ]]; then
            stamp "SKIP NaVILA val_seen chunk ${chunk_idx}"
            continue
        fi
        stamp "START NaVILA val_seen chunk ${chunk_idx}"
        CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n navila-eval \
            python run.py \
            --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
            --run-type eval \
            --num-chunks 10 \
            --chunk-idx "${chunk_idx}" \
            EVAL_CKPT_PATH_DIR "${MODEL_PATH}" \
            EVAL.SPLIT val_seen \
            EVAL.EPISODE_COUNT -1 \
            RESULTS_DIR "${SEEN_ROOT}" \
            VIDEO_OPTION "[]" >"${log_file}" 2>&1
        [[ -s "${result_file}" ]]
        stamp "DONE NaVILA val_seen chunk ${chunk_idx}"
    done
    conda run -n navila-eval python scripts/eval_jsons.py \
        "${SEEN_ROOT}/navila-llama3-8b-8f/VLN-CE-v1/val_seen" 10 \
        >"${SEEN_ROOT}/aggregate_metrics.json"
    stamp "DONE NaVILA val_seen aggregate"
}

run_navila_degnav() {
    local split="$1"
    local run_root="$2"
    local chunk_idx result_file log_file
    mkdir -p "${run_root}/logs"
    cd "${NAVILA_EVAL_ROOT}"
    for chunk_idx in {0..9}; do
        result_file="${run_root}/navila-llama3-8b-8f/VLN-CE-v1/${split}/${split}_10-${chunk_idx}.json"
        log_file="${run_root}/logs/chunk_${chunk_idx}.log"
        if [[ -s "${result_file}" ]]; then
            stamp "SKIP NaVILA+DEGNAV ${split} chunk ${chunk_idx}"
            continue
        fi
        stamp "START NaVILA+DEGNAV ${split} chunk ${chunk_idx}"
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="${HABITAT_ROOT}:${PYTHONPATH:-}" \
        conda run --no-capture-output -n navila-eval \
            python run.py \
            --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
            --run-type eval \
            --num-chunks 10 \
            --chunk-idx "${chunk_idx}" \
            EVAL_CKPT_PATH_DIR "${MODEL_PATH}" \
            EVAL.SPLIT "${split}" \
            EVAL.EPISODE_COUNT -1 \
            EVAL.DEGNAV_ADAPTER.ENABLED True \
            RESULTS_DIR "${run_root}" \
            VIDEO_OPTION "[]" >"${log_file}" 2>&1
        [[ -s "${result_file}" ]]
        stamp "DONE NaVILA+DEGNAV ${split} chunk ${chunk_idx}"
    done
    conda run -n navila-eval python scripts/eval_jsons.py \
        "${run_root}/navila-llama3-8b-8f/VLN-CE-v1/${split}" 10 \
        >"${run_root}/aggregate_metrics.json"
    stamp "DONE NaVILA+DEGNAV ${split} aggregate"
}

run_navila_degnav_smoke() {
    local result_file="${DEGNAV_SMOKE_ROOT}/navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json"
    mkdir -p "${DEGNAV_SMOKE_ROOT}/logs"
    if [[ -s "${result_file}" ]]; then
        stamp "SKIP NaVILA+DEGNAV online smoke"
        return 0
    fi
    stamp "START NaVILA+DEGNAV online smoke"
    cd "${NAVILA_EVAL_ROOT}"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${HABITAT_ROOT}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n navila-eval \
        python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
        --run-type eval --num-chunks 1 --chunk-idx 0 \
        EVAL_CKPT_PATH_DIR "${MODEL_PATH}" \
        EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT 1 \
        EVAL.DEGNAV_ADAPTER.ENABLED True \
        RESULTS_DIR "${DEGNAV_SMOKE_ROOT}" VIDEO_OPTION "[]" \
        >"${DEGNAV_SMOKE_ROOT}/logs/smoke.log" 2>&1
    python -c "import json; p='${result_file}'; d=json.load(open(p)); assert len(d)==1; assert 'degnav_adapter' in next(iter(d.values()))"
    stamp "DONE NaVILA+DEGNAV online smoke"
}

summarize_navila_degnav() {
    local split="$1"
    local baseline_root="$2"
    local adapted_root="$3"
    local output_root="${FORMAL_ROOT}/navila_degnav_comparison_20260901/${split}"
    cd "${HABITAT_ROOT}"
    python examples/narrow_passage_rl/summarize_navila_degnav.py \
        --baseline-dir "${baseline_root}/navila-llama3-8b-8f/VLN-CE-v1/${split}" \
        --adapted-dir "${adapted_root}/navila-llama3-8b-8f/VLN-CE-v1/${split}" \
        --split "${split}" --chunks 10 --output-dir "${output_root}" \
        >"${adapted_root}/logs/paired_summary.log" 2>&1
    stamp "DONE paired NaVILA/DEGNAV summary ${split}"
}

run_nonlearning() {
    local agent="$1"
    local split="$2"
    local out_root="${FORMAL_ROOT}/r2r_nonlearning_20260901"
    local output="${out_root}/stats_${agent}_${split}.json"
    local log_file="${out_root}/logs/${agent}_${split}.log"
    mkdir -p "${out_root}/logs"
    if [[ -s "${output}" ]]; then
        stamp "SKIP ${agent} ${split}"
        return 0
    fi
    stamp "START ${agent} ${split}"
    (
        cd "${NAVILA_EVAL_ROOT}"
        CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n navila-eval \
            python run.py \
            --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
            --run-type eval \
            EVAL.EVAL_NONLEARNING True \
            EVAL.NONLEARNING.AGENT "${agent}" \
            EVAL.SPLIT "${split}" \
            EVAL.EPISODE_COUNT -1 >"${log_file}" 2>&1
        mv "stats_${agent}_${split}.json" "${output}"
    )
    [[ -s "${output}" ]]
    stamp "DONE ${agent} ${split}"
}

mine_mp3d_dataset() {
    local anchors_train="${MP3D_RUN_ROOT}/datasets/anchors_train.csv"
    local anchors_val="${MP3D_RUN_ROOT}/datasets/anchors_val.csv"
    local anchors_train_partial="${anchors_train}.partial"
    local anchors_val_partial="${anchors_val}.partial"
    local anchors_manifest="${MP3D_RUN_ROOT}/datasets/anchors_manifest.json"
    local anchors_manifest_partial="${anchors_manifest}.partial"
    local train_data="${MP3D_RUN_ROOT}/datasets/train/train.json.gz"
    local val_data="${MP3D_RUN_ROOT}/datasets/val/val.json.gz"
    mkdir -p "${MP3D_RUN_ROOT}/datasets/train" "${MP3D_RUN_ROOT}/datasets/val"
    if [[ ! -s "${anchors_train}" || ! -s "${anchors_val}" ]]; then
        stamp "START MP3D 90-scene narrow-passage mining"
        if [[ -e "${anchors_train}" && ! -s "${anchors_val}" ]]; then
            mv "${anchors_train}" "${anchors_train}.incomplete.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${anchors_val}" && ! -s "${anchors_train}" ]]; then
            mv "${anchors_val}" "${anchors_val}.incomplete.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${anchors_train_partial}" ]]; then
            mv "${anchors_train_partial}" "${anchors_train_partial}.failed.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${anchors_val_partial}" ]]; then
            mv "${anchors_val_partial}" "${anchors_val_partial}.failed.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${anchors_manifest_partial}" ]]; then
            mv "${anchors_manifest_partial}" "${anchors_manifest_partial}.failed.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${train_data}" ]]; then
            mv "${train_data}" "${train_data}.stale.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${val_data}" ]]; then
            mv "${val_data}" "${val_data}.stale.$(date +%Y%m%d_%H%M%S)"
        fi
        cd "${HABITAT_ROOT}"
        CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
            python examples/narrow_passage_rl/mine_habitat_passages.py \
            --scenes-dir "${MP3D_ROOT}" \
            --scene-glob '*/*.glb' \
            --episode-prefix mp3d_narrow \
            --train-frac 0.8 \
            --target-episodes 600 \
            --max-per-scene 15 \
            --attempts-per-scene 1200 \
            --robot-radius 0.18 \
            --false-feasible-fraction 0.20 \
            --require-target \
            --seed 20260901 \
            --out-train "${anchors_train_partial}" \
            --out-val "${anchors_val_partial}" \
            >"${MP3D_RUN_ROOT}/logs/mining.log" 2>&1
        conda run --no-capture-output -n habitat \
            python examples/narrow_passage_rl/validate_mp3d_anchors.py \
            --train "${anchors_train_partial}" \
            --val "${anchors_val_partial}" \
            --expected-train 480 --expected-val 120 \
            --robot-radius 0.18 --max-per-scene 15 \
            --output "${anchors_manifest_partial}" \
            >>"${MP3D_RUN_ROOT}/logs/mining.log" 2>&1
        mv "${anchors_train_partial}" "${anchors_train}"
        mv "${anchors_val_partial}" "${anchors_val}"
        mv "${anchors_manifest_partial}" "${anchors_manifest}"
        stamp "DONE MP3D mining"
    else
        stamp "SKIP MP3D mining"
        if [[ ! -s "${anchors_manifest}" ]]; then
            cd "${HABITAT_ROOT}"
            conda run --no-capture-output -n habitat \
                python examples/narrow_passage_rl/validate_mp3d_anchors.py \
                --train "${anchors_train}" --val "${anchors_val}" \
                --expected-train 480 --expected-val 120 \
                --robot-radius 0.18 --max-per-scene 15 \
                --output "${anchors_manifest}"
        fi
    fi
    if [[ ! -s "${train_data}" ]]; then
        cd "${HABITAT_ROOT}"
        python examples/narrow_passage_rl/generate_habitat_episodes.py \
            --anchors "${anchors_train}" --split train --output "${train_data}"
    fi
    if [[ ! -s "${val_data}" ]]; then
        cd "${HABITAT_ROOT}"
        python examples/narrow_passage_rl/generate_habitat_episodes.py \
            --anchors "${anchors_val}" --split val --output "${val_data}"
    fi
    stamp "READY MP3D datasets: ${train_data} ${val_data}"
}

run_pointnav_mp3d() {
    local val_data="${MP3D_RUN_ROOT}/datasets/val/val.json.gz"
    local out="${MP3D_RUN_ROOT}/results"
    local raw_csv="${out}/pointnav_ppo_raw.csv"
    local normalized_csv="${out}/pointnav_ppo.csv"
    mkdir -p "${out}"
    run_logged "${raw_csv}" "MP3D PointNav PPO transfer" \
        "${MP3D_RUN_ROOT}/logs/pointnav_ppo.log" \
        env CUDA_VISIBLE_DEVICES=0 \
        HABITAT_EVAL_DETERMINISTIC=1 \
        HABITAT_EVAL_EPISODE_CSV="${raw_csv}" \
        conda run --no-capture-output -n habitat \
        python -m habitat_baselines.run \
        --config-name=pointnav/ppo_pointnav_mp3d_formal.yaml \
        habitat_baselines.evaluate=True \
        habitat_baselines.load_resume_state_config=False \
        habitat_baselines.eval_ckpt_path_dir="${POINTNAV_MODEL}" \
        habitat_baselines.test_episode_count=-1 \
        habitat_baselines.num_environments=1 \
        habitat.dataset.data_path="${val_data}" \
        habitat.dataset.split=val \
        habitat_baselines.eval.split=val \
        habitat.simulator.agents.main_agent.radius=0.18 \
        habitat.simulator.habitat_sim_v0.allow_sliding=False \
        habitat_baselines.eval.video_option='[]'

    run_logged "${normalized_csv}" "Normalize MP3D PointNav PPO metrics" \
        "${MP3D_RUN_ROOT}/logs/pointnav_ppo_normalize.log" \
        conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/normalize_pointnav_eval.py \
        --raw-csv "${raw_csv}" --dataset "${val_data}" \
        --max-steps 500 --output-csv "${normalized_csv}"
}

train_mp3d_rl() {
    local train_data="${MP3D_RUN_ROOT}/datasets/train/train.json.gz"
    mkdir -p "${MP3D_DIRECT_DIR}" "${MP3D_RECURRENT_DIR}"
    cd "${HABITAT_ROOT}"

    run_logged "${MP3D_DIRECT_MODEL}" "MP3D Direct-control PPO fine-tune" \
        "${MP3D_RUN_ROOT}/logs/direct_control_ppo_train.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/train_habitat_sb3.py \
        --algo ppo --data-path "${train_data}" --split train \
        --total-steps "${MP3D_FINETUNE_STEPS}" --eval-episodes 0 \
        --load-model "${DIRECT_MODEL}" --save-dir "${MP3D_DIRECT_DIR}" \
        --action-space synthetic --agent-radius 0.18 \
        --max-steps 500 --seed 0 --device cuda

    run_logged "${MP3D_RECURRENT_MODEL}" "MP3D Recurrent PPO fine-tune" \
        "${MP3D_RUN_ROOT}/logs/recurrent_ppo_train.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/train_habitat_sb3.py \
        --algo recurrent_ppo --data-path "${train_data}" --split train \
        --total-steps "${MP3D_FINETUNE_STEPS}" --eval-episodes 0 \
        --load-model "${RECURRENT_MODEL}" --save-dir "${MP3D_RECURRENT_DIR}" \
        --action-space synthetic --agent-radius 0.18 \
        --max-steps 500 --seed 0 --device cuda
}

run_mp3d_methods() {
    local val_data="${MP3D_RUN_ROOT}/datasets/val/val.json.gz"
    local out="${MP3D_RUN_ROOT}/results"
    mkdir -p "${out}"
    cd "${HABITAT_ROOT}"

    run_logged "${out}/geometry_only_apf_gap.csv" "MP3D Geometry-only APF+Gap" \
        "${MP3D_RUN_ROOT}/logs/geometry_only_apf_gap.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_apf_gap.py \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --num-episodes -1 --max-steps 500 \
        --out-csv "${out}/geometry_only_apf_gap.csv"

    run_logged "${out}/direct_control_ppo.csv" "MP3D Direct-control PPO" \
        "${MP3D_RUN_ROOT}/logs/direct_control_ppo.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_sb3.py \
        --algo ppo --model "${MP3D_DIRECT_MODEL}" --action-space synthetic \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --num-episodes -1 --max-steps 500 \
        --output-csv "${out}/direct_control_ppo.csv"

    run_logged "${out}/recurrent_ppo.csv" "MP3D Recurrent PPO" \
        "${MP3D_RUN_ROOT}/logs/recurrent_ppo.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_sb3.py \
        --algo recurrent_ppo --model "${MP3D_RECURRENT_MODEL}" --action-space synthetic \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --num-episodes -1 --max-steps 500 \
        --output-csv "${out}/recurrent_ppo.csv"

    run_logged "${out}/degnav_no_memory.csv" "MP3D DEGNAV without memory" \
        "${MP3D_RUN_ROOT}/logs/degnav_no_memory.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --use-memory 0 --num-episodes -1 --max-steps 500 \
        --output-csv "${out}/degnav_no_memory.csv"

    run_logged "${out}/episodic_knn_memory.csv" "MP3D episodic kNN memory" \
        "${MP3D_RUN_ROOT}/logs/episodic_knn_memory.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --use-memory 1 --memory-backend knn \
        --knn-k 5 --knn-min-neighbors 2 --knn-reject-threshold 0.80 \
        --num-episodes -1 --max-steps 500 \
        --output-csv "${out}/episodic_knn_memory.csv"

    run_logged "${out}/degnav_memory.csv" "MP3D DEGNAV with failure memory" \
        "${MP3D_RUN_ROOT}/logs/degnav_memory.log" \
        env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
        python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
        --data-path "${val_data}" --split val --agent-radius 0.18 \
        --use-memory 1 --memory-trigger-count 1 --memory-reject-count 3 \
        --num-episodes -1 --max-steps 500 \
        --output-csv "${out}/degnav_memory.csv"

    if [[ ! -s "${out}/feasibility_ablations/summary.csv" ]]; then
        stamp "START MP3D uncertainty/yaw ablations"
        if [[ -e "${out}/feasibility_ablations.partial" ]]; then
            mv "${out}/feasibility_ablations.partial" \
                "${out}/feasibility_ablations.failed.$(date +%Y%m%d_%H%M%S)"
        fi
        if [[ -e "${out}/feasibility_ablations" ]]; then
            mv "${out}/feasibility_ablations" \
                "${out}/feasibility_ablations.failed.$(date +%Y%m%d_%H%M%S)"
        fi
        CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n habitat \
            python examples/narrow_passage_rl/eval_habitat_feasibility_ablations.py \
            --methods full_dynamic_uncertainty no_uncertainty no_yaw_prior \
            --data-path "${val_data}" --split val \
            --scenes-dir "${MP3D_ROOT}" --skip-scene-dataset-config \
            --agent-radius 0.18 --num-episodes -1 --max-steps 500 \
            --seed 20260901 \
            --output-dir "${out}/feasibility_ablations.partial" \
            >"${MP3D_RUN_ROOT}/logs/feasibility_ablations.log" 2>&1
        mv "${out}/feasibility_ablations.partial" "${out}/feasibility_ablations"
        stamp "DONE MP3D uncertainty/yaw ablations"
    else
        stamp "SKIP MP3D uncertainty/yaw ablations"
    fi

    python examples/narrow_passage_rl/summarize_formal_mp3d.py \
        --dataset "${val_data}" \
        --input "Geometry-only (APF+Gap)=${out}/geometry_only_apf_gap.csv" \
        --input "PointNav PPO (pretrained transfer)=${out}/pointnav_ppo.csv" \
        --input "Direct-control PPO (MP3D fine-tuned)=${out}/direct_control_ppo.csv" \
        --input "Recurrent PPO (MP3D fine-tuned)=${out}/recurrent_ppo.csv" \
        --input "DEGNAV w/o memory=${out}/degnav_no_memory.csv" \
        --input "Episodic kNN memory=${out}/episodic_knn_memory.csv" \
        --input "DEGNAV + failure memory=${out}/degnav_memory.csv" \
        --output-dir "${out}/summary" \
        >"${MP3D_RUN_ROOT}/logs/summary.log" 2>&1
    stamp "DONE MP3D formal summary"
}

main() {
    stamp "QUEUE START"
    wait_for_unseen
    if [[ ! -s "${SEEN_ROOT}/aggregate_metrics.json" ]]; then
        run_navila_seen
    else
        stamp "SKIP NaVILA val_seen aggregate"
    fi
    run_navila_degnav_smoke
    if [[ ! -s "${DEGNAV_UNSEEN_ROOT}/aggregate_metrics.json" ]]; then
        run_navila_degnav val_unseen "${DEGNAV_UNSEEN_ROOT}"
    else
        stamp "SKIP NaVILA+DEGNAV val_unseen aggregate"
    fi
    if [[ ! -s "${DEGNAV_SEEN_ROOT}/aggregate_metrics.json" ]]; then
        run_navila_degnav val_seen "${DEGNAV_SEEN_ROOT}"
    else
        stamp "SKIP NaVILA+DEGNAV val_seen aggregate"
    fi
    summarize_navila_degnav val_unseen "${UNSEEN_ROOT}" "${DEGNAV_UNSEEN_ROOT}"
    summarize_navila_degnav val_seen "${SEEN_ROOT}" "${DEGNAV_SEEN_ROOT}"
    run_nonlearning RandomAgent val_unseen
    run_nonlearning HandcraftedAgent val_unseen
    run_nonlearning RandomAgent val_seen
    run_nonlearning HandcraftedAgent val_seen
    mine_mp3d_dataset
    run_pointnav_mp3d
    train_mp3d_rl
    run_mp3d_methods
    stamp "QUEUE COMPLETE"
}

main "$@"
