#!/usr/bin/env bash
# Explicit commands only: this script does not launch a stage automatically.
# Run inside tmux so a terminal disconnect does not stop the workload.
set -euo pipefail
pgmt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pgmt_root"
pgmt_mode="${1:?usage: run_revised.sh collect-smoke|collect|stage1-smoke|stage1|stage2-smoke|stage2|eval-stage1|eval-stage2 GPU_INDEX}"
pgmt_gpu="${2:?physical GPU index is required}"
[[ "$pgmt_gpu" =~ ^[0-9]+$ ]] || { echo 'GPU index must be an integer.' >&2; exit 2; }
pgmt_python="${PGMT_PYTHON:-/data/jxc/envs/pbfm_isaaclab/bin/python}"
pgmt_assets="${PGMT_ASSETS:-/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets}"
pgmt_reference="$pgmt_root/data/processed/lafan1_g1_mesh_v2"
pgmt_pool="$pgmt_root/data/processed/fall_pool_v2.pt"
pgmt_run="$pgmt_root/runs/revised_v2/$pgmt_mode"
mkdir -p "$pgmt_run"
exec 9>"$pgmt_run/launch.lock"
flock -n 9 || { echo 'This workload is already running.' >&2; exit 2; }
[[ ! -e "$pgmt_run/started_at" ]] || { echo 'Previous attempt exists; inspect its result before choosing a new output directory.' >&2; exit 2; }
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'Verify physical GPU mapping: this launcher expects CUDA_VISIBLE_DEVICES unset.' >&2; exit 2; }
pgmt_busy=$(nvidia-smi -i "$pgmt_gpu" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$pgmt_busy" ]] || { echo "GPU $pgmt_gpu is busy; no workload was started." >&2; exit 2; }
pgmt_temp=$(nvidia-smi -i "$pgmt_gpu" --query-gpu=temperature.gpu --format=csv,noheader,nounits)
[[ "$pgmt_temp" =~ ^[0-9]+$ ]] || { echo 'GPU health query failed.' >&2; exit 2; }
pgmt_common=(--device "cuda:$pgmt_gpu" --asset "$pgmt_assets/usd/g1.usd" --urdf "$pgmt_assets/urdf/g1.urdf" --reference-data "$pgmt_reference")
case "$pgmt_mode" in
  collect|collect-smoke)
    pgmt_count=64; pgmt_states=2048
    if [[ "$pgmt_mode" == collect-smoke ]]; then pgmt_count=8; pgmt_states=16; pgmt_pool="$pgmt_run/fall_pool.pt"; fi
    pgmt_cmd=("$pgmt_python" -u -m pgmt.train.collect_fall_pool "${pgmt_common[@]}" --num-envs "$pgmt_count" --states "$pgmt_states" --output "$pgmt_pool");;
  stage1|stage1-smoke)
    if [[ "$pgmt_mode" == stage1-smoke && ! -f "$pgmt_pool" ]]; then pgmt_pool="$pgmt_root/runs/revised_v2/collect-smoke/fall_pool.pt"; fi
    [[ -f "$pgmt_pool" ]] || { echo 'Collect the physical fall pool first.' >&2; exit 2; }
    pgmt_count=256; pgmt_updates=1000
    if [[ "$pgmt_mode" == stage1-smoke ]]; then pgmt_count=8; pgmt_updates=2; fi
    pgmt_cmd=("$pgmt_python" -u -m pgmt.train.train_stage1 "${pgmt_common[@]}" --backend isaaclab --fall-pool "$pgmt_pool" --num-envs "$pgmt_count" --steps-per-env 24 --updates "$pgmt_updates" --learning-epochs 5 --mini-batches 4 --seed 0 --checkpoint "$pgmt_run/policy.pt" --metrics "$pgmt_run/metrics.json");;
  stage2|stage2-smoke)
    pgmt_source="$pgmt_root/runs/revised_v2/stage1/policy.pt"
    if [[ "$pgmt_mode" == stage2-smoke ]]; then pgmt_source="$pgmt_root/runs/revised_v2/stage1-smoke/policy.pt"; fi
    [[ -f "$pgmt_source" ]] || { echo 'A new physical Stage 1 checkpoint is required.' >&2; exit 2; }
    pgmt_count=256; pgmt_updates=1000
    if [[ "$pgmt_mode" == stage2-smoke ]]; then pgmt_count=5; pgmt_updates=2; fi
    pgmt_cmd=("$pgmt_python" -u -m pgmt.train.train_stage2 "${pgmt_common[@]}" --backend isaaclab --stage1-checkpoint "$pgmt_source" --num-envs "$pgmt_count" --steps-per-env 24 --updates "$pgmt_updates" --learning-epochs 5 --mini-batches 4 --seed 0 --checkpoint "$pgmt_run/policy.pt" --metrics "$pgmt_run/metrics.json");;
  eval-stage1|eval-stage2)
    pgmt_stage="${pgmt_mode#eval-stage}"
    pgmt_cmd=("$pgmt_python" -u -m pgmt.eval.run "${pgmt_common[@]}" --stage "$pgmt_stage" --checkpoint "$pgmt_root/runs/revised_v2/stage$pgmt_stage/policy.pt" --manifest "$pgmt_root/eval/manifests/matched_9600_v2.json" --output "$pgmt_run/episodes.jsonl");;
  *) echo "Unknown workload: $pgmt_mode" >&2; exit 2;;
esac
date --iso-8601=seconds >"$pgmt_run/started_at"
printf '%q ' "${pgmt_cmd[@]}" >"$pgmt_run/command.txt"
printf '\n' >>"$pgmt_run/command.txt"
trap 'pgmt_rc=$?; printf "%s\n" "$pgmt_rc" >"$pgmt_run/exit_code"; date --iso-8601=seconds >"$pgmt_run/finished_at"' EXIT
"${pgmt_cmd[@]}" >"$pgmt_run/training.log" 2>&1
