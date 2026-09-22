#!/usr/bin/env bash
# Manual launcher: one probe OR one 16x20 PPO run, then exit. Never auto-chain.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: bash setup/run_actuator_check.sh neutral|reference|ppo asset_effort_v1|legacy_uniform120 GPU_UUID NEW_OUTPUT_DIR'
  echo 'Requires PGMT_RENDER_GPU to be set to the current Vulkan index; no default GPU is selected.'
  exit 2
fi
pgmt_mode=$1
pgmt_profile=$2
pgmt_gpu=$3
pgmt_output=$(realpath -m "$4")
case "$pgmt_mode" in neutral|reference|ppo) ;; *) echo 'Invalid mode'; exit 2;; esac
case "$pgmt_profile" in asset_effort_v1|legacy_uniform120) ;; *) echo 'Invalid actuator profile'; exit 2;; esac
: "${PGMT_RENDER_GPU:?Set PGMT_RENDER_GPU after checking the current GPU assignment}"
if [[ -e "$pgmt_output" ]]; then
  echo 'Use a fresh output directory to preserve earlier evidence.'
  exit 2
fi
cd /data/jxc/ld/PGMT
mkdir -p "$pgmt_output"
export CUDA_VISIBLE_DEVICES="$pgmt_gpu"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pgmt_python=/data/jxc/envs/pbfm_isaaclab/bin/python
pgmt_common=(--device cuda:0 --asset /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/usd/g1.usd
  --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf
  --reference-data data/processed/lafan1_g1_mesh_v2 --actuator-profile "$pgmt_profile" --seed "${PGMT_PROBE_SEED:-17}")
if [[ "$pgmt_mode" == ppo ]]; then
  pgmt_command=("$pgmt_python" -m pgmt.train.train_stage1 "${pgmt_common[@]}"
    --backend isaaclab --num-envs 16 --steps-per-env 24 --updates 20 --lr-schedule-updates 1000
    --critic-completion --fall-pool runs/curriculum_v4_20260921/collect256/fall_pool.pt
    --initial-checkpoint "$pgmt_output/initial.pt" --checkpoint "$pgmt_output/policy.pt"
    --metrics "$pgmt_output/metrics.json" --diagnostics-dir "$pgmt_output/diagnostics")
else
  pgmt_command=("$pgmt_python" -m setup.check_actuator_response "${pgmt_common[@]}"
    --num-envs 4 --steps 150 --target-mode "$pgmt_mode" --reset-mode "${PGMT_RESET_MODE:-nominal}"
    --output "$pgmt_output/response.json")
fi
printf '%q ' "${pgmt_command[@]}" > "$pgmt_output/command.sh"
printf '\n' >> "$pgmt_output/command.sh"
"$pgmt_python" - "$pgmt_output" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
out = Path(sys.argv[1])
sources = sorted({p for folder in ('pgmt', 'data', 'setup') for p in Path(folder).glob('**/*.py')
                  if '.external' not in p.parts and 'raw' not in p.parts})
sources.append(Path('setup/run_actuator_check.sh'))
manifest = {'gpu_evaluation_chained': False, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'vulkan_index': os.environ['PGMT_RENDER_GPU'], 'command': (out / 'command.sh').read_text(),
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
set +e
"${pgmt_command[@]}" > "$pgmt_output/run.log" 2>&1
pgmt_status=$?
set -e
# Kit shutdown can exit Python before an exception propagates. A zero process
# code is insufficient; require the requested result to be complete as well.
if [[ "$pgmt_status" -eq 0 ]]; then
  if ! "$pgmt_python" - "$pgmt_output" "$pgmt_mode" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1]) / ('metrics.json' if sys.argv[2] == 'ppo' else 'response.json')
if not path.is_file() or json.loads(path.read_text()).get('status') != 'completed':
    print('Probe/training did not complete; inspect the report and run.log.', file=sys.stderr)
    sys.exit(1)
PY
  then
    pgmt_status=1
  fi
fi
printf '%s\n' "$pgmt_status" > "$pgmt_output/exit_code"
exit "$pgmt_status"
