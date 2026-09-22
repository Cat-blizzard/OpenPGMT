"""Run exactly two serial 16-env/20-update diagnostics, then exit and free CUDA.

Call with CUDA_VISIBLE_DEVICES='' so this supervisor stays CPU-only. The child
GPU UUID and Vulkan index are explicit. No evaluation or longer run is chained.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--render-gpu',type=int,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-data',type=Path,required=True)
    p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    a=p.parse_args()
    if a.root.exists():p.error('use a fresh physical pair directory')
    a.root.mkdir(parents=True)
    # Preserve runnable source; dataset/assets remain external and hashed.
    for folder in ('pgmt','setup','data'):
        for source in Path(folder).rglob('*.py'):
            if 'raw' in source.parts or '.external' in source.parts:continue
            target=a.root/'source_snapshot'/source
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for mode in ('nominal','reference_state'):
        free=subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip()
        if int(free)<10000:raise RuntimeError(f'insufficient free GPU memory before {mode}: {free} MiB')
        before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader'],text=True)
        (a.root/f'{mode}_gpu_before.csv').write_text(before)
        output=a.root/mode
        command=[sys.executable,'-m','pgmt.train.fixed_clip_diagnostic','--reset-mode',mode,
            '--backend','isaaclab','--device','cuda:0','--num-envs','16','--steps-per-env','24',
            '--updates','20','--seed','0','--output',str(output)]
        for key in ('manifest','reference_data','asset','urdf'):
            command+=['--'+key.replace('_','-'),str(getattr(a,key))]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'PGMT_RENDER_GPU':str(a.render_gpu),
             'PGMT_ISAACLAB_LOG_DIR':str((a.root/f'{mode}_isaaclab_logs').resolve()),
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
        (a.root/f'{mode}_command.json').write_text(json.dumps(command,indent=2)+'\n')
        print(f'START {mode}',flush=True)
        with (a.root/f'{mode}.log').open('x') as log:
            code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (a.root/f'{mode}_exit_code').write_text(str(code)+'\n')
        # Kit can mask Python exceptions during shutdown: validate persisted
        # evidence independently of the process exit status.
        metrics=json.loads((output/'metrics.json').read_text())
        if code or metrics['status']!='completed' or metrics['completed_updates']!=20 or len(metrics['updates'])!=20:
            raise RuntimeError(f'{mode} failed; inspect {output}/metrics.json and log')
        print(f'COMPLETED {mode}: 20 updates',flush=True)
    after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader'],text=True)
    (a.root/'gpu_after.csv').write_text(after)
    print('PAIR FINISHED; both child processes exited; no GPU follow-up scheduled',flush=True)


if __name__=='__main__':main()
