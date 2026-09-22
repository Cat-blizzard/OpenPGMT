"""Run at most 9600 transitions in two serial no-learning replay processes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(a):
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for folder in ('pgmt', 'setup', 'data'):
        for p in Path(folder).rglob('*.py'):
            if 'raw' in p.parts or '.external' in p.parts: continue
            dest=a.output/'source_snapshot'/p; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    ledger={'maximum_transitions':9600,'ppo_updates':0,'runs':[],'actual_transitions':0}
    process=None
    def save(): (a.output/'execution_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    try:
        for condition in ('control','candidate'):
            free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<10000: raise RuntimeError('insufficient free memory; no replay started')
            command=[sys.executable,'-m','setup.replay_standing_events','--original',str(a.original/condition),
                '--fixture',str(a.fixture),'--tape',str(a.tapes/(condition+'.pt')),'--output',str(a.output/condition)]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'PGMT_RENDER_GPU':str(a.render_gpu),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                'PGMT_ISAACLAB_LOG_DIR':str((a.output/(condition+'_isaac_logs')).resolve())}
            row={'condition':condition,'command':command,'started_unix':time.time()}; ledger['runs'].append(row)
            with (a.output/(condition+'.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=process.pid;save(); print('START',condition,process.pid,flush=True)
                row['exit_code']=process.wait(timeout=1200)
            m=json.loads((a.output/condition/'metrics.json').read_text())
            row.update(status=m['status'],transitions=m['transitions'],finished_unix=time.time())
            ledger['actual_transitions']+=m['transitions'];save()
            if m['transitions']>4800 or ledger['actual_transitions']>9600:raise RuntimeError('budget violated')
            if m['status'] not in ('completed','trajectory_diverged'):raise RuntimeError('replay failed; no automatic retry')
            print('END',condition,m['status'],m['transitions'],flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        alive=[r['pid'] for r in ledger['runs'] if 'pid' in r and Path('/proc',str(r['pid'])).exists()]
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
        (a.output/'gpu_release.json').write_text(json.dumps({'task_pids_alive':alive,'compute_apps':apps,
            'gpu_uuid':a.gpu_uuid,'checked_unix':time.time()},indent=2)+'\n')
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','fixture','tapes','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--render-gpu',type=int,required=True)
    run(p.parse_args())
