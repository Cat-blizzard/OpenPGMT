"""Run the frozen two-condition, 15360-transition budget and release CUDA."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for folder in ('pgmt','setup','data'):
        for source in Path(folder).rglob('*.py'):
            if 'raw' in source.parts or '.external' in source.parts:continue
            dest=a.output/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    ledger={'budget_transitions':15360,'runs':[],'total_transitions':0,'evaluation_chained':False}
    process=None
    def save(): (a.output/'execution_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    try:
        for condition in ('control','candidate'):
            free=subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(free.strip())<10000:raise RuntimeError('insufficient free GPU memory')
            command=[sys.executable,'-m','pgmt.train.fixed_clip_diagnostic','--manifest',str(a.fixture/'manifest.json'),
                '--reference-data',str(a.fixture/'reference'),'--standing-fixture',str(a.fixture),'--standing-condition',condition,
                '--urdf',str(a.urdf),'--asset',str(a.asset),'--output',str(a.output/condition),
                '--reset-mode','nominal','--backend','isaaclab','--device','cuda:0','--num-envs','16',
                '--steps-per-env','24','--updates','20','--lr-schedule-updates','1000','--seed','0']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'PGMT_RENDER_GPU':str(a.render_gpu),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                'PGMT_ISAACLAB_LOG_DIR':str((a.output/(condition+'_isaac_logs')).resolve())}
            row={'condition':condition,'command':command,'started_unix':time.time()};ledger['runs'].append(row)
            with (a.output/(condition+'.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=process.pid;save();print('START',condition,process.pid,flush=True)
                row['exit_code']=process.wait(timeout=1800)
            m=json.loads((a.output/condition/'metrics.json').read_text())
            row.update(status=m['status'],completed_updates=m['completed_updates'],
                       transitions=m.get('executed_transitions',sum(u['steps'] for u in m['updates'])),finished_unix=time.time())
            ledger['total_transitions']+=row['transitions'];save()
            if row['exit_code'] or row['status']!='completed' or row['completed_updates']!=20 or row['transitions']!=7680:
                raise RuntimeError('standing run failed; no automatic retry or extra training')
            print('END',condition,'20 updates / 7680 transitions',flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
        alive=[r['pid'] for r in ledger['runs'] if 'pid' in r and Path('/proc',str(r['pid'])).exists()]
        (a.output/'gpu_release.json').write_text(json.dumps({'task_pids_alive':alive,'compute_apps':apps,
            'gpu_uuid':a.gpu_uuid,'checked_unix':time.time()},indent=2)+'\n')
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixture','output','urdf','asset'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--render-gpu',type=int,required=True)
    run(p.parse_args())
