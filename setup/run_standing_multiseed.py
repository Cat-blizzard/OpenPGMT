"""Six serial physical training runs, strict budget, no chained evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.standing_multiseed_protocol import validate_plan, check_pair, TRAIN_TRANSITIONS


def stop_group(process):
    if process is None:return
    # Every child is launched into its own session; never signal unrelated jobs.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()


def run(a):
    plan=validate_plan(a.plan);fixture=a.plan.parent/'fixture'
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    # Advisory lease among PGMT jobs. Other users' workloads are only observed.
    lease=open(Path('/tmp')/f'pgmt-standing-{a.gpu_uuid}.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger={'status':'running','budget_transitions':TRAIN_TRANSITIONS,'total_transitions':0,
            'plan_sha256':sha256(a.plan),'evaluation_chained':False,'runs':[],'pairs':[]}
    sources=[p for folder in ('pgmt','setup','data') for p in Path(folder).rglob('*.py')
             if 'raw' not in p.parts and '.external' not in p.parts]
    for p in sources:
        dest=a.output/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # New offline-only scripts may be added while training; existing dependencies stay frozen.
    hashes={str(p):sha256(p) for p in sources}
    _write_metrics(a.output/'source_manifest.json',hashes)
    process=None
    def save():_write_metrics(a.output/'execution_ledger.json',ledger)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'received signal {signum}')
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for job in plan['runs']:
            validate_plan(a.plan)
            if any(sha256(p)!=h for p,h in hashes.items()):raise ValueError('training source changed during frozen batch')
            free=subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(free.strip())<10000:raise RuntimeError('insufficient free GPU memory')
            name=job['name'];folder=a.output/name
            command=[sys.executable,'-m','pgmt.train.fixed_clip_diagnostic','--manifest',str(a.plan.parent/f"seed{job['seed']}.json"),
                '--reference-data',str(fixture/'reference'),'--standing-fixture',str(fixture),'--standing-condition',job['condition'],
                '--urdf',plan['urdf'],'--asset',plan['asset'],'--output',str(folder),
                '--reset-mode','nominal','--backend','isaaclab','--device','cuda:0','--num-envs','16',
                '--steps-per-env','24','--updates','100','--lr-schedule-updates','1000','--seed',str(job['seed'])]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'PGMT_RENDER_GPU':str(a.render_gpu),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                'PGMT_ISAACLAB_LOG_DIR':str((a.output/(name+'_isaac_logs')).resolve())}
            row={**job,'command':command,'started_unix':time.time()};ledger['runs'].append(row);save()
            try:
                with (a.output/(name+'.log')).open('x') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    row['pid']=process.pid;save();print('START',name,process.pid,flush=True)
                    row['exit_code']=process.wait(timeout=3600)
            finally:
                stop_group(process);process=None
                metrics=folder/'metrics.json'
                if metrics.exists():
                    m=json.loads(metrics.read_text());row.update(status=m['status'],completed_updates=m['completed_updates'],
                        transitions=m.get('executed_transitions',sum(u['steps'] for u in m['updates'])))
                row['finished_unix']=time.time()
                ledger['total_transitions']=sum(r.get('transitions',0) for r in ledger['runs']);save()
            if row.get('exit_code')!=0 or row.get('status')!='completed' or row.get('completed_updates')!=100 or row.get('transitions')!=38400:
                raise RuntimeError('incomplete run; batch stopped without retry')
            if job['condition']=='candidate':ledger['pairs'].append(check_pair(a.output,job['seed']))
            save();print('END',name,'100 updates / 38400 transitions',flush=True)
        if ledger['total_transitions']!=TRAIN_TRANSITIONS:raise RuntimeError('incorrect final transition count')
        ledger['status']='completed'
    except BaseException:
        ledger['status']='failed';ledger['error']=traceback.format_exc();raise
    finally:
        stop_group(process)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
        alive=[r['pid'] for r in ledger['runs'] if 'pid' in r and Path('/proc',str(r['pid'])).exists()]
        _write_metrics(a.output/'gpu_release.json',{'task_pids_alive':alive,'compute_apps':apps,
            'gpu_uuid':a.gpu_uuid,'checked_unix':time.time()})
        save();lease.close()
        for s,handler in old.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--render-gpu',type=int,required=True)
    run(p.parse_args())
