"""Bounded three-seed single-motion check; fresh initial/final evaluation processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','reference-data','asset','urdf'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--render-gpu',type=int,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    a=p.parse_args()
    if a.root.exists():p.error('use a new output directory')
    free=int(subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<8000*a.workers:raise RuntimeError('insufficient GPU memory for the requested bounded batch')
    a.root.mkdir(parents=True)
    plan={'seeds':[0,1,2],'num_envs':16,'steps_per_env':24,'updates_per_seed':100,
        'total_training_transitions':115200,'evaluation':'16 first episodes, stochastic and deterministic, initial/final; 5s horizon',
        'workers':a.workers,'gpu_uuid':a.gpu_uuid,'no_formal_training_or_v4_evaluation':True}
    with (a.root/'plan.json').open('x') as f:json.dump(plan,f,indent=2)
    for folder in ('pgmt','setup','data'):
        for source in Path(folder).rglob('*.py'):
            if 'raw' in source.parts or '.external' in source.parts:continue
            target=a.root/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    common=[]
    for key in ('manifest','reference_data','asset','urdf'):common+=['--'+key.replace('_','-'),str(getattr(a,key))]

    def execute(name,module,arguments,updates=None):
        output=a.root/name;cmd=[sys.executable,'-m',module,*common,'--output',str(output),*arguments]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu_uuid,'PGMT_RENDER_GPU':str(a.render_gpu),
            'PGMT_ISAACLAB_LOG_DIR':str((a.root/(name+'_isaaclog')).resolve()),
            'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (a.root/(name+'_command.json')).open('x') as f:json.dump(cmd,f,indent=2)
        with (a.root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            with (a.root/(name+'_pid')).open('x') as f:f.write(str(proc.pid)+'\n')
            print('START',name,proc.pid,flush=True);code=proc.wait()
        with (a.root/(name+'_exit.json')).open('x') as f:json.dump({'pid':proc.pid,'exit_code':code},f)
        m=json.loads((output/'metrics.json').read_text())
        if code or m['status']!='completed':raise RuntimeError(f'{name} failed: {m.get("error")}')
        if updates is not None and m['completed_updates']!=updates:raise RuntimeError('incomplete training budget')
        print('COMPLETED',name,flush=True)
        return output

    def seed_job(seed):
        train=execute(f'seed{seed}_train','pgmt.train.fixed_clip_diagnostic',[
            '--seed',str(seed),'--reset-mode','reference_state','--actor-init','clip_start',
            '--num-envs','16','--steps-per-env','24','--updates','100','--device','cuda:0'],100)
        for when,filename in (('initial','initial.pt'),('final','policy.pt')):
            execute(f'seed{seed}_{when}_eval','setup.evaluate_simple_diagnostic',[
                '--checkpoint',str(train/filename),'--eval-seed',str(10000+seed),'--num-envs','16','--device','cuda:0'])
        return seed

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(seed_job,seed) for seed in (0,1,2)]
        for job in jobs:job.result()
    print('BATCH COMPLETED; all GPU children exited',flush=True)


if __name__=='__main__':main()
