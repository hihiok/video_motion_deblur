"""Bounded real-data DDP trials, immutable resume snapshot, measured GPU selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from .data import sha256
from .protocol import validate_training_settings
from .runtime import choose_candidate


def monitor(gpus, samples, stop):
    while not stop.is_set():
        try:
            result=subprocess.run(['nvidia-smi','-i',','.join(gpus),
                '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if result.returncode==0:
                stamp=time.time()
                for row in csv.reader(result.stdout.splitlines()):
                    idx,util,used,total=[v.strip() for v in row]
                    samples.append(dict(time=stamp,gpu=idx,utilization=float(util),
                                        memory_used_MiB=float(used),memory_total_MiB=float(total)))
        except (ValueError,subprocess.TimeoutExpired):
            pass
        stop.wait(2)


def trial(a, gpus, checkpointing, workers, snapshot, output):
    label=f'w{len(gpus)}_checkpoint_{checkpointing}_workers{workers}'
    result_path=output/f'{label}.json'
    log=output/f'{label}.log'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',
        '-m','shift500.train','--config',a.config,'--variant',a.variant,'--resume',str(snapshot),
        '--activation-checkpointing',checkpointing,'--workers',str(workers),'--prefetch-factor','2',
        '--benchmark-report',str(result_path),'--benchmark-warmup','20','--benchmark-steps','80']
    samples=[];stop=threading.Event()
    watcher=threading.Thread(target=monitor,args=(gpus,samples,stop),daemon=True)
    print(json.dumps({'trial_start':label,'gpus':gpus}),flush=True)
    watcher.start()
    try:
        with log.open('w') as handle:
            proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                # Only this disposable child process group, never another training job.
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(f'Trial did not exit after SIGTERM: process group {proc.pid}; inspect {log}')
                code=124
    finally:
        stop.set();watcher.join(timeout=12)
    (output/f'{label}_gpu.json').write_text(json.dumps(samples,indent=2)+'\n')
    if code or not result_path.exists():
        report=dict(status='FAILED',world_size=len(gpus),activation_checkpointing=checkpointing,
                    workers_per_rank=workers,exit_code=code,log=str(log))
    else:
        report=json.loads(result_path.read_text())
        active=[s for s in samples if report['measurement_start_time']<=s['time']<=report['measurement_end_time']]
        groups={gpu:[s['utilization'] for s in active if s['gpu']==gpu] for gpu in gpus}
        means={gpu:sum(values)/len(values) for gpu,values in groups.items() if values}
        report.update(gpu_util_per_rank=means,
                      gpu_util_mean=sum(means.values())/len(means) if len(means)==len(gpus) else None,
                      gpu_util_min_rank=min(means.values()) if len(means)==len(gpus) else None)
    report.update(gpus=gpus,label=label)
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--resume',required=True)
    p.add_argument('--gpus',required=True,help='4 or 8 verified available physical indices, current pair first')
    p.add_argument('--output',required=True,help='New directory outside either variant directory')
    p.add_argument('--variant',choices=['quality','compact'],default='quality')
    a=p.parse_args()
    gpus=[g.strip() for g in a.gpus.split(',')]
    if len(gpus) not in (4,8) or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError('Provide 4 or 8 distinct physical GPU indices')
    c=json.loads(Path(a.config).read_text());validate_training_settings(c)
    output=Path(a.output).resolve()
    for v in ('quality','compact'):
        production=(Path(c['output'])/v).resolve()
        if output==production or production in output.parents:raise ValueError('Trials must stay outside model output directories')
    if output.exists():raise FileExistsError('Use a new trial directory; preserve previous measurements')
    resume=Path(a.resume).resolve()
    if not resume.is_file():raise FileNotFoundError(resume)
    config_hash=sha256(a.config);checkpoint_hash=sha256(resume)
    output.mkdir(parents=True)
    snapshot=output/'resume_snapshot.pth';shutil.copy2(resume,snapshot)
    if sha256(snapshot)!=checkpoint_hash:raise ValueError('Checkpoint changed during copy; stop old training first')
    # Exactly the same starting model/optimizer/scaler and sample indices in every trial.
    results=[trial(a,gpus[:2],'on',2,snapshot,output)]
    for world in (4,8):
        if world>len(gpus):continue
        for checkpointing in ('on','off'):
            results.append(trial(a,gpus[:world],checkpointing,2,snapshot,output))
    healthy=[r for r in results if r['status']=='PASS' and r['world_size']>=4
             and r['amp_skipped_measured']==0 and r['peak_memory_fraction']<.90]
    if healthy:
        fastest=min(healthy,key=lambda r:r['seconds_per_update'])
        if fastest['max_rank_data_wait_fraction']>.10 or (fastest['gpu_util_mean'] or 0)<70:
            # One bounded loader tuning trial. Maximum 32 single-thread workers on 8 GPUs.
            results.append(trial(a,fastest['gpus'],fastest['activation_checkpointing'],4,snapshot,output))
    if sha256(resume)!=checkpoint_hash or sha256(a.config)!=config_hash:
        raise RuntimeError('Production checkpoint/config changed during trials; do not launch competing jobs')
    (output/'trials.json').write_text(json.dumps(results,indent=2)+'\n')
    try:selected=choose_candidate(results)
    except ValueError as error:
        (output/'selection_blocked.txt').write_text(str(error)+'\n')
        raise
    selected.update(source_checkpoint_sha256=checkpoint_hash,config_sha256=config_hash)
    (output/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    # All values are constrained integers/enumerations; never include proxy credentials.
    (output/'selected_env.sh').write_text(
        'export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport CUDA_VISIBLE_DEVICES='+','.join(selected['gpus'])+'\n'+
        'export SHIFT500_CHECKPOINTING='+selected['activation_checkpointing']+'\n'+
        'export SHIFT500_WORKERS='+str(selected['workers_per_rank'])+'\n'+
        'export SHIFT500_PREFETCH=2\n')
    print(json.dumps({'selected':selected,'next':'Source selected_env.sh and resume existing launcher/config'}),flush=True)

if __name__=='__main__':main()
