"""Execution controls; changing device count preserves the global sample stream."""

def accumulation(global_batch, world):
    if world not in (1,2,4,8) or global_batch % world:
        raise ValueError('Use 1, 2, 4 or 8 GPUs dividing the global batch')
    return global_batch//world


def rank_indices(start, end, rank, world, global_batch=8):
    accumulation(global_batch,world)
    if not 0<=rank<world:raise ValueError('Invalid rank')
    return range(start*global_batch+rank,end*global_batch,world)


def choose_candidate(reports):
    valid=[r for r in reports if r.get('status')=='PASS' and r.get('amp_skipped_measured',r.get('amp_skipped_during_trial',1))==0
           and r.get('gpu_util_mean') is not None and r.get('gpu_util_min_rank') is not None
           and r.get('peak_memory_fraction',1)<.90]
    base=next((r for r in valid if r['world_size']==2 and r['activation_checkpointing']=='on'),None)
    if base is None:raise ValueError('Need a healthy two-GPU control trial')
    four=[r for r in valid if r['world_size']==4 and r['gpu_util_mean']>=60 and r['gpu_util_min_rank']>=40]
    if not four:raise ValueError('No healthy, sufficiently utilized four-GPU candidate; retain checkpoint and report trial failures')
    best4=min(four,key=lambda r:r['seconds_per_update'])
    # Eight cards need substantial incremental throughput and sustained activity.
    eight=[r for r in valid if r['world_size']==8
           and best4['seconds_per_update']/r['seconds_per_update']>=1.30
           and r['gpu_util_mean']>=70 and r['gpu_util_min_rank']>=50]
    selected=min(eight,key=lambda r:r['seconds_per_update']) if eight else best4
    if base['seconds_per_update']/selected['seconds_per_update']<1.15:
        raise ValueError('Scale-out speedup below 15%; keep checkpoint, report data wait/utilization bottleneck')
    return selected
