"""Freeze data/weights/settings before scoring; isolate model imports in subprocesses."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml
from .data import EXPECTED, prepare_dataset, overlap_audit, verify_files, rgb
from .metrics import PROTOCOL, score, collect_sequence, reference_check
from rtf_temporal.data import sha256
from rtf_temporal.posttrain import checkpoint_audit, write_json

CODE=Path(__file__).resolve().parents[1]


def git_identity(repo):
    repo=Path(repo)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=repo)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
    h=hashlib.sha256(diff)
    for name in sorted(untracked):
        p=repo/name
        if p.suffix in ('.py','.cu','.cpp','.h','.yaml','.yml'):
            h.update(name.encode());h.update(p.read_bytes())
    return {'commit':commit,'working_changes_sha256':h.hexdigest(),'tracked_dirty':bool(diff)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def init_config(base, output):
    base=Path(base).resolve();run=base/'runs/rtfocuser_causal_temporal_finetune_v1'
    manifest=json.loads((run/'manifest.json').read_text()) if (run/'manifest.json').exists() else {'roots':{}}
    def root(domain,default):
        p=Path(manifest.get('roots',{}).get(domain,str(base/'datasets'/default)))
        return p.parent if p.name.lower() in ('train','training') else p
    def dataset(domain,default,variant):
        p=root(domain,default)
        if domain=='bsd':
            candidates=[base/'datasets/BSD_3ms24ms',base/'datasets/BSD/BSD_3ms24ms',p]
            p=next((q for q in candidates if (q/'test').is_dir()),p)
        test=p/'test'
        children={q.name.lower():q.name for q in test.iterdir() if q.is_dir()} if test.is_dir() else {}
        layout='split_modalities' if 'blur' in children else 'sequence_modalities'
        lq='blur';gt='gt' if layout=='split_modalities' else 'sharp'
        if layout=='split_modalities':
            gt=next((children[k] for k in ('gt','sharp') if k in children),gt)
        return {'root':str(p),'layout':layout,'lq_dir':lq,'gt_dir':gt,'variant':variant,
                **({'variant_evidence':str(p) if '3ms24ms' in str(p).lower() else ''} if domain=='bsd' else {})}
    file_index={}
    for folder in [base/'weights',base/'benchmark/weights',base/'envs']:
        if folder.is_dir():
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.pth','.pt','.tar','.ckpt'):
                    file_index.setdefault(p.name,[]).append(str(p.resolve()))
    def weight(names,fallback):
        matches=sorted({p for name in names for p in file_index.get(name,[])})
        return matches[0] if len(matches)==1 else str(base/'weights'/fallback)
    spec={
        'shiftnet':('shiftnet_repo','https://github.com/dasongli1/Shift-Net',{'clip_length':48}),
        'dstnet':('dstnet_repo','https://github.com/xuboming8/DSTNet',{'clip_length':30,'tile':0,'tile_overlap':64}),
        'bsstnet':('bsstnet_repo','https://github.com/huicongzhang/BSSTNet',{'clip_length':48,'tile_overlap':64}),
        'rvrt':('rvrt_repo','https://github.com/JingyunLiang/RVRT',{'tile':[30,256,256],'tile_overlap':[2,20,20]}),
        'turtle':('turtle_repo','https://github.com/Ascend-Research/Turtle',{'tile':320,'tile_overlap':128,'architecture':'turtle_t1_arch.py'})}
    jobs=[]
    for domain in EXPECTED:
        jobs.append({'id':f'rtf_official_{domain}','kind':'rtf_official','domain':domain,
                     'checkpoint':str(base/'weights/GoPro_RT_Focuser_Standard_256.pth'),
                     'training_domains':['gopro'],'training_evidence':'Official GoPro_RT_Focuser_Standard_256 checkpoint',
                     'causality':'single_frame','inference':{},'python':sys.executable})
        jobs.append({'id':f'rtf_temporal_{domain}','kind':'rtf_temporal','domain':domain,
                     'checkpoint':str(run/'checkpoints/best_stable.pth'),
                     'training_domains':['gopro','bsd_3ms24ms','dvd'],'training_evidence':'Frozen training checkpoint + original manifest',
                     'causality':'causal','inference':{},'python':sys.executable})
    for kind,(folder,url,options) in spec.items():
        repo=base/'envs'/folder
        for domain in EXPECTED:
            source=domain if domain!='bsd' else ('gopro' if kind in ('shiftnet','bsstnet','rvrt') else 'unknown')
            names={'shiftnet':{'gopro':'net_gopro_deblur.pth','dvd':'net_dvd_deblur.pth'},
                'dstnet':{'gopro':'GOPRO.pth','dvd':'DVD.pth','unknown':'BSD.pth'},
                'bsstnet':{'gopro':'BSST_gopro.pth','dvd':'BSST_dvd.pth'},
                'rvrt':{'gopro':'005_RVRT_videodeblurring_GoPro_16frames.pth','dvd':'004_RVRT_videodeblurring_DVD_16frames.pth'},
                'turtle':{'gopro':'TURTLE_GOPRO_REQUIRES_VERIFIED_PATH.pth','dvd':'NO_VERIFIED_TURTLE_DVD_CHECKPOINT.pth','unknown':'TURTLE_BSD_REQUIRES_VERIFIED_EXPOSURE.pth'}}
            filename=names[kind][source]
            path=weight([filename],f'{kind}/{filename}')
            # DSTNet BSD.pth's published test YAML points to 1ms8ms. Do NOT assume 3ms24ms.
            job={'id':f'{kind}_{domain}','kind':kind,'domain':domain,'repo':str(repo),'repo_url':url,
                 'checkpoint':path,'training_domains':[source],
                 'training_evidence':url+'/blob/main/README.md' if source!='unknown' else 'UNKNOWN: verify checkpoint exposure/training source before classification',
                 'causality':'causal' if kind=='turtle' else 'offline','inference':copy.deepcopy(options),
                 'python':sys.executable,'reference_map':None}
            if kind=='bsstnet':job['auxiliary']={'raft':weight(['raft-things.pth'],'raft-things.pth')}
            if kind=='turtle':
                job['auxiliary']={'config':str(repo/'options'/('Turtle_Deblur_Gopro.yml' if domain=='gopro' else 'CHECKPOINT_MATCHED_CONFIG_REQUIRED.yml'))}
                if domain=='bsd':job['inference']['architecture']='turtle_arch.py'
            jobs.append(job)
    cfg={'protocol':'fair_video_deblur_rgb8_v1','training_run':str(run),
         'official_checkpoint':str(base/'weights/GoPro_RT_Focuser_Standard_256.pth'),
         'datasets':{d:dataset(d,{'gopro':'GoPro','bsd':'BSD_3ms24ms','dvd':'DVD'}[d],{'gopro':'blur','bsd':'3ms24ms','dvd':'standard'}[d]) for d in EXPECTED},
         'jobs':jobs,'note':'Review paths and provenance BEFORE prepare. No selection based on test PSNR.'}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    return cfg


def snapshot(path, folder):
    source=Path(path).expanduser().resolve()
    digest=sha256(source);dest=folder/(digest[:16]+'_'+source.name)
    if not dest.exists():shutil.copy2(source,dest)
    if sha256(dest)!=digest or sha256(source)!=digest:raise RuntimeError('Source changed while snapshotting')
    return str(dest),digest


def prepare(config, out):
    config=Path(config).resolve();cfg=yaml.safe_load(config.read_text())
    if cfg['protocol']!='fair_video_deblur_rgb8_v1' or set(cfg['datasets'])!=set(EXPECTED):
        raise ValueError('Exactly gopro/bsd/dvd required in this protocol')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Prepare requires a new output directory')
    run=Path(cfg['training_run']).resolve()
    if out==run or run in out.parents:raise ValueError('Do not write into training run')
    out.mkdir(parents=True);(out/'snapshots').mkdir()
    frozen={'protocol':cfg['protocol'],'metric_protocol':PROTOCOL,'configuration_sha256':sha256(config),
            'evaluation_code':git_identity(CODE),'datasets':{},'dataset_errors':{},'jobs':[]}
    shutil.copy2(config,out/'configuration.yaml')
    try:
        audit,training=checkpoint_audit(run,cfg['official_checkpoint'],out/'training_audit')
        frozen['training_audit']=audit
    except Exception as exc:
        training=None;frozen['training_audit_error']=str(exc)
        traceback.print_exc()
    for domain,spec in cfg['datasets'].items():
        try:frozen['datasets'][domain]=prepare_dataset(domain,spec,out/'datasets'/domain)
        except Exception as exc:
            frozen['dataset_errors'][domain]=str(exc);traceback.print_exc()
    if training is not None:
        overlap=overlap_audit(training,frozen['datasets'])
    else:
        overlap={'datasets':{d:{'status':'UNVERIFIABLE','reason':'training checkpoint/manifest audit failed'} for d in frozen['datasets']}}
    write_json(out/'overlap_audit.json',overlap);frozen['overlap_audit']=overlap
    ids=set()
    for source_job in cfg['jobs']:
        job=copy.deepcopy(source_job)
        if job['id'] in ids or not all(c.isalnum() or c in '_-' for c in job['id']):
            raise ValueError('Job IDs must be unique and path safe')
        ids.add(job['id'])
        try:
            if job['domain'] not in frozen['datasets']:raise ValueError('DATASET_UNAVAILABLE')
            if job['kind']=='rtf_temporal':
                if training is None:raise ValueError('TRAINING_PROVENANCE_UNVERIFIABLE')
                status=overlap['datasets'][job['domain']]['status']
                if status!='CLEAN':raise ValueError('TEST_NOT_INDEPENDENT:'+status)
                expected=frozen['training_audit']['checkpoints']['best_stable']['sha256']
                if sha256(job['checkpoint'])!=expected:raise ValueError('Not the audited best_stable checkpoint')
            if job['kind']=='rtf_official':
                from rtf_temporal.posttrain import OFFICIAL_SHA
                if sha256(job['checkpoint'])!=OFFICIAL_SHA:raise ValueError('Official RT checkpoint identity mismatch')
            if not job.get('training_domains') or not job.get('training_evidence'):raise ValueError('Training provenance fields required')
            job['checkpoint_snapshot'],job['checkpoint_sha256']=snapshot(job['checkpoint'],out/'snapshots')
            if job.get('repo'):
                job['repo']=str(Path(job['repo']).resolve());job['repo_identity']=git_identity(job['repo'])
            job['auxiliary_snapshots']={};job['auxiliary_sha256']={}
            for key,path in job.get('auxiliary',{}).items():
                job['auxiliary_snapshots'][key],job['auxiliary_sha256'][key]=snapshot(path,out/'snapshots')
            if job.get('reference_map'):
                mapping=json.loads(Path(job['reference_map']).read_text())
                for row in mapping:row['official_prediction_sha256']=sha256(row['official_prediction'])
                job['frozen_reference_map']=mapping
            job['status']='READY'
        except Exception as exc:job['status']='UNAVAILABLE';job['reason']=str(exc)
        job['fingerprint']=fingerprint(job);frozen['jobs'].append(job)
    write_json(out/'frozen.json',frozen);(out/'frozen.sha256').write_text(sha256(out/'frozen.json')+'\n')
    return frozen


def load_frozen(out):
    out=Path(out).resolve()
    if sha256(out/'frozen.json')!=(out/'frozen.sha256').read_text().strip():raise ValueError('Frozen protocol changed')
    f=json.loads((out/'frozen.json').read_text())
    if git_identity(CODE)!=f['evaluation_code']:raise ValueError('Evaluation code changed after freeze; use original checkout')
    return out,f


def verify_job(job):
    if sha256(job['checkpoint_snapshot'])!=job['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if job.get('repo') and git_identity(job['repo'])!=job['repo_identity']:raise ValueError('Upstream code changed after freeze')
    for k,p in job['auxiliary_snapshots'].items():
        if sha256(p)!=job['auxiliary_sha256'][k]:raise ValueError('Auxiliary checkpoint/config changed')


def sequence_job(out,job,seq,device):
    folder=out/'jobs'/job['id']/seq['name'];done=folder/'done.json';pred=folder/'frames'
    if done.exists():
        old=json.loads(done.read_text())
        if old['job_fingerprint']!=job['fingerprint']:raise ValueError('Stale sequence output')
        if {p.name:sha256(p) for p in pred.iterdir() if p.is_file()}!=old['prediction_hashes']:
            raise ValueError('Completed prediction files were modified')
    else:
        if folder.exists():
            diagnostic=out/'failed_attempts'/job['id']/(seq['name']+'_'+str(time.time_ns()))
            diagnostic.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(diagnostic))
        folder.mkdir(parents=True);job_path=folder/'job.json';write_json(job_path,job)
        command=[job.get('python',sys.executable),str(CODE/'tools/infer_fair_video.py'),'--job',str(job_path),
                 '--input',str(Path(seq['frames'][0]['blur']).parent),'--output',str(pred),'--device',device]
        write_json(folder/'command.json',command)
        with (folder/'inference.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        collect_sequence(seq,pred) # Exact names, dimensions and RGB8 format before marking complete.
        write_json(done,{'job_fingerprint':job['fingerprint'],'prediction_hashes':{p.name:sha256(p) for p in pred.iterdir()}})
    dest=out/'predictions'/job['id']/seq['name'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(pred,target_is_directory=True)
    print(f'SEQUENCE_COMPLETE {job["id"]}/{seq["name"]}',flush=True)


def run_jobs(root,phase='full',only=None,device='cuda:0'):
    out,frozen=load_frozen(root);statuses=[]
    for ds in frozen['datasets'].values():verify_files(ds)
    for job in frozen['jobs']:
        if only and job['id'] not in only:continue
        status={'id':job['id'],'domain':job['domain'],'status':job['status'],'reason':job.get('reason')}
        if job['status']=='READY':
            try:
                verify_job(job);dataset=frozen['datasets'][job['domain']]
                sequences=dataset['sequences'][:2] if phase=='smoke' else dataset['sequences']
                for seq in sequences:sequence_job(out,job,seq,device)
                if job.get('frozen_reference_map'):
                    for row in job['frozen_reference_map']:
                        if sha256(row['official_prediction'])!=row['official_prediction_sha256']:raise ValueError('Official reference changed')
                    anchor=reference_check(dataset,out/'predictions'/job['id'],job['frozen_reference_map'])
                    write_json(out/'jobs'/job['id']/'reference_check.json',anchor)
                    if not anchor['passes']:raise ValueError('OFFICIAL_ADAPTER_REFERENCE_MISMATCH')
                status['status']='SMOKE_COMPLETE' if phase=='smoke' else 'INFERENCE_COMPLETE'
            except Exception as exc:
                status['status']='FAILED';status['reason']=str(exc);traceback.print_exc()
        statuses.append(status)
        write_json(out/f'run_{phase}_status.json',statuses)
    return statuses


def summarize(root):
    out,frozen=load_frozen(root);rows=[]
    for domain,ds in frozen['datasets'].items():
        verify_files(ds)
        m=score(ds,out/'unused',out/'scores'/('input_'+domain),input_baseline=True)
        rows.append({'id':'input_'+domain,'domain':domain,'kind':'input','training_group':'input','causality':'single_frame','status':'MEASURED',**m})
    for job in frozen['jobs']:
        domain=job['domain'];target='bsd_3ms24ms' if domain=='bsd' else domain
        train=job.get('training_domains',[])
        group='joint_finetuned' if job['kind']=='rtf_temporal' else ('unknown_training' if 'unknown' in train else ('same_domain_pretrained' if target in train else 'cross_domain_pretrained'))
        row={'id':job['id'],'kind':job['kind'],'domain':domain,'training_group':group,'training_domains':train,
             'causality':job.get('causality'),'checkpoint_sha256':job.get('checkpoint_sha256'),
             'inference':job.get('inference'),'status':job['status'],'reason':job.get('reason')}
        if job['status']=='READY':
            try:
                verify_job(job)
                dataset=frozen['datasets'][domain]
                # Hash verification prevents using stale or partly overwritten predictions.
                for seq in dataset['sequences']:
                    folder=out/'jobs'/job['id']/seq['name'];done=json.loads((folder/'done.json').read_text())
                    if done['job_fingerprint']!=job['fingerprint'] or done['prediction_hashes']!={p.name:sha256(p) for p in (folder/'frames').iterdir()}:
                        raise ValueError('Prediction provenance mismatch')
                result=score(dataset,out/'predictions'/job['id'],out/'scores'/job['id'])
                row.update(result);row['status']='MEASURED'
                ref=out/'jobs'/job['id']/'reference_check.json'
                row['official_reference']='NOT_CHECKED'
                if ref.exists():
                    anchor=json.loads(ref.read_text());row['official_reference']='PASS' if anchor['passes'] else 'MISMATCH'
                    if not anchor['passes']:row['status']='PROTOCOL_MISMATCH'
                inp=next(r for r in rows if r['id']=='input_'+domain)
                row['delta_vs_input_db']=float(row['psnr_frame_mean'])-float(inp['psnr_frame_mean'])
                if row['delta_vs_input_db']<0:row['review_note']='Below input: inspect pairing, checkpoint, pixels and official anchor; do not explain away as domain gap'
            except Exception as exc:row['status']='INCOMPLETE_OR_FAILED';row['reason']=str(exc)
        rows.append(row)
    overall='BENCHMARK_COMPLETE' if not frozen['dataset_errors'] and all(r['status']=='MEASURED' for r in rows) else 'BENCHMARK_PARTIAL'
    report={'status':overall,'metric_protocol':PROTOCOL,'frozen_sha256':sha256(out/'frozen.json'),'rows':rows,
            'dataset_errors':frozen['dataset_errors'],
            'interpretation':'Same test data and metric, heterogeneous training and context. Not an architecture-only controlled comparison. No paper numbers mixed into measurements.'}
    write_json(out/'benchmark_report.json',report)
    fields=['id','domain','training_group','causality','status','frames','psnr_frame_mean','psnr_sequence_mean','ssim_frame_mean','official_reference','reason']
    with (out/'benchmark_table.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    text=['# Unified full-frame RGB8 benchmark','',PROTOCOL,'',
          'Training groups and causal/offline contexts must remain visible. Missing/failed cells have no numeric score. Official-anchor mismatches remain diagnostic only.','',overall,'']
    for domain in EXPECTED:
        text += [f'## {domain}','', '| Method | Training group | Context | Status | Frames | PSNR (frame mean) | PSNR (sequence mean) | Official anchor |', '|---|---|---|---|---:|---:|---:|---|']
        for r in rows:
            if r['domain']==domain and r['status']=='MEASURED':
                vals=[r['id'],r['training_group'],r.get('causality'),r['status'],r.get('frames','—'),r.get('psnr_frame_mean','—'),r.get('psnr_sequence_mean','—'),r.get('official_reference','—')]
                text.append('| '+' | '.join(str(x) for x in vals)+' |')
        text.append('')
        unavailable=[r for r in rows if r['domain']==domain and r['status']!='MEASURED']
        if unavailable:
            text += ['Unavailable / excluded cells:', ''] + [f"- {r['id']}: {r['status']}; {r.get('reason') or 'official-reference mismatch; inspect JSON diagnostics'}" for r in unavailable] + ['']
    (out/'benchmark_report.md').write_text('\n'.join(text)+'\n')
    return report


def add_anchor(root, job_id, mapping_path):
    """Append official-run evidence without changing frozen data, weights or inference settings."""
    out,frozen=load_frozen(root)
    job=next(j for j in frozen['jobs'] if j['id']==job_id)
    verify_job(job)
    evidence=json.loads(Path(mapping_path).read_text())
    if evidence['checkpoint_sha256']!=job['checkpoint_sha256']:
        raise ValueError('Official anchor uses a different checkpoint')
    if job.get('repo_identity') and evidence['repo_commit']!=job['repo_identity']['commit']:
        raise ValueError('Official anchor source revision differs')
    if not evidence.get('command') or not evidence.get('protocol_note'):
        raise ValueError('Record the actual official command and evaluation protocol')
    mapping=evidence['rows']
    folder=out/'anchors'/job_id;folder.mkdir(parents=True,exist_ok=False)
    for row in mapping:
        dest,digest=snapshot(row['official_prediction'],folder)
        row['official_prediction']=dest;row['official_prediction_sha256']=digest
    write_json(folder/'evidence.json',evidence)
    result=reference_check(frozen['datasets'][job['domain']],out/'predictions'/job_id,mapping)
    result['evidence_sha256']=sha256(folder/'evidence.json')
    write_json(out/'jobs'/job_id/'reference_check.json',result)
    return result
