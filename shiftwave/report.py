"""Strict joint acceptance. A computed FLOPs pass is never a PSNR pass."""
import argparse
import json
from pathlib import Path
from shift500.data import sha256
from .model import MODEL_ID


def acceptance(c, report, profile):
    g=report['summary'].get('gopro',{})
    rr=[r for r in report['sequences'] if r['domain']=='gopro']
    used=sum(r['frames'] for r in rr)
    factor=sum(r['windows']*12 for r in rr)/used if used else float('inf')
    cost=profile['arithmetic_GFLOPs_per_output']*factor
    complete=g.get('frames')==1111 and g.get('sequences')==11
    quality=complete and g.get('rgb8_psnr',-1)>=c['target_gopro_rgb8_psnr']
    budget=cost<c['target_gflops'] and profile['arithmetic_GFLOPs_per_output']<c['target_gflops']
    return dict(status='TARGET_MET' if quality and budget else 'TARGET_NOT_MET',
                gopro_test_complete=complete,gopro_rgb8_psnr=g.get('rgb8_psnr'),
                gopro_float_psnr=g.get('float_psnr'),
                parameters_M=profile['parameters_M'],
                gflops_1080p_full_output=profile['arithmetic_GFLOPs_per_output'],
                gflops_1080p_per_useful_gopro_output=cost,tail_multiplier=factor,
                quality_pass=quality,flops_pass=budget,
                supplied_ours_s_reference_gflops=1490.,
                reference_note='User reference is FLOPs, not MACs. Its difference from our pinned 16/12 calculation is unresolved; do not relabel it.',
                human_action_required=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    c=json.loads((run/'config.json').read_text());r=json.loads((run/'test_student.json').read_text())
    if r.get('model_id')!=MODEL_ID or r['manifest_sha256']!=c['manifest_sha256'] or r['checkpoint_sha256']!=sha256(run/'student/latest.pth'):
        raise ValueError('Evaluation provenance mismatch')
    profiles=json.loads((run/'profiles.json').read_text())
    result=acceptance(c,r,profiles['student']);result['datasets']=r['summary']
    result['measured_teacher_gflops']=profiles['teacher']['arithmetic_GFLOPs_per_output']
    tp=run/'test_teacher_gopro.json'
    if tp.exists():
        t=json.loads(tp.read_text())
        def keys(rep):return [(s['sequence'],f['filename']) for s in rep['sequences'] if s['domain']=='gopro' for f in s['per_frame']]
        if t['protocol']!=r['protocol'] or keys(t)!=keys(r) or t['manifest_sha256']!=r['manifest_sha256'] or t['checkpoint_sha256']!=c['teacher_sha256']:
            raise ValueError('Teacher comparison protocol/provenance mismatch')
        result['teacher_gopro']=t['summary']['gopro']
    else:result['teacher_gopro']='NOT_AVAILABLE; see teacher evaluation log'
    (run/'FINAL_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
