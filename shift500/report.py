"""Compare identical test frame sets, explicitly report goal met/unmet."""
import argparse
import json
from pathlib import Path
from .data import DOMAINS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    reports={v:json.loads((run/f'test_{v}.json').read_text()) for v in ('teacher','quality','compact')}
    anchor=reports['teacher']
    def keys(r):return [(s['domain'],s['sequence'],f['filename']) for s in r['sequences'] for f in s['per_frame']]
    rows=[]
    for v,r in reports.items():
        if keys(r)!=keys(anchor) or r['protocol']!=anchor['protocol'] or r['chunk_outputs']!=anchor['chunk_outputs']:
            raise ValueError('Cannot compare mismatched protocols/frame sets')
        prof=json.loads((run/f'profile_{v}.json').read_text())
        generated=sum(s['windows']*r['chunk_outputs'] for s in r['sequences'])
        used=sum(s['frames'] for s in r['sequences'])
        row={'model':v,'parameters_M':prof['parameters_M'],
             '1080p_GFLOPs_per_output_full_chunk':prof['arithmetic_GFLOPs_per_output'],
             '1080p_GFLOPs_per_useful_test_output':prof['arithmetic_GFLOPs_per_output']*generated/used}
        for d in DOMAINS:
            row[d+'_psnr']=r['summary'][d]['psnr']
            row[d+'_delta_teacher']=r['summary'][d]['psnr']-anchor['summary'][d]['psnr']
            row[d+'_temporal_l1']=r['summary'][d]['gt_relative_unwarped_temporal_l1']
        row['goal_met']=v!='teacher' and row['gopro_psnr']>=35 and row['1080p_GFLOPs_per_useful_test_output']<=500
        rows.append(row)
    (run/'comparison.json').write_text(json.dumps({'target':'GoPro >=35 dB and <=500 GFLOPs/useful output', 'results':rows},indent=2)+'\n')
    lines=['| Model | Params M | GFLOPs/full output | GFLOPs/useful output | GoPro | DVD | BSD | Goal met |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f'| {r["model"]} | {r["parameters_M"]:.3f} | {r["1080p_GFLOPs_per_output_full_chunk"]:.2f} | {r["1080p_GFLOPs_per_useful_test_output"]:.2f} | {r["gopro_psnr"]:.3f} | {r["dvd_psnr"]:.3f} | {r["bsd_psnr"]:.3f} | {r["goal_met"]} |')
    (run/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
