#!/usr/bin/env python3
"""Create/freeze/run/score a three-dataset, no-ablation benchmark."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fair_benchmark.workflow import init_config,prepare,run_jobs,summarize,add_anchor


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--base',default='/data/pub/z00919662/motion_deblur');q.add_argument('--config',required=True)
    q=sub.add_parser('prepare');q.add_argument('--config',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--output',required=True);q.add_argument('--phase',choices=['smoke','full'],default='full');q.add_argument('--only',nargs='+');q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('anchor');q.add_argument('--output',required=True);q.add_argument('--job',required=True);q.add_argument('--mapping',required=True)
    q=sub.add_parser('score');q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='init':
        init_config(a.base,a.config);print('CONFIG_CREATED: review paths and checkpoint provenance before prepare')
    elif a.command=='prepare':
        f=prepare(a.config,a.output);print('FROZEN:',a.output);print('DATASET_ERRORS:',f['dataset_errors']);print('READY_JOBS:',sum(j['status']=='READY' for j in f['jobs']))
    elif a.command=='run':
        statuses=run_jobs(a.output,a.phase,a.only,a.device)
        return int(any(s['status']=='FAILED' for s in statuses))
    elif a.command=='anchor':
        r=add_anchor(a.output,a.job,a.mapping);print(r);return int(not r['passes'])
    else:
        r=summarize(a.output);print('REPORT:',str(Path(a.output)/'benchmark_report.md'))
        print('STATUS:',r['status']);return int(r['status']!='BENCHMARK_COMPLETE')
    return 0


if __name__=='__main__':raise SystemExit(main())
