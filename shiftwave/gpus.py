"""Select exactly two idle A100s or validate an explicit pair; never kill jobs."""
import argparse
import csv
import io
import subprocess


def query(kind, fields):
    return subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}',
                                    '--format=csv,noheader,nounits'],text=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate');a=p.parse_args()
    busy={r[0].strip() for r in csv.reader(io.StringIO(query('compute-apps','gpu_uuid'))) if r}
    available=[]
    for r in csv.reader(io.StringIO(query('gpu','index,uuid,name,memory.free,utilization.gpu'))):
        idx,uuid,name,free,util=[x.strip() for x in r]
        if 'A100' in name and uuid not in busy and int(free)>=60000 and int(util)<=10:
            available.append(idx)
    if a.validate:
        ids=a.validate.split(',')
        if len(ids)!=2 or len(set(ids))!=2 or not set(ids)<=set(available):
            raise SystemExit(f'Need two distinct idle A100s with >=60000 MiB free; available={available}')
    else:
        ids=available[:2]
        if len(ids)!=2:raise SystemExit(f'TWO_IDLE_GPUS_UNAVAILABLE; available={available}')
    print(','.join(ids))

if __name__=='__main__':main()
