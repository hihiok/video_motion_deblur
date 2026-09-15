"""Select exactly two idle GPUs without touching existing jobs."""
import argparse
import csv
import io
import subprocess


def choose(requested=None):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    busy=set(line.strip() for line in busy.splitlines())
    available=[]
    for row in csv.reader(io.StringIO(raw)):
        i,uuid,name,used,total,util=[v.strip() for v in row]
        if uuid not in busy and float(used)<2048 and float(total)>75000 and float(util)<=10:
            available.append(i)
    selected=requested.split(',') if requested else available[:2]
    if len(selected)!=2 or len(set(selected))!=2 or not set(selected)<=set(available):
        raise RuntimeError(f'Need two idle >=80GB GPUs; idle eligible physical indices: {available}')
    return ','.join(selected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus');a=p.parse_args();print(choose(a.gpus))
