#!/usr/bin/env python3
"""Print idle A100 indices only; never stop or take over existing jobs."""
import argparse
import csv
import io
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-gpus', type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    occupied = {line.strip() for line in apps.splitlines() if line.strip()}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True)
    idle = []
    for row in csv.reader(io.StringIO(raw)):
        index, uuid, name, memory, utilization = [v.strip() for v in row]
        if 'A100' in name and int(memory) >= 70000 and int(utilization) < 10 and uuid not in occupied:
            idle.append(index)
    if not idle:
        raise SystemExit('NO_IDLE_A100_80GB: do not interrupt existing jobs')
    print(','.join(idle[:args.max_gpus]))


if __name__ == '__main__':
    main()
