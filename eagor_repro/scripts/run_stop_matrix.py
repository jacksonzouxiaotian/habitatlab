"""Run fresh matched conditions, retaining every process log and exit status."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['smoke','matrix','videos'],required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    stage=args.root/args.phase;stage.mkdir(parents=True,exist_ok=False)
    manifest=Path('eagor_outputs/online_navigation_20260911/episode_manifest.json')
    if not (args.root/'episode_manifest.json').exists():shutil.copyfile(manifest,args.root/'episode_manifest.json')
    methods=['oracle_direction','eagor','grid','centroid','exploration'] if args.phase=='matrix' else ['eagor']
    jobs=[]
    if args.phase=='videos':
        cases=[('eagor','pLe4wQe7qrG_0'),('oracle_direction','EU6Fwq7SyZv_0'),('oracle_direction','8194nk5LbLH_0')]
    else:cases=[(m,None if args.phase=='matrix' else 'pLe4wQe7qrG_0') for m in methods]
    for method,key in cases:
        for stop in ('area','oracle_distance'):
            name=method+'__'+stop+('__'+key if key and args.phase=='videos' else '')
            command=[sys.executable,'-m','eagor_repro.scripts.run_online_navigation','--output',str(stage/name),
                '--methods',method,'--stop',stop,'--override','stop_diagnosis.enabled=true']
            if key:command+=['--episodes',key]
            if args.phase=='videos':command+=['--video']
            jobs.append(dict(name=name,command=command,log=str(stage/(name+'.log')),status='pending',attempt=1))
    ledger=stage/'processes.json'
    ledger.write_text(json.dumps(jobs,indent=2))
    def run(job):
        job['started_utc']=datetime.now(timezone.utc).isoformat()
        job['status']='running'
        with Path(job['log']).open('w') as f:
            done=subprocess.run(job['command'],stdout=f,stderr=subprocess.STDOUT)
        job.update(returncode=done.returncode,status='completed' if done.returncode==0 else 'failed',
            finished_utc=datetime.now(timezone.utc).isoformat())
        return job
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for future in as_completed(futures):
            job=future.result();ledger.write_text(json.dumps(jobs,indent=2))
            print(job['name'],job['status'],flush=True)
    assert all(j['status']=='completed' for j in jobs),f'Failed jobs retained in {ledger}; do not omit them.'
    print('Completed',len(jobs),'conditions in',stage,flush=True)


if __name__=='__main__':main()
