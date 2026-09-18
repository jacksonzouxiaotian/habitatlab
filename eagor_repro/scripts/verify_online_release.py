"""Save executable regression/preflight evidence and unchanged-math hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('eagor_outputs/online_navigation_20260911/diagnostic_v1/source_snapshot'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    commands=[('tests',[sys.executable,'-m','eagor_repro.scripts.run_tests']),
        ('compileall',[sys.executable,'-m','compileall','-q','eagor_repro']),
        ('preflight',[sys.executable,'-m','eagor_repro.scripts.evaluate_eagor','--overlay','configs/eagor/mp3d_oracle.yaml','--preflight'])]
    for name,command in commands:
        completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (args.output/(name+'.log')).write_text(completed.stdout)
        results.append(dict(check=name,argv=command,returncode=completed.returncode))
        assert completed.returncode==0,(name,completed.stdout)
        if name=='preflight':
            start=completed.stdout.index('{');data=json.loads(completed.stdout[start:]);assert data['ready'],data
        print(name,'PASS',flush=True)
    unchanged=[]
    for directory in ('spherical','policies','controllers','sensors','perception'):
        for path in sorted((Path('eagor_repro')/directory).glob('*.py')):
            previous=args.reference/path
            assert previous.read_bytes()==path.read_bytes(),path
            unchanged.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'verification.json').write_text(json.dumps(dict(checks=results,unchanged_modules=unchanged,
        reference=str(args.reference),note='Reference is pre-final-diagnostic snapshot, not a pre-user git commit; dirty/untracked original repository is preserved.'),indent=2))
    for source in ('eagor_repro','configs/eagor'):
        shutil.copytree(source,args.output/'source_snapshot'/source,ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for source in ('eagor_repro','configs/eagor') for p in Path(source).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (args.output/'provenance.json').write_text(json.dumps(dict(argv=sys.argv,file_hashes=hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    (args.output/'workspace.diff').write_text(subprocess.check_output(['git','diff','--binary'],text=True))
    (args.output/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print('Unchanged math/policy/controller/sensor/perception files:',len(unchanged))


if __name__=='__main__':main()
