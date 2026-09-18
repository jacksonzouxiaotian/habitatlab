"""Export metadata, or explicitly stage all local research assets for migration.

Default mode DOES NOT copy large data or modify source environments.
No delete, overwrite flag, package installation, upload, or robot command.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
DISK = Path('/media/xiaotian/ACD525D1B7A093D9')
CONDA = Path('/home/xiaotian/miniconda3')
SOURCES = {
    'habitat_lab': REPO,
    'vla': Path('/home/xiaotian/vla'),
    'narrow_passage_nav': Path('/home/xiaotian/navigation/narrow-passage-nav'),
    'habitat_data': DISK/'habitat_data',
    'pointnav_assets': DISK/'robot_nav_data/pointnav_assets',
    'vln': DISK/'robot_nav_data/vln',
}
ENVS = {name: CONDA/'envs'/name for name in ('habitat','navila','navila-eval')}
ENVS['navid'] = SOURCES['vln']/'reproduction/envs/navid'

def is_inside(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def validate_output(out, sources=SOURCES):
    if out.resolve() in (Path('/'),Path.home().resolve()):
        raise ValueError('Output must be a new dedicated directory, not a broad root')
    for source in sources.values():
        if is_inside(out, source):
            raise ValueError('Output inside a copied source would recurse: %s' % source)
    if out.exists():
        raise ValueError('Refusing existing output. Choose a new snapshot/staging directory.')

def record_command(cmd, directory, name, timeout=180):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run([str(x) for x in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=timeout)
        (directory/(name+'.txt')).write_text(result.stdout)
        (directory/(name+'.stderr.txt')).write_text(result.stderr)
        status={'argv':list(map(str,cmd)), 'returncode':result.returncode}
    except (OSError, subprocess.TimeoutExpired) as e:
        status={'argv':list(map(str,cmd)), 'error':str(e)}
    (directory/(name+'.status.json')).write_text(json.dumps(status,indent=2))
    return status

def export_env(item, out):
    name, prefix = item
    dest=out/'environments'/name
    if not (prefix/'bin/python').exists():
        return {'name':name,'error':'missing_python','prefix':str(prefix)}
    cmds={
        'conda-explicit': [CONDA/'bin/conda','list','-p',prefix,'--explicit'],
        'environment-full': [CONDA/'bin/conda','env','export','-p',prefix],
        'environment-history': [CONDA/'bin/conda','env','export','-p',prefix,'--from-history'],
        'pip-freeze': [prefix/'bin/python','-m','pip','freeze','--all'],
        'pip-list': [prefix/'bin/python','-m','pip','list','--format=json'],
        'pip-check': [prefix/'bin/python','-m','pip','check'],
    }
    return {'name':name,'prefix':str(prefix),'commands':{k:record_command(v,dest,k) for k,v in cmds.items()}}

def scan_links(root):
    rows=[]
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','build','node_modules','.cache','cache'}]
        for name in dirs+files:
            p=Path(base)/name
            if p.is_symlink():
                rows.append({'path':str(p),'target':os.readlink(p),'resolved':str(p.resolve()),'exists':p.exists()})
    return rows

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--copy-files',action='store_true',help='Stage all six source trees, no symlink dereference')
    p.add_argument('--pack-envs',action='store_true',help='Also conda-pack four environments; requires disk space')
    args=p.parse_args()
    out=args.output.absolute()
    validate_output(out)
    out.mkdir(parents=True)
    state={'timestamp':datetime.now(timezone.utc).isoformat(),'sources':{k:str(v) for k,v in SOURCES.items()},
           'environments':{k:str(v) for k,v in ENVS.items()},'mode':{'copy_files':args.copy_files,'pack_envs':args.pack_envs},
           'missing_sources':[str(v) for v in SOURCES.values() if not v.exists()]}
    (out/'migration_manifest.json').write_text(json.dumps(state,indent=2))
    for name,cmd in {'gpu':['nvidia-smi'],'os':['uname','-a'], 'disk':['df','-h',str(DISK)],
                     'mount':['findmnt','-T',str(DISK)]}.items():
        record_command(cmd,out/'system',name)
    repos={'habitat_lab':REPO,'NaVILA':SOURCES['vla']/'NaVILA', 'Open-Nav':SOURCES['vla']/'Open-Nav',
           'VLN-CE':SOURCES['vla']/'VLN-CE','habitat-lab-017':SOURCES['vla']/'habitat-lab-v0.1.7',
           'habitat-sim-017':SOURCES['vla']/'habitat-sim-v0.1.7','narrow_passage_nav':SOURCES['narrow_passage_nav']}
    for name in ('NaVid-VLN-CE','Uni-NaVid','AwareVLN'):
        repos[name]=SOURCES['vln']/'reproduction/sources'/name
    for name,repo in repos.items():
        if not (repo/'.git').exists(): continue
        for label,cmd in {'commit':['rev-parse','HEAD'],'status':['status','--short'],
                          'tracked-diff':['diff','--binary','HEAD'],
                          'untracked-files':['ls-files','--others','--exclude-standard'],
                          'submodules':['submodule','status','--recursive']}.items():
            record_command(['git','-C',repo]+cmd,out/'git'/name,label)
    with ThreadPoolExecutor(max_workers=4) as pool:
        state['environment_exports']=list(pool.map(lambda item:export_env(item,out),ENVS.items()))
    # Source link inventory; datasets are transferred whole, not recursively dereferenced.
    links=scan_links(REPO)+scan_links(SOURCES['vla'])
    (out/'source_symlinks.json').write_text(json.dumps(links,ensure_ascii=False,indent=2))
    fingerprints={}
    candidates=list((REPO/'configs/eagor').glob('*.yaml'))
    candidates += list((SOURCES['pointnav_assets']/'mp3d_narrow_v1').glob('*/*.json.gz'))
    candidates += [SOURCES['pointnav_assets']/'mp3d_v1/val/val.json.gz']
    candidates += list((SOURCES['vln']/'data/datasets/R2R_VLNCE_v1-3_preprocessed').glob('*/*.json.gz'))
    candidates += list((REPO/'examples/narrow_passage_rl/results/degnav_e2e_mp3d_narrow_v1_20260904').glob('seed*/train/best.pt'))
    candidates += [REPO/'eagor_outputs/stop_diagnosis_20260911_161253/episode_manifest.json']
    for path in candidates:
        if path.is_file(): fingerprints[str(path)]={'bytes':path.stat().st_size,'sha256':sha256(path)}
    (out/'critical_fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
    commands=[]
    for name,src in SOURCES.items():
        dest=out/'files'/name
        commands.append(['rsync','-a','--info=progress2',str(src)+'/',str(dest)+'/'])
    (out/'copy_commands.json').write_text(json.dumps(commands,indent=2))
    state['copy_results']=[]
    if args.copy_files:
        for cmd in commands:
            Path(cmd[-1]).mkdir(parents=True,exist_ok=True)
            code=subprocess.call(cmd)
            state['copy_results'].append({'argv':cmd,'returncode':code})
            if code: break
    state['pack_results']=[]
    if args.pack_envs:
        (out/'packed_envs').mkdir()
        for name,prefix in ENVS.items():
            archive=out/'packed_envs'/(name+'.tar.gz')
            # Editable source is deliberately included in files/, not magically
            # relocated by conda-pack. Rebind it after extraction (see README).
            cmd=[CONDA/'bin/conda-pack','-p',prefix,'-o',archive,'--ignore-editable-packages']
            status=record_command(cmd,out/'pack_logs',name,timeout=7200)
            if status.get('returncode')==0: status['sha256']=sha256(archive)
            state['pack_results'].append(status)
    (out/'migration_manifest.json').write_text(json.dumps(state,indent=2))
    print('Snapshot:',out)
    print('Source copies performed:',len(state['copy_results']))
    print('Environment packs attempted:',len(state['pack_results']))
    failed=bool(state['missing_sources']) or any(r['returncode'] for r in state['copy_results']) or any(r.get('returncode')!=0 for r in state['pack_results'])
    raise SystemExit(1 if failed else 0)

if __name__=='__main__': main()
