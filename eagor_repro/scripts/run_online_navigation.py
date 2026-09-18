"""Reproducible runner for the frozen ten-episode online-planning comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf
from eagor_repro.config import load_config
from eagor_repro.evaluation.habitat_evaluator import evaluate_objectnav


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',default=['oracle_direction'])
    parser.add_argument('--stop',choices=['area','area_depth','oracle_distance'],default='oracle_distance')
    parser.add_argument('--episodes',nargs='+',default=[])
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--override',action='append',default=[])
    args=parser.parse_args()
    cfg=OmegaConf.merge(load_config('configs/eagor/base.yaml'),OmegaConf.load('configs/eagor/mp3d_failure_attribution.yaml'),OmegaConf.load('configs/eagor/mp3d_online.yaml'))
    cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist(args.override))
    cfg.output.root=str(args.output);cfg.controller.stop_mode=args.stop;cfg.episode_keys=args.episodes;cfg.evaluation.save_video=args.video
    if args.output.exists():raise FileExistsError('Choose a fresh output directory: '+str(args.output))
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg,args.output/'config.yaml')
    shutil.copyfile(cfg.episode_manifest,args.output/'episode_manifest.json')
    source=args.output/'source_snapshot'
    shutil.copytree('eagor_repro',source/'eagor_repro',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('configs/eagor',source/'configs/eagor')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in ('eagor_repro','configs/eagor') for p in sorted(Path(base).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
    (args.output/'provenance.json').write_text(json.dumps(dict(argv=sys.argv,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),file_hashes=hashes),indent=2))
    (args.output/'workspace.diff').write_text(subprocess.check_output(['git','diff','--binary'],text=True))
    (args.output/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    result={}
    for method in args.methods:
        result[method]=evaluate_objectnav(cfg,method)
    (args.output/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
