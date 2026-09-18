"""Read-only source/data audit; outputs only to the new reproduction directory."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
from datetime import datetime, timezone

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')
DATA = ROOT.parent / 'data'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    out = ROOT / 'audits' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    results = {'timestamp': datetime.now(timezone.utc).isoformat(), 'seed': 42, 'sources': {}, 'datasets': {}}
    for repo in ('NaVid-VLN-CE', 'Uni-NaVid', 'AwareVLN'):
        src = ROOT / 'sources' / repo
        results['sources'][repo] = {arg: subprocess.check_output(['git', '-C', str(src)] + cmd, text=True).strip()
                                    for arg, cmd in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--short']), ('remote', ['remote', 'get-url', 'origin'])]}
        (out / (repo + '.diff')).write_text(subprocess.check_output(['git', '-C', str(src), 'diff'], text=True))
    for cmd, filename in [(['nvidia-smi'], 'gpu.txt'), (['df', '-h', str(ROOT), '/home/xiaotian'], 'disk.txt'), (['free', '-h'], 'memory.txt')]:
        (out / filename).write_text(subprocess.check_output(cmd, text=True))
    source_request = Path('/home/xiaotian/下载/VLN_Reproduction_Prompts_ZH.md')
    results['request_sha256'] = digest(source_request)
    (out / 'request.md').write_bytes(source_request.read_bytes())
    ds = DATA / 'datasets' / 'R2R_VLNCE_v1-3_preprocessed'
    for split in ('train', 'val_seen', 'val_unseen', 'test'):
        episodes = ds / split / (split + '.json.gz')
        gt_path = ds / split / (split + '_gt.json.gz')
        if not episodes.exists():
            results['datasets'][split] = {'available': False}
            continue
        with gzip.open(episodes, 'rt') as f:
            data = json.load(f)
        eps = data['episodes']
        ids = [str(e['episode_id']) for e in eps]
        scenes = sorted({e['scene_id'] for e in eps})
        missing = [s for s in scenes if not (DATA / 'scene_datasets' / s).exists()]
        gt = {}
        if gt_path.exists():
            with gzip.open(gt_path, 'rt') as f:
                gt = json.load(f)
        results['datasets'][split] = {'available': True, 'episode_count': len(eps), 'unique_ids': len(set(ids)), 'scenes': scenes,
            'missing_scenes': missing, 'episode_sha256': digest(episodes), 'gt_sha256': digest(gt_path) if gt else None,
            'missing_gt_ids': [i for i in ids if i not in gt] if gt else ids}
        if split == 'val_unseen':
            # Debug subsets only; no claim that this is official full-split sharding.
            ordered = sorted(eps, key=lambda e: str(e['episode_id']))
            random.Random(42).shuffle(ordered)
            for n in (1, 10, 100, len(ordered)):
                (out / ('r2r_val_unseen_%s_tasks.json' % n)).write_text(json.dumps({'seed':42, 'selection':'sorted string IDs then Python Random(42).shuffle; nested debug subsets', 'source_sha256':digest(episodes), 'episodes':ordered[:n]}, ensure_ascii=False, indent=2))
    results['rxr_present'] = (DATA / 'datasets' / 'RxR_VLNCE_v0').exists()
    (out / 'audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(out), 'sources':results['sources'], 'datasets':{k: {a:b for a,b in v.items() if a not in ('scenes','missing_gt_ids')} for k,v in results['datasets'].items()}}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
