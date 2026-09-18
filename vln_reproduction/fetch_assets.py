"""Pin public asset metadata, then optionally download selected official assets.

This is acquisition, NOT a model/benchmark reproduction result.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')
SPECS = {
    'navid': ('Jzzhang/NaVid', ['navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split/*', 'README.md']),
    'uninavid': ('Jzzhang/Uni-NaVid', ['uninavid-7b-full-224-video-fps-1-grid-2/*', 'test_cases/**', 'Nav-Finetune/open_uninavid_sampled_500.json', 'README.md']),
    'awarevln': ('gwx22/AwareVLN-ck', ['awarevln/**', 'README.md']),
    'samples': ('Jzzhang/Uni-NaVid', ['test_cases/**', 'Nav-Finetune/open_uninavid_sampled_500.json', 'README.md']),
    'train_sample': ('Jzzhang/Uni-NaVid', ['Nav-Finetune/nav_videos/35150_002.mp4', 'Nav-Finetune/open_uninavid_sampled_500.json']),
    'clip_config': ('openai/clip-vit-large-patch14', ['*.json', 'vocab.json', 'merges.txt']),
    'bert_config': ('google-bert/bert-base-uncased', ['*.json', 'vocab.txt']),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('asset', choices=SPECS)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--metadata-only-files', action='store_true')
    args = parser.parse_args()
    repo, patterns = SPECS[args.asset]
    out = ROOT / 'assets' / args.asset
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'revision.json'
    if lock.exists():
        revision = json.loads(lock.read_text())['revision']
    else:
        revision = None
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    meta = {'repo': repo, 'revision': info.sha, 'timestamp': datetime.now(timezone.utc).isoformat(),
            'files': [{'path': f.rfilename, 'size': f.size, 'lfs': dataclasses.asdict(f.lfs) if dataclasses.is_dataclass(f.lfs) else f.lfs} for f in info.siblings]}
    lock.write_text(json.dumps(meta, indent=2))
    print(json.dumps({'repo': repo, 'revision': info.sha, 'file_count': len(info.siblings)}), flush=True)
    if args.download:
        if args.metadata_only_files:
            patterns = ['*.json', '**/*.json', '**/*.model', 'README.md']
        snapshot_download(repo, revision=info.sha, local_dir=str(out / 'snapshot'),
                          allow_patterns=patterns, max_workers=2)
        hashes = {}
        for path in sorted((out / 'snapshot').rglob('*')):
            if not path.is_file() or '.cache' in path.parts:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            hashes[str(path.relative_to(out / 'snapshot'))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        (out / 'download_verified.json').write_text(json.dumps(hashes, indent=2))
        print('DOWNLOAD_VERIFIED', args.asset, len(hashes), flush=True)

if __name__ == '__main__':
    main()
