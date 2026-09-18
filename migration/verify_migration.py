"""Verify the 24 selected critical files, NOT the entire multi-GB dataset tree."""
import argparse
import hashlib
import json
from pathlib import Path

def remap(path, sources, destination):
    path=Path(path)
    for name,old in sorted(sources.items(),key=lambda item:len(item[1]),reverse=True):
        try: return destination/name/path.relative_to(old)
        except ValueError: pass
    raise ValueError('Unmapped critical path: %s'%path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--files-root',type=Path,help='Directory containing habitat_lab/, vla/, habitat_data/, etc.')
    group.add_argument('--original',action='store_true',help='Validate original machine paths instead')
    args=p.parse_args()
    state=json.loads((args.snapshot/'migration_manifest.json').read_text())
    expected=json.loads((args.snapshot/'critical_fingerprints.json').read_text())
    failures=[]
    for old,record in expected.items():
        path=Path(old) if args.original else remap(old,state['sources'],args.files_root)
        try:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            ok=path.stat().st_size==record['bytes'] and h.hexdigest()==record['sha256']
        except OSError: ok=False
        print('PASS' if ok else 'FAIL',path)
        if not ok: failures.append(str(path))
    print(json.dumps({'critical_files':len(expected),'passed':len(expected)-len(failures),'failures':failures},ensure_ascii=False))
    raise SystemExit(bool(failures))

if __name__=='__main__':main()
