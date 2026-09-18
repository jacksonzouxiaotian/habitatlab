"""Restore pinned external CODE into a fresh root; no models or environments.

Defaults to a printed plan. --apply clones new directories only, never resets
or overwrites existing repositories. Use full directory migration instead if
untracked third-party assets or environment/site-packages patches are needed.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent / 'external_sources'

def target_for(row, root):
    old = Path(row['path'])
    for prefix, name in [('/home/xiaotian/vla', 'vla'),
                         ('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction/sources', 'vln_sources')]:
        try:
            return root / name / old.relative_to(prefix)
        except ValueError:
            pass
    raise ValueError('Unexpected source root: ' + str(old))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.absolute()
    if root.exists() or root.is_symlink():
        parser.error('Use a NEW destination root, not existing migrated sources.')
    rows = json.loads((BASE / 'manifest.json').read_text())['repositories']
    for row in rows:
        dest = target_for(row, root)
        commands = [
            ['git', 'clone', '--no-checkout', row['url'], str(dest)],
            ['git', '-C', str(dest), 'checkout', '--detach', row['commit']],
            ['git', '-C', str(dest), 'submodule', 'update', '--init', '--recursive'],
        ]
        if row['patch']:
            patch = str(BASE / row['patch'])
            commands += [['git', '-C', str(dest), 'apply', '--check', patch],
                         ['git', '-C', str(dest), 'apply', patch]]
        if args.apply:
            if dest.exists() or dest.is_symlink():
                raise FileExistsError('Refusing existing destination: ' + str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
        for command in commands:
            print(shlex.join(command), flush=True)
            if args.apply:
                subprocess.run(command, check=True)
    print('Code only. Restore data/model paths and matching environments using README_MIGRATION_CN.md.')

if __name__ == '__main__':
    main()
