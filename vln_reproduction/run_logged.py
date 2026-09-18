"""Run a command without a shell; preserve every attempt and its exit status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--cwd', default=os.getcwd())
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'attempts' / (stamp + '_' + args.label)
    out.mkdir(parents=True)
    record = {'argv': cmd, 'cwd': args.cwd, 'start_utc': stamp,
              'environment_overrides': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_DISABLE_XET', 'CONDA_PKGS_DIRS', 'PIP_CACHE_DIR', 'TMPDIR') if k in os.environ}}
    (out / 'command.json').write_text(json.dumps(record, indent=2))
    started = time.monotonic()
    with (out / 'output.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=args.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        record['pid'] = proc.pid
        (out / 'command.json').write_text(json.dumps(record, indent=2))
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
            record['interrupted'] = True
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-started)
    (out / 'result.json').write_text(json.dumps(record, indent=2))
    print('ATTEMPT_RECORD', out, flush=True)
    sys.exit(code if code >= 0 else 128-code)

if __name__ == '__main__':
    main()
