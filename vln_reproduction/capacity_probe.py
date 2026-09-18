"""GPU byte-capacity diagnostic only: does NOT load or run a model.

Allocates the published weight byte count as a uint8 buffer, then immediately
releases it. Success is not evidence that inference fits (no activations/KV).
Failure proves only that this byte allocation failed in this runtime/device state.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--allocate', action='store_true')
    args = p.parse_args()
    assets = ROOT / 'assets'
    nav_index = next((assets / 'navid/snapshot').glob('*/pytorch_model.bin.index.json'))
    nav_bytes = json.loads(nav_index.read_text())['metadata']['total_size']
    aware_index = assets / 'awarevln/snapshot/awarevln/llm/model.safetensors.index.json'
    aware_llm = json.loads(aware_index.read_text())['metadata']['total_size']
    aware_meta = json.loads((assets / 'awarevln/revision.json').read_text())
    # Component file sizes include small headers, explicitly an approximation.
    aware_aux = sum(f['size'] for f in aware_meta['files'] if f['path'] in ['awarevln/vision_tower/model.safetensors', 'awarevln/mm_projector/model.safetensors'])
    torch.cuda.init()
    record = {'kind':'capacity_diagnostic_not_model_inference', 'torch':torch.__version__, 'cuda':torch.version.cuda,
              'device':torch.cuda.get_device_name(), 'mem_get_info_after_context':torch.cuda.mem_get_info(), 'results':[]}
    for name, nbytes in [('NaVid_published_tensor_bytes', nav_bytes), ('AwareVLN_llm_tensor_bytes_plus_aux_file_bytes', aware_llm+aware_aux)]:
        result = {'name':name, 'bytes':nbytes, 'GiB':nbytes/2**30}
        if args.allocate:
            try:
                buffer = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
                torch.cuda.synchronize()
                result['allocation'] = 'passed_not_an_inference_pass'
                del buffer
            except torch.cuda.OutOfMemoryError as exc:
                result['allocation'] = 'cuda_out_of_memory'
                result['exception'] = str(exc)
            finally:
                torch.cuda.empty_cache()
        record['results'].append(result)
    out = ROOT / 'capacity' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    (out / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print('CAPACITY_RECORD', out)

if __name__ == '__main__':
    main()
