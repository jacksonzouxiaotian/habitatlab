"""Validate downloaded official sample data and preprocessing, without generation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, AutoTokenizer
import imageio.v2 as imageio

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')

def main():
    assets = ROOT / 'assets'
    src = ROOT / 'sources/NaVid-VLN-CE'
    processor = CLIPImageProcessor.from_pretrained(str(src / 'navid/processor/clip-patch14-224'), local_files_only=True)
    output = {'kind':'data_and_preprocessing_audit_not_model_inference', 'samples':{}}
    for name in ('vln_1','tracking_1'):
        sample = assets / 'samples/snapshot/test_cases' / name
        files = sorted((sample/'images').glob('*.jpg'), key=lambda p:int(p.stem))
        frames = [np.array(Image.open(f).convert('RGB')) for f in files]
        batch = processor.preprocess(np.stack(frames), return_tensors='np')['pixel_values']
        output['samples'][name] = {'frames':len(files), 'indices':[int(p.stem) for p in files],
            'original_shapes':sorted(set(str(f.shape) for f in frames)), 'processed_shape':list(batch.shape),
            'processed_finite':bool(np.isfinite(batch).all()), 'input_color_order':'RGB',
            'capture_timestamps':'not_provided_in_official_sample; numeric filenames are frame indices, not measured time',
            'instruction':json.loads((sample/'instruction.json').read_text())}
    annotation = assets / 'samples/snapshot/Nav-Finetune/open_uninavid_sampled_500.json'
    records = json.loads(annotation.read_text())
    actions = Counter()
    invalid = []
    for sample in records:
        turns = sample['conversations']
        if [t['from'] for t in turns] != ['human','gpt']:
            invalid.append(sample['id'])
        answer = turns[-1]['value'].split()
        actions.update(answer)
        if len(answer)!=4 or any(x not in ('left','right','forward','stop') for x in answer):
            invalid.append(sample['id'])
    output['training_annotations'] = {'count':len(records), 'unique_ids':len({r['id'] for r in records}),
        'task_id_prefixes':dict(Counter(r['id'].split('_')[2] for r in records)), 'action_counts':dict(actions),
        'invalid_ids':invalid, 'sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
        'full_training_reproduction':False, 'loss_mask_and_backward_test':'not_run'}
    video = assets / 'train_sample/snapshot/Nav-Finetune' / records[0]['video']
    if video.exists():
        reader = imageio.get_reader(video)
        meta = reader.get_meta_data()
        frames = [frame for frame in reader]
        reader.close()
        output['training_video_smoke'] = {'sample_id':records[0]['id'], 'file':str(video), 'frames_decoded':len(frames),
            'fps':meta.get('fps'), 'duration':meta.get('duration'), 'shape':list(frames[0].shape),
            'sha256':hashlib.sha256(video.read_bytes()).hexdigest(), 'kind':'decode_only_not_training'}
    for name, subfolder in [('awarevln','awarevln/llm'),('navid','navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split')]:
        path = assets / name / 'snapshot' / subfolder
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=False)
        special = ['<BEGIN_OF_REASONING>','<BEGIN_OF_ACTION>'] if name=='awarevln' else ['<video_special>','</video_special>','<image_special>','</image_special>','[Navigation]','<image_sep>']
        output[name+'_tokens'] = {s:{'id':tokenizer.convert_tokens_to_ids(s),'encoded_without_bos':tokenizer.encode(s,add_special_tokens=False)} for s in special}
    out = ROOT/'sample_audits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    (out/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))
    print('SAMPLE_AUDIT',out)

if __name__ == '__main__': main()
