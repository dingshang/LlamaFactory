"""Reproduce streaming preprocessing pipeline to find where pixel_values shrinks."""
import sys, os
sys.path.insert(0, 'src')
os.environ['DISABLE_VERSION_CHECK'] = '1'

from datasets import IterableDataset
from llamafactory.data.loader import _iter_dynamic_jsonl

ds = IterableDataset.from_generator(_iter_dynamic_jsonl, gen_kwargs={'file_path': 'data/streaming_mllm_demo.jsonl', 'is_drop_cache': False})

def conv(ex):
    return {'_prompt': ex['messages'][:-1], '_response': ex['messages'][-1:], '_images': ex['images'], '_system': None, '_tools': None, '_videos': [], '_audios': []}

column_names = list(next(iter(ds)).keys())
ds = ds.map(conv, batched=False, remove_columns=column_names)
column_names = list(next(iter(ds)).keys())

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('/mnt/data/models/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)

class PP:
    def __init__(self):
        self.calls = 0
    def __call__(self, examples):
        self.calls += 1
        from PIL import Image
        images = examples['_images'][0] or []
        decoded = []
        for img in images:
            if isinstance(img, str):
                # resolve relative path against data/ dir like converter does
                if not os.path.isabs(img):
                    img = os.path.join('data', img)
                decoded.append(Image.open(img))
            else:
                decoded.append(img)
        if decoded:
            pix = processor.image_processor(decoded, return_tensors='pt')
            pv_shape = tuple(pix['pixel_values'].shape)
            grid = pix['image_grid_thw'].tolist()
            print(f'  PREPROCESS call {self.calls}: pixel_values.shape={pv_shape} image_grid_thw={grid}')
        return {
            'input_ids': [[1,2,3]],
            'labels': [[1,2,3]],
            'attention_mask': [[1,1,1]],
            'images': examples['_images'],
        }

pp = PP()
ds = ds.map(pp, batched=True, batch_size=1, remove_columns=column_names)

for i, s in enumerate(ds):
    print(f'FINAL iter {i}: images={s["images"]}')
    if i >= 2:
        break