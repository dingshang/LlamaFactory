"""Reproduce streaming preprocessing using actual Qwen2VL process_messages."""
import sys, os
sys.path.insert(0, 'src')
os.environ['DISABLE_VERSION_CHECK'] = '1'

from datasets import IterableDataset
from llamafactory.data.loader import _iter_dynamic_jsonl

ds = IterableDataset.from_generator(_iter_dynamic_jsonl, gen_kwargs={'file_path': 'data/streaming_mllm_demo.jsonl', 'is_drop_cache': False})

# Convert messages to _prompt/_response like align_dataset does
# (manual implementation, skipping the real DatasetAttr setup)

def conv(ex):
    msgs = ex['messages']
    images = ex['images']
    # Manually resolve paths
    resolved_images = []
    for img in images:
        if isinstance(img, str) and not os.path.isabs(img):
            img = os.path.join('data', img)
        resolved_images.append(img)
    return {
        '_prompt': msgs[:-1],
        '_response': msgs[-1:],
        '_images': resolved_images,
        '_system': None,
        '_tools': None,
        '_videos': [],
        '_audios': [],
    }

column_names = list(next(iter(ds)).keys())
ds = ds.map(conv, batched=False, remove_columns=column_names)
column_names = list(next(iter(ds)).keys())

# Now use real Qwen2VL mm_plugin
from transformers import AutoProcessor, AutoTokenizer
from llamafactory.data.template import Template
from llamafactory.data.mm_plugin import Qwen2VLPlugin

processor = AutoProcessor.from_pretrained('/mnt/data/models/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
plugin = Qwen2VLPlugin(image_token='<|image_pad|>', video_token=None, audio_token=None)

# Test process_messages directly with sample 0 (after fix)
for i, s in enumerate(ds):
    print(f'=== SAMPLE {i} ===')
    print(f'  _images = {s["_images"]}')

    # This is what _encode_data_example does
    images = s['_images']
    messages = s['_prompt'] + s['_response']

    # Call process_messages — this is where pixel_values get decoded and discarded
    processed_messages = plugin.process_messages(messages, images, [], [], processor)
    print(f'  After process_messages, messages[0].content[:100] = {processed_messages[0]["content"][:100]}...')

    # Now call get_mm_inputs with what collator would do
    mm = plugin.get_mm_inputs(
        images=images, videos=[], audios=[],
        imglens=[len(images)], vidlens=[0], audlens=[0],
        batch_ids=[[1,2,3]],  # dummy, just to test mm_inputs call
        processor=processor,
    )
    pv_shape = tuple(mm['pixel_values'].shape)
    grid = mm['image_grid_thw'].tolist()
    print(f'  After get_mm_inputs: pixel_values.shape={pv_shape} image_grid_thw={grid}')

    if i >= 2:
        break