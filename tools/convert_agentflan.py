import json
import os
import sys
from datasets import Dataset


file_path = sys.argv[1]  # /xxx/internlm/Agent-Flan/data
if file_path.endswith('/'):
    file_path = file_path[:-1]

ds = []
for file in os.listdir(file_path):
    if not file.endswith('.jsonl'):
        continue
    with open(os.path.join(file_path, file)) as f:
        dataset = f.readlines()
        for item in dataset:
            conv = json.loads(item)
            conv['messages'] = conv.pop('conversation')
            if 'id' in conv:
                conv.pop('id')
            ds.append(conv)

ds = Dataset.from_list(ds)
ds.save_to_disk(f'{file_path}_converted')
