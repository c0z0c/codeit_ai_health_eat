import os
import json


annotation_root = 'D:/project_example/data/ai04-level1-project/train_annotations'
# annotation_root = 'D:/project_example/data/ai04-level1-project/1.Training/라벨링데이터/경구약제조합_5000종/train_annotations'

for root, dirs, files in os.walk(annotation_root):
    for file_name in files:
        # print(file_name)
        if file_name.endswith('.json'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # print(data['annotations'][0]['bbox'])
            if len(data['annotations'][0]['bbox'])<4:
                print(file_name)
                print(data['annotations'][0]['bbox'])
