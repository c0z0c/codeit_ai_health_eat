import json
import matplotlib.pyplot as plt
import os
from matplotlib import patches


with open('./data/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

image_dir = './data/ai04-level1-project/test_images'


for data_dict in data:
    # print(data_dict.keys())
    file_name = data_dict['image_id'] + '.png'
    image_path = os.path.join(image_dir, file_name)
    x_min, y_min, w, h = data_dict['bbox']
    x_max = x_min + w
    y_max = y_min + h
    # Bounding Box 추가
    rect = patches.Rectangle(
        (x_min, y_min), w, h, linewidth=2, edgecolor="red", facecolor="none"
    )
    break