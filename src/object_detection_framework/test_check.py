import json
import matplotlib.pyplot as plt
import os
from matplotlib import patches
import cv2
from pycocotools.coco import COCO

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False


train_annotation_path = './data/labels/train/train_pseudo.json'
image_dir = './data/images/train'

with open(train_annotation_path, 'r', encoding='utf-8') as f:
    coco = COCO()
    coco.dataset = json.load(f)
    coco.createIndex()
image_ids = coco.getImgIds()
for image_id in image_ids:
    print(image_id)
    image_info = coco.loadImgs(image_id)
    print(image_info)
    # anno_info = coco.loadAnns(image_id)
    annIds = coco.getAnnIds(imgIds=image_id)
    print(annIds)
    annotations = coco.loadAnns(annIds)
    print(annotations)
    image_filename = image_info[0]['file_name']
    print(image_filename)
    img = cv2.imread(os.path.join(image_dir,image_filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Create a figure and axes for plotting
    fig, ax = plt.subplots(1)

    # Display the image on the axes
    ax.imshow(img)

    if annotations:
        for ann in annotations:
            category_name = coco.loadCats(ann['category_id'])[0]['name']
            print(f"Category: {category_name}")

            # COCO bbox format is [x, y, width, height]
            xmin, ymin, width, height = ann['bbox']
            xmax = xmin + width
            ymax = ymin + height
            print(f"Bbox: [{xmin}, {ymin}, {xmax}, {ymax}]")

            # Create a rectangle patch
            try:

                if ann['pseudo_label']:
                    rect = patches.Rectangle((xmin, ymin), width, height,
                                         linewidth=2, edgecolor='blue', facecolor='none')
            except KeyError:
                rect = patches.Rectangle((xmin, ymin), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
            # Add the rectangle to the axes
            ax.add_patch(rect)
            try:
                if ann['pseudo_label']:
                # Add the category name as text
                    plt.text(xmin, ymin - 10, category_name, color='blue', fontsize=12)
            except KeyError:
                # pseudo_label 키가 없으면(정식 레이블로 간주) 빨간색으로 텍스트 표시
                plt.text(xmin, ymin - 10, category_name, color='red', fontsize=12)

        # Set the plot title and display
        plt.title(f"Image: {image_filename}")
        plt.axis('off')  # Turn off axes
        plt.show()

    else:
        print("해당 이미지에 대한 Annotation이 없습니다.")