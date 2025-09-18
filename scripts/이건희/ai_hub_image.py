import os
import shutil
import json

# image_root = 'D:/project_example/data/ai04-level1-project/1.Training/원천데이터/경구약제조합_5000종/train_images_folder'
# output_dir = 'D:/project_example/data/ai04-level1-project/1.Training/원천데이터/경구약제조합_5000종/train_images'
#
# os.makedirs(output_dir,exist_ok=True)
# for root, _, files in os. walk(image_root):
#     # print(root, files)
#     # break
#     for file_name in files:
#         if not file_name.endswith('index.png'):
#             file_path = os.path.join(root, file_name)
#             # print(file_path)
#             # break
#             destination_path = os.path.join(output_dir, file_name)
#             if os.path.exists(output_dir):
#                 shutil.copy(file_path, destination_path)


# annotation_root = 'D:/project_example/data/ai04-level1-project/1.Training/라벨링데이터/경구약제조합_5000종/train_annotations'
annotation_root = 'D:/project_example/data/ai04-level1-project/train_annotations'


image_id_map = {}
next_image_id = 1
next_annotation_id = 1

for root, dirs, files in os.walk(annotation_root):
    for file_name in files:
        # print(file_name)
        if file_name.endswith('.json'):
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 이미지 id 새로 만들기 (1번부터)
                img_name = data['images'][0]['file_name']

                if img_name not in image_id_map:
                    image_id_map[img_name] = next_image_id
                    next_image_id += 1

                data["images"][0]["id"] = image_id_map[img_name]

                remove_file = False
                for ann in data.get('annotations', []):
                    if 'bbox' not in ann or len(ann['bbox']) != 4 or any(v <= 0 for v in ann['bbox']):
                        print(len(ann['bbox']))
                        remove_file = True
                        break
                if remove_file:
                    print(f"삭제: {file_path}")
                    os.remove(file_path)
                    continue  # 다음 파일로 넘어감


                # 해당 이미지 id를 annotataions에 기입
                data['annotations'][0]['image_id'] = image_id_map[img_name]
                data['annotations'][0]['id'] = next_annotation_id
                data['annotations'][0]['category_id'] = int(data['images'][0]['dl_idx'])


                # new_annotations = []
                # for ann in data['annotations']:
                #     if "bbox" in ann and all(dim > 0 for dim in ann["bbox"][2:]):  # width, height > 0
                #         ann['image_id'] = image_id_map[img_name]
                #         ann['id'] = next_annotation_id
                #         ann['category_id'] = int(data['images'][0]['dl_idx'])
                #         next_annotation_id += 1
                #         new_annotations.append(ann)

                # data['annotations'] = new_annotations


                # categories 데이터 기입
                data['categories'][0]['id'] = int(data['images'][0]['dl_idx'])
                data['categories'][0]['name'] = data['images'][0]['dl_name']
                
                next_annotation_id += 1                

                # if data['categories'][0]['id'] == 1 or data['categories'][0]['name']=='Drug':
                #     print(data['categories'][0]['name'])
                # data['categories'][0]['id'] = data['images'][0]['dl_idx']
                # data['categories'][0]['name'] = data['images'][0]['dl_name']
                # data['annotations'][0]['category_id'] = data['images'][0]['dl_idx']

                # # print(data['images'][0] )

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

            except Exception as e:
                print(f"오류 발생: {file_path} - {e}")
#         # break

# '''check data'''
# #
# annotation_root = 'D:/project_example/data/ai04-level1-project/train_annotations'
# for root, _, files in os.walk(annotation_root):
#     # print(files)
#     for file_name in files:
#         # print(file_name)
#         if file_name.endswith('.json'):
#             file_path = os.path.join(root, file_name)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                 if data['categories'][0]['id'] == 1 or data['categories'][0]['name']=='Drug':
#                     print(data['categories'][0]['name'])
#
#             except Exception as e:
#                 print(f"오류 발생: {file_path} - {e}")