from ultralytics import YOLO
import torch
import os
import json
import cv2


def main(args):

    # 학습된 모델 불러오기
    trained_model = YOLO(args.model_path)

    # 추론 실행
    if args.predict_one_image:
        #이미지 하나만
        results = trained_model.predict(
            source=args.test_image_path,
            save=False,                             # 결과 이미지 저장
            show=True,
            conf=0.5                                # confidence threshold
        )
        # results 객체에서 첫 번째 이미지의 결과를 가져옴
        result = results[0]

        # result.plot() 메서드는 바운딩 박스와 라벨이 그려진 NumPy 배열을 반환
        plotted_image = result.plot()

        # OpenCV를 사용해 이미지를 화면에 띄움
        cv2.imshow("Predicted Image", plotted_image)

        # 키를 누를 때까지 창을 유지
        cv2.waitKey(0)

    else: # 이미지 폴더 전체를 활용할 경우에만, json파일로 저장하는 형식 (여기에 하나만 넣더라도 가능은 함)
        results = trained_model.predict(
            source=args.test_image_folder,
            save=True,
            conf=0.5
        )

        with open(args.label2id_path, 'r', encoding='utf-8') as f:
            label2id_dict = json.load(f)

        data = []
        # 결과 확인
        for r in results:
            img_path = r.path
            for cls_label, cls_score, bbox in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh):
                image_id = img_path.split('\\')[-1]
                image_id = os.path.splitext(image_id)[0]
                label = int(cls_label)
                score = float(cls_score)
                x_mid, y_mid, width, height = bbox.tolist()
                x_min = x_mid - (width/2)
                y_min = y_mid - (height/2)
                # print(int(label2id_dict[str(label)]), bbox)
                r_dict = {'image_id' : image_id,
                          # 'category_id' : int(label),
                          'category_id' : int(label2id_dict[str(label)]),
                          'bbox': [x_min, y_min, width, height],
                          'score' : score}
                data.append(r_dict)
        with open('./data/test.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("=====json파일 저장 완료======")


if __name__ == "__main__":
    class Args_yolo:
        def __init__(self):
            self.model_path = './runs/detect/train4/weights/best.pt'
            self.label2id_path = './data/label2id.json'
            # 이미지 하나만 사용할 경우 True 아니면 False
            self.predict_one_image = True
            self.test_image_folder = './data/ai04-level1-project/test_images'
            self.test_image_path = './data/ai04-level1-project/test_images/1.png'

    args = Args_yolo()
    main(args)
