from ultralytics import YOLO
import torch
import os
import json
import cv2


def main(args):

    # 새로운 폴더 구조에서 모델 찾기
    base_project_dir = "./object-detection"
    yolo_project_dir = os.path.join(base_project_dir, "yolo")
    if args.model_path:
          model_path = args.model_path
          print(f"선택된 모델: {model_path}")

    else:
        if os.path.exists(yolo_project_dir):
            existing_runs = [d for d in os.listdir(yolo_project_dir) if os.path.isdir(os.path.join(yolo_project_dir, d))]
            if existing_runs:
                print(f"사용 가능한 YOLO 모델들: {existing_runs}")
                # 가장 최신 폴더 자동 선택
                latest_run = sorted(existing_runs)[-1]
                selected_run_dir = os.path.join(yolo_project_dir, latest_run)
                model_path = os.path.join(selected_run_dir, "weights", "best.pt")

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

                print(f"선택된 모델: {model_path}")
            else:
                raise FileNotFoundError(f"학습된 YOLO 모델을 찾을 수 없습니다: {yolo_project_dir}")
        else:
            raise FileNotFoundError(f"YOLO 프로젝트 폴더가 존재하지 않습니다: {yolo_project_dir}")

    # 학습된 모델 불러오기
    trained_model = YOLO(model_path)

    # 추론 실행
    if args.predict_one_image:
        # 이미지 하나만
        results = trained_model.predict(
            source=args.test_image_path,
            save=False,
            show=True,
            conf=0.5
        )

        result = results[0]
        plotted_image = result.plot()
        cv2.imshow("Predicted Image", plotted_image)
        cv2.waitKey(0)

    else:
        # 이미지 폴더 전체 추론
        results = trained_model.predict(
            source=args.test_image_folder,
            save=True,
            conf=0.5,
            project=selected_run_dir,  # 해당 모델 폴더에 저장
            name="test_visualizations",  # test_visualizations 폴더에 저장
            exist_ok=True
        )

        # 결과를 JSON으로 저장
        with open(args.label2id_path, 'r', encoding='utf-8') as f:
            label2id_dict = json.load(f)

        data = []
        for r in results:
            img_path = r.path
            for cls_label, cls_score, bbox in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh):
                image_id = img_path.split('\\')[-1]
                image_id = os.path.splitext(image_id)[0]
                label = int(cls_label)
                score = float(cls_score)
                x_mid, y_mid, width, height = bbox.tolist()
                x_min = x_mid - (width / 2)
                y_min = y_mid - (height / 2)

                r_dict = {
                    'image_id': image_id,
                    'category_id': int(label2id_dict[str(label)]),
                    'bbox': [x_min, y_min, width, height],
                    'score': score
                }
                data.append(r_dict)

        # JSON 파일을 해당 모델 폴더에 저장
        predictions_dir = os.path.join(selected_run_dir, "test_predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        output_json = os.path.join(predictions_dir, 'test.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"=====JSON 파일 저장 완료: {output_json}======")
        print(f"=====시각화 결과 저장: {os.path.join(selected_run_dir, 'test_visualizations')}======")

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
