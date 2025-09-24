from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import torch
import io
import json
from models import load_model
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = './object-detection/yolo/yolo_train1/weights/best.pt'
label2name = './data/label2name.json'
label2id = './data/label2id.json'

with open(label2name, 'r', encoding='utf-8') as f:
    label2name_dict = json.load(f)
with open(label2id, 'r', encoding='utf-8') as f:
    label2id_dict = json.load(f)

model = YOLO(model_path)


img_path = './data/images/val/K-000250-000573-002483-006192_0_2_0_2_75_000_200.png'
image = Image.open(img_path)
# print(image.)
a = model.predict(image)
# print(a[0].boxes.xyxy)

#
@app.post("/api/detect")
async def detect_pills(file: UploadFile = File(...)):
# def detect_pills(label2name_dict):
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        # image = Image.open(img_path)

        # 모델 예측
        results = model.predict(
            source=image,
            save=False,
            # show=True,
            conf=0.5
        )

        # 결과 포맷팅
        formatted_results = format_results(results, label2name_dict)

        return {"success": True, "results": formatted_results}
    except Exception as e:
        return {"success": False, "error": str(e)}
#
#
def format_results(raw_results, label2name_dict):
    # 여러분 모델의 출력을 앱이 이해할 수 있는 형태로 변환
    formatted = []
    for detection in raw_results:
        for box, conf, cls in zip(detection.boxes.xywh, detection.boxes.conf, detection.boxes.cls):
            formatted.append({
                "name": label2name_dict[str(int(cls.item()))],
                "code": label2id_dict[str(int(cls.item()))],
                "confidence": round(conf.item() * 100, 1),
                "bbox": {
                    "x": float(box[0])-(float(box[2])/2),
                    "y": float(box[1])-(float(box[3])/2),
                    "width": float(box[2]),
                    "height": float(box[3])
                },
                # 추가 정보들...
            })
    return formatted

# print(detect_pills(label2name_dict))