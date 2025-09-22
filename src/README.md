---
layout: default
title: "알약 분석 프로그램 사용 가이드"
description: "알약 분석 프로그램 사용 가이드"
date: 2025-09-22
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 알약 분석 프로그램 사용 가이드

알약 이미지를 분석하여 약물 정보와 병용금기 정보를 제공하는 PyQt5 기반 GUI 프로그램입니다.

## 시연 동영상

[![알약 분석 프로그램 시연](https://img.youtube.com/vi/L_HuRQvuV9E/0.jpg)](https://youtu.be/L_HuRQvuV9E)

**[▶️ 시연 동영상 보기](https://youtu.be/vRsNxjkodzU)**

## 목차
1. [설치 요구사항](#1-설치-요구사항) <br/>
2. [프로그램 실행 방법](#2-프로그램-실행-방법) <br/>
3. [모델 엔진 변경 방법](#3-모델-엔진-변경-방법) <br/>
4. [출력 결과 형식](#4-출력-결과-형식) <br/>
5. [문제 해결](#5-문제-해결) <br/>

## 1. 설치 요구사항

### 1.1 필수 라이브러리

```bash
# Python 버전
Python 3.8 이상

# 필수 라이브러리 설치
pip install PyQt5>=5.15.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install Pillow>=9.0.0
pip install opencv-python>=4.7.0
pip install numpy>=1.24.0
pip install pandas>=1.5.0
pip install matplotlib>=3.6.0
pip install pytz
```

### 1.2 선택적 라이브러리 (모델에 따라)

```bash
# YOLO 모델 사용 시
pip install ultralytics>=8.0.0

# EfficientNet 모델 사용 시  
pip install timm
```

### 1.3 한번에 설치하기

`requirements.txt` 파일 생성:

```txt
PyQt5>=5.15.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.6.0
pytz
ultralytics>=8.0.0
timm
```

일괄 설치:

```bash
pip install -r requirements.txt
```

## 2. 프로그램 실행 방법

### 2.1 기본 실행

```bash
python main.py
```

### 2.2 실행 전 확인사항

다음 파일 구조가 필요합니다:

```
프로젝트폴더/
├── main.py
├── PillAnalysisEngine.py
└── python_modules/
    ├── modeling/
    │   └── fasterrcnn_resnet101/
    │       └── best.pt
    ├── data/
    │   ├── df_drug_118.pkl
    │   └── df_병용금기약물_20240813.pkl
    └── sampledata/
        └── 1.png
```

### 2.3 GUI 사용법

1. **이미지 선택**: "이미지 파일 선택" 버튼 클릭 (PNG, JPG, JPEG, BMP 지원)
2. **분석 시작**: "분석 시작" 버튼 클릭
3. **결과 확인**:
   - 왼쪽 상단: 탐지된 알약 박스가 표시된 원본 이미지
   - 왼쪽 하단: 개별 알약 이미지
   - 오른쪽: 약물 정보 및 병용금기 경고

## 3. 모델 엔진 변경 방법

### 3.1 1단계 모델만 사용 (탐지 + 분류)

`main.py`의 `init_engine()` 메서드 수정:

```python
def init_engine(self):
    try:
        py_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 모델 경로 변경
        model_1_stage_path = os.path.join(py_dir, "python_modules", "modeling", 
                                          "fasterrcnn_resnet101", "best.pt")
        
        # 또는 절대 경로
        # model_1_stage_path = "C:/models/my_model.pt"
        
        if os.path.exists(model_1_stage_path):
            self.engine = PillAnalysisEngine(model_1_stage_path)
            self.result_text.setText("✅ 분석 엔진이 준비되었습니다.")
```

### 3.2 2단계 모델 사용 (탐지 + 분류 분리)

```python
def init_engine(self):
    try:
        py_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1단계: 객체 탐지
        model_1_stage_path = os.path.join(py_dir, "python_modules", "modeling", 
                                          "yolo_v8", "best.pt")
        
        # 2단계: 분류
        model_2_stage_path = os.path.join(py_dir, "python_modules", "modeling",
                                          "efficientnet_b3", "best.pth")
        
        if os.path.exists(model_2_stage_path):
            self.engine = PillAnalysisEngine(model_1_stage_path, model_2_stage_path)
        else:
            self.engine = PillAnalysisEngine(model_1_stage_path)
```

### 3.3 모델 동작 방식

**1단계 모델만 사용:**
- 하나의 모델이 탐지와 분류 수행
- Faster R-CNN 등 사용

**2단계 모델 사용:**
- `model_1_stage`: 객체 탐지 (YOLO)
- `model_2_stage`: 분류 (EfficientNet, ResNet)

## 4. 출력 결과 형식

### 4.1 JSON 형식

```json
{
  "img_path": "이미지 경로",
  "bboxs": [
    {
      "class_id": 3543,
      "class_name": "타이레놀",
      "xyxy": [10, 20, 50, 60],
      "xywh": [10, 20, 40, 40],
      "detect_score": 0.95,
      "class_score": 0.92,
      "img": "base64_encoded_image",
      "drug_info": {
        "drug_N": "아세트아미노펜",
        "dl_name": "타이레놀정 500mg"
      },
      "ddi": null,
      "ddi_drug": {...}
    }
  ]
}
```

### 4.2 GUI 텍스트 결과

```
📊 분석 완료 - 2개의 알약이 탐지되었습니다.

📋 알약 #1
  • 분류: 타이레놀
  • 신뢰도: 0.92
  • 약물명: 아세트아미노펜
  • 제품명: 타이레놀정 500mg
  ⚠️ 다른 약물과의 상호작용 주의가 필요합니다!

🚨 중요 안내사항
• 병용금기 약물이 탐지되었습니다.
• 복용 전 반드시 의사나 약사와 상담하시기 바랍니다.
```

## 5. 문제 해결

### 5.1 일반적인 오류

**모델 파일을 찾을 수 없음**
- 경로 확인: `model_1_stage_path` 경로가 정확한지 확인
- 파일 존재 확인: 모델 파일이 실제로 존재하는지 확인

**메모리 부족**
- 더 작은 모델 사용
- GPU 메모리 확인

**분석 시간 지연**
- GPU 작동 확인
- 가벼운 모델로 변경

### 5.2 디버깅

콘솔 로그를 확인하여 문제 진단이 가능합니다.

## 6. 추가 정보

- 프로젝트 상세: [README.md](README.md)
- 엔진 코드: [PillAnalysisEngine.py](PillAnalysisEngine.py)
- 개발팀: 코드잇 AI 4기 4팀