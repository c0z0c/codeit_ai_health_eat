# Object Detection Framework

의료용 알약 검출을 위한 객체 검출 프레임워크입니다. Faster R-CNN과 YOLO 두 가지 모델을 지원하며, 데이터 전처리부터 모델 학습, 평가, 추론까지의 완전한 파이프라인을 제공합니다.

## 📋 실행 순서

프로젝트를 처음 시작할 때 다음 순서대로 실행하세요:

1. **데이터 다운로드**: `download_data.py`
2. **데이터 전처리**: `preprocessing.py`
3. **모델 학습**: `train.py`
4. **모델 추론**: `test.py`


## 📚 파일별 상세 설명

### 🚀 실행 파일들

#### 1. `download_data.py`
- **목적**: Kaggle에서 AI Hub 데이터셋을 자동으로 다운로드
- **기능**:
  - Kaggle API를 통한 데이터셋 다운로드
  - ZIP 파일 자동 압축 해제
- **사전 준비**: Kaggle API 키 설정 필요

#### 2. `preprocessing.py`
- **목적**: 원본 데이터를 모델 학습용 형태로 변환
- **주요 기능**:
  - 개별 JSON 파일들을 COCO 형식으로 병합
  - 학습/검증 데이터셋 분할 (기본 8:2 비율)
  - YOLO 형식 어노테이션 생성
  - 클래스 ID // Label // Name 매핑 파일 생성
  - 데이터셋 YAML 설정 파일 생성

#### 3. `train.py`
- **목적**: 모델 학습 실행의 진입점
- **지원 모델**:
  - Faster R-CNN (ResNet-50, ResNet-101 백본)
  - YOLO (YOLOv8 시리즈)
- **기능**: 사용자 입력을 받아 적절한 trainer 모듈 호출

#### 4. `test.py`
- **목적**: 학습된 모델을 사용한 추론 실행
- **기능**:
  - 단일 이미지 또는 폴더 단위 추론
  - 결과 시각화 및 JSON 형식 저장

### 🧠 모델 관련 파일들

#### 5. `models.py`
- **목적**: 모델 아키텍처 정의 및 팩토리 패턴 구현
- **포함 모델**:
  - `CustomFasterRCNN`: ResNet-50 기반
  - `FasterRCNN_resnet101`: ResNet-101 기반
- **기능**: 모델명으로 해당 모델 인스턴스 반환

#### 6. `trainer_fasterrcnn.py`
- **목적**: Faster R-CNN 모델 학습 파이프라인
- **주요 기능**:
  - 학습/검증 루프 구현
  - mAP, Precision, Recall 등 상세 평가 지표 계산
  - 체크포인트 저장/로드
  - WandB 연동 지원
  - 학습 곡선 시각화

#### 7. `trainer_yolo.py`
- **목적**: YOLO 모델 학습 파이프라인
- **기능**: Ultralytics YOLO 패키지를 활용한 간단한 학습 인터페이스

### 🧪 테스트 및 평가 파일들

#### 8. `tester_fasterrcnn.py`
- **목적**: Faster R-CNN 모델 추론 및 결과 생성
- **기능**:
  - 배치 단위 추론 처리
  - 신뢰도 임계값 적용
  - JSON 형식 예측 결과 저장
  - 시각화 결과 생성

#### 9. `tester_yolo.py`
- **목적**: YOLO 모델 추론
- **기능**: Ultralytics 패키지를 활용한 추론 및 결과 저장

#### 10. `metrics.py`
- **목적**: 객체 검출 전용 평가 지표 계산
- **포함 메트릭**:
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - 클래스별 AP (Average Precision)
  - Precision, Recall, F1-Score
  - IoU 기반 성능 분석
- **시각화**: 성능 지표 그래프 자동 생성

### 🛠️ 유틸리티 파일들

#### 11. `dataset.py`
- **목적**: 데이터 로딩 및 전처리 클래스
- **주요 클래스**:
  - `CustomCocoDataset`: COCO 형식 데이터 로더
  - `custom_collate_fn`: 배치 처리 함수
- **기능**: 이미지-라벨 매핑, 데이터 증강 지원

#### 12. `utils.py`
- **목적**: 공통 유틸리티 함수 모음
- **주요 기능**:
  - 어노테이션 파일 로딩
  - 예측 결과 시각화
  - 클래스 매핑 파일 생성
  - 이미지 검증 및 표시

### 🔧 기타 파일들

#### 13. `ai_hub_image.py`
- **목적**: AI Hub 데이터 전처리 전용 스크립트
- **기능**: 원본 데이터의 ID 매핑 및 정리

#### 14. `data_check.py`
- **목적**: 데이터 무결성 검사
- **기능**: 어노테이션 파일의 오류 검출

#### 15. `json2csv.py`
- **목적**: 예측 결과를 CSV 형식으로 변환
- **기능**: 제출용 CSV 파일 생성

#### 16. `test_check.py`
- **목적**: 테스트 결과 시각화 확인
- **기능**: 예측 결과의 바운딩 박스 시각화


## 🎯 사용법

### 1. 데이터 준비
```bash
python download_data.py
python preprocessing.py
```

### 2. 모델 학습
```bash
python train.py
# 프롬프트에서 모델 선택: FasterRCNN 또는 Yolo
# Faster R-CNN의 경우 백본 선택: Resnet-50 또는 resnet-101
# yolo의 경우, 현재 v8 버전 사용
```

### 3. 모델 평가 및 추론
```bash
python test.py
# 프롬프트에서 모델 및 백본 선택
```

## 📊 성능 평가

학습 과정에서 다음 지표들이 자동으로 계산됩니다:

- **mAP@0.5:0.95**: COCO 표준 평균 정밀도
- **클래스별 성능**: 각 클래스의 개별 성능 분석
- **Precision/Recall/F1**: 전통적인 분류 지표

## 📁 결과 파일

### 학습 결과
- `checkpoints/`: Faster-RCNN 모델 체크포인트
- `runs/`: YOLO 학습 로그 및 결과
- 학습 곡선 그래프 및 성능 분석 차트

### 추론 결과  
- `predictions/`: JSON 형식 예측 결과
- `visualizations/`: 바운딩 박스가 그려진 이미지
- CSV 형식 제출 파일

## ⚙️ 설정 옵션

각 모듈의 `Args` 클래스에서 다음 항목들을 조정할 수 있습니다:

- **학습 파라미터**: 에포크, 배치 크기, 학습률
- **모델 설정**: 백본 네트워크, 클래스 수
- **평가 설정**: 신뢰도 임계값, IoU 임계값
- **저장 옵션**: 체크포인트 주기, 시각화 여부

## 🤝 지원 모델

### Faster R-CNN
- ResNet-50 백본 (CustomFasterRCNN)
- ResNet-101 백본 (FasterRCNN_resnet101)
- 사전 훈련된 가중치 사용 가능

### YOLO
- YOLOv8 시리즈 (n, s, m, l, x)
- Ultralytics 구현 활용

## 📈 모니터링

- **WandB 연동**: 실시간 학습 모니터링
- **자동 시각화**: 성능 지표 그래프 생성
- **체크포인트 관리**: 최고 성능 모델 자동 저장

---

이 프레임워크는 의료용 알약 검출을 위한 완전한 솔루션을 제공하며, 연구 및 실무 환경에서 직접 활용 가능한 수준으로 구현되어 있습니다.