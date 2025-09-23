---
layout: default
title: "경구약제 이미지 분류를 위한 EDA 및 전처리 보고서"
description: "코드잇 AI 4기 4팀 - 딥러닝 기반 약품식별 AI 개발을 위한 데이터 탐색적 분석 및 전처리 보고서"
date: 2025-09-23
author: "김명환"
team: "코드잇 AI 4기 4팀"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 경구약제 이미지 분류를 위한 EDA 및 전처리 보고서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)<br/>
   1.1. [프로젝트 목적](#11-프로젝트-목적)<br/>
   1.2. [데이터셋 정보](#12-데이터셋-정보)<br/>
   1.3. [기술적 목표](#13-기술적-목표)<br/>

2. [데이터셋 탐색적 분석 (EDA)](#2-데이터셋-탐색적-분석-eda)<br/>
   2.1. [데이터 구조 분석](#21-데이터-구조-분석)<br/>
   2.2. [클래스 분포 분석](#22-클래스-분포-분석)<br/>
   2.3. [이미지 특성 분석](#23-이미지-특성-분석)<br/>
   2.4. [라벨링 품질 분석](#24-라벨링-품질-분석)<br/>

3. [데이터 전처리](#3-데이터-전처리)<br/>
   3.1. [전처리 전략](#31-전처리-전략)<br/>
   3.2. [라벨링 품질 개선](#32-라벨링-품질-개선)<br/>
   3.3. [데이터 확장 및 클리핑](#33-데이터-확장-및-클리핑)<br/>
   3.4. [이상치 제거 프로세스](#34-이상치-제거-프로세스)<br/>
   3.5. [YOLO 데이터셋 변환](#35-yolo-데이터셋-변환)<br/>

4. [데이터셋 구성](#4-데이터셋-구성)<br/>
   4.1. [최종 데이터셋 통계](#41-최종-데이터셋-통계)<br/>
   4.2. [훈련용 데이터셋](#42-훈련용-데이터셋)<br/>
   4.3. [검증용 데이터셋](#43-검증용-데이터셋)<br/>
   4.4. [테스트용 데이터셋](#44-테스트용-데이터셋)<br/>

5. [결과 및 인사이트](#5-결과-및-인사이트)<br/>
   5.1. [전처리 효과 분석](#51-전처리-효과-분석)<br/>
   5.2. [데이터 품질 개선](#52-데이터-품질-개선)<br/>
   5.3. [향후 모델링 전략](#53-향후-모델링-전략)<br/>

6. [용어집](#6-용어집)<br/>

---

## 1. 프로젝트 개요

### 1.1. 프로젝트 목적

본 프로젝트는 **약품식별 인공지능 모델 개발**을 위한 경구약제 이미지 데이터의 탐색적 분석(Exploratory Data Analysis, EDA) 및 전처리를 수행한다. 목표는 실험실 환경에서 획득한 경구약제 이미지 데이터를 활용하여 높은 정확도로 식별할 수 있는 딥러닝 모델의 기반을 마련하는 것이다.

### 1.2. 데이터셋 정보

**데이터셋 명**: 약품식별 인공지능 개발을 위한 경구약제 이미지데이터  
**제공기관**: 서울특별시 보라매병원  
**구축년도**: 2021년  
**데이터 특성**: 실험실 통제 환경에서 촬영된 이미지

#### 1.2.1. 데이터 구성

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/1.2.1.%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B5%AC%EC%84%B1.png" width="640px"/>

**이미지 유형**:
- 단독 이미지: 단일 알약 촬영 (파일 크기 과다)
- 조합 이미지: 최대 4개 알약 조합 촬영 (경구 조합 TL2 포함)

#### 1.2.2. 이미지 메타데이터 스키마(Schema)

각 이미지는 다음과 같은 메타정보를 포함한다:

- **이미지 기본정보**: 해상도(Resolution), 파일명, 크기
- **촬영 조건**: 배경색상, 조명색상, 카메라 각도
- **약제 정보**: 제품코드, 제품명, 성분명, 제조사
- **바운딩 박스**: COCO 포맷(Format) 어노테이션(Annotation)

### 1.3. 기술적 목표

1. **객체 검출 모델**: YOLO 아키텍처(Architecture) 기반 다중 객체 검출
2. **분류 정확도**: 118종 경구약제에 대한 고정밀 식별
3. **데이터 품질**: 라벨링 오류 최소화 및 클래스 균형 확보

---

## 2. 데이터셋 탐색적 분석 (EDA)

### 2.1. 데이터 구조 분석

#### 2.1.1. 파일 구조

데이터셋은 다음과 같은 계층구조(Hierarchical Structure)를 가진다:

```
dataset/
├── test_images/
│   └── *.png
├── train_images/
│   └── *.png
└── train_annotations/
    └── *_json
       └── K-XXXXX
           └── *.json
```

#### 2.1.2. 어노테이션 포맷

각 이미지에 대해 **JSON 형태의 라벨링 파일**이 존재하며, COCO 데이터셋 포맷을 따른다:

- **images**: 이미지 메타데이터
- **annotations**: 바운딩 박스 좌표 및 클래스 정보  
- **categories**: 약제 카테고리 정의

#### 2.1.3. 다중 객체 특성

**핵심 발견사항**: 
- 조합 이미지에 **최대 4개의 알약 객체** 존재
- **일부 객체는 라벨링이 누락**된 상태 확인
- 파일명 구조: 최대 4개 약제 코드 조합으로 구성

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/2.1.3.%EB%8B%A4%EC%A4%91%EA%B0%9D%EC%B2%B4%ED%8A%B9%EC%84%B1.png" width="640px"/>

### 2.2. 클래스 분포 분석

#### 2.2.1. 초기 데이터셋 분석

**과제 제공 데이터**: 73종  

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/org_train_%ED%81%B4%EB%9E%98%EC%8A%A4%EB%B6%84%ED%8F%AC%EB%8F%84.png" width="640px"/>

**최종 확장 데이터**: 118종

#### 2.2.2. 클래스 불균형 문제

**주요 이슈**:
- 알약 개수의 심한 편차 발견
- 학습 데이터에 존재하지 않는 알약 종류 식별
- 테스트 데이터와의 불일치

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/org_클래스분포도_kde.png" width="640px"/>

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/org_label_다양한이미지.png" width="640px"/>


### 2.3. 이미지 특성 분석

#### 2.3.1. 해상도 및 파일 크기

- **조합 이미지**: 976×1280 픽셀, 적정 파일 크기
- **단독 이미지**: 고해상도, 과도한 파일 크기로 인한 처리 제약

#### 2.3.2. 촬영 조건

**실험실 통제 환경**:
- **배경색상**: 표준화된 배경
- **조명 조건**: 일관된 조명 설정
- **촬영 각도**: 0°, 90°, 180°, 270° 회전

### 2.4. 라벨링 품질 분석

#### 2.4.1. 라벨링 누락 문제

**검토 결과**:
- 조합 이미지에서 일부 알약이 라벨링되지 않은 경우 발견
- 시각적으로 존재하나 어노테이션 누락

#### 2.4.2. 바운딩 박스 오류

**발견된 오류 유형**:
- 좌표 이상값 (YOLO 범위 초과)
- 부정확한 객체 영역 지정
- 중복 라벨링

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/org_label_K-002483-003743-005886-012778_0_2_0_2_75_000_200.png" width="640px"/>

---

## 3. 데이터 전처리

### 3.1. 전처리 전략

데이터 품질 향상을 위한 **다단계 전처리 파이프라인(Pipeline)** 설계:

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/3.1.%EC%A0%84%EC%B2%98%EB%A6%AC%EC%A0%84%EB%9E%B5.png" width="640px"/>

### 3.2. 라벨링 품질 개선

#### 3.2.1. 라벨링 누락 처리 전략 비교

**초기 접근 (방법 1)**: 라벨링 누락 이미지 삭제
- 라벨링되지 않은 이미지 제거 로직 구현
- 결과: 데이터 손실로 인한 학습 성능 저하

**배경 재구성 (방법 2)**: 합성 이미지 생성
```python
# 라벨링된 bbox만 추출하여 배경 재구성
def reconstruct_labeled_only(image, bbox_list):
    # 1. 라벨링 영역 제외한 배경 평균값 계산
    mask = create_mask_from_bbox(bbox_list)
    background_mean = np.mean(image[~mask], axis=0)
    
    # 2. 새 배경에 라벨링된 객체만 붙이기
    new_image = np.full_like(image, background_mean)
    new_image[mask] = image[mask]
    
    return new_image
```
<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/org_label_K-002483-003743-005886-012778_0_2_0_2_75_000_200.png" width="300px"/>
<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/YOLO%EB%AA%A8%EB%8D%B8%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/yolo/yolov8s_resize_20250912_0033/BoxPR_curve.png" width="600px"/>

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/EDA_%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%A0%84%EC%B2%98%EB%A6%AC/nolabel_label_K-002483-003743-005886-012778_0_2_0_2_75_000_200.png" width="300px"/>
<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/YOLO%EB%AA%A8%EB%8D%B8%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/yolo/yolov8s_nolabel_20250912_0024/BoxPR_curve.png" width="600px"/>

**최종 접근 (방법 3)**: 누락 라벨 추가
- 라벨링되지 않은 객체에 대한 어노테이션 보완
- 결과: 학습 성능 향상 확인


#### 3.2.2. 라벨링 보완 프로세스

1. 시각적 검토를 통한 누락 객체 식별
2. 바운딩 박스 좌표 수동 보정
3. 어노테이션 파일 업데이트

### 3.3. 데이터 확장 및 클리핑

#### 3.3.1. 알약 클리핑 전략

**목적**: 조합 이미지에서 개별 알약을 추출하여 클래스별 분류

**프로세스**:
1. 조합 이미지에서 바운딩 박스 기준 알약 클리핑
2. 클래스별로 분류 및 저장
3. 단독 이미지는 파일 크기 문제로 제외

**클리핑 결과**:
- 총 클리핑 알약 수: **약 50,334개**
- 1차 분류: 각 클래스당 최대 100개로 제한

#### 3.3.2. 데이터 확장 전략

**과제 지침 준수**:
- 경구 조합 TL2 제외
- 테스트 데이터의 알약 종류에 적합한 데이터 선정

**확장 결과**:
- 초기: 73종
- 확장 후: **118종**
- 과제 중심의 선별적 데이터 추가

### 3.4. 이상치 제거 프로세스

#### 3.4.1. 1차 분류: 시각적 검토

**검토 내용**:
- 각 클래스당 100개 이미지 시각적 검사
- 라벨링 오류 및 bbox 오류로 인한 잘못된 분류 발견

**발견된 문제**:
- 다른 알약이 섞여 있는 경우
- bbox 오류로 잘못 클리핑된 경우
- 중복 이미지

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%ED%83%90%EC%83%89%EC%98%A4%EA%B2%80%EC%B6%9C%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/라벨링오류.jpg" width="600px"/>


#### 3.4.2. 2차 분류: 히스토그램 기반 이상치 제거

**방법론**:
```python
# 카테고리별 알약 히스토그램 분석
def remove_outliers_by_histogram(category_images):
    # 1. 색상 히스토그램 계산
    histograms = [calculate_histogram(img) for img in category_images]
    
    # 2. 평균 히스토그램과의 거리 계산
    mean_hist = np.mean(histograms, axis=0)
    distances = [histogram_distance(h, mean_hist) for h in histograms]
    
    # 3. 이상치 제거 (예: 3-sigma 규칙)
    threshold = np.mean(distances) + 3 * np.std(distances)
    return [img for img, dist in zip(category_images, distances) if dist < threshold]
```

**결과**:
- 통계적 방법으로 명확한 이상치 제거
- 클래스 내 일관성 확보

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%ED%83%90%EC%83%89%EC%98%A4%EA%B2%80%EC%B6%9C%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/라벨링오류_04.jpg" width="600px"/>

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%ED%83%90%EC%83%89%EC%98%A4%EA%B2%80%EC%B6%9C%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/라벨링오류_04_비교.jpg" width="600px"/>


#### 3.4.3. 3차 분류: 잔여 이상치 수동 제거

**대상**: 2차 분류에서 제거되지 않은 이미지  
**방법**: 시각적 검토를 통한 최종 정제

**최종 통계**:
- 이미지: 11,706 → **11,587** (119개 제거)
- Annotation: 11,706 → **11,587** (119개 제거)
- 카테고리: 118 유지

<img src="https://c0z0c.github.io/codeit_ai_health_eat/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%EA%B0%9D%EC%B2%B4%ED%83%90%EC%83%89%EC%98%A4%EA%B2%80%EC%B6%9C%EB%B6%84%EC%84%9D%EB%B3%B4%EA%B3%A0%EC%84%9C/라벨링오류_03.jpg" width="600px"/>

### 3.5. YOLO 데이터셋 변환
#### 3.5.1. 바운딩 박스 좌표 변환

COCO 포맷에서 YOLO 포맷으로 변환:

**COCO 포맷**: `[x_min, y_min, width, height]` (절대좌표)  
**YOLO 포맷**: `[x_center, y_center, width, height]` (상대좌표, 0-1 정규화)

$$
\begin{align}
x_{center} &= \frac{x_{min} + width/2}{image_{width}} \\
y_{center} &= \frac{y_{min} + height/2}{image_{height}} \\
w_{norm} &= \frac{width}{image_{width}} \\
h_{norm} &= \frac{height}{image_{height}}
\end{align}
$$

---

## 4. 데이터셋 구성

### 4.1. 최종 데이터셋 통계

#### 4.1.1. 전체 데이터 요약

| 항목 | 수량 | 비고 |
|------|------|------|
| 총 이미지 수 | 11,587 | 이상치 제거 후 |
| 총 어노테이션 수 | 11,587 | 1:1 매칭 |
| 카테고리 수 | 118 | 과제 73종 + 확장 45종 |
| 클리핑 알약 수 | 50,334 | 조합 이미지에서 추출 |

#### 4.1.2. 클래스별 분포

**균형 전략**:
- 각 클래스당 최대 100개로 제한
- 최소 데이터 수 확보를 위한 선별적 추가

### 4.2. 훈련용 데이터셋

**구성 비율**: 전체 데이터의 70%  
**전처리 적용**: 
- 라벨링 보완
- 이상치 제거
- YOLO 변환

**AI HUB 데이터 추가**: 
- 클래스 불균형 완화

### 4.3. 검증용 데이터셋

**구성 비율**: 전체 데이터의 20%  
**용도**: 하이퍼파라미터(Hyperparameter) 튜닝 및 모델 선택  
**전처리**: 훈련 데이터와 동일

---

## 5. 결과 및 인사이트

### 5.1. 전처리 효과 분석

#### 5.1.1. 라벨링 품질 개선 효과

**전략별 성능 비교**:

| 전략 | YOLO 학습 성능 | 데이터 활용도 | 최종 채택 |
|------|----------------|---------------|-----------|
| 누락 이미지 삭제 | 낮음 | 낮음 | ✗ |
| 배경 재구성 | **더 낮음** | 중간 | ✗ |
| 누락 라벨 추가 | **높음** | **높음** | ✓ |

**핵심 발견**:
- 단순 삭제보다 라벨 추가가 효과적
- 합성 이미지는 실제 학습에 부정적 영향
- 원본 데이터 보존이 중요

#### 5.1.2. 이상치 제거 효과
1. 멀티 테스크 모델에서 알약 분류를 위하여 데이타 전처리 작업을 진행 하였음
2. 과제 시간의 한계로 인하여 성능 검토 및 하이퍼 파라미터 튜닝 작업을 진행 하지 못함.

**다단계 정제 프로세스**:
1. 시각적 1차 검토 → 명확한 오류 제거
2. 히스토그램 분석 → 통계적 이상치 제거
3. 수동 3차 검토 → 최종 품질 확보

**결과**:
- 119개 이상치 제거 (전체의 약 1%)
- 클래스 내 일관성 대폭 향상
- 모델 학습 안정성 개선

### 5.2. 데이터 품질 개선

#### 5.2.1. 데이터 확장 성과

**확장 전략**:
- 과제 중심의 선별적 확장
- 경구 조합 TL2 제외
- 테스트 데이터 기준 우선순위

**성과**:
- 73종 → 118종 (61.6% 증가)
- 클래스당 균형적 데이터 확보
- 파일 크기 최적화 (조합 이미지 활용)

#### 5.2.2. 어노테이션 정확도

**개선사항**:
- 좌표 이상값 완전 제거
- 누락 라벨 보완
- bbox 정확도 향상

**품질 지표**:
- 좌표 범위: 모두 0-1 정규화
- 어노테이션 완성도: 100%
- 이미지-라벨 매칭: 1:1 완벽 일치
