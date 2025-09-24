# --- 이미지 처리 ---
import cv2
from PIL import Image, ImageFilter, ImageDraw

# --- PyTorch: 딥러닝 관련 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from ultralytics import YOLO

# --- 딥러닝 모델 ---
import timm

# --- 기본 라이브러리 ---
import os
import sys
import json
from pathlib import Path

# --- 데이터 분석 및 시각화 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 시간 관련 ---
import pytz
from python_modules.utils.debug_log import *

class PillAnalysisEngine:
    def __init__(self, model_1_stage_path, model_2_stage_path=None):
        def drive_root():
            """
            실행 파일(main.py)을 기준으로 2개 상위 디렉토리를 반환하는 함수입니다.
            - 로컬 환경: main.py가 있는 디렉토리의 2단계 상위
            - Colab 환경: /content/drive/MyDrive
            프로젝트 내에서 데이터, 모델, 설정 파일 등 경로를 일관되게 관리할 때 사용합니다.
            """    
            try:
                import google.colab
                from google.colab import drive
                COLAB_AVAILABLE = True
            except ImportError:
                COLAB_AVAILABLE = False
            
            if COLAB_AVAILABLE:
                root_path = os.path.join("/content/drive/MyDrive")
            else:
                # 실행 파일(main.py)의 경로를 기준으로 2단계 상위 디렉토리
                main_script_path = os.path.abspath(sys.argv[0])
                main_script_dir = os.path.dirname(main_script_path)  # src 디렉토리
                project_root = os.path.dirname(main_script_dir)      # codeit_ai_health_eat 디렉토리
                root_path = os.path.dirname(project_root)            # GoogleDrive 디렉토리
            
            return root_path
        
        self.DEBUG_ON = True
        # --- 시간대 설정 ---
        self.__kst = pytz.timezone('Asia/Seoul')

        # --- GPU 설정 ---
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__device_cpu = torch.device('cpu')
        
        self.main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.project_path = os.path.join(drive_root(), "codeit_ai_health_eat")
        self.modeling_path = os.path.join(self.main_script_dir, "python_modules", "modeling")
        self.data_path = os.path.join(self.main_script_dir, "python_modules", "data")
        self.model_1_stage_path = model_1_stage_path
        self.model_2_stage_path = model_2_stage_path

        if self.DEBUG_ON:
            DLOG.log(LV.TRACE,f"실행 스크립트: {sys.argv[0]}")
            DLOG.log(LV.TRACE,f"실행 스크립트 디렉토리: {self.main_script_dir}")
            DLOG.log(LV.TRACE,f"Drive Root: {drive_root()}")
            DLOG.log(LV.TRACE,f"프로젝트 경로: {self.project_path}")
            DLOG.log(LV.TRACE,f"모델링 경로: {self.modeling_path}")
            DLOG.log(LV.TRACE,f"데이터 경로: {self.data_path}")
            DLOG.log(LV.TRACE,f"1단계 모델 경로: {self.model_1_stage_path}")
            DLOG.log(LV.TRACE,f"2단계 모델 경로: {self.model_2_stage_path}")
            
        self.categorys = []
        self.database = self.init_database() # DB 초기화 클래스 개수 확인을 위하여 가장 먼저 로딩되어야함.

        if self.model_2_stage_path is None:
            self.model_1_stage = self.load_1_stage_model_fasterrcnn_resnet101()
            self.model_2_stage = None
        else:
            self.model_2_stage = self.load_2_stage_model_resnet()
            self.model_2_stage = self.load_2_stage_model_efficientnet_b3()
        
        self.transform_fasterrcnn_resnet101 = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
        self.transform = transforms.Compose([
            transforms.Resize(224),           # 짧은 변 기준 224로 리사이즈 (비율 유지)
            transforms.CenterCrop(224),       # 중앙에서 224x224 crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_image(self, image) -> dict:
        """입력 이미지를 분석하여 약물(알약) 탐지 및 분류를 수행합니다.
        Args:
            image
                문자열 파일 경로 (str): 이미지 파일의 경로
                numpy 배열 (np.ndarray): 이미지 데이터 (예: cv2.imread()로 읽은 배열)
                PIL 이미지 객체 (PIL.Image.Image)
                torch.Tensor: 이미지 텐서 (채널 순서 및 정규화 필요)            

        Returns:
            dict: 분석 결과
        """
        
        
        # # 1단계: 이미지 검증
        # validated_image = await self.validate_image(image)
        
        # # 2단계: 객체 탐지
        # detections = await self.detect_pills(validated_image)
        
        # # 3단계: 객체 분류
        # classifications = await self.classify_pills(validated_image, detections)
        
        # # 4단계: 결과 검증 및 DB 조회
        # results = await self.validate_and_enrich(classifications)

        # 1단계: 이미지 검증
        validated_image = self.validate_image(image)
        
        # 2단계: 객체 탐지
        if self.model_2_stage:
            detections = self.detect_pills(validated_image)
        else:
            detections = self.detect_pills_fasterrcnn_resnet101(validated_image)
            
        detections['img_path'] = image
        """
        detections
        ├─ org_img [PIL.Image.Image]
        └─ bboxs [list]
            ├─ [0] [dict]
            │   ├─ class_id [int]
            │   ├─ class_name [int]
            │   ├─ xyxy [list] x1, y1, x2, y2
            │   ├─ xywh [list] x1, y1, w, h
            │   ├─ score [float]
            │   └─ img [PIL.Image.Image]
            ├─ [1] [dict]
            │   └─ ... (다음 박스 정보)
            └─ [N] [dict]
                └─ ... (다음 박스 정보)
        """

        # 3단계: 객체 분류
        classifications = self.classify_pills(validated_image, detections)
        """
        ├─ org_img [Image]
        ├─ bboxs [list]
        │  ├─ [0] [dict]
        │  │  ├─ class_id [int]
        │  │  ├─ class_name [int]
        │  │  ├─ xyxy [list] x1, y1, x2, y2
        │  │  ├─ xywh [list] x1, y1, w, h
        │  │  ├─ detect_score [float]
        │  │  ├─ img [Image]
        │  │  ├─ class_probabilitie [ndarray]
        │  │  ├─ class_score [float32]
        """
     
        # 4단계: 결과 검증 및 DB 조회
        drug_info = self.validate_and_enrich(classifications)
        return self.format_response(drug_info)

    def validate_image(self, image) -> bool:
        """입력 이미지를 검증합니다.
        Args:
            image: 검증할 이미지

        Returns:
            bool: 이미지 검증 결과
        """
        # image가 경로 일수 있음.
        # PIL 이미지로 변환
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("지원되지 않는 이미지 형식입니다. (str, np.ndarray, PIL.Image.Image, torch.Tensor 중 하나여야 함)")
        
        # 필요하다면 전처리
        return image
    
    def detect_pills_fasterrcnn_resnet101(self, validated_image):
        """
        detections
        ├─ org_img [Image]
        ├─ bboxs [list]
        │  ├─ [0] [dict]
        │  │  ├─ class_id [int]
        │  │  ├─ class_name [int]
        │  │  ├─ xyxy [list] x1, y1, x2, y2
        │  │  ├─ xywh [list] x1, y1, w, h
        │  │  ├─ detect_score [float]
        │  │  ├─ img [Image]
        │  │  ├─ class_probabilitie [ndarray]
        │  │  ├─ class_score [float32]
        ├─ [1] [dict]
        │   └─ ... (다음 박스 정보)
        └─ [N] [dict]
                └─ ... (다음 박스 정보)
        """
        image = validated_image
        # 이미지 텐서 변환
        self.model_1_stage.eval()
        with torch.no_grad():
            image_tensor = self.transform_fasterrcnn_resnet101(image)
            images_batch = [image_tensor]
            
            result_detections = self.model_1_stage(images_batch)
            # DLOG.log(LV.TRACE,"result_detections",result_detections)
            
            detections={}
            detections['org_img'] = image
            detections['bboxs'] = []
            
            for res in result_detections:
                i = 0
                for label in res['labels']:
                    label = label.cpu().numpy().astype(int)
                    box = res['boxes'][i].cpu().numpy().astype(int)
                    score = res['scores'][i].cpu().numpy().astype(float)
                    #DLOG.log(LV.TRACE,f"label: {label}, box: {box}, score: {score}")
                    
                    x1, y1, x2, y2 = box
                    w = x2-x1
                    h = y2-y1
                    cropped = image.crop((x1, y1, x2, y2))
                    
                    class_name = self.database['categorys'][label]
                    class_probabilitie = score
                    
                    bbox_info ={
                        'class_id': label,
                        'class_name': label,
                        'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                        'xywh': [int(x1), int(y1), int(w), int(h)],
                        'detect_score': float(score),
                        'img': cropped,
                        'class_name': class_name,
                        'class_probabilitie': [class_probabilitie,],
                        'class_score': score,
                    }
                    detections['bboxs'].append(bbox_info)
                    #DLOG.log(LV.TRACE,bbox_info)
                    i += 1

        return detections
    
    def detect_pills(self, validated_image):
        """
        2단계: 객체 탐지
        설명:
          - 이 함수는 1단계 모델(YOLO)을 사용하여 이미지 내 알약(객체)을 탐지합니다.
          - 입력(validated_image)은 PIL.Image 또는 numpy.ndarray 가능.
          - 반환값(detections)은 다음 구조의 dict:
            {
              'org_img': PIL.Image.Image,   # 원본(혹은 모델이 내부적으로 리사이즈/패딩한) 이미지
              'bboxs': [                    # 각 탐지된 박스의 리스트
                {
                  'class_id': int,          # 모델이 반환한 클래스 인덱스 (초기값)
                  'class_name': int,        # 현재는 class_id와 동일 (후속 분류에서 실제 이름으로 교체)
                  'xyxy': [x1,y1,x2,y2],    # 좌표 (정수)
                  'xywh': [x,y,w,h],        # x,y는 왼쪽상단
                  'detect_score': float,    # 탐지 신뢰도
                  'img': PIL.Image.Image    # 박스 크롭 이미지
                }, ...
              ]
            }
        """
        import matplotlib.patches as mpatches
        if self.model_1_stage is None:
            raise ValueError("1단계 모델이 로드되지 않았습니다.")
        
        # 이미지가 PIL 이미지인 경우 numpy 배열로 변환
        if isinstance(validated_image, Image.Image):
            validated_image = np.array(validated_image)
            
        # 객체 탐지 수행
        with torch.no_grad():
            result_detections = self.model_1_stage(validated_image, verbose=False)

        current_img = Image.fromarray(result_detections[0].orig_img.copy())
        img_width, img_height = current_img.size

        detections={}
        detections['org_img'] = current_img
        detections['bboxs'] = []
        for box in result_detections[0].boxes:
            #DLOG.log(LV.TRACE,box.xyxy, box.conf, box.cls)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls.item())
            score = box.conf.item()
            label = f" {score:.2f}"
            color = 'red'  # 빨강색
            x1, y1, x2, y2 = xyxy
            w = x2-x1
            h = y2-y1
            
            # int()로 확실하게 정수 변환
            rect = mpatches.Rectangle((int(x1), int(y1)), int(w), int(h),
                                linewidth=2, edgecolor=color, facecolor='none')
            cropped = current_img.crop((x1, y1, x2, y2))
            bbox_info={
                'class_id': cls,
                'class_name': cls,
                'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                'xywh': [int(x1), int(y1), int(w), int(h)],
                'detect_score': float(score),
                'img': cropped
            }
            detections['bboxs'].append(bbox_info)
            
            # 개별 객체 출력
            # plt.figure(figsize=(4, 4))
            # plt.imshow(cropped)
            # plt.title(f"Crop {x1},{y1},{x2},{y2}")
            # plt.axis('off')
            # plt.show()
            
        """
        detections
        ├─ org_img [PIL.Image.Image]
        └─ bboxs [list]
            ├─ [0] [dict]
            │   ├─ class_id [int]
            │   ├─ class_name [int]
            │   ├─ xyxy [list] x1, y1, x2, y2
            │   ├─ xywh [list] x1, y1, w, h
            │   ├─ score [float]
            │   └─ img [PIL.Image.Image]
            ├─ [1] [dict]
            │   └─ ... (다음 박스 정보)
            └─ [N] [dict]
                └─ ... (다음 박스 정보)
        """
        return detections
    
        # 3단계: 객체 분류
    def classify_pills(self, validated_image, detections):
        """
        3단계: 객체 분류
        설명:
          - 2단계(탐지)에서 잘라낸 각 박스 이미지를 2단계 분류기(model_2_stage)에 넣어
            약(알약) 클래스 확률 및 최종 클래스 예측을 수행합니다.
          - model_2_stage가 로드되어 있지 않으면 입력 detections를 그대로 반환합니다.
        입력:
          - validated_image: 원본 이미지 (사용하지 않지만 인터페이스 일관성을 위해 전달됨)
          - detections: detect_pills/ detect_pills_fasterrcnn_resnet101의 반환 dict
              {
                'org_img': PIL.Image.Image,
                'bboxs': [
                  {'img': PIL.Image.Image, 'class_id': ..., 'class_name': ..., 'detect_score': ...}, ...
                ]
              }
        반환:
          - 동일한 detections dict에 아래 필드들을 추가/갱신하여 반환:
            - class_id: 예측 클래스 인덱스 (int)
            - class_name: 데이터베이스의 카테고리 이름/ID (str 또는 int)
            - class_probabilitie: 클래스별 확률 배열 (numpy.ndarray, 소수점 반올림)
            - class_score: 예측된 클래스의 확률 (float)
        """
        
        if not self.model_2_stage:
            return detections
        
        for box in detections['bboxs']:
            image = box['img']
            input_tensor = self.transform(image).unsqueeze(0).to(self.__device)    
            with torch.no_grad():
                result_classify = self.model_2_stage(input_tensor)
            
            probabilities = torch.nn.functional.softmax(result_classify[0], dim=0)
            class_id = torch.argmax(probabilities).item()
            class_name = self.database['categorys'][class_id]
            class_probabilitie = probabilities.detach().cpu().numpy()
            box['class_id'] = class_id
            box['class_name'] = class_name
            box['class_probabilitie'] = np.round(class_probabilitie, 4)
            box['class_score'] = float(np.round(class_probabilitie[class_id], 4))
    
        """
        ├─ org_img [Image]
        ├─ bboxs [list]
        │  ├─ [0] [dict]
        │  │  ├─ class_id [int]
        │  │  ├─ class_name [int]
        │  │  ├─ xyxy [list] x1, y1, x2, y2
        │  │  ├─ xywh [list] x1, y1, w, h
        │  │  ├─ detect_score [float]
        │  │  ├─ img [Image]
        │  │  ├─ class_probabilitie [ndarray]
        │  │  ├─ class_score [float32]
        """    
        return detections

    def validate_and_enrich(self, classifications):
        """분류된 객체를 검증하고 DB 조회를 통해 추가 정보를 얻습니다.
        Args:
            classifications: 분류된 객체 목록

        Returns:
            list: 검증 및 추가 정보가 포함된 객체 목록
        """
        category_ids = []
        for box in classifications['bboxs']:
            category_id = box['class_name']
            category_ids.append(category_id)

        for box in classifications['bboxs']:
            box['drug_info'] = None
            box['ddi'] = None
            box['ddi_drug'] = None
            
        ddi_result = {}
        if len(category_ids) > 0:
            ddi_result = self.find_ddi(category_ids)
            
            for drug in ddi_result['drug']:
                #DLOG.log(LV.TRACE,drug['category_id'], drug['di_edi_code'], drug['drug_N'], drug['dl_name'])
                for box in classifications['bboxs']:
                    if box['class_name'] == drug['category_id']:
                        box['drug_info'] = drug
                
            for ddi in ddi_result['ddi']:
                #DLOG.log(LV.TRACE,ddi['category_id'], ddi['제품코드A'], ddi['제품명A'], ddi['제품코드B'], ddi['제품명B'])
                for box in classifications['bboxs']:
                    if box['class_name'] == ddi['category_id']:
                        box['ddi'] = ddi
                
            for ddi_drug in ddi_result['ddi_drug']:
                #DLOG.log(LV.TRACE,ddi_drug['category_id'], ddi_drug['제품코드A'], ddi_drug['제품명A'], ddi_drug['제품코드B'], ddi_drug['제품명B'])
                for box in classifications['bboxs']:
                    if box['class_name'] == ddi_drug['category_id']:
                        box['ddi_drug'] = ddi_drug            
            
        return classifications

    def find_ddi(self, category_ids):
        """
        한글 성분명 리스트를 받아서 병용금기 약물 조합을 찾는 함수
        
        Args:
            df: 병용금기 데이터프레임
            korean_ingredient_list: 사진에서 인식된 한글 성분명 리스트
        
        Returns:
            병용금기 조합 정보가 담긴 리스트 (중복 제거됨)
        """
        df_drug = self.database['df_drug']
        df = self.database['td_interaction']
        
        df_drug = df_drug[df_drug['category_id'].isin(category_ids)]
        # df_drug.head_att(10)
        
        # DLOG.log(LV.TRACE,'-' * 80)

        df_drug['code'] = df_drug['di_edi_code'].str.split(',').str[0].astype(int)
        codes = df_drug['code'].tolist()
        code_to_category = dict(zip(df_drug['code'], df_drug['category_id']))
        
        df_ddi = pd.DataFrame()        
        df_drug_ddi = pd.DataFrame()
        for code_a in codes:
            mask_a = df['제품코드A'] == code_a
            for code_b in codes:
                mask_b = df['제품코드B'] == code_b
                if code_a == code_b: 
                    continue
                # 매칭되는 행들을 가져오기
                matched_rows = df[mask_a & mask_b].copy()
                if not matched_rows.empty:
                    # category_ids 컬럼 추가
                    matched_rows['category_id'] = code_to_category.get(code_a, None)
                    df_ddi = pd.concat([df_ddi, matched_rows], axis=0)
                
                # ddi_drug에도 추가
                matched_drug_rows_a = df[mask_a].copy()
                if not matched_drug_rows_a.empty:
                    matched_drug_rows_a['category_id'] = code_to_category.get(code_a, None)
                    df_drug_ddi = pd.concat([df_drug_ddi, matched_drug_rows_a], axis=0)

                matched_drug_rows_b = df[mask_b].copy()
                if not matched_drug_rows_b.empty:
                    matched_drug_rows_b['category_id'] = code_to_category.get(code_b, None)
                    df_drug_ddi = pd.concat([df_drug_ddi, matched_drug_rows_b], axis=0)

        result = {}
        result['drug']=df_drug.drop_duplicates().copy().to_dict(orient='records')
        result['ddi'] = df_ddi.drop_duplicates().copy().to_dict(orient='records')
        result['ddi_drug'] = df_drug_ddi.drop_duplicates().copy().to_dict(orient='records')
        
        # DLOG.log(LV.TRACE,'ddi=',len(result['ddi']))
        # result['ddi'].head_att(20)
        # DLOG.log(LV.TRACE,'-' * 80)
        # DLOG.log(LV.TRACE,'ddi_drug=',len(result['ddi_drug']))
        # result['ddi_drug'].head_att(20)
        # DLOG.log(LV.TRACE,'=' * 80)
        
        return result

    def format_response(self, drug_info):
        """
        응답용으로 약물 정보를 포맷합니다.
        
        설명:
          - drug_info(dict)를 받아 내부의 PIL 이미지들을 base64 문자열로 변환하고
            numpy / pandas 등 직렬화 불가능한 객체를 JSON 직렬화 가능한 형태로 변환합니다.
          - 반환값은 utf-8 한글을 유지하는 JSON 문자열입니다.
        
        Args:
          - drug_info (dict): validate_and_enrich() 또는 classify_pills()의 반환 구조와 동일.
        
        Returns:
          - str: JSON 문자열 (ensure_ascii=False, indent=2)
        """
        import json
        import base64
        from io import BytesIO
        from PIL import Image        
        def pil_to_base64_str(pil_img, format='PNG'):
            """PIL 이미지를 base64 인코딩된 문자열로 변환합니다."""
            buffered = BytesIO()
            pil_img.save(buffered, format=format)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
                
        def clear_object(obj):
            """
            drug_info 내부의 비직렬화 객체를 재귀적으로 변환합니다.
            변환 규칙:
              - 'org_img' 키는 결과에 포함하지 않습니다 (원본 이미지는 보통 크기 큼).
              - numpy.ndarray -> list
              - numpy scalar -> python 기본형
              - dict/list 내부는 재귀 처리
            """
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k in ['org_img']:
                        continue
                    result[k] = clear_object(v)
                return result
            elif isinstance(obj, list):
                return [clear_object(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        for box in drug_info['bboxs']:
            box['img'] = pil_to_base64_str(box['img'])
            
        drug_info_no_img = clear_object(drug_info)
        json_str = json.dumps(drug_info_no_img, ensure_ascii=False, indent=2)
        return json_str

    def init_database(self):
        """
        데이터베이스를 초기화하고 필요한 데이터프레임을 로드합니다.

        동작:
          - self.data_path에서 약물 정보(df_drug_118.pkl)와 병용금기 데이터(df_병용금기약물_20240813.pkl)를 읽어옵니다.
          - df_drug에서 카테고리 ID 목록(categorys)을 생성하여 데이터베이스에 저장합니다.
          - 실패 시 로그를 남기고 예외를 발생시킵니다.

        Returns:
          dict: {
            "categorys": list,            # 정렬된 고유 category_id 리스트
            "df_drug": pandas.DataFrame,  # 약물 정보 데이터프레임
            "td_interaction": pandas.DataFrame  # 병용금기(상호작용) 데이터프레임
          }
        """
        #df_drug_116
        #df_drug = pd.read_pickle(os.path.join(self.data_path, "df_drug.pkl"))
        #df_drug = pd.read_pickle(os.path.join(self.data_path, "df_drug_116.pkl"))
        df_drug = pd.read_pickle(os.path.join(self.data_path, "df_drug_118.pkl"))
        df_interaction = pd.read_pickle(os.path.join(self.data_path, "df_병용금기약물_20240813.pkl"))
        
        # df_drug.head_att(10)        
        df_drug_sorted = df_drug.sort_values('category_id')
        categorys = df_drug_sorted['category_id'].unique().tolist()
        
        # DLOG.log(LV.TRACE,"categorys:", categorys)
        
        database = {
            "categorys": categorys,
            "df_drug": df_drug,
            "td_interaction": df_interaction
        }
        
        return database
    
    def load_1_stage_model(self):
        """
        YOLO 모델을 로드하여 1단계 객체 탐지 모델로 설정합니다.
        이 모델은 약물(알약) 객체를 탐지하는 데 사용됩니다.
        """
        # 모델 파일 경로 설정 (생성자에서 전달된 경로 사용)
        model_path = self.model_1_stage_path
        
        if self.DEBUG_ON:
            DLOG.log(LV.TRACE,os.path.exists(model_path), model_path)
        
        model_1_stage = YOLO(model_path)
        model_1_stage.to(self.__device)
        model_1_stage.eval()
        return model_1_stage
    
    def load_1_stage_model_fasterrcnn_resnet101(self):
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))  # src 폴더를 경로에 추가
        from object_detection_framework.models import CustomFasterRCNN, FasterRCNN_resnet101, get_model
        #model_1_stage = get_model(model_name='fasterrcnn_resnet101', num_classes=74)
        
        categorys = [
            3543, 10220, 16547, 29344, 3482, 20237, 25468, 30307, 16231, 34596,
            19606, 21025, 6562, 23202, 27732, 35205, 2482, 13394, 23222, 25437,
            22346, 5093, 19551, 3350, 3831, 16261, 27652, 3742, 5885, 25366,
            19231, 22073, 20876, 31884, 36636, 27776, 1899, 33207, 16550, 27925,
            12777, 22361, 12419, 29666, 33879, 12080, 21324, 18109, 18356, 18146,
            27992, 4377, 6191, 4542, 32309, 31704, 28762, 38161, 12246, 22626,
            20013, 44198, 31862, 24849, 29450, 33877, 33008, 19860, 13899, 21770,
            29870, 41767, 16687
        ]
        self.database['categorys'] = categorys
        
        model_path = self.model_1_stage_path
        
        if self.DEBUG_ON:
            DLOG.log(LV.TRACE,os.path.exists(model_path), model_path)
        
        model_1_stage = get_model(model_name='fasterrcnn_resnet101', num_classes=74)
        
        model_1_stage_state = torch.load(model_path, map_location=self.__device)
        
        model_1_stage.load_state_dict(model_1_stage_state)
        
        model_1_stage.to(self.__device)        
        
        model_1_stage.eval()
        #DLOG.log(LV.TRACE,"load_1_stage_model_fasterrcnn_resnet101:", model_1_stage)
        
        return model_1_stage
    
    def load_2_stage_model_efficientnet_b3(self):
        """
        EfficientNet B3 모델을 로드하여 2단계 분류 모델로 설정합니다.

        설명:
          - 저장된 체크포인트(best.pth)를 읽어 모델 state를 복원하고 평가 모드로 설정합니다.
          - timm의 efficientnet_b3 모델을 사용하며, 클래스 수는 DB에 로드된 카테고리 수로 설정합니다.
          - 모델은 self.__device(CUDA 가능 시 GPU)로 이동됩니다.

        반환:
          - model_2_stage: 로드된 PyTorch 모델 (eval 모드)
        """
        
        def load_model_dict(path, pth_name=None):
            """
            save_model_dict로 저장한 모델을 불러오는 함수
            반환값: (model_state, model_info)
            """
            import torch
            load_path = path
            if pth_name is not None:
                load_path = os.path.join(path, f"{pth_name}.pth")
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)  # <-- 여기 추가
            model_state = checkpoint.get('model_state')
            model_info = checkpoint.get('model_info')
            model_info['file_name'] = os.path.basename(load_path)
            return model_state, model_info
        
        model_path = os.path.join(self.modeling_path,
                                            "efficientnet_b3",
                                            "efficientnet_b3_pill_classify_250921_2228",
                                            "best.pth"
                                            )
        if self.DEBUG_ON:
            DLOG.log(LV.TRACE,os.path.exists(model_path), model_path)

        NUM_CLASSES = len(self.database['categorys'])
        model_2_stage_state, model_2_stage_info = load_model_dict(model_path)
        model_2_stage = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
        model_2_stage.load_state_dict(model_2_stage_state)
        model_2_stage.to(self.__device)
        model_2_stage.eval()
        return model_2_stage

    def load_2_stage_model_resnet(self):
        """
        EfficientNet B3 모델을 로드하여 2단계 분류 모델로 설정합니다.

        설명:
          - 저장된 체크포인트(best.pth)를 읽어 모델 state를 복원하고 평가 모드로 설정합니다.
          - timm의 efficientnet_b3 모델을 사용하며, 클래스 수는 DB에 로드된 카테고리 수로 설정합니다.
          - 모델은 self.__device(CUDA 가능 시 GPU)로 이동됩니다.

        반환:
          - model_2_stage: 로드된 PyTorch 모델 (eval 모드)
        """
        
        def load_model_dict(path, pth_name=None):
            """
            save_model_dict로 저장한 모델을 불러오는 함수
            반환값: (model_state, model_info)
            """
            import torch
            load_path = path
            if pth_name is not None:
                load_path = os.path.join(path, f"{pth_name}.pth")
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)  # <-- 여기 추가
            model_state = checkpoint.get('model_state')
            model_info = checkpoint.get('model_info')
            model_info['file_name'] = os.path.basename(load_path)
            return model_state, model_info
        
        model_path = self.model_2_stage_path
        
        if self.DEBUG_ON:
            DLOG.log(LV.TRACE,os.path.exists(model_path), model_path)

        NUM_CLASSES = len(self.database['categorys'])
        model_2_stage_state, model_2_stage_info = load_model_dict(model_path)
        
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        model_2_stage = resnet101(pretrained=False)

        # 마지막 FC layer를 원하는 클래스 수로 변경
        model_2_stage.fc = nn.Sequential(
            # nn.Dropout(0.3),  # 적절한 정규화
            nn.Dropout(0.1),  # 적절한 정규화
            nn.Linear(model_2_stage.fc.in_features, NUM_CLASSES)
        )

        model_2_stage.load_state_dict(model_2_stage_state)
        model_2_stage.to(self.__device)
        model_2_stage.eval()
        return model_2_stage
    
    
    
    def image_classify(self, classify):
        """이미지 분류 결과를 시각화하여 PIL 이미지로 반환합니다.

        설명:
          - 입력 classify는 analyze_image -> format_response 이전의 내부 구조(dict)와 동일합니다:
            {
              'org_img': PIL.Image.Image,   # 원본 이미지
              'bboxs': [                    # 탐지된 박스 리스트
                {
                  'class_id': int,
                  'class_name': str|int,
                  'xyxy': [x1,y1,x2,y2],
                  'xywh': [x,y,w,h],
                  'detect_score': float,
                  'img': PIL.Image.Image,    # 크롭된 알약 이미지
                  'class_probabilitie': ndarray,
                  'class_score': float
                }, ...
              ]
            }
          - 원본 이미지에 바운딩박스와 레이블을 그린 상단 영역과,
            하단에 각 크롭 이미지를 좌우로 이어 붙여 레이블을 표시한 이미지를 생성합니다.
        Args:
          - classify (dict): 위 구조의 분류/탐지 결과 dict
        Returns:
          - PIL.Image.Image: 시각화된 결과 이미지 (PNG 형식으로 메모리 저장 후 PIL로 반환)
        """
        
        import json
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
        import base64
        from io import BytesIO    
        # 원본 이미지와 박스 정보
        org_img = classify['org_img']
        bboxs = classify['bboxs']
        n = len(bboxs)
        
        # Figure 생성 (2행: 위=원본+박스, 아래=Crop들)
        fig = plt.figure(figsize=(4, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # 1행: 원본 이미지 + 박스
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(org_img)
        for box in bboxs:
            xyxy = box['xyxy']
            class_name = box['class_name']
            class_score = box['class_score']
            x1, y1, x2, y2 = xyxy
            w, h = x2-x1, y2-y1
            rect = mpatches.Rectangle((int(x1), int(y1)), int(w), int(h),
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            label = f"{class_name} {class_score:.2f}"
            ax1.text(x1, y1-5, label, color='red', fontsize=10, backgroundcolor='white', alpha=0.8)
        ax1.axis('off')
        ax1.set_title('이미지 + 박스', pad=5)  # 제목과 이미지 간 여백 최소화

        # 2행: 각 박스 Crop 이미지 한 행에 나란히
        ax2 = fig.add_subplot(gs[1])
        crop_imgs = [np.array(box['img']) for box in bboxs]
        if n > 0:
            heights = [img.shape[0] for img in crop_imgs]
            max_h = max(heights)
            resized_imgs = []
            crop_positions = []
            x_offset = 0
            for i, img in enumerate(crop_imgs):
                if img.shape[0] != max_h:
                    from PIL import Image
                    pil_img = Image.fromarray(img)
                    ratio = max_h / img.shape[0]
                    new_w = int(img.shape[1] * ratio)
                    pil_img = pil_img.resize((new_w, max_h))
                    img = np.array(pil_img)
                resized_imgs.append(img)
                crop_positions.append((x_offset, img.shape[1]))
                x_offset += img.shape[1]
            concat_img = np.concatenate(resized_imgs, axis=1)
            ax2.imshow(concat_img)
            ax2.axis('off')
            ax2.set_title('알약 이미지들', pad=5)
            # 각 crop 이미지 위에 라벨 출력
            for i, (start_x, width) in enumerate(crop_positions):
                class_name = bboxs[i]['class_name']
                class_score = bboxs[i]['class_score']
                label = f"{class_name}\n{class_score:.2f}"
                ax2.text(start_x + width // 2, 10, label, color='red', fontsize=10,
                        backgroundcolor='white', ha='center', va='top', alpha=0.8)
            
        plt.subplots_adjust(hspace=0, top=1, bottom=0)
        plt.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf)
        return pil_img

    def image_result(self, result_json):
        """이미지 분류 결과를 시각화하여 PIL 이미지로 반환합니다.
        
        설명:
          - result_json(JSON 문자열)에서 박스 정보와 (선택적)원본 이미지 경로를 읽어
            바운딩박스 및 레이블이 그려진 시각화 이미지를 생성하여 PIL.Image로 반환합니다.
          - 반환 이미지는 PNG로 메모리 저장된 후 PIL로 로드됩니다.
        """
        
        import json
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
        import base64
        from io import BytesIO

        # JSON 문자열 파싱
        result = json.loads(result_json)
        bboxs = result['bboxs']
        img_path = result.get('img_path', None)

        # 원본 이미지 로드
        if img_path is not None:
            org_img = Image.open(img_path).convert("RGB")
        else:
            org_img = None

        n = len(bboxs)
        fig = plt.figure(figsize=(4, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # 1행: 원본 이미지 + 박스
        ax1 = fig.add_subplot(gs[0])
        if org_img is not None:
            ax1.imshow(org_img)
            for box in bboxs:
                xyxy = box['xyxy']
                class_name = box['class_name']
                class_score = box['class_score']
                drug_N = box['drug_info']['drug_N']
                dl_name = box['drug_info']['dl_name']
                x1, y1, x2, y2 = xyxy
                w, h = x2-x1, y2-y1
                rect = mpatches.Rectangle((int(x1), int(y1)), int(w), int(h),
                                        linewidth=2, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)
                label = f"{class_name} {class_score:.2f}\n{drug_N}\n{dl_name}"
                ax1.text(x1, y1-5, label, color='red', fontsize=10, backgroundcolor='white', alpha=0.8)
            ax1.axis('off')
            ax1.set_title('이미지 + 박스', pad=5)
        else:
            ax1.set_title('원본 이미지 없음')
            ax1.axis('off')

        # 2행: 각 박스 Crop 이미지 한 행에 나란히
        ax2 = fig.add_subplot(gs[1])
        crop_imgs = []
        for box in bboxs:
            # base64 이미지를 PIL로 변환
            img_b64 = box['img']
            img_bytes = base64.b64decode(img_b64)
            crop_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            crop_imgs.append(np.array(crop_img))

        if n > 0:
            heights = [img.shape[0] for img in crop_imgs]
            max_h = max(heights)
            resized_imgs = []
            crop_positions = []
            x_offset = 0
            for i, img in enumerate(crop_imgs):
                if img.shape[0] != max_h:
                    pil_img = Image.fromarray(img)
                    ratio = max_h / img.shape[0]
                    new_w = int(img.shape[1] * ratio)
                    pil_img = pil_img.resize((new_w, max_h))
                    img = np.array(pil_img)
                resized_imgs.append(img)
                crop_positions.append((x_offset, img.shape[1]))
                x_offset += img.shape[1]
            concat_img = np.concatenate(resized_imgs, axis=1)
            ax2.imshow(concat_img)
            ax2.axis('off')
            ax2.set_title('알약 이미지들', pad=5)
            # 각 crop 이미지 위에 라벨 출력
            for i, (start_x, width) in enumerate(crop_positions):
                class_name = bboxs[i]['class_name']
                class_score = bboxs[i]['class_score']
                label = f"{class_name}\n{class_score:.2f}"
                ax2.text(start_x + width // 2, 10, label, color='red', fontsize=10,
                        backgroundcolor='white', ha='center', va='top', alpha=0.8)
        else:
            ax2.set_title('알약 없음')
            ax2.axis('off')

        # 여백 최소화
        plt.subplots_adjust(hspace=0, top=1, bottom=0)
        plt.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf)
        return pil_img

    def show_img(self, img):
        """이미지를 시각화하여 보여줍니다.
        
        Args:
          - img: PIL.Image.Image 또는 numpy.ndarray (cv2로 읽은 BGR 가능)
        """
        
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        dpi = mpl.rcParams['figure.dpi']
        w, h = img.size
        plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    def save_img(self, img, out_image_path):
        """이미지를 저장합니다.
        Args:
            img (PIL.Image.Image 또는 numpy.ndarray): 저장할 이미지
            out_image_path (str): 저장할 파일 경로
        """
        
        # JPEG은 RGBA(알파채널) 저장 불가 → RGB로 변환 필요
        import os
        from PIL import Image
        if hasattr(img, 'save'):
            # 파일 확장자 확인
            ext = os.path.splitext(out_image_path)[1].lower()
            if ext in ['.jpg', '.jpeg'] and img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(out_image_path)
        else:
            # numpy 배열 등 다른 타입이면 PIL로 변환 후 저장
            pil_img = Image.fromarray(img)
            ext = os.path.splitext(out_image_path)[1].lower()
            if ext in ['.jpg', '.jpeg'] and pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            pil_img.save(out_image_path)

    def result_json_to_csv_rows(self, result_json, img_path, annotation_id=1):
        import os
        import json
        rows = []
        result = json.loads(result_json)
        image_id = os.path.basename(img_path).split('.')[0]
        for box in result['bboxs']:
            # xywh: [x, y, w, h]
            row = {
                'annotation_id': annotation_id,
                'image_id': image_id,
                'category_id': box['class_name'],
                'bbox_x': box['xywh'][0],
                'bbox_y': box['xywh'][1],
                'bbox_w': box['xywh'][2],
                'bbox_h': box['xywh'][3],
                'score': box['class_score']
            }
            rows.append(row)
            annotation_id += 1
        return annotation_id, rows
    
# # 이미지 파일 경로를 이미지로 변환

if __name__ == "__main__":
    
    py_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_1_stage_path=None
    model_2_stage_path=None
    
    model_1_stage_path = os.path.join(py_dir,"python_modules","modeling","fasterrcnn_resnet101","best.pt")
    sample_path = os.path.join(py_dir,"python_modules","sampledata","1.png")
    
    engine = PillAnalysisEngine(model_1_stage_path)
    result_json = engine.analyze_image(sample_path)
    DLOG.log(LV.TRACE,result_json)
    
