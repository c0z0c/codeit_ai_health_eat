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

class PillAnalysisEngine:    
    def __init__(self):
        def drive_root():
            """
            Google Drive의 최상위 경로를 반환하는 함수입니다.
            - 로컬 환경(Windows): D:\GoogleDrive
            - Colab 환경: /content/drive/MyDrive
            프로젝트 내에서 데이터, 모델, 설정 파일 등 경로를 일관되게 관리할 때 사용합니다.
            """    
            try:
                import google.colab
                from google.colab import drive
                COLAB_AVAILABLE = True
            except ImportError:
                COLAB_AVAILABLE = False
            
            root_path = os.path.join(Path.cwd().drive + '\\', "GoogleDrive")
            if COLAB_AVAILABLE:
                root_path = os.path.join("/content/drive/MyDrive")
            return root_path
        
        self.DEBUG_ON = True
        # --- 시간대 설정 ---
        self.__kst = pytz.timezone('Asia/Seoul')

        # --- GPU 설정 ---
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__device_cpu = torch.device('cpu')
        self.project_path = os.path.join(drive_root(), "codeit_ai_health_eat")
        self.modeling_path = os.path.join(self.project_path, "src", "python_modules", "modeling")
        self.data_path = os.path.join(self.project_path, "src", "python_modules", "data")
        
        if self.DEBUG_ON:
            print(self.project_path)
            print(self.modeling_path)
            print(self.data_path)
            
        self.database = self.init_database() # DB 초기화 클래스 개수 확인을 위하여 가장 먼저 로딩되어야함.
        self.model_1_stage = self.load_1_stage_model()
        self.model_2_stage = self.load_2_stage_model_resnet152()
        # self.model_2_stage = self.load_2_stage_model_efficientnet_b3()  # EfficientNet 사용
    
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
        detections = self.detect_pills(validated_image)
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
    
    def detect_pills(self, validated_image):
        import matplotlib.patches as mpatches
        if self.model_1_stage is None:
            raise ValueError("1단계 모델이 로드되지 않았습니다.")
        
        # 이미지가 PIL 이미지인 경우 numpy 배열로 변환
        if isinstance(validated_image, Image.Image):
            validated_image = np.array(validated_image)
            
        # 객체 탐지 수행
        result_detections = self.model_1_stage(validated_image, verbose=False)

        current_img = Image.fromarray(result_detections[0].orig_img.copy())
        img_width, img_height = current_img.size

        detections={}
        detections['org_img'] = current_img
        detections['bboxs'] = []
        for box in result_detections[0].boxes:
            #print(box.xyxy, box.conf, box.cls)
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
        for box in detections['bboxs']:
            image = box['img']
            input_tensor = self.transform(image).unsqueeze(0).to(self.__device)    
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
                #print(drug['category_id'], drug['di_edi_code'], drug['drug_N'], drug['dl_name'])
                for box in classifications['bboxs']:
                    if box['class_name'] == drug['category_id']:
                        box['drug_info'] = drug
                
            for ddi in ddi_result['ddi']:
                #print(ddi['category_id'], ddi['제품코드A'], ddi['제품명A'], ddi['제품코드B'], ddi['제품명B'])
                for box in classifications['bboxs']:
                    if box['class_name'] == ddi['category_id']:
                        box['ddi'] = ddi
                
            for ddi_drug in ddi_result['ddi_drug']:
                #print(ddi_drug['category_id'], ddi_drug['제품코드A'], ddi_drug['제품명A'], ddi_drug['제품코드B'], ddi_drug['제품명B'])
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
        
        # print('-' * 80)

        df_drug['code'] = df_drug['di_edi_code'].str.split(',').str[0].astype(int)
        codes = df_drug['code'].tolist()
        code_to_category = dict(zip(df_drug['code'], df_drug['category_id']))
        
        df_ddi = pd.DataFrame()        
        df_drug_ddi = pd.DataFrame()
        for code_a in codes:
            mask_a = df['제품코드A'] == code_a
            for code_b in codes:
                mask_b = df['제품코드B'] == code_b
                if code_a == code_b: continue
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
        
        # print('ddi=',len(result['ddi']))
        # result['ddi'].head_att(20)
        # print('-' * 80)
        # print('ddi_drug=',len(result['ddi_drug']))
        # result['ddi_drug'].head_att(20)
        # print('=' * 80)
        
        return result

    def format_response(self, drug_info):
        import json
        import base64
        from io import BytesIO
        from PIL import Image        
        def pil_to_base64_str(pil_img, format='PNG'):
            """PIL 이미지를 base64 문자열로 변환"""
            buffered = BytesIO()
            pil_img.save(buffered, format=format)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
                
        def clear_object(obj):
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
        df_drug = pd.read_pickle(os.path.join(self.data_path, "df_drug.pkl"))
        df_interaction = pd.read_pickle(os.path.join(self.data_path, "df_병용금기약물_20240813.pkl"))
        
        df_drug_sorted = df_drug.sort_values('category_id')
        categorys = df_drug_sorted['category_id'].unique().tolist()
        
        database = {
            "categorys": categorys,
            "df_drug": df_drug,
            "td_interaction": df_interaction
        }
        return database
    
    def load_1_stage_model(self):
        model_path = os.path.join(self.modeling_path, 
                                            "yolo",
                                            "yolov8m_yolo_noresize_one_class_20250915_0858",
                                            "weights",
                                            "best.pt"
                                            )

        if self.DEBUG_ON:
            print(os.path.exists(model_path), model_path)
        
        model_1_stage = YOLO(model_path)
        model_1_stage.to(self.__device)
        model_1_stage.eval()
        return model_1_stage
    
    def load_2_stage_model_efficientnet_b3(self):
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
                                            "efficientnet_b3_experiment_20250917_200857",
                                            "best.pth"
                                            )
        if self.DEBUG_ON:
            print(os.path.exists(model_path), model_path)

        NUM_CLASSES = len(self.database['categorys'])
        model_2_stage_state, model_2_stage_info = load_model_dict(model_path)
        model_2_stage = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
        model_2_stage.load_state_dict(model_2_stage_state)
        model_2_stage.to(self.__device)
        model_2_stage.eval()
        return model_2_stage

    def load_2_stage_model_resnet152(self):
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
                                            "resnet152_20250918_203333",
                                            "best.pth"
                                            )
        if self.DEBUG_ON:
            print(os.path.exists(model_path), model_path)

        NUM_CLASSES = len(self.database['categorys'])
        model_2_stage_state, model_2_stage_info = load_model_dict(model_path)
        
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        model_2_stage = resnet152(pretrained=False)

        # 마지막 FC layer를 원하는 클래스 수로 변경
        model_2_stage.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model_2_stage.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )        
        
        model_2_stage.load_state_dict(model_2_stage_state)
        model_2_stage.to(self.__device)
        model_2_stage.eval()
        return model_2_stage
    
    
    
    def image_classify(self, classify):
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
# image_path = test_images[0]
# self = PillAnalysisEngine()
# validated_image = self.validate_image(image_path)    