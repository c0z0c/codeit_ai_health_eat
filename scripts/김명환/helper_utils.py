# --- Scikit-learn: 데이터 전처리, 모델, 평가 ---
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    fetch_california_housing, load_iris, make_moons, make_circles,
    load_breast_cancer, load_wine
)
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score

# --- 이미지 처리 ---
import cv2
from PIL import Image, ImageFilter, ImageDraw
import albumentations as A

# --- PyTorch: 딥러닝 관련 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from torchvision.datasets import CocoDetection
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

# --- COCO 데이터셋 관련 ---
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

# --- 딥러닝 모델 ---
import timm

# --- 기본 라이브러리 ---
import os
import sys
import re
import csv
import copy
import json
import math
import random
import yaml
import shutil
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# --- 데이터 분석 및 시각화 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 시간 관련 ---
from datetime import datetime, timezone, timedelta
import pytz

# --- 진행률 표시 ---
import IPython.display
from tqdm.notebook import tqdm

try:
    import google.colab
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

def get_tqdm_kwargs():
    """Widget 오류를 방지하는 안전한 tqdm 설정"""
    return {
        'disable': False,
        'leave': True,
        'file': sys.stdout,
        'ascii': True,  # ASCII 문자만 사용
        'dynamic_ncols': False,
#        'ncols': 80  # 고정 폭
    }

def drive_root():
    """
    Google Drive의 최상위 경로를 반환하는 함수입니다.
    - 로컬 환경(Windows): D:\GoogleDrive
    - Colab 환경: /content/drive/MyDrive
    프로젝트 내에서 데이터, 모델, 설정 파일 등 경로를 일관되게 관리할 때 사용합니다.
    """
    root_path = os.path.join(Path.cwd().drive + '\\', "GoogleDrive")
    if COLAB_AVAILABLE:
        root_path = os.path.join("/content/drive/MyDrive")
    return root_path

def get_path_modeling(add_path = None):
    """
    get_path_modeling() 함수는 모델링 관련 파일(예: 학습 결과, 체크포인트, 로그 등)을 저장할 경로를 반환합니다.
    기본적으로 Google Drive의 루트 경로(drive_root()) 아래 "modeling_yolo" 폴더를 기준으로 경로를 생성합니다.
    추가 하위 경로가 필요할 경우 add_path 인자를 통해 세부 폴더까지 지정할 수 있습니다.
    예시:
    get_path_modeling() → modeling(로컬)
    get_path_modeling("exp1") → D:\GoogleDrive\modeling\exp1
    Colab 환경에서는 /content/drive/MyDrive/modeling반환됩니다.
    """

    modeling_path = "modeling"
    path = os.path.join(drive_root(),modeling_path)
    if add_path is not None:
        path = os.path.join(path,add_path)
    return path

def get_path_modeling_release(add_path = None):
    """
    get_path_modeling_release() 함수는 모델링 결과물(예: 학습 결과, 체크포인트, 로그 등)을 저장할 경로를 반환합니다.
    기본적으로 Google Drive의 루트 경로(drive_root()) 아래 "modeling_yolo" 폴더를 기준으로 경로를 생성합니다.
    추가 하위 경로가 필요할 경우 add_path 인자를 통해 세부 폴더까지 지정할 수 있습니다.
    예시:
    get_path_modeling_release() → modeling (로컬)
    get_path_modeling_release("exp1") → D:\GoogleDrive\modeling\exp1
    Colab 환경에서는 /content/drive/MyDrive/modeling_yolo로 반환됩니다._
    """
    modeling_path = "modeling"
    path = os.path.join(drive_root(),modeling_path)
    if add_path is not None:
        path = os.path.join(path,add_path)
    return path

def print_dir_tree(root, max_depth=2, list_count=3, indent=""):
    """
    지정한 폴더(root) 하위의 디렉토리 구조를 트리 형태로 출력하는 함수입니다.

    Args:
        root: 시작 경로(폴더)
        max_depth: 출력할 최대 깊이(디폴트 2)
        list_count: 파일 개수가 많을 때 몇 개만 출력할지(디폴트 3)
        indent: 들여쓰기(재귀적으로 사용)
    """
    import os
    if max_depth < 0:
        return
    try:
        items = os.listdir(root)
    except Exception as e:
        print(indent + f"[Error] {e}")
        return

    img_count = len([f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.xml', '.inf', '.txt'))])
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print(indent + "|-- "+ item)
            # 이미지 파일 개수만 출력
            img_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.xml', '.inf', '.txt'))])
            if img_count > list_count:
                print(indent + "   "+ f"[데이터파일: {img_count}개]")
            print_dir_tree(root=path, max_depth=max_depth-1, list_count=list_count, indent=indent + "   ")
        else:
            if list_count < img_count and item.lower().endswith(('.jpg', '.jpeg', '.png', '.xml', '.inf', '.txt')):
                continue
            print(indent + "|-- "+ item)

def save_model_dict(model, path, pt_name, kwargs=None):
    """모델 state_dict와 추가 정보를 저장"""
    def safe_makedirs(path):
        """안전한 디렉토리 생성"""
        if os.path.exists(path) and not os.path.isdir(path):
            os.remove(path)  # 파일이면 삭제
        os.makedirs(path, exist_ok=True)

    # 디렉토리 생성
    safe_makedirs(path)

    # 모델 구조 정보 추출
    model_info = {
        'class_name': model.__class__.__name__,
        'init_args': {},
        'str': str(model),
        'repr': repr(model),
        'modules': [m.__class__.__name__ for m in model.modules()],
    }

    # 생성자 인자 자동 추출(가능한 경우)
    if hasattr(model, '__dict__'):
        for key in ['in_ch', 'base_ch', 'num_classes', 'out_ch']:
            if hasattr(model, key):
                model_info['init_args'][key] = getattr(model, key)

    # kwargs 처리
    extra_info = {}
    if kwargs is not None:
        if isinstance(kwargs, str):
            extra_info = json.loads(kwargs)
        elif isinstance(kwargs, dict):
            extra_info = kwargs

    model_info.update(extra_info)

    # 저장할 dict 구성
    save_dict = {
        'model_state': model.state_dict(),
        'class_name': model.__class__.__name__,
        'model_info': model_info,
    }

    save_path = os.path.join(path, f"{pt_name}.pt")
    torch.save(save_dict, save_path)
    return save_path

def load_model_dict(path, pt_name=None):
    """
    save_model_dict로 저장한 모델을 불러오는 함수
    반환값: (model_state, model_info)
    """
    import torch
    load_path = path
    if pt_name is not None:
        load_path = os.path.join(path, f"{pt_name}.pt")
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)  # <-- 여기 추가
    model_state = checkpoint.get('model_state')
    model_info = checkpoint.get('model_info')
    model_info['file_name'] = os.path.basename(load_path)
    return model_state, model_info


def search_pt_files(base_path):
    """
    입력된 경로의 하위 폴더들에서 pt 파일들을 검색
    """
    pt_files = []

    if not os.path.exists(base_path):
        print(f"경로가 존재하지 않습니다: {base_path}")
        return pt_files

    print(f"pt 파일 검색 시작: {base_path}")

    # 하위 폴더들을 순회하며 pt 파일 검색
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pt'):
                pt_path = os.path.join(root, file)
                pt_files.append(pt_path)

    # 결과 정리 및 출력
    if pt_files:
        print(f"\n발견된 pt 파일들 ({len(pt_files)}개):")
        for i, pt_file in enumerate(pt_files, 1):
            # 상대 경로로 표시 (base_path 기준)
            rel_path = os.path.relpath(pt_file, base_path)
            print(f" {i:2d}. {rel_path}")
    else:
        print("pt 파일을 찾을 수 없습니다.")

    return pt_files

def print_json_tree(data, indent="", max_depth=4, _depth=0, list_count=2, print_value=True):
    """
    JSON 객체를 지정한 단계(max_depth)까지 트리 형태로 출력
    - list 타입은 3개 이상일 때 개수만 출력
    - 하위 노드가 값일 경우 key(type) 형태로 출력
    - print_value=True일 때 key(type): 값 형태로 출력
    """
    if _depth > max_depth:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent}|-- {key}")
                print_json_tree(value, indent + "    ", max_depth, _depth + 1, list_count, print_value)
            else:
                if print_value:
                    print(f"{indent}|-- {key}({type(value).__name__}): {value if len(str(value)) < 100 else f'{str(value)[:30]}...'}")
                else:
                    print(f"{indent}|-- {key}({type(value).__name__})")
    elif isinstance(data, list):
        if len(data) > list_count:
            print(f"{indent}|-- [list] ({len(data)} items)")
        else:
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    print(f"{indent}|-- [{i}]")
                    print_json_tree(item, indent + "    ", max_depth, _depth + 1, list_count, print_value)
                else:
                    if print_value:
                        print(f"{indent}|-- [{i}]({type(item).__name__}): {item if len(str(item)) < 100 else f'{str(item)[:30]}...'}")
                    else:
                        print(f"{indent}|-- [{i}]({type(item).__name__})")
    else:
        if print_value:
            print(f"{indent}{type(data).__name__}: {data if len(str(data)) < 100 else f'{str(data)[:30]}...'}")
        else:
            print(f"{indent}{type(data).__name__}")

def print_dict_tree(data, indent="", max_depth=3, _depth=0):
    """
    PyTorch tensor/딕셔너리/리스트를 git tree 스타일로 출력
    """
    import torch
    import numpy as np

    if _depth > max_depth:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}├─ {key} [{type(value).__name__}]")
            print_git_tree(value, indent + "│  ", max_depth, _depth + 1)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            print(f"{indent}├─ [{i}] [{type(item).__name__}]")
            print_git_tree(item, indent + "│  ", max_depth, _depth + 1)
    elif torch.is_tensor(data):
        shape = tuple(data.shape)
        dtype = str(data.dtype)
        preview = str(data)
        preview_str = preview[:80] + ("..." if len(preview) > 80 else "")
        print(f"{indent}└─ Tensor shape={shape} dtype={dtype} preview={preview_str}")
    elif isinstance(data, np.ndarray):
        shape = data.shape
        dtype = data.dtype
        preview = str(data)
        preview_str = preview[:80] + ("..." if len(preview) > 80 else "")
        print(f"{indent}└─ ndarray shape={shape} dtype={dtype} preview={preview_str}")
    else:
        val_str = str(data)
        print(f"{indent}└─ {type(data).__name__}: {val_str[:80]}{'...' if len(val_str)>80 else ''}")

def make_zip_with_progress(src_folder, zip_file_path):
    """
    src_folder의 모든 파일을 zip_file_path로 압축하며 진행 상황을 tqdm으로 표시합니다.

    Args:
        src_folder (str): 압축할 폴더 경로
        zip_file_path (str): 생성할 zip 파일 경로

    Returns:
        str: 생성된 zip 파일 경로
    """
    import zipfile
    file_list = []
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_list.append(os.path.join(root, file))

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(file_list, desc="Zipping files", unit="file"):
            arcname = os.path.relpath(file, src_folder)
            zipf.write(file, arcname=arcname)
    #print(f"ZIP 파일 생성 완료: {zip_file_path}")
    return zip_file_path

def zip_move_shared(zip_src_path, zip_name, dst_folder="shared_datasets", ignore=True):
    """
    YOLO 데이터셋 폴더(zip_src_path)를 zip으로 압축하여 공유 폴더(dst_folder)로 이동합니다.
    - 이미 zip 파일이 있으면 ignore=True일 때 덮어쓰고, False면 기존 파일을 반환합니다.
    - 진행 상황을 tqdm으로 표시합니다.

    Args:
        zip_src_path (str): YOLO 데이터셋 폴더 경로
        zip_name (str): 압축 파일명(확장자 제외)
        dst_folder (str, optional): 공유 폴더 경로. 기본값 "shared_datasets"
        ignore (bool, optional): 기존 zip 파일 덮어쓰기 여부. 기본값 True

    Returns:
        str: 공유 폴더에 생성된 zip 파일 경로
    """
    dest_path = os.path.join(dst_folder, f"{zip_name}.zip")
    if os.path.exists(dest_path):
        if ignore:
            os.remove(dest_path)
        else:
            return dest_path

    # Google Drive의 공유 폴더로 이동
    if os.path.exists(dst_folder) == False :
        os.makedirs(dst_folder, exist_ok=True)

    src_folder = os.path.join(zip_src_path)
    zip_file_path = os.path.join(zip_src_path, '..', f"{zip_name}.zip")

    # zip 압축 생성
    zip_path = make_zip_with_progress(src_folder, zip_file_path)
    #zip_path = shutil.make_archive(base_name=zip_file_path.replace('.zip', ''), format='zip', root_dir=src_folder)
    print(f"ZIP 파일 생성: {zip_path}")

    # shutil.move(zip_path, dest_path)
    # print(f"ZIP 파일을 공유 폴더로 이동: {dest_path}")
    return dest_path


print("유틸리티 함수 로드 완료")