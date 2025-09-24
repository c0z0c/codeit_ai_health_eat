import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import copy

from models import get_model
from trainer_fasterrcnn import get_transforms


def compute_iou(box1, box2):
    """두 bounding box 간의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class PseudoLabelDataset(Dataset):
    """Pseudo-labeling을 위한 데이터셋"""
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        self.coco = COCO()
        self.coco.dataset = self.coco_data
        self.coco.createIndex()
        
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            return None
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, img_info, img_id


def load_trained_model(model_name, model_path, num_classes, device):
    """학습된 모델 로드"""
    model = get_model(model_name=model_name, num_classes=num_classes)
    
    # 체크포인트에서 모델 가중치만 추출
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"모델 로드 완료: {model_path}")
    return model


def generate_pseudo_labels(model, dataset, dataloader,label2id_dict, device, confidence_threshold=0.9, iou_threshold=0.5):
    """모델을 사용하여 pseudo-label 생성"""
    model.eval()
    pseudo_annotations = []
    next_annotation_id = 1
    
    # 기존 annotation ID들 중 최대값 찾기
    for ann in dataset.coco_data['annotations']:
        next_annotation_id = max(next_annotation_id, ann['id'] + 1)
    
    print("Pseudo-label 생성 중...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Generating pseudo-labels")):
            if batch_data is None:
                continue
            
            images, img_infos, img_ids = batch_data
            images = [img.to(device) for img in images]
            
            predictions = model(images)
            
            for pred, img_info, img_id in zip(predictions, img_infos, img_ids):
                # 높은 confidence score를 가진 예측만 선택
                high_conf_mask = pred['scores'] >= confidence_threshold
                
                if not torch.any(high_conf_mask):
                    continue
                
                pred_boxes = pred['boxes'][high_conf_mask].cpu().numpy()
                pred_labels = pred['labels'][high_conf_mask].cpu().numpy()
                pred_scores = pred['scores'][high_conf_mask].cpu().numpy()
                
                # 기존 annotation과 겹치는지 확인
                existing_anns = dataset.coco.getAnnIds(imgIds=img_id)
                existing_boxes = []
                
                for ann_id in existing_anns:
                    ann = dataset.coco.loadAnns(ann_id)[0]
                    x, y, w, h = ann['bbox']
                    existing_boxes.append([x, y, x+w, y+h])
                
                # 각 예측에 대해 기존 annotation과 IoU 확인
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    x1, y1, x2, y2 = box
                    
                    # 기존 annotation과 겹치는지 확인
                    overlaps = False
                    for existing_box in existing_boxes:
                        if compute_iou([x1, y1, x2, y2], existing_box) > iou_threshold:
                            overlaps = True
                            break
                    
                    # 겹치지 않는 경우만 pseudo-label로 추가
                    if not overlaps:
                        pseudo_ann = {
                            'id': next_annotation_id,
                            'image_id': int(img_id),
                            'category_id': label2id_dict[str(label-1)],
                            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                            'area': float((x2-x1) * (y2-y1)),
                            'iscrowd': 0,
                            'pseudo_label': True,  # pseudo-label임을 표시
                            'confidence': float(score)
                        }
                        pseudo_annotations.append(pseudo_ann)
                        next_annotation_id += 1
    
    return pseudo_annotations


def create_pseudo_labeled_dataset(original_annotation_file, pseudo_annotations, output_file):
    """원본 데이터셋에 pseudo-label을 추가하여 새로운 데이터셋 생성"""
    with open(original_annotation_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 새로운 데이터셋 생성
    new_data = copy.deepcopy(original_data)
    
    # pseudo-label 추가
    new_data['annotations'].extend(pseudo_annotations)
    
    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"새로운 데이터셋 생성 완료: {output_file}")
    print(f"원본 annotation 수: {len(original_data['annotations'])}")
    print(f"추가된 pseudo-label 수: {len(pseudo_annotations)}")
    print(f"총 annotation 수: {len(new_data['annotations'])}")


def custom_collate_fn(batch):
    """배치 처리를 위한 collate function"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    images, img_infos, img_ids = zip(*batch)
    return list(images), list(img_infos), list(img_ids)


def main():
    # 설정 파라미터
    class Config:
        def __init__(self):
            # 모델 설정
            # self.model_name = 'CustomFasterRCNN'  # 또는 'FasterRCNN_resnet101'
            self.model_name = 'FasterRCNN_resnet101'
            self.num_classes = 93  # 학습 시 사용한 클래스 수
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 데이터 경로
            self.train_image_dir = "./data/images/train"
            self.valid_image_dir = "./data/images/val"
            self.train_annotation_path = "./data/labels/train/train.json"
            self.valid_annotation_path = "./data/labels/val/valid.json"
            self.label2id_path = "./data/label2id.json"

            # 모델 경로 (자동으로 찾거나 직접 지정)
            self.model_path = None  # None이면 자동으로 최신 모델 찾기
            # self.model_path = "./object-detection/CustomFasterRCNN/train1/best_model_map_0.1234.pth"
            
            # Pseudo-labeling 설정
            self.confidence_threshold = 0.90 # pseudo-label로 사용할 최소 confidence
            self.iou_threshold = 0.5  # 기존 label과 겹치는 것으로 판단할 IoU 임계값
            self.batch_size = 4
            self.num_workers = 4
            
            # 출력 파일 경로
            self.output_train_path = "./data/labels/train/train_pseudo.json"
            self.output_valid_path = "./data/labels/val/valid_pseudo.json"
    
    config = Config()
    
    # 모델 경로 자동 설정
    if config.model_path is None:
        base_project_dir = "./object-detection"
        model_project_dir = os.path.join(base_project_dir, config.model_name)
        
        if os.path.exists(model_project_dir):
            existing_runs = [d for d in os.listdir(model_project_dir) if os.path.isdir(os.path.join(model_project_dir, d))]
            if existing_runs:
                latest_run = sorted(existing_runs)[-1]
                selected_run_dir = os.path.join(model_project_dir, latest_run)
                
                # best 모델 찾기
                best_model_files = [f for f in os.listdir(selected_run_dir) if f.startswith('best_model_map_')]
                if best_model_files:
                    config.model_path = os.path.join(selected_run_dir, best_model_files[-1])
                else:
                    # checkpoint에서 최신 모델 찾기
                    checkpoint_dir = os.path.join(selected_run_dir, "checkpoints")
                    if os.path.exists(checkpoint_dir):
                        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
                        if checkpoint_files:
                            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                            config.model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    if config.model_path is None or not os.path.exists(config.model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {config.model_path}")
    
    print(f"사용할 모델: {config.model_path}")
    print(f"Confidence threshold: {config.confidence_threshold}")
    print(f"IoU threshold: {config.iou_threshold}")
    
    # 모델 로드
    model = load_trained_model(config.model_name, config.model_path, config.num_classes, config.device)
    
    # Train 데이터셋 처리
    print("\n=== Train 데이터셋 Pseudo-labeling ===")
    train_dataset = PseudoLabelDataset(
        config.train_image_dir, 
        config.train_annotation_path,
        transforms=get_transforms(train=False)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )
    with open(config.label2id_path, 'r', encoding='utf-8') as f:
        label2id_dict = json.load(f)

    train_pseudo_annotations = generate_pseudo_labels(
        model, train_dataset, train_dataloader, label2id_dict,config.device,
        config.confidence_threshold, config.iou_threshold
    )

    create_pseudo_labeled_dataset(
        config.train_annotation_path, 
        train_pseudo_annotations, 
        config.output_train_path
    )
    
    # Validation 데이터셋 처리
    print("\n=== Validation 데이터셋 Pseudo-labeling ===")
    valid_dataset = PseudoLabelDataset(
        config.valid_image_dir,
        config.valid_annotation_path,
        transforms=get_transforms(train=False)
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )
    
    valid_pseudo_annotations = generate_pseudo_labels(
        model, valid_dataset, valid_dataloader,label2id_dict, config.device,
        config.confidence_threshold, config.iou_threshold
    )
    
    create_pseudo_labeled_dataset(
        config.valid_annotation_path,
        valid_pseudo_annotations,
        config.output_valid_path
    )
    
    print("\n=== Pseudo-labeling 완료 ===")
    print(f"Train pseudo-labeled dataset: {config.output_train_path}")
    print(f"Valid pseudo-labeled dataset: {config.output_valid_path}")


if __name__ == "__main__":
    main()
