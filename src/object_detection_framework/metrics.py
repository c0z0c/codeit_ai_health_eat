import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    두 바운딩 박스 간의 IoU(Intersection over Union) 계산
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형태의 바운딩 박스
    
    Returns:
        IoU 값 (0~1 사이)
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 면적 계산
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 = 두 면적의 합 - 교집합
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_iou(box1, box2):
    """
    두 bounding box의 IoU 계산
    box format: [x_min, y_min, width, height]
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # 교집합 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Average Precision (AP) 계산
    VOC 2010+ 방식 (11-point interpolation이 아닌 모든 point 사용)
    
    Args:
        recall: 정렬된 recall 값들
        precision: 해당 recall에서의 precision 값들
    
    Returns:
        Average Precision 값
    """
    # recall = 0과 recall = 1 지점 추가
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # precision을 단조감소하도록 조정
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # recall이 변하는 지점들 찾기
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    
    # AP 계산 (면적의 합)
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return ap


def evaluate_detection(model, data_loader, device, class_names: List[str], 
                      confidence_threshold: float = 0.5, 
                      iou_thresholds: List[float] = None) -> Dict[str, Any]:
    """
    객체 검출 모델의 mAP, precision, recall 평가
    
    Args:
        model: 평가할 모델
        data_loader: 검증 데이터 로더
        device: 디바이스 (cuda/cpu)
        class_names: 클래스 이름 리스트 ['background', 'class1', 'class2', ...]
        confidence_threshold: 예측 신뢰도 임계값
        iou_thresholds: IoU 임계값들 (None이면 [0.5:0.95:0.05] 사용)
    
    Returns:
        평가 결과 딕셔너리
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # COCO 스타일: 0.5, 0.55, ..., 0.95
    
    model.eval()
    
    # 각 클래스별로 예측과 정답을 저장
    all_predictions = defaultdict(list)  # {class_id: [(confidence, box, image_id), ...]}
    all_ground_truths = defaultdict(list)  # {class_id: [(box, image_id), ...]}
    
    print("모델 예측 수집 중...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 예측 수행
            predictions = model(images)
            
            # 배치의 각 이미지에 대해 처리
            for i, (prediction, target) in enumerate(zip(predictions, targets)):
                image_id = batch_idx * len(images) + i
                
                # CPU로 이동
                prediction = {k: v.cpu() for k, v in prediction.items()}
                target = {k: v.cpu() for k, v in target.items()}
                
                # Ground Truth 수집
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                for box, label in zip(gt_boxes, gt_labels):
                    if label > 0:  # 배경 클래스 제외
                        all_ground_truths[label.item()].append((box.numpy(), image_id))
                
                # 예측 결과 수집 (confidence threshold 적용)
                pred_boxes = prediction['boxes']
                pred_labels = prediction['labels']
                pred_scores = prediction['scores']
                
                # 신뢰도가 임계값보다 높은 예측만 선택
                high_conf_mask = pred_scores >= confidence_threshold
                
                for box, label, score in zip(pred_boxes[high_conf_mask], 
                                           pred_labels[high_conf_mask], 
                                           pred_scores[high_conf_mask]):
                    if label > 0:  # 배경 클래스 제외
                        all_predictions[label.item()].append((score.item(), box.numpy(), image_id))
    
    print("평가 지표 계산 중...")
    
    # 각 IoU 임계값과 클래스에 대해 AP 계산
    results = {
        'per_class_ap': {},
        'per_class_metrics': {},
        'mAP': {},
        'overall_metrics': {}
    }
    
    # 클래스별 AP 계산
    for class_id in range(1, len(class_names)):  # 배경 클래스(0) 제외
        class_name = class_names[class_id]
        
        if class_id not in all_ground_truths:
            print(f"클래스 '{class_name}'에 대한 Ground Truth가 없습니다.")
            continue
            
        if class_id not in all_predictions:
            print(f"클래스 '{class_name}'에 대한 예측이 없습니다.")
            results['per_class_ap'][class_name] = {f'AP@{iou:.2f}': 0.0 for iou in iou_thresholds}
            continue
        
        # 예측을 confidence 순으로 정렬
        predictions = sorted(all_predictions[class_id], key=lambda x: x[0], reverse=True)
        ground_truths = all_ground_truths[class_id]
        
        # 각 IoU 임계값에 대해 AP 계산
        class_aps = {}
        
        for iou_threshold in iou_thresholds:
            # 각 이미지별로 GT 매칭 여부 추적 - 수정된 버전
            gt_matched = {}
            
            # 각 이미지별 GT 개수 미리 계산
            image_gt_counts = defaultdict(int)
            for _, gt_image_id in ground_truths:
                image_gt_counts[gt_image_id] += 1
            
            # 각 이미지별로 매칭 배열 초기화
            for image_id, count in image_gt_counts.items():
                gt_matched[image_id] = np.zeros(count, dtype=bool)
            
            true_positives = []
            false_positives = []
            
            for confidence, pred_box, pred_image_id in predictions:
                # 같은 이미지의 GT들과 비교 - 인덱스 매핑 수정
                image_gts = []
                local_idx = 0
                for global_idx, (gt_box, gt_image_id) in enumerate(ground_truths):
                    if gt_image_id == pred_image_id:
                        image_gts.append((gt_box, local_idx))
                        local_idx += 1
                
                best_iou = 0
                best_gt_local_idx = -1
                
                for gt_box, local_gt_idx in image_gts:
                    iou = compute_iou(torch.tensor(pred_box), torch.tensor(gt_box))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_local_idx = local_gt_idx
                
                # 매칭 확인 및 업데이트
                if (best_iou >= iou_threshold and 
                    pred_image_id in gt_matched and 
                    best_gt_local_idx < len(gt_matched[pred_image_id]) and
                    not gt_matched[pred_image_id][best_gt_local_idx]):
                    # True Positive
                    true_positives.append(1)
                    false_positives.append(0)
                    gt_matched[pred_image_id][best_gt_local_idx] = True
                else:
                    # False Positive
                    true_positives.append(0)
                    false_positives.append(1)
            
            # 누적합 계산
            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)
            
            # Precision과 Recall 계산
            num_gt = len(ground_truths)
            recalls = tp_cumsum / max(num_gt, 1)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
            
            # AP 계산
            ap = compute_ap(recalls, precisions)
            class_aps[f'AP@{iou_threshold:.2f}'] = ap
            
        results['per_class_ap'][class_name] = class_aps
        
        # 클래스별 전체 메트릭 (IoU@0.5 기준)
        if predictions and ground_truths:
            final_precision = precisions[-1] if len(precisions) > 0 else 0.0
            final_recall = recalls[-1] if len(recalls) > 0 else 0.0
            f1_score = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-8)
            
            results['per_class_metrics'][class_name] = {
                'precision': final_precision,
                'recall': final_recall,
                'f1_score': f1_score,
                'num_predictions': len(predictions),
                'num_ground_truths': len(ground_truths)
            }
    
    # 전체 mAP 계산
    for iou_threshold in iou_thresholds:
        iou_key = f'AP@{iou_threshold:.2f}'
        aps = []
        for class_name in results['per_class_ap']:
            if iou_key in results['per_class_ap'][class_name]:
                aps.append(results['per_class_ap'][class_name][iou_key])
        
        results['mAP'][iou_key] = np.mean(aps) if aps else 0.0
    
    # mAP@0.5:0.95 (COCO 스타일)
    results['mAP']['mAP@0.5:0.95'] = np.mean([results['mAP'][f'AP@{iou:.2f}'] for iou in iou_thresholds])
    
    # 전체 성능 요약
    all_precisions = [metrics['precision'] for metrics in results['per_class_metrics'].values()]
    all_recalls = [metrics['recall'] for metrics in results['per_class_metrics'].values()]
    all_f1s = [metrics['f1_score'] for metrics in results['per_class_metrics'].values()]
    
    results['overall_metrics'] = {
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'mean_f1_score': np.mean(all_f1s) if all_f1s else 0.0,
        'total_predictions': sum([metrics['num_predictions'] for metrics in results['per_class_metrics'].values()]),
        'total_ground_truths': sum([metrics['num_ground_truths'] for metrics in results['per_class_metrics'].values()])
    }
    
    return results


def print_evaluation_results(results: Dict[str, Any]):
    """평가 결과를 보기 좋게 출력"""
    print("\n" + "="*80)
    print("객체 검출 평가 결과")
    print("="*80)
    
    # 전체 mAP
    print("\n📊 전체 mAP:")
    print(f"  mAP@0.5:0.95: {results['mAP']['mAP@0.5:0.95']:.4f}")
    print(f"  mAP@0.5:     {results['mAP'].get('AP@0.50', 0.0):.4f}")
    print(f"  mAP@0.75:    {results['mAP'].get('AP@0.75', 0.0):.4f}")
    
    # 전체 성능 요약
    print(f"\n📈 전체 성능 요약:")
    overall = results['overall_metrics']
    print(f"  평균 Precision: {overall['mean_precision']:.4f}")
    print(f"  평균 Recall:    {overall['mean_recall']:.4f}")
    print(f"  평균 F1-Score:  {overall['mean_f1_score']:.4f}")
    print(f"  총 예측 수:     {overall['total_predictions']}")
    print(f"  총 Ground Truth: {overall['total_ground_truths']}")
    
    # 클래스별 상세 결과
    print(f"\n📋 클래스별 상세 결과:")
    print("-"*80)
    print(f"{'클래스':<15} {'AP@0.5':<8} {'AP@0.75':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
    print("-"*80)
    
    for class_name in results['per_class_ap']:
        ap_data = results['per_class_ap'][class_name]
        metrics_data = results.get('per_class_metrics', {}).get(class_name, {})
        
        ap_50 = ap_data.get('AP@0.50', 0.0)
        ap_75 = ap_data.get('AP@0.75', 0.0)
        precision = metrics_data.get('precision', 0.0)
        recall = metrics_data.get('recall', 0.0)
        f1_score = metrics_data.get('f1_score', 0.0)
        
        print(f"{class_name:<15} {ap_50:<8.4f} {ap_75:<9.4f} {precision:<10.4f} {recall:<8.4f} {f1_score:<8.4f}")
    
    print("-"*80)


def plot_evaluation_results(results: Dict[str, Any], save_path: str = None):
    """평가 결과를 그래프로 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('객체 검출 성능 평가 결과', fontsize=16, fontweight='bold')
    
    # 1. 클래스별 AP@0.5
    if results['per_class_ap']:
        classes = list(results['per_class_ap'].keys())
        ap_values = [results['per_class_ap'][cls].get('AP@0.50', 0.0) for cls in classes]
        
        axes[0, 0].bar(classes, ap_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('클래스별 AP@0.5')
        axes[0, 0].set_ylabel('Average Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 클래스별 Precision, Recall, F1-Score
    if results['per_class_metrics']:
        classes = list(results['per_class_metrics'].keys())
        precisions = [results['per_class_metrics'][cls]['precision'] for cls in classes]
        recalls = [results['per_class_metrics'][cls]['recall'] for cls in classes]
        f1_scores = [results['per_class_metrics'][cls]['f1_score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 1].bar(x - width, precisions, width, label='Precision', alpha=0.7)
        axes[0, 1].bar(x, recalls, width, label='Recall', alpha=0.7)
        axes[0, 1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
        
        axes[0, 1].set_title('클래스별 Precision, Recall, F1-Score')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. IoU 임계값별 mAP
    iou_keys = [k for k in results['mAP'].keys() if k.startswith('AP@')]
    if iou_keys:
        iou_thresholds = [float(k.split('@')[1]) for k in iou_keys]
        map_values = [results['mAP'][k] for k in iou_keys]
        
        axes[1, 0].plot(iou_thresholds, map_values, 'o-', linewidth=2, markersize=6)
        axes[1, 0].set_title('IoU 임계값별 mAP')
        axes[1, 0].set_xlabel('IoU Threshold')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0.5, 0.95)
    
    # 4. 전체 성능 요약 (도넛 차트)
    overall = results['overall_metrics']
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [overall['mean_precision'], overall['mean_recall'], overall['mean_f1_score']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    wedges, texts, autotexts = axes[1, 1].pie(values, labels=metrics, colors=colors, autopct='%1.3f',
                                             startangle=90, wedgeprops=dict(width=0.5))
    axes[1, 1].set_title('전체 성능 요약')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"평가 결과 그래프 저장: {save_path}")
    
    plt.show()


# ====================================================================
# trainer_fasterrcnn.py에서 사용할 통합 평가 함수
# ====================================================================

def evaluate_with_metrics(model, data_loader, device, class_names, 
                         confidence_threshold=0.5, compute_detailed_metrics=True):
    """
    기존 evaluate 함수를 확장하여 loss와 detection metrics를 모두 계산
    
    Args:
        model: 평가할 모델
        data_loader: 검증 데이터 로더  
        device: 디바이스
        class_names: ['background', 'class1', 'class2', ...] 형태의 클래스 이름 리스트
        confidence_threshold: 예측 신뢰도 임계값
        compute_detailed_metrics: 상세 메트릭 계산 여부 (시간이 오래 걸림)
    
    Returns:
        avg_loss: 평균 검증 손실
        detailed_results: mAP, precision, recall 등 상세 결과 (None일 수 있음)
    """
    model.eval()
    total_loss = 0.0
    detailed_results = None
    
    # Loss 계산
    print("검증 손실 계산 중...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Computing Loss"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Loss 계산을 위해 일시적으로 train 모드
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    # 상세 메트릭 계산 (선택적)
    if compute_detailed_metrics:
        print("상세 검출 메트릭 계산 중...")
        detailed_results = evaluate_detection(
            model, data_loader, device, class_names, confidence_threshold
        )
        print_evaluation_results(detailed_results)
    
    return avg_loss, detailed_results


# ====================================================================
# trainer_fasterrcnn.py의 main 함수 수정 예시
# ====================================================================

def modified_main_loop_example():
    """
    기존 trainer_fasterrcnn.py의 main 함수에서 수정할 부분 예시
    """
    
    # 클래스 이름 생성 (기존 코드에서)
    class_names = ['background'] + [name for name in train_dataset.sequential_label_to_original_name.values()]
    
    # 학습 루프에서 검증 부분 수정
    for epoch in range(start_epoch, args.num_epochs):
        # 학습
        train_loss = train_one_epoch(
            model, optimizer, train_loader, args.device, epoch, args.print_freq
        )
        train_losses.append(train_loss)

        # 상세 메트릭 계산 여부 결정 (예: 5 에포크마다)
        compute_detailed = (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1
        
        # 검증 (수정된 함수 사용)
        val_loss, detailed_results = evaluate_with_metrics(
            model, val_loader, args.device, class_names, 
            confidence_threshold=args.confidence_threshold,
            compute_detailed_metrics=compute_detailed
        )
        val_losses.append(val_loss)
        
        # 상세 결과가 있으면 시각화
        if detailed_results and args.visualize_predictions:
            plot_evaluation_results(
                detailed_results, 
                save_path=os.path.join(args.checkpoint_dir, f'metrics_epoch_{epoch+1}.png')
            )

        # 나머지 코드는 동일...