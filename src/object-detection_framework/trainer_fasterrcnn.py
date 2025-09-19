import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms import v2
import re  # 정규 표현식 모듈 추가
import wandb


import models
from dataset import CustomCocoDataset, custom_collate_fn
from models import get_model, CustomFasterRCNN
from utils import visualize_prediction, get_id2name_dict
from metrics import evaluate_with_metrics, plot_evaluation_results
# , log_metrics_to_wandb

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False


def get_transforms(train=True):
    """데이터 증강을 위한 transform 정의"""
    if train:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(p=0.5),
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    return transforms


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """한 에폭 학습 함수"""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)):
        # print(images, targets)
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss


# def evaluate(model, data_loader, device):
#     """검증 함수"""
#     model.eval()
#     total_loss = 0.0
#
#     with torch.no_grad():
#         for images, targets in tqdm(data_loader, desc="Evaluating"):
#             images = [image.to(device) for image in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#             # 평가 모드에서는 targets를 전달하지 않음
#             model.train()  # loss 계산을 위해 일시적으로 train 모드
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             total_loss += losses.item()
#             model.eval()  # 다시 eval 모드로
#
#     avg_loss = total_loss / len(data_loader)
#     print(f"Validation Loss: {avg_loss:.4f}")
#     return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """체크포인트 로드"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss


def visualize_sample_predictions(model, dataset, device, class_names, num_samples=5):
    """샘플 예측 결과 시각화"""
    model.eval()

    # 랜덤하게 샘플 선택
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for idx in indices:
            image, target = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            prediction = model(image_tensor)[0]

            # CPU로 이동
            prediction = {k: v.cpu() for k, v in prediction.items()}
            target = {k: v.cpu() for k, v in target.items()}

            visualize_prediction(image, prediction, class_names, target)


def main(args, model_name):
    print(f'사용중인 모델 명 : {model_name}')
    print(f"Using device: {args.device}")

    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    # # 체크포인트 디렉토리 생성(존재하지 않을 경우)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    ''' train1, train2 새로 훈련할때마다 새로운 폴더 만들어서 저장'''
    base_checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    os.makedirs(base_checkpoint_dir, exist_ok=True)  # 상위 폴더가 없으면 생성

    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        # 학습 재개 시에는 기존 체크포인트가 있는 폴더를 사용
        args.checkpoint_dir = os.path.dirname(args.checkpoint_path)
        print(f"학습 재개: 기존 체크포인트 경로 사용 - {args.checkpoint_dir}")
    else:
        # 새롭게 학습을 시작할 때만 고유한 폴더 생성
        existing_runs = [d for d in os.listdir(base_checkpoint_dir) if re.match(r'^train\d+$', d)]

        if existing_runs:
            # 기존 폴더에서 가장 큰 숫자 찾기
            run_numbers = [int(re.match(r'^train(\d+)$', d).group(1)) for d in existing_runs]
            next_run_number = max(run_numbers) + 1
        else:
            # 기존 폴더가 없으면 1부터 시작
            next_run_number = 1

        run_dir_name = f"train{next_run_number}"
        args.checkpoint_dir = os.path.join(base_checkpoint_dir, run_dir_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"새로운 훈련 시작: 고유한 폴더 생성 - {args.checkpoint_dir}")

    '''wandb 연결'''
    if args.use_wandb:
        # wandb.init() 호출 시 실행 이름을 `run_dir_name`으로 설정하여 폴더명과 일치시킵니다.
        # 기존 학습을 재개하는 경우, 이름은 이미 저장된 `run_dir_name`이 됩니다.
        run_name = os.path.basename(args.checkpoint_dir)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=args
        )
        print("Wandb에 연결되었습니다.")


    # 어노테이션 파일 존재 확인
    if not os.path.exists(args.train_annotation_path):
        raise FileNotFoundError(f"훈련 annotation 파일을 찾을 수 없습니다.: {args.train_annotation_path}")
    if not os.path.exists(args.valid_annotation_path):
        raise FileNotFoundError(f"검증 annotation 파일을 찾을 수 없습니다.: {args.valid_annotation_path}")

    # 데이터셋 생성 - 훈련용과 검증용을 별도로 생성
    print("Loading datasets...")
    train_dataset = CustomCocoDataset(
        image_dir=args.train_image_dir,
        annotation_file=args.train_annotation_path,
        id2label_path=args.id2label_path,
        transforms=get_transforms(train=True),
        # transforms = get_transforms(train=True),
    )

    val_dataset = CustomCocoDataset(
        image_dir=args.valid_image_dir,
        annotation_file=args.valid_annotation_path,
        id2label_path=args.id2label_path,
        transforms=get_transforms(train=False)
    )

    # 클래스 수 설정 (훈련 데이터셋 기준)
    args.num_classes = train_dataset.num_total_classes
    print(f"클래스 수(배경포함): {args.num_classes}")
    # print(f"Class mapping: {train_dataset.sequential_label_to_original_name}")

    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    # 모델 생성
    print("Creating model...")
    model = models.get_model(model_name, args.num_classes)
    model.to(args.device)

    # 옵티마이저 및 스케줄러 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )
    # 체크포인트 로드 (재개 학습)
    start_epoch = 0
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path)
        start_epoch += 1

    class_names = ['background'] + list(train_dataset.id2label.values())
    best_map = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, args.num_epochs):
        # 학습
        train_loss = train_one_epoch(
            model, optimizer, train_loader, args.device, epoch, args.print_freq
        )
        train_losses.append(train_loss)

        # 상세 평가 여부 결정 (설정 에포크마다 또는 마지막 에포크)
        compute_detailed = (epoch + 1) % args.eval_freq == 0 or epoch == args.num_epochs - 1

        # 검증 (새로운 함수 사용)
        val_loss, detailed_results = evaluate_with_metrics(
            model, val_loader, args.device, class_names,
            confidence_threshold=args.confidence_threshold,
            compute_detailed_metrics=compute_detailed
        )
        val_losses.append(val_loss)

        # 최고 성능 모델 저장
        if detailed_results:
            current_map = detailed_results['mAP']['mAP@0.5:0.95']
            if current_map > best_map:
                best_map = current_map
                best_model_path = os.path.join(args.checkpoint_dir, f"best_model_map_{current_map:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"🎉 새로운 최고 성능! mAP@0.5:0.95: {current_map:.4f}")

            # 평가 결과 시각화 저장
            if hasattr(args, 'save_metric_plots') and args.save_metric_plots:
                plot_path = os.path.join(args.checkpoint_dir, f'metrics_epoch_{epoch + 1}.png')
                plot_evaluation_results(detailed_results, save_path=plot_path)
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }

            if detailed_results:
                log_dict.update({
                    "mAP@0.5:0.95": detailed_results['mAP']['mAP@0.5:0.95'],
                    # "mAP@0.5": detailed_results['mAP']['mAP@0.5'],
                    # "mAP@0.75": detailed_results['mAP']['mAP@0.75'],
                })
            if args.use_wandb:
                wandb.log(log_dict)
        
        # 스케줄러 업데이트
        scheduler.step()

        # 체크포인트 저장
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)



        print(f"Epoch {epoch + 1}/{args.num_epochs} completed\n")
        if detailed_results:
            print(f"Current mAP@0.5:0.95: {detailed_results['mAP']['mAP@0.5:0.95']:.4f}, Best: {best_map:.4f}")
        # 이미지 시각화
    if args.visualize_predictions:
        visualize_sample_predictions(
            model, val_dataset, args.device, class_names, args.vis_num_samples)
'''val_loss만 계산하는 train과정'''
    # # 학습 루프
    # train_losses = []
    # val_losses = []
    #
    # print("Starting training...")
    # for epoch in range(start_epoch, args.num_epochs):
    #     # 학습
    #     train_loss = train_one_epoch(
    #         model, optimizer, train_loader, args.device, epoch, args.print_freq
    #     )
    #     train_losses.append(train_loss)
    #
    #     # 검증
    #     val_loss = evaluate(model, val_loader, args.device)
    #     val_losses.append(val_loss)
    #
    #     # 스케줄러 업데이트
    #     scheduler.step()
    #
    #     # 체크포인트 저장
    #     if (epoch + 1) % args.save_freq == 0:
    #         checkpoint_path = os.path.join(
    #             args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
    #         )
    #         save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
    #
    #     print(f"Epoch {epoch + 1}/{args.num_epochs} completed\n")
    #
    # # 최종 모델 저장
    # final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    # torch.save(model.state_dict(), final_model_path)
    # print(f"Final model saved: {final_model_path}")
    #
    # # 학습 곡선 시각화
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'))
    # plt.show()
    #
    # # 샘플 예측 시각화
    # if args.visualize_predictions:
    #     class_names = ['background'] + [name for name in train_dataset.sequential_label_to_original_name.values()]
    #     visualize_sample_predictions(
    #         model, val_dataset, args.device, class_names, args.vis_num_samples
    #     )
    #
    # print("Training completed!")


if __name__ == "__main__":
    class Args_rcnn:
        def __init__(self):
            # Data paths - 새로운 구조에 맞게 수정
            self.train_image_dir = "./data/images/train"  # 훈련 이미지 폴더
            self.valid_image_dir = "./data/images/val"  # 검증 이미지 폴더
            self.train_annotation_path = "./data/labels/train/train.json"  # 훈련 어노테이션 파일
            self.valid_annotation_path = "./data/labels/val/valid.json"  # 검증 어노테이션 파일
            self.checkpoint_dir = "./checkpoints/CustomFasterRCNN"  # checkpoint 모델 경로

            # Training parameters
            self.batch_size = 4  # 배치 크기
            self.num_epochs = 11  # 에포크 수
            self.learning_rate = 0.005  # 학습률
            self.weight_decay = 0.0005  # 학습률 변화
            self.momentum = 0.9  # 모멘텀
            self.step_size = 3
            self.gamma = 0.1

            # Model parameters
            self.num_classes = None  # dataset에서 자동으로 결정, 손수 결정할 때만 입력
            self.model_name = 'customfasterrcnn'  # ['Yolov#', 'CustomFasterRCNN', ]

            # Training settings
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 디바이스 설정
            self.num_workers = 4  # Dataloader의 num_worker 수 설정
            self.print_freq = 10  # 얼마마다 loss 프린트 할 것인지
            self.save_freq = 1  # 체크포인트를 몇 epoch마다 저장할 것인지

            # Resume training
            # 중간에 학습 멈추고 다시 시작할때, checkpoint 불러오는 설정
            # self.resume = False                                                                   # False는 안불러옴 - 처음부터 학습함(대신 이전에 학습해서 checkpoint있으면, 덮어씀.
            # self.checkpoint_path = None
            self.resume = True  # True는 불러옴
            self.checkpoint_path = "./checkpoints/CustomFasterRCNN/checkpoint_epoch_10.pth"  # 불러올 checkpoint 경로

            # Validation - 이제 별도 파일로 제공되므로 필요없음
            # self.val_split = 0.2

            # Visualization
            self.visualize_predictions = True  # validation 이미지 보여 주는가
            self.vis_num_samples = 5  # 이미지 갯수

            # WandB settings
            self.use_wandb = True  # wandb 연결
            self.wandb_project = "object-detection"
            self.wandb_entity = 'AI-team4'
            self.wandb_run_name = None  # None으로 설정되면 아무렇게나 자동으로 저장됨

            # Evaluation settings
            self.eval_freq = 1  # 몇 epoch마다 평가하나
            self.confidence_threshold = 0.5  # classification된 박스를 제거하는 기준
            self.nms_threshold = 0.5  # IoU를 기준으로 박스를 제거하는 기준
            self.save_metric_plots = True                                                       # 메트릭 시각화 저장 여부


    args = Args_rcnn()
    main(args)