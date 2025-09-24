import os
import matplotlib.pyplot as plt
import re
import wandb

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우
plt.rcParams['axes.unicode_minus'] = False

from ultralytics import YOLO

def main(args):
    base_project_dir = "./object-detection"
    yolo_project_dir = base_project_dir + "/yolo"
    os.makedirs(yolo_project_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(yolo_project_dir) if re.match(rf'^yolo_train\d+$', d)]

    if existing_runs:
        # 기존 폴더에서 가장 큰 숫자 찾기
        run_numbers = [int(re.match(rf'^yolo_train(\d+)$', d).group(1)) for d in existing_runs]
        next_run_number = max(run_numbers) + 1
    else:
        # 기존 폴더가 없으면 1부터 시작
        next_run_number = 1


    # 새로운 실행 폴더명 생성
    run_name = f"yolo_train{next_run_number}"
    run_dir = os.path.join(yolo_project_dir, run_name)


    # WandB 설정
    if not args.wandb_run_name:
        args.wandb_run_name = run_name

    if args.use_wandb:
        wandb.login()
    print(yolo_project_dir)
    print(run_name)

    model = YOLO(args.model_name)
    results = model.train(data=args.yaml_path,
                          epochs=args.num_epochs,
                          imgsz=640,
                          # degrees=args.degrees,
                          batch=args.batch_size,
                          device=0,
                          project=yolo_project_dir,  # object-detection/yolo로 설정
                          name=run_name,             # yolo_train{num}으로 설정
                          exist_ok=True,
                          lr0=0.005,
                          optimizer='Adam'

                          )

    print(f"학습 완료! 결과 저장 위치: {run_dir}")
    print(f"best.pt 저장 위치: {os.path.join(run_dir, 'weights', 'best.pt')}")

    return run_dir  # 추론에서 사용할 수 있도록 경로 반환
# if __name__ == "__main__":
#     args = Args()
#     main(args)
