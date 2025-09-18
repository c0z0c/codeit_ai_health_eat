# use_wandb_yolo.py
import os, wandb, torch, multiprocessing as mp
from wandb_yolo import WanDBYolo

def main():
    # W&B 환경 & 선점 init 
    os.environ["WANDB_PROJECT"] = "ai04-level1-project"
    os.environ["WANDB_ENTITY"]  = "dksksj12-student"
    os.environ["WANDB_MODE"]    = "online"
    
    tune_name = "yolov8s-tune"

    # 튠 런
    wandb.init(project=os.environ["WANDB_PROJECT"],
               entity=os.environ["WANDB_ENTITY"],
               name=tune_name,
               reinit=True)

    # 먼저 workers=0로 테스트 (Windows 안정화)
    wandb_yolo_model = WanDBYolo(
        name=tune_name,
        workers=2,
        epochs=100,
        patience=25
    )

    # 최적 하이퍼 탐색
    wandb_yolo_model.tune(iterations=15, epochs=15, patience=5, save=True, use_ray=False, name=tune_name)      # 튠에서만 사용할 파라미터 값을 지울시 설정 값

    wandb.finish()  # 튠 런 종료(다음 런과 분리), 주석처리시 위에서 파라미터에서 사용한 설정(epoch, patience)

    # 2) 방금 생성된 best_hyperparameters.yaml로 학습
    #    (경로를 안 넘기면 자동으로 "가장 최근" 파일을 찾습니다)
    # wandb.finish()  # 튠 런 종료 (원하면 같은 런 유지해도 무방)
    wandb.init(project=os.environ["WANDB_PROJECT"], 
               entity=os.environ["WANDB_ENTITY"],
               name="yolov8s-train-best",             
               reinit=True)
    

    wandb_yolo_model.train_with_best()

    wandb.finish()

if __name__ == "__main__":
    mp.freeze_support()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

# job.val(split="test")
# job.predict(source=r"E:\ai04-level1-project\datasets\pills\images\test")
# job.export(format="onnx")
