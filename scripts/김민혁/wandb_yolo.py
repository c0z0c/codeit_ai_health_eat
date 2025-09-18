# wandb_yolo.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any
import os
import platform
import wandb
import yaml
from ultralytics import YOLO

PathLike = Union[str, Path]


@dataclass
class WanDBYolo:
    # --- W&B ---
    wandb_project: str = os.getenv("WANDB_PROJECT", "ai04-level1-project")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", "dksksj12-student")
    # None => online(기본), "offline" 또는 "disabled"를 원할 때만 지정
    wandb_mode: Optional[str] = None

    # --- 학습 관련 ---
    model: PathLike = "yolov8s.pt"                                  # .pt 또는 run 폴더/weights 폴더(사용할 yolo모델 파일)
    data_yaml: PathLike = "./data/my_dataset_config.yaml"       
    epochs: Optional[int] = None                                    # epoch수(None인 이유는 사용할때 따로 지정했기 때문.)
    iterations: int = 60
    patience: int = 8                                               # n epoch 이상 개선이 없으면 중지
    imgsz: int = 640
    name: str = "yolov8s-basic"
    device: Union[int, str] = 0                                     # GPU index 또는 "cpu"
    workers: int = 2                                                # 기본은 0이지만 가능하면 늘려서 속도 향상
    batch: Optional[int] = None
    lr0: Optional[float] = None
    project_dir: Optional[PathLike] = None                          # Ultralytics 'project' 루트
    runs_root: PathLike = "./runs/detect/"                          # 최신 best 탐색용 디렉터리

    # ---------- 생성 시 경로 정리 ----------
    def __post_init__(self):
        # model: 로컬 파일이면 절대경로로, 별칭(yolov8n.pt/.yaml)은 그대로 두기
        m = str(self.model)
        if Path(m).is_file():
            self.model = str(Path(m).resolve())
        else:
            self.model = m  # 별칭은 그대로 → Ultralytics가 다운로드/로딩 처리

        self.data_yaml = self._to_abs(self.data_yaml)
        if self.project_dir is not None:
            self.project_dir = self._to_abs(self.project_dir)

        # data yaml은 존재 확인
        assert Path(self.data_yaml).exists(), f"YAML not found: {self.data_yaml}"
        # epochs 폴백은 train()/tune()에서 일관 처리

    # ---------- 유틸 ----------
    def _to_abs(self, p: PathLike) -> str:
        p = Path(p)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return str(p)

    def _apply_wandb_env(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        if self.wandb_entity:
            os.environ["WANDB_ENTITY"] = self.wandb_entity
        if self.wandb_mode is None:
            os.environ.pop("WANDB_MODE", None)  # online 기본
        else:
            os.environ["WANDB_MODE"] = self.wandb_mode
        os.environ["WANDB_NAME"] = self.name

    def _ensure_wandb_run(self):
        if wandb.run is None:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, name=self.name, reinit=True)

    # --- per-epoch W&B 로깅 콜백 등록 ---
    def _register_callbacks(self, m: YOLO):
        # 에폭 종료마다 W&B에 지표 로깅 + 콘솔 한 줄 요약
        def on_fit_epoch_end(trainer):
            metrics: Dict[str, Any] = getattr(trainer, "metrics", {}) or {}
            if not metrics:
                return
            data = {}
            for k, v in metrics.items():
                try:
                    data[k] = float(v)
                except Exception:
                    pass
            epoch = int(getattr(trainer, "epoch", 0))
            self._ensure_wandb_run()
            wandb.log(data, step=epoch)

            # W&B 런이 없으면 만들어 두기
            if wandb.run is None:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "ai04-level1-project"),
                    entity=os.environ.get("WANDB_ENTITY", "dksksj12-student"),
                    name=os.environ.get("WANDB_NAME", self.name),
                    reinit=True,
                )
            wandb.log(data, step=epoch)

            # 콘솔 한 줄 요약 (원하면 주석 처리)
            p     = data.get("metrics/precision(B)")
            r     = data.get("metrics/recall(B)")
            m50   = data.get("metrics/mAP50(B)")
            m5095 = data.get("metrics/mAP50-95(B)")
            if all(x is not None for x in (p, r, m50, m5095)):
                print(f"\r[VAL {epoch+1}] P={p:.3f} R={r:.3f} mAP50={m50:.3f} mAP50-95={m5095:.3f}   ",
                    end="", flush=True)

        # ← 여기! 콜백은 이렇게 등록해야 함
        m.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def _build(self) -> YOLO:
        self._apply_wandb_env()
        return YOLO(self.model)
    
    # ---------- 최신 best.pt 스캔 ----------
    def latest_best(self) -> Optional[str]:
        """
        runs_root 디렉터리 전체에서 가장 최근의 weights/best.pt를 찾아서 반환.
        """
        root = Path(self._to_abs(self.runs_root))
        if not root.exists():
            return None
        cands = sorted(root.rglob("weights/best.pt"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
        return str(cands[0]) if cands else None

    # ---------- 실행 메서드 ----------
    def train(self, **overrides):
        m = self._build()
        self._register_callbacks(m)

        args = dict(
            data=self.data_yaml,
            epochs=self.epochs,
            patience=self.patience,
            imgsz=self.imgsz,
            name=self.name,
            device=self.device,
            workers=self.workers,
            val=True,
        )

        if self.batch is not None: args["batch"] = self.batch
        if self.lr0 is not None: args["lr0"] = self.lr0
        if self.project_dir is not None: args["project"] = self.project_dir
        args.update(overrides)
        print(f"[train] data={self.data_yaml}\n[train] model={self.model}\n[train] name={self.name}\n[train] workers={self.workers}")

        out = m.train(**args)
        print()

        # 이번 런의 best 경로 출력(가능한 경우)
        try:
            this_best = getattr(getattr(m, "trainer", None), "best", None)
            if this_best:
                print(f"[train] this run best: {this_best}")
        except Exception:
            pass

        # 학습 종료 후 한 번 더 평가(플롯 보장)
        final_metrics = m.val(data=self.data_yaml, split="val", imgsz=self.imgsz, device=self.device, plots=True)
        try:
            self._ensure_wandb_run()
            wandb.log({k: float(v) for k, v in final_metrics.results_dict.items()}, step=int(self.epochs))
        except Exception:
            pass

        # 전체 runs에서 가장 최근 best도 안내
        latest = self.latest_best()
        if latest:
            print(f"[latest_best] {latest}")

        return out
        
    def val(self, split="val", **overrides):
        m = self._build()
        return m.val(data=self.data_yaml, split=split, imgsz=self.imgsz, device=self.device, plots=True, **overrides)

    def predict(self, source: PathLike, **overrides):
        m = self._build()
        return m.predict(source=str(source), imgsz=self.imgsz, device=self.device, save=True, **overrides)

    def export(self, format="onnx", **overrides):
        m = self._build()
        return m.export(format=format, **overrides)

    def resume(self, last_weights: PathLike):
        self._apply_wandb_env()
        return YOLO(self._to_abs(last_weights)).train(resume=True)
    
    # best_hyperparameters.yaml 찾는 헬퍼
    def _find_latest_best_hparams(self) -> Optional[str]:
        # runs_root(디렉터리) 기준으로 탐색
        base = Path(self._to_abs(self.runs_root))
        if not base.exists():
            base = Path(__file__).resolve().parent
        cands = sorted(
            base.rglob("best_hyperparameters.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return str(cands[0]) if cands else None
    
    
    def tune(self, iterations, use_ray=False, **overrides):
        """
        내장 HPO 실행: 여러 조합으로 학습/검증을 돌려 best_hyperparameters.yaml 생성.
        iterations: 시도 횟수(예: 30~100)
        """
        m = self._build()
        args = dict(
            data=self.data_yaml,
            epochs=self.epochs,   # 각 시도(run) 당 에폭
            iterations=iterations,
            imgsz=self.imgsz,
            device=self.device,
            val=True,                   # 지표 필수
            use_ray=use_ray,            # 고급: Ray Tune 사용 (GPU 1개면 False 권장)
            plots=False, save=False,
            # optimizer="AdamW",        # 원하면 고정 가능 (미설정이면 auto/기본)
        )
        args.update(overrides)
        print(f"[tune] iterations={iterations} epochs/try={args['epochs']} imgsz={self.imgsz} device={self.device}")
        return m.tune(**args)

    def train_with_best(self, best_yaml_path: Optional[str] = None, **overrides):
        """
        튠 결과(best_hyperparameters.yaml)를 읽어들여 그 값으로 train.
        best_yaml_path가 None이면 최신 파일을 자동으로 탐색.
        """
        if best_yaml_path is None:
            best_yaml_path = self._find_latest_best_hparams()
        assert best_yaml_path, "best_hyperparameters.yaml 을 찾지 못했습니다."

        with open(best_yaml_path, "r", encoding="utf-8") as f:
            best = yaml.safe_load(f) or {}

        # Ultralytics의 train 인자와 겹치는 키만 남기기(안전)
        allow = {
            "lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr",
            "box","cls","dfl","hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective",
            "flipud","fliplr","mosaic","mixup","copy_paste","erasing"
        }
        best = {k: v for k, v in best.items() if k in allow}

        print(f"[train_with_best] using {best_yaml_path}")
        return self.train(**best, **overrides)


    # ---------- 편의 ----------
    @staticmethod
    def latest_best(runs_root: PathLike = "./runs/detect") -> Optional[str]:
        root = Path(runs_root)
        cands = sorted(root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(cands[0]) if cands else None
