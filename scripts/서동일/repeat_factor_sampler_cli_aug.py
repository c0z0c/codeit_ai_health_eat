# repeat_factor_sampler_cli_aug.py
# - CLI로 경로/하이퍼 지정
# - transforms: GaussianBlur(p≈0.1), CLAHE(L채널만 약하게), Unsharp Mask(미세 샤프닝)
# - sampler: RFS(Repeat-Factor) 또는 Class-Aware Batch Sampler 선택
# - --dry-run: 반복계수 요약/CSV 저장만

'''# 1) 설정 요약만 + RFS 반복계수 CSV 저장 (학습/로더 생성 X)
python repeat_factor_sampler_cli_aug.py `
  --ann "C:\Users\dev\data\cocoimage\train.json" ` 예시
  --images-dir "C:\Users\dev\data\images\train" `
  --t 1e-3 --alpha 0.5 --rmax 10 `
  --save-weights "C:\Users\dev\data\rfs_out\weights.csv" `
  --dry-run

# 2) RFS + 증강 적용해서 DataLoader 준비 - 희소 클래스를 높은확률로 추가
python repeat_factor_sampler_cli_aug.py `
  --ann "C:\Users\dev\data\cocoimage\train.json" `
  --images-dir "C:\Users\dev\data\images\train" `
  --sampler rfs --t 5e-4 --alpha 0.5 --rmax 8 ` / t = 몇퍼센트 이하의 클래스를 더 많이 추가 시킬건지
  --batch 12 --num-workers 8 `
  --p-blur 0.1 --p-clahe 0.2 --clahe-alpha 0.5 --p-unsharp 0.3
    GaussianBlur(p≈0.1)
    CLAHE (라이트 채널에만 약하게): LAB의 L 채널에 clipLimit 작게(예: 1.5~2.5)
    Unsharp Mask(미세 샤프닝): radius 작게(1.0~1.5), amount 0.5~1.0

# 3) Class-Aware 배치 샘플러(+증강)로 준비 - 희소 클래스를 무조건 추가
python repeat_factor_sampler_cli_aug.py `
  --ann "C:\Users\dev\data\cocoimage\train.json" `
  --images-dir "C:\Users\dev\data\images\train" `
  --sampler classaware --rare-frac-thresh 0.01 --rare-ratio 0.5 `
  --batch 8 --epoch-batches 1200 `
  --p-blur 0.1 --p-clahe 0.2 --p-unsharp 0.3
  
  # 경로
    ap.add_argument("--ann", type=Path, required=True, help="COCO train.json 경로")
    ap.add_argument("--images-dir", type=Path, required=True, help="file_name 기준 이미지 루트")
    # 샘플러 공통/선택
    ap.add_argument("--sampler", choices=["rfs", "classaware"], default="rfs", help="샘플러 선택")
    ap.add_argument("--len-factor", type=float, default=1.0, help="epoch 샘플 수 배율(RFS 전용)")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--num-workers", type=int, default=4)
    # RFS 하이퍼
    ap.add_argument("--t", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--save-weights", type=Path, default=None, help="RFS 반복계수 CSV 저장(선택)")
    ap.add_argument("--dry-run", action="store_true", help="반복계수 요약만(로더 미생성)")
    # Class-Aware 하이퍼
    ap.add_argument("--rare-frac-thresh", type=float, default=0.01, help="희소 클래스 판정 임계(이미지 등장 비율)")
    ap.add_argument("--rare-ratio", type=float, default=0.5, help="배치 내 희소 이미지 비율")
    ap.add_argument("--epoch-batches", type=int, default=1000, help="Class-Aware 한 에폭의 배치 수")
    # Transforms 하이퍼
    ap.add_argument("--p-blur", type=float, default=0.1)
    ap.add_argument("--blur-sigma-min", type=float, default=0.5)
    ap.add_argument("--blur-sigma-max", type=float, default=1.5)
    ap.add_argument("--p-clahe", type=float, default=0.2)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--clahe-alpha", type=float, default=0.5)
    ap.add_argument("--p-unsharp", type=float, default=0.3)
    ap.add_argument("--unsharp-radius", type=float, default=1.0)
    ap.add_argument("--unsharp-percent", type=int, default=50)
    ap.add_argument("--unsharp-threshold", type=int, default=3)
  '''

from pathlib import Path
import argparse, json, math, statistics, random
from collections import defaultdict
from typing import Dict, Set, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from PIL import Image, ImageFilter

try:
    import cv2  # CLAHE용 (없으면 CLAHE 건너뜀)
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def collate_fn(batch):
    return tuple(zip(*batch))


# ------------------------- Photo Transforms -------------------------
class PhotoTransforms:
    """
    - GaussianBlur: p_blur 확률로 약하게
    - CLAHE (L채널만): p_clahe 확률로 약하게(블렌딩)
    - Unsharp Mask: p_unsharp 확률로 미세 샤프닝
    박스 좌표는 안 바뀌므로 target은 그대로 반환.
    """
    def __init__(
        self,
        p_blur: float = 0.1,
        blur_sigma_min: float = 0.5,
        blur_sigma_max: float = 1.5,
        p_clahe: float = 0.2,
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
        clahe_alpha: float = 0.5,   # 0~1: CLAHE 결과와 원본 블렌딩 비율
        p_unsharp: float = 0.3,
        unsharp_radius: float = 1.0,
        unsharp_percent: int = 50,
        unsharp_threshold: int = 3,
    ):
        self.p_blur = p_blur
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.p_clahe = p_clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.clahe_alpha = np.clip(clahe_alpha, 0.0, 1.0)
        self.p_unsharp = p_unsharp
        self.unsharp_radius = unsharp_radius
        self.unsharp_percent = unsharp_percent
        self.unsharp_threshold = unsharp_threshold

    def _gaussian_blur(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _clahe_lightness(self, img: Image.Image) -> Image.Image:
        if not _HAS_CV2:
            return img  # cv2 없으면 패스
        arr = np.asarray(img)  # RGB uint8
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        out = Image.fromarray(rgb2)
        # 약하게 적용: 원본과 블렌딩
        if 0.0 < self.clahe_alpha < 1.0:
            out = Image.blend(img, out, self.clahe_alpha)
        return out

    def _unsharp(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.UnsharpMask(
            radius=self.unsharp_radius,
            percent=self.unsharp_percent,
            threshold=self.unsharp_threshold,
        ))

    def __call__(self, img: Image.Image, target: dict):
        # 순서: CLAHE(톤) → Blur(가끔) → Unsharp(가끔)
        if self.p_clahe > 0 and random.random() < self.p_clahe:
            img = self._clahe_lightness(img)
        if self.p_blur > 0 and random.random() < self.p_blur:
            img = self._gaussian_blur(img)
        if self.p_unsharp > 0 and random.random() < self.p_unsharp:
            img = self._unsharp(img)
        return img, target


# ------------------------- Dataset -------------------------
class COCODetectionDataset(Dataset):
    """COCO 포맷 -> torchvision detection 호환 Dataset (+transforms 적용)"""
    def __init__(self, coco_like: dict, images_root: Path, transforms=None):
        self.coco = coco_like
        self.images_root = images_root
        self.transforms = transforms

        self.images = {im["id"]: im for im in self.coco["images"]}
        self.img_ids = [im["id"] for im in self.coco["images"]]

        self.anns_by_img = defaultdict(list)
        for a in self.coco["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

        # 각 인덱스가 가진 클래스 집합(클래스 어웨어 샘플러용)
        self.classes_by_idx: List[Set[int]] = []
        for img_id in self.img_ids:
            cls = {int(a["category_id"]) for a in self.anns_by_img.get(img_id, [])}
            self.classes_by_idx.append(cls)

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.images[img_id]
        img_path = (self.images_root / im["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes_xyxy, labels = [], []
        for a in self.anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])   # COCO xywh -> xyxy
            labels.append(int(a["category_id"]))

        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id], dtype=torch.int64)}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


# ------------------------- Samplers -------------------------
def compute_repeat_factors(images: list, img_classes: Dict[int, Set[int]], f: Dict[int, float],
                           t=1e-3, alpha=0.5, r_min=1.0, r_max=10.0):
    """r_i = max_{c in img_i} (t / f_c) ** alpha"""
    repeats = {}
    for im in images:
        img_id = im["id"]
        cls_set = img_classes.get(img_id, set())
        if not cls_set:
            r = 1.0
        else:
            r = 1.0
            for cid in cls_set:
                fc = max(f.get(cid, 0.0), 1e-12)
                r = max(r, (t / fc) ** alpha)
        r = min(r, r_max) if r_max is not None else r
        repeats[img_id] = max(r_min, r)
    return repeats


class ClassAwareBatchSampler(Sampler[List[int]]):
    """
    희소/일반 인덱스를 따로 두고, 배치마다 rare_ratio 비율로 섞는다.
    """
    def __init__(self, rare_idx: List[int], common_idx: List[int], batch_size=8,
                 rare_ratio=0.5, epoch_batches=1000, seed=42):
        self.rare = list(rare_idx)
        self.common = list(common_idx)
        self.batch_size = batch_size
        self.k_rare = max(1, int(round(batch_size * rare_ratio)))
        self.k_common = batch_size - self.k_rare
        self.epoch_batches = epoch_batches
        self.rng = random.Random(seed)

    def __iter__(self):
        rpool = self.rare[:]
        cpool = self.common[:]
        self.rng.shuffle(rpool)
        self.rng.shuffle(cpool)
        ri = ci = 0
        for _ in range(self.epoch_batches):
            if ri + self.k_rare > len(rpool):
                self.rng.shuffle(rpool); ri = 0
            if ci + self.k_common > len(cpool):
                self.rng.shuffle(cpool); ci = 0
            batch = rpool[ri:ri+self.k_rare] + cpool[ci:ci+self.k_common]
            ri += self.k_rare; ci += self.k_common
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.epoch_batches


def split_rare_common_by_frac(cls_imgset: Dict[int, Set[int]], N_img: int, frac_thresh=0.01,
                              imgid_to_idx: Dict[int, int] = None):
    """클래스 '이미지 등장 비율' <= frac_thresh면 희소 클래스로 보고, 그 클래스를 가진 이미지들을 희소 이미지로 분류"""
    rare_cls = {cid for cid, s in cls_imgset.items() if (len(s) / max(N_img, 1)) <= frac_thresh}
    rare_img_ids = set()
    for cid in rare_cls:
        rare_img_ids |= set(cls_imgset[cid])
    rare_idx, common_idx = [], []
    for img_id, idx in imgid_to_idx.items():
        (rare_idx if img_id in rare_img_ids else common_idx).append(idx)
    return rare_idx, common_idx, rare_cls


# ------------------------- Main (CLI) -------------------------
def main():
    ap = argparse.ArgumentParser(description="RFS + Class-Aware + Photo Transforms — CLI")
    # 경로
    ap.add_argument("--ann", type=Path, required=True, help="COCO train.json 경로")
    ap.add_argument("--images-dir", type=Path, required=True, help="file_name 기준 이미지 루트")
    # 샘플러 공통/선택
    ap.add_argument("--sampler", choices=["rfs", "classaware"], default="rfs", help="샘플러 선택")
    ap.add_argument("--len-factor", type=float, default=1.0, help="epoch 샘플 수 배율(RFS 전용)")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--num-workers", type=int, default=4)
    # RFS 하이퍼
    ap.add_argument("--t", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--rmax", type=float, default=10.0)
    ap.add_argument("--save-weights", type=Path, default=None, help="RFS 반복계수 CSV 저장(선택)")
    ap.add_argument("--dry-run", action="store_true", help="반복계수 요약만(로더 미생성)")
    # Class-Aware 하이퍼
    ap.add_argument("--rare-frac-thresh", type=float, default=0.01, help="희소 클래스 판정 임계(이미지 등장 비율)")
    ap.add_argument("--rare-ratio", type=float, default=0.5, help="배치 내 희소 이미지 비율")
    ap.add_argument("--epoch-batches", type=int, default=1000, help="Class-Aware 한 에폭의 배치 수")
    # Transforms 하이퍼
    ap.add_argument("--p-blur", type=float, default=0.1)
    ap.add_argument("--blur-sigma-min", type=float, default=0.5)
    ap.add_argument("--blur-sigma-max", type=float, default=1.5)
    ap.add_argument("--p-clahe", type=float, default=0.2)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--clahe-alpha", type=float, default=0.5)
    ap.add_argument("--p-unsharp", type=float, default=0.3)
    ap.add_argument("--unsharp-radius", type=float, default=1.0)
    ap.add_argument("--unsharp-percent", type=int, default=50)
    ap.add_argument("--unsharp-threshold", type=int, default=3)

    args = ap.parse_args()

    coco = json.loads(args.ann.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns   = coco.get("annotations", [])
    cats   = coco.get("categories", [])

    print(f"[INFO] images={len(images)}  anns={len(anns)}  classes={len(cats)}")
    print(f"[PATH] ann='{args.ann}'")
    print(f"[PATH] images_dir='{args.images_dir}'")
    print(f"[CFG ] sampler={args.sampler} | t={args.t} alpha={args.alpha} rmin={args.rmin} rmax={args.rmax} "
          f"| batch={args.batch} workers={args.num_workers} len_factor={args.len_factor}")
    print(f"[AUG ] blur p={args.p_blur} σ∈[{args.blur_sigma_min},{args.blur_sigma_max}] | "
          f"clahe p={args.p_clahe} clip={args.clahe_clip} tile={args.clahe_tile} α={args.clahe_alpha} "
          f"| unsharp p={args.p_unsharp} radius={args.unsharp_radius} percent={args.unsharp_percent} thr={args.unsharp_threshold}")
    if args.p_clahe > 0 and not _HAS_CV2:
        print("[WARN] OpenCV(cv2) 미설치 → CLAHE는 건너뜁니다. pip install opencv-python")

    # 이미지별/클래스별 집합
    img_classes = defaultdict(set)
    cls_imgset  = defaultdict(set)
    for a in anns:
        img_id = a["image_id"]; cid = a["category_id"]
        img_classes[img_id].add(cid)
        cls_imgset[cid].add(img_id)

    N_img = len(images) if len(images) else 1
    f = {cid: (len(imgs) / N_img) for cid, imgs in cls_imgset.items()}  # f_c

    # transforms 구성
    tf = PhotoTransforms(
        p_blur=args.p_blur, blur_sigma_min=args.blur_sigma_min, blur_sigma_max=args.blur_sigma_max,
        p_clahe=args.p_clahe, clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile, clahe_alpha=args.clahe_alpha,
        p_unsharp=args.p_unsharp, unsharp_radius=args.unsharp_radius, unsharp_percent=args.unsharp_percent,
        unsharp_threshold=args.unsharp_threshold,
    )

    # Dry-run: RFS 통계/CSV만
    if args.dry_run:
        repeats = compute_repeat_factors(images, img_classes, f, t=args.t, alpha=args.alpha,
                                         r_min=args.rmin, r_max=args.rmax)
        if args.save_weights:
            args.save_weights.parent.mkdir(parents=True, exist_ok=True)
            with args.save_weights.open("w", encoding="utf-8") as fcsv:
                fcsv.write("image_id,file_name,repeat\n")
                for im in images:
                    img_id = im["id"]
                    fcsv.write(f"{img_id},{im.get('file_name')},{repeats.get(img_id,1.0):.6f}\n")
            print(f"[OK  ] saved weights CSV → {args.save_weights}")
        vals = list(repeats.values()) or [1.0]
        print(f"[SUM ] repeat stats: min={min(vals):.3f} max={max(vals):.3f} "
              f"mean={statistics.mean(vals):.3f} median={statistics.median(vals):.3f}")
        top = sorted(((im["file_name"], repeats[im["id"]]) for im in images), key=lambda x: x[1], reverse=True)[:8]
        print("[TOP ] highest repeat examples:")
        for fn, rv in top:
            print(f"       r={rv:.2f}  {fn}")
        return

    # Dataset
    dataset = COCODetectionDataset(coco, args.images_dir, transforms=tf)

    # Sampler
    if args.sampler == "rfs":
        repeats = compute_repeat_factors(images, img_classes, f, t=args.t, alpha=args.alpha,
                                         r_min=args.rmin, r_max=args.rmax)
        weights = [repeats.get(img_id, 1.0) for img_id in dataset.img_ids]
        num_samples = max(1, int(len(dataset) * args.len_factor))
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

        loader = DataLoader(dataset,
                            batch_size=args.batch,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers)
        print("[OK  ] RFS + transforms → DataLoader 준비됨.")

    else:  # classaware
        imgid_to_idx = {img_id: i for i, img_id in enumerate(dataset.img_ids)}
        rare_idx, common_idx, rare_cls = split_rare_common_by_frac(
            cls_imgset, N_img, frac_thresh=args.rare_frac_thresh, imgid_to_idx=imgid_to_idx
        )
        print(f"[INFO] rare classes={len(rare_cls)}  rare imgs={len(rare_idx)}  common imgs={len(common_idx)}")
        batch_sampler = ClassAwareBatchSampler(
            rare_idx=rare_idx, common_idx=common_idx,
            batch_size=args.batch, rare_ratio=args.rare_ratio,
            epoch_batches=args.epoch_batches
        )
        loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers)
        print("[OK  ] Class-Aware + transforms → DataLoader 준비됨.")

    # 이제 loader를 학습 루프에 연결


if __name__ == "__main__":
    main()
