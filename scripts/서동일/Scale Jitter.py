# 1) 사이즈 스케줄
MS_SIZES = [448, 512, 576, 640, 704, 736]
CHANGE_EVERY = 10  # 배치 10번마다 사이즈 교체

# 2) 동적 트랜스폼 팩토리(Albumentations 예)
import albumentations as A
import random

def make_multiscale_tf(sz):
    return A.Compose([
        A.LongestMaxSize(max_size=sz),
        A.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0, value=(114,114,114)),
        # (선택) 약한 밝기/샤프닝 등
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_area=4, min_visibility=0.1))

# 3) 학습 루프에서 주기적으로 갱신
current_tf = make_multiscale_tf(random.choice(MS_SIZES))
for step, batch in enumerate(loader, 1):
    if step % CHANGE_EVERY == 0:
        current_tf = make_multiscale_tf(random.choice(MS_SIZES))
    # Dataset.__getitem__ 또는 collate_fn에서 current_tf를 참조해서 변환 적용
