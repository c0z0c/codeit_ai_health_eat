import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # weights 가져오기
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchinfo import summary

class FasterRCNN_resnet101(torch.nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN_resnet101, self).__init__()
        # 1. ResNet-101 + FPN backbone 생성
        self.backbone = resnet_fpn_backbone('resnet101', pretrained=True)

        # 2. Faster R-CNN 생성 (num_classes는 원하는 클래스 수)
        self.model = FasterRCNN(self.backbone, num_classes=num_classes)  # COCO는 91

    def forward(self, images, targets=None):
        return self.model(images, targets)


model = FasterRCNN_resnet101(92)

summary(model)