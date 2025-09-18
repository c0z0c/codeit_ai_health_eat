import tester_yolo
import tester_fasterrcnn
import torch


class Args_fasterrcnn:
    def __init__(self):
        # Data paths
        self.test_image_dir = "./data/ai04-level1-project/test_images"          ##
        self.model_path = "./checkpoints/CustomFasterRCNN/final_model.pth"      ##
        self.label2name = './data/label2name.json'
        self.label2id = './data/label2id.json'

        # Inference parameters
        self.predict_one_image = False                                       ## 이미지 하나만 예측할때 True
        self.one_image_path = './data/ai04-level1-project/test_images/1.png'

        self.batch_size = 4
        self.num_workers = 4
        self.confidence_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        ## 디바이스 설정

        # Model parameters
        self.num_classes = 93       ## 반드시 모델 학습 시 사용한 클래스 수로 맞춰야 함

        # Output settings
        self.save_predictions = True                        ## json 데이터 저장 여부
        self.prediction_output_path = "./predictions"       ## json 데이터 저장 폴더 설정
        self.save_visualizations = True                         ## 이미지 데이터 저장 여부
        self.visualization_output_path = "./visualizations"     ## 이미지 데이터 저장 폴더
        self.vis_num_samples = 20                               ## 이미지 데이터 저장 갯수


class Args_yolo:
    def __init__(self):
        self.model_path = './runs/detect/train4/weights/best.pt'
        self.label2id_path = './data/label2id.json'
        # 이미지 하나만 사용할 경우 True 아니면 False
        self.predict_one_image = True
        self.test_image_folder = './data/ai04-level1-project/test_images'
        self.test_image_path = './data/ai04-level1-project/test_images/1.png'


if __name__ == "__main__":
    model_name = input('사용할 모델 명을 선택하세요 [FasterRCNN, Yolo] : ')
    lower_model_name = str.lower(model_name)
    if lower_model_name == 'fasterrcnn':
        args = Args_fasterrcnn()
        tester_fasterrcnn.main(args)


    if lower_model_name == 'yolo':
        args = Args_yolo()
        tester_yolo.main(args)