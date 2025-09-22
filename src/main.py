import sys
import json
import os
import base64
from io import BytesIO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QFileDialog, QLabel, QScrollArea, 
                            QTextEdit, QSplitter, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QColor
from PIL import Image
import numpy as np
from PillAnalysisEngine import PillAnalysisEngine

def print_json_tree(data, indent="", max_depth=4, _depth=0, list_count=10, print_value=True, max_length=30):
    """
    JSON 객체를 지정한 단계(max_depth)까지 트리 형태로 출력
    - list 타입은 list_count개 이상일 때 개수만 출력
    - 하위 노드가 값일 경우 key(type) 형태로 출력
    - print_value=True일 때 key(type): 값 형태로 출력
    - max_length: 문자열 값의 출력 최대 길이 (초과시 ... 표시)
    """
    if _depth > max_depth:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent}|-- {key}")
                print_json_tree(value, indent + "    ", max_depth, _depth + 1, list_count, print_value, max_length)
            else:
                if print_value:
                    if isinstance(value, str) and len(value) > max_length:
                        display_value = f'{value[:max_length]}...'
                    else:
                        display_value = value
                    print(f"{indent}|-- {key}({type(value).__name__}): {display_value}")
                else:
                    print(f"{indent}|-- {key}({type(value).__name__})")
    elif isinstance(data, list):
        if len(data) > list_count:
            print(f"{indent}|-- [list] ({len(data)} items)")
        else:
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    print(f"{indent}|-- [{i}]")
                    print_json_tree(item, indent + "    ", max_depth, _depth + 1, list_count, print_value, max_length)
                else:
                    if print_value:
                        if isinstance(item, str) and len(item) > max_length:
                            display_item = f'{item[:max_length]}...'
                        else:
                            display_item = item
                        print(f"{indent}|-- [{i}]({type(item).__name__}): {display_item}")
                    else:
                        print(f"{indent}|-- [{i}]({type(item).__name__})")
    else:
        if print_value:
            if isinstance(data, str) and len(data) > max_length:
                display_data = f'{data[:max_length]}...'
            else:
                display_data = data
            print(f"{indent}{type(data).__name__}: {display_data}")
        else:
            print(f"{indent}{type(data).__name__}")

class AnalysisWorker(QThread):
    """분석 작업을 위한 워커 스레드"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, engine, image_path):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
    
    def run(self):
        try:
            result_json = self.engine.analyze_image(self.image_path)

            # 문자열이면 파싱해서 트리 출력, dict면 그대로 출력
            if isinstance(result_json, str):
                try:
                    parsed = json.loads(result_json)
                except json.JSONDecodeError:
                    parsed = result_json  # 비-JSON 문자열이면 그대로 둠
            else:
                parsed = result_json

            print_json_tree(parsed, max_length=30)  # 문자열 값만 max_length 적용

            # finished 시그널은 str이므로 dict면 JSON 문자열로 변환
            if isinstance(result_json, (dict, list)):
                self.finished.emit(json.dumps(result_json, ensure_ascii=False))
            else:
                self.finished.emit(result_json)
        except Exception as e:
            self.error.emit(str(e))

class PillAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.current_image_path = None
        self.full_pixmap = QPixmap()
        self.init_ui()
        self.init_engine()
    
    def init_ui(self):
        self.setWindowTitle("알약 분석기")
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 파일 선택 버튼
        file_layout = QHBoxLayout()
        self.select_btn = QPushButton("이미지 파일 선택")
        self.select_btn.clicked.connect(self.select_file)
        self.analyze_btn = QPushButton("분석 시작")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        
        file_layout.addWidget(self.select_btn)
        file_layout.addWidget(self.analyze_btn)
        file_layout.addStretch()
        
        main_layout.addLayout(file_layout)
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        # self.progress_bar.setRange(0, 0)
        main_layout.addWidget(self.progress_bar)
        
        # 스플리터로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)
        
        # 왼쪽: 이미지 영역
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 원본 이미지 + 박스
        self.image_label = QLabel("이미지를 선택하세요")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        
        # 스크롤 영역에 이미지 라벨 추가
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        left_layout.addWidget(self.scroll_area)
        
        # 알약 크롭 이미지들
        self.crop_label = QLabel("분석 결과가 여기에 표시됩니다")
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setMinimumSize(400, 150)
        self.crop_label.setStyleSheet("border: 1px solid gray;")
        left_layout.addWidget(self.crop_label)
        
        splitter.addWidget(left_widget)
        
        # 오른쪽: 결과 텍스트
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        result_label = QLabel("분석 결과 및 주의사항")
        result_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setMinimumWidth(400)
        right_layout.addWidget(self.result_text)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
    
    def init_engine(self):
        """엔진 초기화"""
        try:
            py_dir = os.path.dirname(os.path.abspath(__file__))
            model_1_stage_path = os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101", "best.pt")
            
            if os.path.exists(model_1_stage_path):
                self.engine = PillAnalysisEngine(model_1_stage_path)
                self.result_text.setText("분석 엔진이 준비되었습니다.")
            else:
                self.result_text.setText(f"모델 파일을 찾을 수 없습니다: {model_1_stage_path}")
                
        except Exception as e:
            self.result_text.setText(f"엔진 초기화 실패: {str(e)}")
    
    def select_file(self):
        """파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_original_image(file_path)
            self.analyze_btn.setEnabled(True)
            self.result_text.setText(f"선택된 파일: {os.path.basename(file_path)}")
    
    def display_original_image(self, image_path):
        """원본 이미지 표시"""
        try:
            pixmap = QPixmap(image_path)
            # 전체 해상도 이미지를 보관하고 화면 크기에 맞게 갱신
            self.full_pixmap = pixmap
            self.update_main_image()
        except Exception as e:
            QMessageBox.warning(self, "오류", f"이미지 로드 실패: {str(e)}")
    
    def analyze_image(self):
        """이미지 분석"""
        if not self.engine or not self.current_image_path:
            return
        
        # UI 상태 변경
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 프로그레스
        self.result_text.setText("🔍 이미지 분석 중...")
        
        # 워커 스레드 시작
        self.worker = AnalysisWorker(self.engine, self.current_image_path)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
    
    def on_analysis_finished(self, result_json):
        """분석 완료"""
        try:
            self.display_results(result_json)
            
            # UI 상태 복원
            self.analyze_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 1) # 무한 프로그래스 해제
            
        except Exception as e:
            self.on_analysis_error(str(e))
    
    def on_analysis_error(self, error_msg):
        """분석 오류"""
        self.result_text.setText(f"분석 실패: {error_msg}")
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 1) # 무한 프로그래스 해제
        QMessageBox.critical(self, "오류", f"분석 중 오류가 발생했습니다:\n{error_msg}")
    
    def display_results(self, result_json):
        """결과 표시"""
        try:
            result = json.loads(result_json)
            bboxs = result.get('bboxs', [])
            
            # 1. 원본 이미지에 박스 그리기
            self.draw_boxes_on_image(bboxs)
            
            # 2. 크롭 이미지들 표시
            self.display_crop_images(bboxs)
            
            # 3. 텍스트 결과 표시
            self.display_text_results(bboxs)
            
        except Exception as e:
            self.result_text.setText(f"결과 표시 실패: {str(e)}")
    
    def draw_boxes_on_image(self, bboxs):
        """원본 이미지에 박스 그리기"""
        if not self.current_image_path:
            return
            
        # 원본 이미지 로드
        pixmap = QPixmap(self.current_image_path)
        
        # QPainter로 박스 그리기
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(255, 0, 0), 3))  # 빨간색 3px
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        
        for i, box in enumerate(bboxs):
            x1, y1, x2, y2 = box['xyxy']
            class_name = box.get('class_name', 'Unknown')
            score = box.get('class_score', 0.0)
            
            # 박스 그리기
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            
            # 라벨 그리기
            label = f"{class_name} {score:.2f}"
            painter.fillRect(x1, y1-25, len(label)*8, 25, QColor(255, 0, 0))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawText(x1+2, y1-5, label)
            painter.setPen(QPen(QColor(255, 0, 0), 3))
        
        painter.end()
        
        # 전체 해상도 이미지를 저장해 두고, 화면 크기에 맞게 꽉 차게 표시
        self.full_pixmap = pixmap
        self.update_main_image()

    def update_main_image(self):
        """스크롤 영역 viewport에 맞춰 비율 유지 '맞춤' 표시(과도한 확대 방지)"""
        try:
            if self.full_pixmap.isNull():
                return
            if not hasattr(self, "scroll_area"):
                return
            vsize = self.scroll_area.viewport().size()
            if vsize.width() <= 0 or vsize.height() <= 0:
                return
            
            vsize -= QSize(0, 10)  # 테두리 여유 공간

            # 원본보다 크게는 키우지 않고, 영역 안에 '맞춤(Contain)'으로 표시
            ow, oh = self.full_pixmap.width(), self.full_pixmap.height()
            if ow <= vsize.width() and oh <= vsize.height():
                pix = self.full_pixmap  # 업스케일 금지
            else:
                pix = self.full_pixmap.scaled(vsize, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(pix)
            self.image_label.setFixedSize(pix.size())
        except Exception:
            fallback = self.full_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(fallback)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 창/레이아웃 크기 변화에 맞춰 이미지 갱신
        self.update_main_image()
    
    def display_crop_images(self, bboxs):
        """크롭된 알약 이미지들 표시"""
        if not bboxs:
            self.crop_label.setText("탐지된 알약이 없습니다.")
            return
        
        try:
            # 크롭 이미지들을 가로로 연결
            crop_images = []
            for box in bboxs:
                img_b64 = box['img']
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                crop_images.append(np.array(pil_img))
            
            if crop_images:
                # 높이를 맞춰서 연결
                max_height = max(img.shape[0] for img in crop_images)
                resized_images = []
                
                for img in crop_images:
                    if img.shape[0] != max_height:
                        pil_img = Image.fromarray(img)
                        ratio = max_height / img.shape[0]
                        new_width = int(img.shape[1] * ratio)
                        pil_img = pil_img.resize((new_width, max_height))
                        img = np.array(pil_img)
                    resized_images.append(img)
                
                # 이미지 연결
                combined_img = np.concatenate(resized_images, axis=1)
                
                # PIL로 변환 후 QPixmap으로 변환
                pil_combined = Image.fromarray(combined_img)
                
                # 임시 파일로 저장 후 QPixmap으로 로드
                temp_path = "temp_crop.png"
                pil_combined.save(temp_path)
                
                pixmap = QPixmap(temp_path)
                scaled_pixmap = pixmap.scaled(600, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.crop_label.setPixmap(scaled_pixmap)
                
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            self.crop_label.setText(f"크롭 이미지 표시 실패: {str(e)}")
    
    def display_text_results(self, bboxs):
        """텍스트 결과 표시"""
        if not bboxs:
            self.result_text.setText("분석 완료\n\n탐지된 알약이 없습니다.")
            return
        
        result_text = f"분석 완료 - {len(bboxs)}개의 알약이 탐지되었습니다.\n\n"
        
        for i, box in enumerate(bboxs, 1):
            class_name = box.get('class_name', 'Unknown')
            score = box.get('class_score', 0.0)
            
            result_text += f"알약 #{i}\n"
            result_text += f"  • 분류: {class_name}\n"
            result_text += f"  • 신뢰도: {score:.2f}\n"
            
            # 약물 정보
            drug_info = box.get('drug_info')
            if drug_info:
                result_text += f"  • 약물명: {drug_info.get('drug_N', 'N/A')}\n"
                result_text += f"  • 제품명: {drug_info.get('dl_name', 'N/A')}\n"
            
            # 병용금기 정보
            ddi = box.get('ddi')
            if ddi:
                result_text += f"  병용금기 약물이 발견되었습니다!\n"
            
            ddi_drug = box.get('ddi_drug')
            if ddi_drug:
                """
                    "ddi_drug": {
                        "성분명A": "atorvastatin calcium (as atorvastatin)",
                        "성분코드A": "111501ATB",
                        "제품코드A": 642100980,
                        "제품명A": "아토르바정10밀리그램(아토르바스타틴칼슘삼수화물)_(10.85mg/1정)",
                        "업체명A": "(주)유한양행",
                        "급여구분A": "급여",
                        "성분명B": "sodium fusidate",
                        "성분코드B": "229101ATB",
                        "제품코드B": 642703980,
                        "제품명B": "후시딘정(퓨시드산나트륨정)_(0.25g/1정)",
                        "업체명B": "동화약품(주)",
                        "급여구분B": "급여",
                        "공고번호": 20180088,
                        "공고일자": "2018-11-02",
                        "금기사유": "횡문근융해와 같은 중증의 근육이상 보고",
                        "category_id": 21324
                    }
                """                
                result_text += "\n" + "="*50 + "\n"
                result_text += f"  다른 약물과의 상호작용 주의가 필요합니다!\n"
                result_text += f"  • {ddi_drug.get('성분명A', 'N/A')} ({ddi_drug.get('제품명A', 'N/A')})\n"
                result_text += f"  • {ddi_drug.get('성분명B', 'N/A')} ({ddi_drug.get('제품명B', 'N/A')})\n"
                result_text += f"  • 금기사유: {ddi_drug.get('금기사유', 'N/A')}\n"
                result_text += f"  • 공고일자: {ddi_drug.get('공고일자', 'N/A')}\n"

            result_text += "\n" + "-"*50 + "\n\n"
        
        # 전체 주의사항
        has_ddi = any(box.get('ddi') or box.get('ddi_drug') for box in bboxs)
        if has_ddi:
            result_text += "중요 안내사항\n"
            result_text += "="*50 + "\n"
            result_text += "• 병용금기 약물이나 상호작용 가능성이 있는 약물이 탐지되었습니다.\n"
            result_text += "• 복용 전 반드시 의사나 약사와 상담하시기 바랍니다.\n"
            result_text += "• 동시 복용 시 부작용이나 효과 감소가 있을 수 있습니다.\n"
        
        self.result_text.setText(result_text)

def download_model_files(url, target):
    """url의 파일을 target 경로에 다운로드 (이미 있으면 건너뜀, 폴더 자동 생성)"""
    import urllib.request
    import os
    import sys
    import time

    # 폴더 생성
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)

    # 파일이 이미 존재하면 건너뜀
    if os.path.exists(target):
        print(f"파일이 이미 존재합니다: {os.path.basename(target)}")
        return True

    try:
        filename = os.path.basename(target)
        # print(f"다운로드 시작: {filename}")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                # 프로그레스 바 생성 (50자 길이)
                bar_length = 50
                filled_length = int(bar_length * percent // 100)
                bar = '=' * filled_length + '.' * (bar_length - filled_length)
                
                # 콘솔에 한 줄로 출력 (이전 줄 덮어쓰기)
                sys.stdout.write(f'\r{filename}: |{bar}| {percent:3.0f}% ({downloaded_mb:.1f}MB/{total_mb:.1f}MB)')
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, target, reporthook=progress_hook)
        print('')
        #print(f"\n다운로드 완료: {filename}")
        return True
        
    except Exception as e:
        print(f"\n다운로드 실패: {filename} - {e}")
        if os.path.exists(target):
            os.remove(target)
        return False

def extract_model_from_split_files():
    """분할된 tar 파일들을 합쳐서 best.pt 모델 파일을 생성"""
    try:
        import glob
        import tarfile
        py_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101")
        model_path = os.path.join(model_dir, "best.pt")
        
        # best.pt 파일이 이미 존재하면 종료
        if os.path.exists(model_path):
            print(f"모델 파일이 이미 존재합니다: {model_path}")
            return True
            
        # 분할 파일들 찾기
        split_files = glob.glob(os.path.join(model_dir, "best.tar.*"))
        if not split_files:
            print(f"분할 파일을 찾을 수 없습니다: {model_dir}/best.tar.*")
            return False
            
        # 파일명 정렬 (best.tar.001, best.tar.002, ...)
        split_files.sort()
        print(f"찾은 분할 파일들: {[os.path.basename(f) for f in split_files]}")
        
        # 분할된 파일들을 하나로 합치기
        merged_tar_path = os.path.join(model_dir, "best.tar")
        print(f"분할 파일들을 합치는 중...")
        
        with open(merged_tar_path, 'wb') as merged_file:
            for split_file in split_files:
                print(f"  - {os.path.basename(split_file)} 합치는 중...")
                with open(split_file, 'rb') as f:
                    merged_file.write(f.read())
        
        print(f"합친 tar 파일 생성 완료: {merged_tar_path}")
        
        # tar 파일 압축 해제
        print("tar 파일 압축 해제 중...")
        with tarfile.open(merged_tar_path, 'r') as tar:
            # tar 파일 내용 확인
            members = tar.getnames()
            print(f"tar 파일 내용: {members}")
            
            # best.pt 파일 찾기
            pt_file = None
            for member in members:
                if member.endswith('best.pt') or 'best.pt' in member:
                    pt_file = member
                    break
            
            if pt_file:
                # best.pt 파일 추출
                tar.extract(pt_file, model_dir)
                extracted_path = os.path.join(model_dir, pt_file)
                
                # 파일이 하위 폴더에 추출된 경우 상위로 이동
                if extracted_path != model_path:
                    os.rename(extracted_path, model_path)
                    # 빈 하위 폴더가 있다면 정리
                    try:
                        parent_dir = os.path.dirname(extracted_path)
                        if parent_dir != model_dir and os.path.exists(parent_dir):
                            os.rmdir(parent_dir)
                    except:
                        pass
                
                print(f"best.pt 파일 추출 완료: {model_path}")
            else:
                print("tar 파일에서 best.pt 파일을 찾을 수 없습니다.")
                return False
        
        # 임시 tar 파일 삭제
        if os.path.exists(merged_tar_path):
            os.remove(merged_tar_path)
            print("임시 tar 파일 삭제 완료")
            
        return os.path.exists(model_path)
        
    except Exception as e:
        print(f"모델 파일 추출 중 오류 발생: {str(e)}")
        return False

def main():
    # 모델 파일 자동 다운로드 및 준비
    print("="*60)
    print("알약 분석기 시작")
    print("="*60)
    
    py_dir = os.path.dirname(os.path.abspath(__file__))
    urls = [{'url' : "https://raw.githubusercontent.com/c0z0c/codeit_ai_health_eat_data/refs/heads/master/modeling/fasterrcnn_resnet101/best.tar.001",
              'target' : os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101", "best.tar.001")},
             {'url' : "https://raw.githubusercontent.com/c0z0c/codeit_ai_health_eat_data/refs/heads/master/modeling/fasterrcnn_resnet101/best.tar.002",
              'target' : os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101", "best.tar.002")},
             {'url' : "https://raw.githubusercontent.com/c0z0c/codeit_ai_health_eat_data/refs/heads/master/modeling/fasterrcnn_resnet101/best.tar.003",
              'target' : os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101", "best.tar.003")},
             {'url' : "https://raw.githubusercontent.com/c0z0c/codeit_ai_health_eat_data/refs/heads/master/modeling/yolo8m/best.pt",
              'target' : os.path.join(py_dir, "python_modules", "modeling", "yolo8m", "best.pt")},
        ]
    
    
    # 1. 모델 파일이 존재하는지 확인
    py_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(py_dir, "python_modules", "modeling", "fasterrcnn_resnet101", "best.pt")
    
    if not os.path.exists(model_path):
        print("모델 파일이 없습니다. 자동으로 다운로드를 시작합니다...")
        print("\n다운로드할 파일들:")
        for url in urls:
            print(f"- {os.path.basename(url['target'])}")
        print("\n이 파일들은 GitHub 파일 크기 제한으로 인해 분할되어 저장되었습니다.")
        
        # 2. 모델 파일들 다운로드
        for url in urls:
            for i in range(3):  # 최대 3회 재시도
                if download_model_files(url['url'], url['target']):
                    break
                print(f"재시도 {i+1}/3...")
    
    # 3. 분할된 tar 파일들을 합쳐서 best.pt 모델 파일 생성
    print("\n모델 파일 확인 및 추출 중...")
    if not extract_model_from_split_files():
        print("모델 파일 추출에 실패했습니다.")
        input("Enter 키를 눌러 종료하세요...")
        return
    
    print("모델 파일 준비 완료")
    print("="*60)
    
    # 4. GUI 애플리케이션 시작
    app = QApplication(sys.argv)
    window = PillAnalysisUI()
    window.resize(1920, 1080)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()