import sys

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QFont, QPainter, QLinearGradient, QColor

class ScrollingTextLabel(QLabel):
    """스크롤링 텍스트를 표시하는 커스텀 라벨"""
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.full_text = text
        self.current_text = ""
        self.char_index = 0
        
        # 타이핑 애니메이션 타이머
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.update_text)
        
    def start_animation(self):
        """타이핑 애니메이션 시작"""
        self.char_index = 0
        self.current_text = ""
        self.setText("")
        self.typing_timer.start(100)  # 100ms마다 한 글자씩
        
    def update_text(self):
        """텍스트를 한 글자씩 추가"""
        if self.char_index < len(self.full_text):
            self.current_text = self.full_text[:self.char_index + 1]
            self.setText(self.current_text)
            self.char_index += 1
        else:
            self.typing_timer.stop()

class AnimatedLabel(QLabel):
    """페이드 인 애니메이션이 있는 라벨"""
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._opacity = 0.0
        self.setStyleSheet("color: rgba(255, 255, 255, 0);")
        
    def get_opacity(self):
        """현재 투명도 값을 반환합니다.

        Returns:
            float: 현재 투명도 (0.0 ~ 1.0)
        """
        return self._opacity
    
    def set_opacity(self, opacity):
        """투명도 값을 설정합니다.

        Args:
            opacity (float): 설정할 투명도 (0.0 ~ 1.0)
        """
        self._opacity = opacity
        alpha = int(255 * opacity)
        self.setStyleSheet(f"color: rgba(52, 73, 94, {alpha});")
    
    opacity = pyqtProperty(float, get_opacity, set_opacity)

class TeamInfoDialog(QDialog):
    """팀 정보를 표시하는 모달리스 다이얼로그"""
    
    def __init__(self, parent=None, window_ref=None):
        super().__init__(parent)
        self.window_ref = window_ref  # 메인 창 참조 저장
        self.download_finished = False
        self.setWindowTitle("코드잇 AI 4기 4팀 - 헬스잇(Health Eat)")
        self.setFixedSize(800, 900)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # 화면 중앙에 배치
        self.center_on_screen()
        
        # UI 구성
        self.init_ui()
        
        # 애니메이션 시작
        self.start_animations()
        
        # 자동 닫기 타이머 (애니메이션 완료 후 시작하도록 변경)
        self.close_timer = QTimer()
        self.close_timer.timeout.connect(self.close)
        self.close_timer.setSingleShot(True)
        #self.close_timer.start(25000)  # 25초 후 자동 닫기
    
    def closeEvent(self, event):
        """다이얼로그가 닫힐 때 메인 창 표시"""
        if self.window_ref:
            self.window_ref.show()
        super().closeEvent(event)
            
    def center_on_screen(self):
        """다이얼로그를 화면 중앙에 배치"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def init_ui(self):
        """UI 구성"""
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 제목 (좁게)
        self.title_label = QLabel("🏥 경구약제 이미지 인식 AI 프로젝트")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("맑은 고딕", 14, QFont.Bold))
        self.title_label.setFixedHeight(60)  # 높이 제한으로 좁게
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                padding: 10px 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 30px;
                border: 3px solid #4c63d2;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.title_label)
        
        # 팀 정보 (넓게, 스크롤링 텍스트)
        team_text = """
⚡ 시스템을 준비하고 있습니다...
⏰ 준비 후 자동으로 시작됩니다.
        
🚀 코드잇 AI 엔지니어 4기 4팀

✨ 헬스케어 스타트업 "헬스잇(Health Eat)" ✨

👥 Team Members:
🎯 팀장/파이프라인: 이건희
📊 데이터분석/아키텍처 설계: 김명환  
🔬 모델실험설계/성능튜닝: 김민혁

🎨 Innovation Through Technology
💡 Health × AI × Future

"""

        self.team_info = ScrollingTextLabel(team_text)
        self.team_info.setAlignment(Qt.AlignCenter)
        self.team_info.setFont(QFont("맑은 고딕", 12))
        self.team_info.setMinimumHeight(400)  # 넓게
        self.team_info.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 30px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffecd2, stop:0.5 #fcb69f, stop:1 #ffecd2);
                border-radius: 20px;
                border: 4px solid #ff7675;
                line-height: 1.8;
                font-weight: 500;
            }
        """)
        layout.addWidget(self.team_info)
        
        # 하단 로딩 메시지
        self.loading_label = AnimatedLabel("🌟 Loading Amazing AI System... 🌟")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont("맑은 고딕", 10, QFont.Bold))
        self.loading_label.setFixedHeight(40)
        self.loading_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #a8edea, stop:1 #fed6e3);
                border-radius: 20px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.loading_label)
        
        self.setLayout(layout)
        
        # 다이얼로그 전체 스타일
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffffff, stop:0.5 #f8f9ff, stop:1 #ffffff);
                border-radius: 15px;
            }
        """)
    
    def start_animations(self):
        """애니메이션 시작"""
        
        # 제목 애니메이션 제거 (흔들림 방지)
        
        # 팀 정보 타이핑 애니메이션 (바로 시작)
        self.team_info.start_animation()
        
        # 로딩 라벨 페이드 인 (2초 후 시작)
        self.loading_animation = QPropertyAnimation(self.loading_label, b"opacity")
        self.loading_animation.setDuration(100)
        self.loading_animation.setStartValue(0.0)
        self.loading_animation.setEndValue(1.0)
        self.loading_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 로딩 애니메이션 완료 시 자동 닫기 타이머 시작
        #self.loading_animation.finished.connect(self.start_close_timer)
        self.loading_animation.stateChanged.connect(self.on_animation_state_changed)
        
        QTimer.singleShot(2000, self.loading_animation.start)

    def on_animation_state_changed(self, new_state, old_state):
        """애니메이션 상태 변경 시 호출"""
        if new_state == QPropertyAnimation.Stopped:
            self.start_close_timer()
            
    def start_close_timer(self):
        """모든 애니메이션 완료 후 자동 닫기 타이머 시작"""
        if self.download_finished is False:
            again_timer = QTimer()
            again_timer.timeout.connect(self.start_close_timer)
            again_timer.setSingleShot(True)
            again_timer.start(1000)  # 1초 후 다시 확인
            return  # 다운로드가 완료되지 않았으면 종료 타이머 시작 안 함
        
    def done_download(self):
        """다운로드 완료 시 호출 - 애니메이션 중단하고 모든 텍스트 표시"""
        self.download_finished = True
        
        # 타이핑 애니메이션 중단하고 전체 텍스트 표시
        if hasattr(self, 'team_info') and self.team_info.typing_timer.isActive():
            self.team_info.typing_timer.stop()
            self.team_info.setText(self.team_info.full_text)
        
        # 로딩 애니메이션 중단하고 완전히 표시
        if hasattr(self, 'loading_animation') and self.loading_animation.state() == QPropertyAnimation.Running:
            self.loading_animation.stop()
            self.loading_label.set_opacity(1.0)
        
        # 로딩 메시지 변경
        if hasattr(self, 'loading_label'):
            self.loading_label.setText("🎉 시스템 준비 완료! 곧 시작됩니다... 🎉")
        
        # 5초 후 자동 닫기 시작
        if hasattr(self, 'close_timer'):
            self.close_timer.stop()  # 기존 타이머 중단
            self.close_timer.start(10000)  # 5초 후 닫기        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    team_dialog = TeamInfoDialog()
    team_dialog.show()
    sys.exit(app.exec_())