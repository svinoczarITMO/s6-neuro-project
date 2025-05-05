import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Emotion Analyzer")
        self.setGeometry(100, 100, 800, 600)
        
        # Кнопки и элементы
        self.btn_upload = QPushButton("Upload Video", self)
        self.btn_upload.move(20, 20)
        self.btn_upload.clicked.connect(self.upload_video)
        
        self.result_label = QLabel(self)
        self.result_label.move(20, 100)
        self.result_label.resize(760, 400)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4)")
        if file_path:
            # Обработка видео
            text_emotion = analyze_text_emotion("user_text.txt")
            audio_emotion = analyze_audio_emotion("extracted_audio.wav")
            video_emotion = analyze_video_emotion(file_path)
            
            # Вывод результатов
            result = f"""
            Text Emotion: {text_emotion}
            Audio Emotion: {audio_emotion}
            Video Emotion: {video_emotion}
            """
            self.result_label.setText(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())