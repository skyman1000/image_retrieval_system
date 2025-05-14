# main_app.py
import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget, QVBoxLayout, QLabel, QDesktopWidget 
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from gui.main_window import MainWindow
from core.config import INDEX_PATH, MAPPING_PATH, DATA_DIR, VIT_MODEL_NAME
from core.feature_extractor import ViTFeatureExtractor
from core.searcher import FaissSearcher

# --- 后台初始化工作线程 ---
class BackendInitializerWorker(QThread):

    initialization_finished = pyqtSignal(object, object)
    initialization_error = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.searcher = None

    def run(self):
        try:
            self.progress_updated.emit("正在初始化 ViT 特征提取器...")
            print("[后台初始化] 初始化 ViT 特征提取器...")
            vit_init_start_time = time.time()
            self.feature_extractor = ViTFeatureExtractor(model_name=VIT_MODEL_NAME)
            vit_init_end_time = time.time()
            print(f"  [后台计时] ViTFeatureExtractor 实例化耗时: {vit_init_end_time - vit_init_start_time:.4f} 秒")
            if self.feature_extractor is None:
                raise RuntimeError("ViT 特征提取器初始化失败 (返回 None)")
            print("[后台初始化] ViT 初始化成功。")

            self.progress_updated.emit("正在初始化 Faiss 搜索器 (加载索引)...")
            print("[后台初始化] 初始化 Faiss 搜索器...")
            searcher_init_start_time = time.time()
            self.searcher = FaissSearcher(index_path=INDEX_PATH, mapping_path=MAPPING_PATH)
            searcher_init_end_time = time.time()
            print(f"  [后台计时] FaissSearcher 实例化 (含加载) 耗时: {searcher_init_end_time - searcher_init_start_time:.4f} 秒")
            if self.searcher.get_active_index() is None:
                raise RuntimeError(f"加载 Faiss 索引失败。请检查 '{INDEX_PATH}' 和 '{MAPPING_PATH}'。")
            print(f"[后台初始化] Faiss 初始化成功。{self.searcher.get_index_status()}")

            self.progress_updated.emit("后端组件初始化完成！")
            self.initialization_finished.emit(self.feature_extractor, self.searcher)
        except Exception as e:
            error_msg = f"后端初始化过程中发生错误: {e}"
            print(f"[后台错误] {error_msg}")
            import traceback
            traceback.print_exc()
            self.initialization_error.emit(error_msg)


# --- 启动画面类 ---
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.background_color = QColor("#FFFFFF") # 白色背景
        self.title_text_color = QColor("#2c3e50")   # 深蓝灰色标题
        self.message_text_color = QColor("#7f8c8d") # 中灰色消息

        palette = self.palette()
        palette.setColor(QPalette.Window, self.background_color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40) # 增加边距
        main_layout.setSpacing(18) # 调整间距
        main_layout.setAlignment(Qt.AlignCenter)

        # 应用标题
        self.title_label = QLabel("图像检索系统", self)
        self.title_label.setFont(QFont("Segoe UI Semibold", 24)) # 使用更现代的字体，稍大
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"color: {self.title_text_color.name()};")
        main_layout.addWidget(self.title_label)

        # 状态信息标签
        self.message_label = QLabel("正在启动，请稍候...", self)
        self.message_label.setFont(QFont("Segoe UI", 11)) # 使用更现代的字体
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet(f"color: {self.message_text_color.name()};")
        self.message_label.setWordWrap(True) 
        self.message_label.setMinimumHeight(40) 
        main_layout.addWidget(self.message_label)

        # 根据内容自动调整窗口大小，或者设置一个更合适的固定大小
        self.setFixedSize(420, 200) # 稍微调整大小

        # --- 窗口居中显示 ---
        self.center_on_screen()

    def center_on_screen(self):
        """将窗口居中显示在屏幕上"""
        screen_geometry = QDesktopWidget().screenGeometry() # 获取主屏幕几何信息
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def update_progress_text(self, text: str):
        self.message_label.setText(text)
        QApplication.processEvents() # 确保消息立即显示

    def close_splash(self):
        self.close()
        self.deleteLater()

def check_prerequisites():
    errors = []
    if not os.path.exists(DATA_DIR): errors.append(f"数据目录 '{DATA_DIR}' 不存在。")
    elif not any(fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) for fname in os.listdir(DATA_DIR)):
        errors.append(f"数据目录 '{DATA_DIR}' 为空或不包含图像文件，请放入图片。")
    if not os.path.exists(INDEX_PATH): errors.append(f"Faiss 索引文件 '{INDEX_PATH}' 未找到。")
    if not os.path.exists(MAPPING_PATH): errors.append(f"图像路径映射文件 '{MAPPING_PATH}' 未找到。")

    if errors:
        error_message = "应用程序无法启动，缺少必要文件或目录：\n\n"
        error_message += "\n".join(errors)
        error_message += "\n\n请确保：\n1. 'data' 文件夹存在且包含图像文件。\n2. 已成功运行 'python build_index.py' 来生成索引文件。"
        msg_box = QMessageBox(); msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("启动错误"); msg_box.setText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok); msg_box.exec_()
        return False
    return True

class ApplicationController:
    def __init__(self, app):
        self.app = app
        self.splash = None
        self.main_window = None
        self.backend_initializer = None

    def start(self):
        if not check_prerequisites():
            print("启动检查失败，退出程序。")
            return

        self.splash = SplashScreen()
        self.splash.show()
        self.app.processEvents()

        self.backend_initializer = BackendInitializerWorker()
        self.backend_initializer.initialization_finished.connect(self.on_backend_ready)
        self.backend_initializer.initialization_error.connect(self.on_backend_error)
        self.backend_initializer.progress_updated.connect(self.splash.update_progress_text)
        self.backend_initializer.start()

    def on_backend_ready(self, feature_extractor, searcher):
        print("[主流程] 后端初始化成功完成。")
        self.splash.update_progress_text("加载完成，正在启动主界面...")
        self.app.processEvents()

        self.main_window = MainWindow()
        self.main_window.finish_initialization(feature_extractor, searcher)
        QTimer.singleShot(500, self._show_main_window)

    def _show_main_window(self):
        if self.main_window: self.main_window.show()
        if self.splash: self.splash.close_splash(); self.splash = None
        print("[主流程] 主窗口已显示。")

    def on_backend_error(self, error_message):
        print(f"[主流程] 后端初始化失败: {error_message}")
        if self.splash: self.splash.close_splash(); self.splash = None
        msg_box = QMessageBox(); msg_box.setIcon(QMessageBox.Critical); msg_box.setWindowTitle("后端初始化错误")
        msg_box.setText(f"应用程序后端初始化失败：\n\n{error_message}\n\n请检查配置和依赖。"); msg_box.exec_()
        self.app.quit()


if __name__ == '__main__':
    app_start_time = time.time()
    print("应用程序启动...")
    app = QApplication(sys.argv) 
    controller = ApplicationController(app)
    controller.start()
    exit_code = app.exec_()
    app_end_time = time.time()
    print(f"应用程序退出，代码: {exit_code}。总运行时间: {app_end_time - app_start_time:.2f} 秒。")
    sys.exit(exit_code)