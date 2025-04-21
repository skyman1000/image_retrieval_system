# gui/main_window.py
import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QScrollArea,
                             QGridLayout, QFrame, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from core.config import (INDEX_PATH, MAPPING_PATH, VIT_MODEL_NAME, K_RESULTS,
                         QUERY_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE, GRID_COLS,
                         FAISS_INDEX_TYPE_CPU) # 导入 CPU 类型用于显示得分
from core.feature_extractor import ViTFeatureExtractor
from core.searcher import FaissSearcher

# --- 后台特征提取与搜索线程 ---
class SearchWorker(QThread):
    results_signal = pyqtSignal(object, list, float)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, feature_extractor, searcher, image_path, k):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.searcher = searcher
        self.image_path = image_path
        self.k = k
        self.query_feature = None
        self.total_time = 0.0 # 记录总时间

    def run(self):
        try:
            total_start_time = time.time()

            self.progress_signal.emit("正在提取查询图像特征...")
            extract_start_time = time.time()
            self.query_feature = self.feature_extractor.extract_features(self.image_path)
            extract_end_time = time.time()
            if self.query_feature is None:
                raise ValueError("无法提取查询图像特征。")
            print(f"特征提取耗时: {extract_end_time - extract_start_time:.2f}秒")

            self.progress_signal.emit("正在 Faiss 索引中搜索...")
            search_start_time = time.time()
            if self.searcher.get_active_index() is None or self.searcher.get_active_index().ntotal == 0:
                raise ValueError("Faiss 索引未加载或为空。")
            search_results = self.searcher.search(self.query_feature, k=self.k)
            search_end_time = time.time()
            print(f"Faiss 搜索耗时: {search_end_time - search_start_time:.4f}秒 (使用 {'GPU' if self.searcher.is_gpu_enabled else 'CPU'})")

            total_end_time = time.time()
            self.total_time = total_end_time - total_start_time # 记录总时间
            print(f"总处理时间: {self.total_time:.2f}秒")

            self.results_signal.emit(self.query_feature, search_results, self.total_time)

        except Exception as e:
            print(f"后台线程错误: {e}")
            self.error_signal.emit(str(e))

# --- 主窗口类 ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.searcher = None
        self.search_worker = None
        self.query_file_path = None

        self._init_backend()
        self._init_ui()

    def _init_backend(self):
        print("[后端初始化] 初始化 ViT 特征提取器...")
        try:
            self.feature_extractor = ViTFeatureExtractor(model_name=VIT_MODEL_NAME)
            print("[后端初始化] ViT 初始化成功。")
        except Exception as e:
            self._show_error_message(f"初始化 ViT 模型失败: {e}\n请检查网络连接或模型名称。")

        print("[后端初始化] 初始化 Faiss 搜索器...")
        try:
            self.searcher = FaissSearcher(index_path=INDEX_PATH, mapping_path=MAPPING_PATH)
            # 检查加载和 GPU 初始化是否成功
            if self.searcher.get_active_index() is None:
                 self._show_error_message(f"加载 Faiss 索引失败。\n请确保先运行 'python build_index.py' 脚本，\n并且 'index' 文件夹中有索引文件。")
            else:
                print(f"[后端初始化] Faiss 初始化成功。{self.searcher.get_index_status()}")
        except Exception as e:
            self._show_error_message(f"初始化 Faiss 搜索器失败: {e}")

    def _init_ui(self):
        self.setWindowTitle("基于 ViT 和 Faiss (GPU 加速) 的图像检索系统")
        self.setGeometry(100, 100, 1000, 750)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)

        query_frame = QFrame()
        query_frame.setFrameShape(QFrame.StyledPanel)
        query_layout = QHBoxLayout(query_frame)

        self.upload_button = QPushButton("选择查询图像")
        self.upload_button.setFont(QFont("微软雅黑", 12))
        self.upload_button.setMinimumHeight(40)
        self.upload_button.clicked.connect(self._upload_image)
        if self.feature_extractor is None or self.searcher is None or self.searcher.get_active_index() is None:
             self.upload_button.setEnabled(False)
             self.upload_button.setToolTip("后端或索引未准备好，无法上传")

        self.query_image_label = QLabel("此处显示查询图像")
        self.query_image_label.setFixedSize(QUERY_IMG_DISPLAY_SIZE, QUERY_IMG_DISPLAY_SIZE)
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setFrameShape(QFrame.Box)
        self.query_image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f8f8f8;")

        query_layout.addWidget(self.upload_button)
        query_layout.addSpacing(20)
        query_layout.addWidget(self.query_image_label)
        query_layout.addStretch()
        self.main_layout.addWidget(query_frame)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine); line1.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line1)

        results_title = QLabel("检索结果:")
        results_title.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.main_layout.addWidget(results_title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #ffffff;")
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)
        self.scroll_area.setWidget(self.results_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.status_label = QLabel("状态：请上传一张图像以开始检索")
        self.status_label.setFont(QFont("微软雅黑", 10))
        self.status_label.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(self.status_label)

    def _upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择查询图像", "",
                                                   "图像文件 (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.query_file_path = file_path
            print(f"用户选择了图像: {file_path}")

            if self.feature_extractor is None: self._show_error_message("错误：特征提取器未初始化！"); return
            if self.searcher is None or self.searcher.get_active_index() is None or self.searcher.get_active_index().ntotal == 0:
                self._show_error_message("错误：Faiss 索引未加载或为空！请先运行 build_index.py。"); return

            if self.search_worker and self.search_worker.isRunning():
                print("正在终止上一个搜索线程...")
                self.search_worker.terminate(); self.search_worker.wait()
                print("上一个线程已终止。")

            pixmap = QPixmap(file_path)
            if pixmap.isNull(): self._show_error_message(f"无法加载查询图像: {file_path}"); return
            self.query_image_label.setPixmap(pixmap.scaled(QUERY_IMG_DISPLAY_SIZE, QUERY_IMG_DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            self.status_label.setText(f"状态：正在处理查询图像 {os.path.basename(file_path)}...")
            self._clear_results()
            QApplication.processEvents()

            self.search_worker = SearchWorker(self.feature_extractor, self.searcher, file_path, K_RESULTS)
            self.search_worker.results_signal.connect(self._display_results)
            self.search_worker.error_signal.connect(self._handle_search_error)
            self.search_worker.progress_signal.connect(self._update_status)
            self.search_worker.finished.connect(self._search_finished)
            self.search_worker.start()

            self.upload_button.setEnabled(False)
            self.upload_button.setText("正在搜索中...")

    def _update_status(self, message):
        self.status_label.setText(f"状态：{message}")
        QApplication.processEvents()

    def _clear_results(self):
         while self.results_layout.count():
             item = self.results_layout.takeAt(0)
             widget = item.widget()
             if widget is not None:
                 widget.deleteLater()

    def _display_results(self, query_feature, results: list[tuple[str, float]], duration: float):
        print(f"收到 {len(results)} 个搜索结果。")
        self._clear_results()

        if not results:
            no_results_label = QLabel("未找到相似图像。")
            no_results_label.setAlignment(Qt.AlignCenter)
            no_results_label.setFont(QFont("微软雅黑", 12))
            self.results_layout.addWidget(no_results_label, 0, 0, 1, GRID_COLS)
            return

        row, col = 0, 0
        for img_path, score in results:
            if not os.path.exists(img_path):
                print(f"警告：结果图像路径无效: {img_path}")
                placeholder = QLabel(f"图像丢失:\n{os.path.basename(img_path)}")
                placeholder.setFixedSize(RESULT_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE); placeholder.setFrameShape(QFrame.Box); placeholder.setAlignment(Qt.AlignCenter); placeholder.setWordWrap(True)
                self.results_layout.addWidget(placeholder, row, col)
            else:
                pixmap = QPixmap(img_path)
                if pixmap.isNull():
                    print(f"警告：无法加载结果图像: {img_path}")
                    placeholder = QLabel(f"加载失败:\n{os.path.basename(img_path)}")
                    placeholder.setFixedSize(RESULT_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE); placeholder.setFrameShape(QFrame.Box); placeholder.setAlignment(Qt.AlignCenter); placeholder.setWordWrap(True)
                    self.results_layout.addWidget(placeholder, row, col)
                else:
                    img_label = QLabel()
                    img_label.setPixmap(pixmap.scaled(RESULT_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    # 根据索引类型显示得分
                    if FAISS_INDEX_TYPE_CPU == 'IndexFlatIP': # 余弦相似度
                        similarity_percent = score * 100
                        tooltip_text = f"路径: {os.path.basename(img_path)}\n相似度: {similarity_percent:.2f}%"
                    else: # L2 距离
                         tooltip_text = f"路径: {os.path.basename(img_path)}\n距离: {score:.4f}"
                    img_label.setToolTip(tooltip_text)
                    img_label.setFrameShape(QFrame.Box)
                    img_label.setFixedSize(RESULT_IMG_DISPLAY_SIZE + 4, RESULT_IMG_DISPLAY_SIZE + 4)
                    img_label.setAlignment(Qt.AlignCenter)
                    self.results_layout.addWidget(img_label, row, col)

            col += 1
            if col >= GRID_COLS: col = 0; row += 1

        # 更新最终状态
        self.status_label.setText(f"状态：检索完成！显示 {len(results)} 个结果。总耗时: {duration:.2f} 秒。")


    def _handle_search_error(self, error_message):
        self._show_error_message(f"检索过程中发生错误: {error_message}")
        self._search_finished() # 确保恢复按钮

    def _search_finished(self):
        print("搜索线程结束。")
        self.upload_button.setEnabled(True)
        self.upload_button.setText("选择查询图像")
        # 状态栏的最终文本由 _display_results 或 _handle_search_error 设置
        self.search_worker = None

    def _show_error_message(self, message):
        print(f"错误弹窗: {message}")
        QMessageBox.critical(self, "应用程序错误", message)
        self.status_label.setText(f"状态：错误！{message.splitlines()[0]}")

    def closeEvent(self, event):
        if self.search_worker and self.search_worker.isRunning():
            print("关闭窗口前停止后台线程...")
            self.search_worker.terminate()
            self.search_worker.wait()
        event.accept()