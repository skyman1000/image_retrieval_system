# gui/main_window.py
import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QScrollArea,
                             QGridLayout, QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer # QThread 仍然需要用于 SearchWorker

from core.config import (K_RESULTS, QUERY_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE,
                         GRID_COLS) # FAISS_INDEX_TYPE_CPU 不再需要导入这里

# SearchWorker 现在需要从 main_app.py 导入 (或者定义在 main_window.py 如果更集中)
# 为了保持 main_window.py 的纯UI和主逻辑，我们假设 SearchWorker 仍在 workers.py 或 main_app.py

# --- 为了演示，我们先将 SearchWorker 移到这里，如果它只被 MainWindow 使用 ---
# --- 实际大型项目中，建议保持在 workers.py ---
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
        self.total_time = 0.0

    def run(self):
        try:
            total_start_time = time.time()

            self.progress_signal.emit("正在提取查询图像特征...")
            extract_start_time = time.time()
            self.query_feature = self.feature_extractor.extract_features(self.image_path)
            extract_end_time = time.time()
            if self.query_feature is None:
                raise ValueError("无法提取查询图像特征。")
            # print(f"特征提取耗时: {extract_end_time - extract_start_time:.2f}秒")

            self.progress_signal.emit("正在 Faiss 索引中搜索...")
            search_start_time = time.time()
            if self.searcher.get_active_index() is None or self.searcher.get_active_index().ntotal == 0:
                raise ValueError("Faiss 索引未加载或为空。")
            search_results = self.searcher.search(self.query_feature, k=self.k)
            search_end_time = time.time()

            total_end_time = time.time()
            self.total_time = total_end_time - total_start_time
            # print(f"总处理时间 (特征提取+搜索): {self.total_time:.2f}秒")

            self.results_signal.emit(self.query_feature, search_results, self.total_time)

        except Exception as e:
            print(f"搜索线程错误: {e}")
            self.error_signal.emit(str(e))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        app_init_start_time = time.time()
        print(f"[计时] MainWindow __init__ 开始...")

        self.feature_extractor = None
        self.searcher = None
        self.search_worker = None
        self.query_file_path = None
        self.backend_ready = False

        ui_init_start_time = time.time()
        print(f"  [计时] _init_ui 调用开始...")
        self._init_ui()
        ui_init_end_time = time.time()
        print(f"  [计时] _init_ui 调用完成, 耗时: {ui_init_end_time - ui_init_start_time:.4f} 秒")

        app_init_end_time = time.time()
        print(f"[计时] MainWindow __init__ (仅UI框架) 总耗时: {app_init_end_time - app_init_start_time:.4f} 秒")

    def finish_initialization(self, feature_extractor, searcher):
        print("[主窗口] 接收到后端初始化完成信号。")
        self.feature_extractor = feature_extractor
        self.searcher = searcher
        self.backend_ready = True
        self.upload_button.setEnabled(True)
        self.upload_button.setToolTip("选择一张本地图像进行相似性检索")
        self.status_label.setText("状态：系统准备就绪，请上传查询图像。")
        print(f"[主窗口] 后端组件已设置: ViT {'已加载' if self.feature_extractor else '未加载'}, Faiss {'已加载' if self.searcher and self.searcher.get_active_index() else '未加载'}")
        if self.searcher:
            print(f"  {self.searcher.get_index_status()}")

    def _init_ui(self):
        # ... (样式表和布局与之前相同) ...
        self.setWindowTitle("图像检索系统 (ViT + Faiss)")
        self.setGeometry(100, 100, 1000, 750)
        self.setStyleSheet("""
            QWidget {
                font-family: '微软雅黑', 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #3498db; /* 蓝色 */
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #2980b9; /* 深蓝色 */
            }
            QPushButton:disabled {
                background-color: #bdc3c7; /* 灰色 */
                color: #7f8c8d;
            }
            QLabel#QueryImageLabel, QLabel.ResultImageLabel {
                border: 1px solid #cccccc;
                background-color: #f8f8f8;
                padding: 2px;
            }
            QScrollArea {
                border: 1px solid #dddddd;
                background-color: white;
            }
            QFrame[frameShape="4"], QFrame[frameShape="5"] {
                 color: #cccccc;
            }
            QLabel#StatusLabel {
                padding: 5px;
                background-color: #ecf0f1;
                border-top: 1px solid #cccccc;
            }
            QToolTip {
                background-color: black;
                color: white;
                border: 1px solid white;
            }
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)

        query_frame = QFrame()
        query_layout = QHBoxLayout(query_frame)
        query_layout.setContentsMargins(0,0,0,0)

        self.upload_button = QPushButton("选择查询图像")
        self.upload_button.setMinimumHeight(40)
        self.upload_button.clicked.connect(self._upload_image)
        self.upload_button.setEnabled(False)
        self.upload_button.setToolTip("系统初始化中，请稍候...")

        self.query_image_label = QLabel("此处显示查询图像")
        self.query_image_label.setObjectName("QueryImageLabel")
        self.query_image_label.setFixedSize(QUERY_IMG_DISPLAY_SIZE, QUERY_IMG_DISPLAY_SIZE)
        self.query_image_label.setAlignment(Qt.AlignCenter)

        query_layout.addWidget(self.upload_button)
        query_layout.addSpacing(20)
        query_layout.addWidget(self.query_image_label)
        query_layout.addStretch()
        self.main_layout.addWidget(query_frame)

        line1 = QFrame(); line1.setFrameShape(QFrame.HLine); line1.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line1)

        results_title = QLabel("检索结果:"); results_title.setFont(QFont("微软雅黑", 14, QFont.Bold)); results_title.setStyleSheet("color: #2c3e50;")
        self.main_layout.addWidget(results_title)

        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget); self.results_layout.setSpacing(10)
        self.scroll_area.setWidget(self.results_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.status_label = QLabel("状态：正在初始化系统，请稍候..."); self.status_label.setObjectName("StatusLabel"); self.status_label.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(self.status_label)

    def _upload_image(self):
        # ... (与之前相同) ...
        if not self.backend_ready:
            self._show_error_message("错误：后端组件尚未准备就绪，请稍候。")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择查询图像", "",
                                                   "图像文件 (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.query_file_path = file_path
            print(f"用户选择了图像: {file_path}")

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

            if self.feature_extractor is None or self.searcher is None:
                 self._show_error_message("严重错误：特征提取器或搜索器未正确初始化！")
                 return

            self.search_worker = SearchWorker(self.feature_extractor, self.searcher, file_path, K_RESULTS) # SearchWorker 从本文件定义
            self.search_worker.results_signal.connect(self._display_results)
            self.search_worker.error_signal.connect(self._handle_search_error)
            self.search_worker.progress_signal.connect(self._update_status_from_worker)
            self.search_worker.finished.connect(self._search_finished)
            self.search_worker.start()

            self.upload_button.setEnabled(False)
            self.upload_button.setText("正在搜索中...")


    def _update_status_from_worker(self, message):
        self.status_label.setText(f"状态：{message}")
        QApplication.processEvents()

    def _clear_results(self):
         while self.results_layout.count():
             item = self.results_layout.takeAt(0); widget = item.widget()
             if widget is not None: widget.deleteLater()

    def _display_results(self, query_feature, results: list[tuple[str, float]], duration: float):
        print(f"收到 {len(results)} 个搜索结果。")
        self._clear_results()
        if not results:
            no_results_label = QLabel("未找到相似图像。"); no_results_label.setAlignment(Qt.AlignCenter); no_results_label.setFont(QFont("微软雅黑", 12))
            self.results_layout.addWidget(no_results_label, 0, 0, 1, GRID_COLS); return

        row, col = 0, 0
        for img_path, score in results: # ★ 不再使用 score 来显示 ★
            img_label = QLabel()
            img_label.setObjectName("ResultImageLabel")
            if not os.path.exists(img_path):
                print(f"警告：结果图像路径无效: {img_path}")
                img_label.setText(f"图像丢失:\n{os.path.basename(img_path)}"); img_label.setWordWrap(True)
            else:
                pixmap = QPixmap(img_path)
                if pixmap.isNull():
                    print(f"警告：无法加载结果图像: {img_path}")
                    img_label.setText(f"加载失败:\n{os.path.basename(img_path)}"); img_label.setWordWrap(True)
                else:
                    img_label.setPixmap(pixmap.scaled(RESULT_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            img_label.setFixedSize(RESULT_IMG_DISPLAY_SIZE, RESULT_IMG_DISPLAY_SIZE)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setToolTip(f"路径: {os.path.basename(img_path)}") # ★ Tooltip 中也不再显示得分 ★
            # img_label.setStyleSheet("border: 1px solid #e0e0e0; border-radius: 3px;") # 可以通过setObjectName设置
            self.results_layout.addWidget(img_label, row, col)

            col += 1
            if col >= GRID_COLS: col = 0; row += 1
        self.status_label.setText(f"状态：检索完成！显示 {len(results)} 个结果。总耗时: {duration:.2f} 秒。")

    def _handle_search_error(self, error_message):
        self._show_error_message(f"检索过程中发生错误: {error_message}")
        self._search_finished()

    def _search_finished(self):
        print("搜索线程结束。")
        if self.backend_ready: self.upload_button.setEnabled(True); self.upload_button.setText("选择查询图像")
        self.search_worker = None

    def _show_error_message(self, message):
        print(f"错误弹窗: {message}")
        QMessageBox.critical(self, "应用程序错误", message)
        self.status_label.setText(f"状态：错误！{message.splitlines()[0]}")

    def closeEvent(self, event):
        if self.search_worker and self.search_worker.isRunning():
            print("关闭窗口前停止后台线程...")
            self.search_worker.terminate(); self.search_worker.wait()
        event.accept()