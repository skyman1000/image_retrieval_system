# main_app.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.main_window import MainWindow
from core.config import INDEX_PATH, MAPPING_PATH, DATA_DIR

def check_prerequisites():
    """ 检查运行应用所需的文件和目录 """
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

if __name__ == '__main__':
    print("应用程序启动...")
    # 检查先决条件
    if not check_prerequisites():
        print("启动检查失败，退出程序。")
        sys.exit(1)

    app = QApplication(sys.argv)
    try:
        print("正在创建主窗口...")
        mainWin = MainWindow()
        print("正在显示主窗口...")
        mainWin.show()
        print("进入 Qt 事件循环...")
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "严重错误", f"应用程序遇到严重错误并即将退出：\n{e}")
        print(f"在应用程序执行期间捕获到未处理的异常: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈
        sys.exit(1)