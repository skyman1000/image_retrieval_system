# core/config.py
import os

# --- 基本路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "image_features.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "image_paths.pkl")

# --- 模型配置 ---
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
FEATURE_DIM = 768

# --- Faiss 配置 ---
# 构建和保存时使用CPU索引 (IndexFlatIP 用于余弦相似度)
FAISS_INDEX_TYPE_CPU = "IndexFlatIP"

# --- 搜索配置 ---
K_RESULTS = 5 # 返回结果数量

# --- GUI 配置 ---
QUERY_IMG_DISPLAY_SIZE = 224
RESULT_IMG_DISPLAY_SIZE = 150
GRID_COLS = 3