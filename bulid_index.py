# build_index.py
import os
import time # 导入 time 模块
from core.config import (DATA_DIR, INDEX_PATH, MAPPING_PATH, VIT_MODEL_NAME,
                         FEATURE_DIM, FAISS_INDEX_TYPE_CPU, INDEX_DIR)
from core.feature_extractor import ViTFeatureExtractor
from core.indexer import FaissIndexer

if __name__ == "__main__":
    print("-" * 60)
    print("--- 开始图像索引构建过程 ---")
    print("-" * 60)

    overall_start_time = time.time()

    print(f"[信息] 确保索引目录存在: {INDEX_DIR}")
    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"[信息] 检查数据目录: {DATA_DIR}")
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"[错误] 数据目录 '{DATA_DIR}' 为空或不存在。")
        print("        请先将图片添加到 'data' 文件夹再运行此脚本。")
        exit(1)
    print("[信息] 数据目录检查通过。")

    # --- 步骤 1: 初始化 ViT 特征提取器 ---
    print(f"\n[步骤 1/4] 初始化 ViT 特征提取器 ({VIT_MODEL_NAME})...")
    step1_start_time = time.time()
    try:
        feature_extractor = ViTFeatureExtractor(model_name=VIT_MODEL_NAME)
        print("[成功] 特征提取器初始化完成。")
    except Exception as e:
        print(f"[错误] 初始化特征提取器失败: {e}")
        exit(1)
    step1_end_time = time.time()
    print(f"  [计时] 步骤 1 耗时: {step1_end_time - step1_start_time:.4f} 秒")


    # --- 步骤 2: 初始化 Faiss 索引器 ---
    print(f"\n[步骤 2/4] 初始化 Faiss 索引器 (CPU类型: {FAISS_INDEX_TYPE_CPU}, 维度: {FEATURE_DIM})...")
    step2_start_time = time.time()
    try:
        indexer = FaissIndexer(feature_dim=FEATURE_DIM)
        print("[成功] Faiss 索引器初始化完成。")
    except Exception as e:
        print(f"[错误] 初始化 Faiss 索引器失败: {e}")
        exit(1)
    step2_end_time = time.time()
    print(f"  [计时] 步骤 2 耗时: {step2_end_time - step2_start_time:.4f} 秒")


    # --- 步骤 3: 从图像构建索引 ---
    print("\n[步骤 3/4] 从图像构建索引 (可能需要较长时间)...")
    step3_start_time = time.time()
    try:
        indexer.build_index(DATA_DIR, feature_extractor)
        if indexer.index_cpu is None or indexer.index_cpu.ntotal == 0:
             print("[错误] 索引构建失败或结果为空索引。")
             exit(1)
        print(f"[成功] 索引构建完成，包含 {indexer.index_cpu.ntotal} 个向量。")
    except Exception as e:
        print(f"[错误] 在索引构建过程中发生错误: {e}")
        exit(1)
    step3_end_time = time.time()
    print(f"  [计时] 步骤 3 (特征提取+索引构建) 耗时: {step3_end_time - step3_start_time:.4f} 秒")


    # --- 步骤 4: 保存 CPU 索引和映射文件 ---
    print("\n[步骤 4/4] 保存 CPU 索引和映射文件...")
    step4_start_time = time.time()
    try:
        indexer.save_index(INDEX_PATH, MAPPING_PATH)
        print(f"[成功] CPU 索引保存至: {INDEX_PATH}")
        print(f"          映射文件保存至: {MAPPING_PATH}")
    except Exception as e:
        print(f"[错误] 保存索引或映射失败: {e}")
        exit(1)
    step4_end_time = time.time()
    print(f"  [计时] 步骤 4 耗时: {step4_end_time - step4_start_time:.4f} 秒")


    overall_end_time = time.time()
    print("-" * 60)
    print(f"--- 索引构建过程成功结束 ---")
    print(f"总耗时: {overall_end_time - overall_start_time:.2f} 秒。")
    print("-" * 60)