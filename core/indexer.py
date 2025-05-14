# core/indexer.py
import faiss
import numpy as np
import os
import pickle
from tqdm import tqdm
from .feature_extractor import ViTFeatureExtractor
from .config import FAISS_INDEX_TYPE_CPU
import time # 导入 time 模块

class FaissIndexer:
    def __init__(self, feature_dim: int):
        init_start_time = time.time() # 开始计时
        self.feature_dim = feature_dim
        if FAISS_INDEX_TYPE_CPU == "IndexFlatIP":
             self.index_cpu = faiss.IndexFlatIP(self.feature_dim)
        elif FAISS_INDEX_TYPE_CPU == "IndexFlatL2":
             self.index_cpu = faiss.IndexFlatL2(self.feature_dim)
        else:
            raise ValueError(f"不支持的 CPU 索引类型: {FAISS_INDEX_TYPE_CPU}")
        print(f"初始化 Faiss CPU 索引 (类型: {FAISS_INDEX_TYPE_CPU}, 维度: {self.feature_dim})")
        self.image_paths = []
        init_end_time = time.time() # 结束计时
        print(f"  [计时] FaissIndexer __init__ 耗时: {init_end_time - init_start_time:.4f} 秒")


    def build_index(self, image_folder: str, feature_extractor: ViTFeatureExtractor):
        # build_index 内部有 tqdm 进度条，可以大致了解特征提取时间
        # 这里主要关注 Faiss add 的时间
        all_features = []
        valid_image_paths = []
        try:
            image_files = [os.path.join(image_folder, f)
                           for f in os.listdir(image_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        except FileNotFoundError:
            print(f"错误：数据文件夹 {image_folder} 未找到！")
            return

        if not image_files:
            print(f"警告：文件夹 {image_folder} 中没有找到支持的图像文件。")
            return

        print(f"在 {image_folder} 中找到 {len(image_files)} 张图片。开始提取特征...")

        for img_path in tqdm(image_files, desc="提取特征中"):
            features = feature_extractor.extract_features(img_path)
            if features is not None:
                all_features.append(features)
                valid_image_paths.append(img_path)

        if not all_features:
            print("错误：未能成功提取任何特征，无法构建索引。")
            return

        features_np = np.array(all_features).astype('float32')
        if features_np.shape[1] != self.feature_dim:
             raise ValueError(f"特征维度不匹配: 期望 {self.feature_dim}, 得到 {features_np.shape[1]}")

        print(f"提取了 {features_np.shape[0]} 个特征。正在构建 Faiss CPU 索引...")
        add_start_time = time.time()
        self.index_cpu.add(features_np)
        add_end_time = time.time()
        print(f"  [计时] Faiss index_cpu.add 耗时: {add_end_time - add_start_time:.4f} 秒")
        self.image_paths = valid_image_paths
        print(f"Faiss CPU 索引构建成功，包含 {self.index_cpu.ntotal} 个向量。")

    def save_index(self, index_path: str, mapping_path: str):
        if not hasattr(self.index_cpu, 'ntotal') or self.index_cpu.ntotal == 0:
            print("索引为空，不执行保存。")
            return
        print(f"正在保存 Faiss CPU 索引到 {index_path}")
        faiss.write_index(self.index_cpu, index_path)
        print(f"正在保存图像路径映射到 {mapping_path}")
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.image_paths, f)

    def load_index(self, index_path: str, mapping_path: str) -> bool:
        load_total_start = time.time()
        if not os.path.exists(index_path):
            print(f"错误：索引文件未找到: {index_path}")
            return False
        if not os.path.exists(mapping_path):
            print(f"错误：映射文件未找到: {mapping_path}")
            return False

        try:
            print(f"正在从 {index_path} 加载 Faiss CPU 索引")
            read_index_start = time.time()
            self.index_cpu = faiss.read_index(index_path)
            read_index_end = time.time()
            print(f"  [计时] faiss.read_index 耗时: {read_index_end - read_index_start:.4f} 秒")

            print(f"正在从 {mapping_path} 加载图像路径映射")
            pickle_load_start = time.time()
            with open(mapping_path, 'rb') as f:
                self.image_paths = pickle.load(f)
            pickle_load_end = time.time()
            print(f"  [计时] pickle.load 耗时: {pickle_load_end - pickle_load_start:.4f} 秒")

            print(f"CPU 索引和映射加载成功。索引包含 {self.index_cpu.ntotal} 个向量。")

            if self.index_cpu.d != self.feature_dim:
                print(f"警告：加载的索引维度 ({self.index_cpu.d}) 与期望维度 ({self.feature_dim}) 不符。将使用加载的维度。")
                self.feature_dim = self.index_cpu.d
            load_total_end = time.time()
            print(f"  [计时] FaissIndexer load_index 总耗时: {load_total_end - load_total_start:.4f} 秒")
            return True
        except Exception as e:
            print(f"错误：加载索引或映射时出错: {e}")
            self.index_cpu = None
            self.image_paths = []
            return False