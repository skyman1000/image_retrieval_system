# core/searcher.py
import faiss
import numpy as np
import pickle
import os
import time
from .config import FAISS_INDEX_TYPE_CPU

class FaissSearcher:
    def __init__(self, index_path: str, mapping_path: str):
        init_start_time = time.time() # 开始计时
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index_cpu = None
        self.index_gpu = None
        self.gpu_resource = None
        self.image_paths = None
        self.is_gpu_enabled = False
        self._load_and_init_gpu()
        init_end_time = time.time() # 结束计时
        print(f"  [计时] FaissSearcher __init__ (含 _load_and_init_gpu) 总耗时: {init_end_time - init_start_time:.4f} 秒")


    def _load_and_init_gpu(self):
        load_total_start = time.time()
        if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
            print(f"错误：索引文件 ({self.index_path}) 或映射文件 ({self.mapping_path}) 未找到。请先构建索引。")
            return False
        try:
            print(f"搜索器：正在从 {self.index_path} 加载 Faiss CPU 索引...")
            read_index_start = time.time()
            self.index_cpu = faiss.read_index(self.index_path)
            read_index_end = time.time()
            print(f"  [计时] faiss.read_index 耗时: {read_index_end - read_index_start:.4f} 秒")
            print(f"搜索器：CPU 索引加载成功，包含 {self.index_cpu.ntotal} 个向量，维度 {self.index_cpu.d}。")

            print(f"搜索器：正在从 {self.mapping_path} 加载图像路径映射...")
            pickle_load_start = time.time()
            with open(self.mapping_path, 'rb') as f:
                self.image_paths = pickle.load(f)
            pickle_load_end = time.time()
            print(f"  [计时] pickle.load 耗时: {pickle_load_end - pickle_load_start:.4f} 秒")
            print("搜索器：图像路径映射加载成功。")

            print("搜索器：尝试将索引转移到 GPU...")
            gpu_init_total_start = time.time()
            try:
                if faiss.get_num_gpus() > 0:
                    print(f"搜索器：检测到 {faiss.get_num_gpus()} 个 GPU。正在使用 GPU 0...")
                    
                    gpu_res_start = time.time()
                    self.gpu_resource = faiss.StandardGpuResources()
                    gpu_res_end = time.time()
                    print(f"    [计时] faiss.StandardGpuResources() 耗时: {gpu_res_end - gpu_res_start:.4f} 秒")

                    cpu_to_gpu_start = time.time()
                    self.index_gpu = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index_cpu)
                    cpu_to_gpu_end = time.time()
                    print(f"    [计时] faiss.index_cpu_to_gpu 耗时: {cpu_to_gpu_end - cpu_to_gpu_start:.4f} 秒")
                    
                    self.is_gpu_enabled = True
                    print("搜索器：索引已成功转移到 GPU。将使用 GPU 进行搜索。")
                else:
                    print("搜索器：未检测到可用 GPU。将使用 CPU 进行搜索。")
                    self.is_gpu_enabled = False
            except AttributeError:
                 print("搜索器：当前 Faiss 版本似乎不支持 GPU (可能是 faiss-cpu 版本)。将使用 CPU 进行搜索。")
                 self.is_gpu_enabled = False
            except Exception as gpu_e:
                print(f"搜索器：将索引转移到 GPU 时出错: {gpu_e}。将使用 CPU 进行搜索。")
                self.is_gpu_enabled = False
            gpu_init_total_end = time.time()
            print(f"  [计时] GPU 初始化尝试总耗时: {gpu_init_total_end - gpu_init_total_start:.4f} 秒")

            load_total_end = time.time()
            print(f"  [计时] FaissSearcher _load_and_init_gpu 总耗时: {load_total_end - load_total_start:.4f} 秒")
            return True
        except Exception as e:
            print(f"错误：加载索引或映射时发生严重错误: {e}")
            self.index_cpu = None
            self.index_gpu = None
            self.image_paths = None
            return False

    def get_active_index(self):
        if self.is_gpu_enabled and self.index_gpu:
            return self.index_gpu
        elif self.index_cpu:
            return self.index_cpu
        else:
            return None

    def search(self, query_feature: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        # 搜索本身的计时已经在 SearchWorker 中
        active_index = self.get_active_index()
        if active_index is None or self.image_paths is None:
            print("错误：索引未成功加载，无法执行搜索。")
            return []
        if active_index.ntotal == 0:
            print("警告：索引为空，无法执行搜索。")
            return []

        query_feature_np = query_feature.astype('float32').reshape(1, -1)
        if query_feature_np.shape[1] != active_index.d:
             raise ValueError(f"查询特征维度 ({query_feature_np.shape[1]}) 与索引维度 ({active_index.d}) 不匹配！")

        print(f"搜索器：使用 {'GPU' if self.is_gpu_enabled else 'CPU'} 执行搜索...")
        # 实际的 active_index.search 计时由 SearchWorker 完成
        distances, indices = active_index.search(query_feature_np, k)

        results = []
        if indices.size > 0:
            for i, dist in zip(indices[0], distances[0]):
                if i == -1 or not (0 <= i < len(self.image_paths)):
                    print(f"警告：搜索返回无效索引 {i}。")
                    continue
                results.append((self.image_paths[i], float(dist)))
        return results

    def get_index_status(self) -> str:
        if self.get_active_index():
            status = f"索引已加载 ({self.get_active_index().ntotal} 向量, 维度 {self.get_active_index().d})。"
            status += f" 当前使用 {'GPU' if self.is_gpu_enabled else 'CPU'} 进行搜索。"
            return status
        else:
            return "索引未加载或加载失败。"