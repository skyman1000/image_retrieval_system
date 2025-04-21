# core/searcher.py
import faiss
import numpy as np
import pickle
import os
import time
from .config import FAISS_INDEX_TYPE_CPU # 需要知道原始CPU类型以处理得分

class FaissSearcher:
    def __init__(self, index_path: str, mapping_path: str):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index_cpu = None # 保存加载的 CPU 索引
        self.index_gpu = None # 保存 GPU 索引 (如果可用)
        self.gpu_resource = None # 保存 GPU 资源
        self.image_paths = None
        self.is_gpu_enabled = False # 标记是否成功启用了 GPU
        self._load_and_init_gpu() # 调用加载和 GPU 初始化方法

    def _load_and_init_gpu(self):
        """ 加载 CPU 索引、映射，并尝试初始化 GPU 索引 """
        if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
            print(f"错误：索引文件 ({self.index_path}) 或映射文件 ({self.mapping_path}) 未找到。请先构建索引。")
            return False
        try:
            # 1. 加载 CPU 索引
            print(f"搜索器：正在从 {self.index_path} 加载 Faiss CPU 索引...")
            self.index_cpu = faiss.read_index(self.index_path)
            print(f"搜索器：CPU 索引加载成功，包含 {self.index_cpu.ntotal} 个向量，维度 {self.index_cpu.d}。")

            # 2. 加载图像路径映射
            print(f"搜索器：正在从 {self.mapping_path} 加载图像路径映射...")
            with open(self.mapping_path, 'rb') as f:
                self.image_paths = pickle.load(f)
            print("搜索器：图像路径映射加载成功。")

            # 3.  尝试初始化 GPU 索引 
            print("搜索器：尝试将索引转移到 GPU...")
            try:
                # 检查是否有可用的 GPU
                if faiss.get_num_gpus() > 0:
                    print(f"搜索器：检测到 {faiss.get_num_gpus()} 个 GPU。正在使用 GPU 0...")
                    self.gpu_resource = faiss.StandardGpuResources() # 初始化 GPU 资源
                    # 将加载的 CPU 索引转移到 GPU 0
                    self.index_gpu = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index_cpu)
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

            return True
        except Exception as e:
            print(f"错误：加载索引或映射时发生严重错误: {e}")
            self.index_cpu = None
            self.index_gpu = None
            self.image_paths = None
            return False

    def get_active_index(self):
        """ 返回当前用于搜索的索引 (GPU 或 CPU) """
        if self.is_gpu_enabled and self.index_gpu:
            return self.index_gpu
        elif self.index_cpu:
            return self.index_cpu
        else:
            return None

    def search(self, query_feature: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
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
        start_time = time.time()
        distances, indices = active_index.search(query_feature_np, k)
        end_time = time.time()
        print(f"搜索器：搜索耗时 {end_time - start_time:.4f} 秒。")


        results = []
        if indices.size > 0:
            for i, dist in zip(indices[0], distances[0]):
                # Faiss 可能返回 -1 表示没有找到足够的邻居
                if i == -1 or not (0 <= i < len(self.image_paths)):
                    print(f"警告：搜索返回无效索引 {i}。")
                    continue
                results.append((self.image_paths[i], float(dist)))

        #  注意: IndexFlatIP 返回的是内积，对于 L2 归一化的向量，内积即余弦相似度。
        #    得分越高越相似。IndexFlatL2 返回的是欧氏距离的平方，得分越低越相似。
        #    这里我们假设使用 IndexFlatIP (余弦相似度)，结果列表默认就是按相似度降序排列的。
        # if FAISS_INDEX_TYPE_CPU == "IndexFlatL2":
        #      results.sort(key=lambda x: x[1]) # 如果是 L2 距离，按距离升序排

        return results

    def get_index_status(self) -> str:
        """ 返回当前索引状态的描述 """
        if self.get_active_index():
            status = f"索引已加载 ({self.get_active_index().ntotal} 向量, 维度 {self.get_active_index().d})。"
            status += f" 当前使用 {'GPU' if self.is_gpu_enabled else 'CPU'} 进行搜索。"
            return status
        else:
            return "索引未加载或加载失败。"