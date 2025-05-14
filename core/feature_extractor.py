# core/feature_extractor.py
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import numpy as np
import time # 导入 time 模块

class ViTFeatureExtractor:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        init_start_time = time.time() # 开始计时
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"特征提取器：使用设备 - {self.device}")

        print(f"  [计时] ViTImageProcessor.from_pretrained('{model_name}') 开始...")
        proc_start_time = time.time()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        proc_end_time = time.time()
        print(f"  [计时] ViTImageProcessor.from_pretrained 完成, 耗时: {proc_end_time - proc_start_time:.4f} 秒")

        print(f"  [计时] ViTModel.from_pretrained('{model_name}').to('{self.device}') 开始...")
        model_load_start_time = time.time()
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        model_load_end_time = time.time()
        print(f"  [计时] ViTModel.from_pretrained.to(device) 完成, 耗时: {model_load_end_time - model_load_start_time:.4f} 秒")

        self.model.eval()
        init_end_time = time.time() # 结束计时
        print(f"  [计时] ViTFeatureExtractor __init__ 总耗时: {init_end_time - init_start_time:.4f} 秒")


    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray | None:
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            normalized_features = features / (norm + 1e-6)
            return normalized_features.flatten()
        except Exception as e:
            print(f"错误：处理图像 {image_path} 时出错: {e}")
            return None