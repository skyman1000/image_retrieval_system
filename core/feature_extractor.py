# core/feature_extractor.py
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import numpy as np

class ViTFeatureExtractor:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"特征提取器：使用设备 - {self.device}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray | None:
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # L2 归一化 (对于 IndexFlatIP/余弦相似度很重要)
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            normalized_features = features / (norm + 1e-6)
            return normalized_features.flatten()
        except Exception as e:
            print(f"错误：处理图像 {image_path} 时出错: {e}")
            return None