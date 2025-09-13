"""医学图像编码器模块

支持多种医学图像编码器，包括biomedclip等专业医学图像模型。
"""

import typing
from abc import ABC, abstractmethod
from pathlib import Path
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEncoder(ABC):
    """图像编码器基类"""
    
    @abstractmethod
    def encode(self, image_path: str | Path) -> list[float]:
        """将图像编码为特征向量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像特征向量
        """
        pass
    
    @abstractmethod
    def encode_batch(self, image_paths: list[str | Path]) -> list[list[float]]:
        """批量编码图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            图像特征向量列表
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回特征向量维度"""
        pass


class BiomedCLIPEncoder(ImageEncoder):
    """BiomedCLIP图像编码器
    
    专门用于医学图像的CLIP模型，在生物医学数据上预训练。
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """初始化BiomedCLIP编码器
        
        Args:
            model_name: HuggingFace模型名称
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
                import torch
                
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
                self._model.eval()
                
                logger.info(f"已加载BiomedCLIP模型: {self.model_name}")
                
            except ImportError as e:
                raise ImportError(
                    "需要安装transformers和torch库：pip install transformers torch"
                ) from e
            except Exception as e:
                logger.error(f"加载BiomedCLIP模型失败: {e}")
                raise
    
    def encode(self, image_path: str | Path) -> list[float]:
        """编码单张图像"""
        self._load_model()
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 获取图像特征
            import torch
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                features = outputs.cpu().numpy().flatten()
                
            return features.tolist()
            
        except Exception as e:
            logger.error(f"编码图像失败 {image_path}: {e}")
            raise
    
    def encode_batch(self, image_paths: list[str | Path]) -> list[list[float]]:
        """批量编码图像"""
        self._load_model()
        
        try:
            # 加载图像
            images = []
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                images.append(image)
            
            # 批量处理
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 获取特征
            import torch
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                features = outputs.cpu().numpy()
                
            return [feat.tolist() for feat in features]
            
        except Exception as e:
            logger.error(f"批量编码图像失败: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """BiomedCLIP特征维度通常为512"""
        return 512


class ResNetEncoder(ImageEncoder):
    """ResNet图像编码器
    
    使用预训练的ResNet模型作为图像编码器的备选方案。
    """
    
    def __init__(self, model_name: str = "resnet50"):
        """初始化ResNet编码器
        
        Args:
            model_name: 模型名称（resnet18, resnet34, resnet50等）
        """
        self.model_name = model_name
        self._model = None
        self._transform = None
        self._device = None
    
    def _load_model(self):
        """加载ResNet模型"""
        if self._model is None:
            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # 加载预训练模型
                if self.model_name == "resnet18":
                    self._model = models.resnet18(pretrained=True)
                elif self.model_name == "resnet34":
                    self._model = models.resnet34(pretrained=True)
                elif self.model_name == "resnet50":
                    self._model = models.resnet50(pretrained=True)
                else:
                    raise ValueError(f"不支持的ResNet模型: {self.model_name}")
                
                # 移除最后的分类层，使用特征
                self._model = torch.nn.Sequential(*list(self._model.children())[:-1])
                self._model.to(self._device)
                self._model.eval()
                
                # 图像预处理
                self._transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"已加载ResNet模型: {self.model_name}")
                
            except ImportError as e:
                raise ImportError("需要安装torch和torchvision：pip install torch torchvision") from e
    
    def encode(self, image_path: str | Path) -> list[float]:
        """编码单张图像"""
        self._load_model()
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert("RGB")
            input_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            # 提取特征
            import torch
            with torch.no_grad():
                features = self._model(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            return features.tolist()
            
        except Exception as e:
            logger.error(f"编码图像失败 {image_path}: {e}")
            raise
    
    def encode_batch(self, image_paths: list[str | Path]) -> list[list[float]]:
        """批量编码图像"""
        self._load_model()
        
        try:
            # 加载图像
            images = []
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                tensor = self._transform(image)
                images.append(tensor)
            
            # 批量处理
            import torch
            batch_tensor = torch.stack(images).to(self._device)
            
            with torch.no_grad():
                features = self._model(batch_tensor)
                features = features.squeeze().cpu().numpy()
                
            return [feat.tolist() for feat in features]
            
        except Exception as e:
            logger.error(f"批量编码图像失败: {e}")
            raise
    
    @property 
    def dimension(self) -> int:
        """ResNet特征维度"""
        if self.model_name in ["resnet18", "resnet34"]:
            return 512
        else:  # resnet50, resnet101, resnet152
            return 2048


def create_image_encoder(encoder_type: str = "biomedclip", **kwargs) -> ImageEncoder:
    """创建图像编码器
    
    Args:
        encoder_type: 编码器类型 ("biomedclip", "resnet")
        **kwargs: 编码器特定参数
        
    Returns:
        图像编码器实例
    """
    if encoder_type == "biomedclip":
        return BiomedCLIPEncoder(**kwargs)
    elif encoder_type == "resnet":
        return ResNetEncoder(**kwargs)
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")


def compute_image_similarity(features1: list[float], features2: list[float]) -> float:
    """计算两个图像特征向量的相似度
    
    Args:
        features1: 第一个图像的特征向量
        features2: 第二个图像的特征向量
        
    Returns:
        余弦相似度 (-1到1之间)
    """
    vec1 = np.array(features1)
    vec2 = np.array(features2)
    
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)