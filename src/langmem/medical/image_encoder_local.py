"""
本地BiomedCLIP模型管理
处理模型下载、缓存和微调
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class LocalBiomedCLIP:
    """本地BiomedCLIP模型管理器"""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """初始化本地BiomedCLIP模型
        
        Args:
            model_name: HuggingFace模型名称
            cache_dir: 模型缓存目录
            device: 计算设备
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/langmem/biomedclip")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._model = None
        self._processor = None
        self._is_finetuned = False
        
        # 创建缓存目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def download_model(self, force_download: bool = False):
        """下载BiomedCLIP模型到本地
        
        Args:
            force_download: 是否强制重新下载
        """
        print(f"📥 下载BiomedCLIP模型到: {self.cache_dir}")
        
        try:
            # 检查是否已存在
            model_path = Path(self.cache_dir) / "model"
            if model_path.exists() and not force_download:
                print("✅ 模型已存在，跳过下载")
                return
            
            # 下载模型和处理器
            print("🔄 正在下载模型文件...")
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(model_path / "processor")
            )
            
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(model_path / "model")
            )
            
            # 保存配置信息
            config = {
                "model_name": self.model_name,
                "download_time": str(torch.utils.data.get_worker_info()),
                "device": self.device,
                "model_size_mb": self._get_model_size(model)
            }
            
            with open(model_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print("✅ BiomedCLIP模型下载完成！")
            print(f"📊 模型大小: {config.get('model_size_mb', 'Unknown')}MB")
            
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise
    
    def _get_model_size(self, model) -> float:
        """计算模型大小(MB)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return round(size_mb, 2)
    
    def load_model(self):
        """加载本地BiomedCLIP模型"""
        if self._model is not None:
            return
            
        print(f"🔄 加载BiomedCLIP模型到 {self.device}...")
        
        try:
            model_path = Path(self.cache_dir) / "model"
            
            if not model_path.exists():
                print("❌ 本地模型不存在，开始下载...")
                self.download_model()
            
            # 加载处理器和模型
            self._processor = AutoProcessor.from_pretrained(
                str(model_path / "processor")
            )
            
            self._model = AutoModel.from_pretrained(
                str(model_path / "model")
            ).to(self.device)
            
            self._model.eval()
            
            print("✅ BiomedCLIP模型加载完成！")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            print("💡 尝试重新下载模型...")
            self.download_model(force_download=True)
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单张图像"""
        if self._model is None:
            self.load_model()
        
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 预处理
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                features = outputs.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"图像编码失败 {image_path}: {e}")
            raise
    
    def prepare_for_finetuning(
        self,
        bone_age_dataset: Optional[Dict] = None,
        learning_rate: float = 1e-5,
        freeze_backbone: bool = True
    ):
        """准备骨龄微调
        
        Args:
            bone_age_dataset: 骨龄数据集
            learning_rate: 学习率
            freeze_backbone: 是否冻结主干网络
        """
        if self._model is None:
            self.load_model()
        
        print("🎯 准备BiomedCLIP骨龄微调...")
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self._model.vision_model.parameters():
                param.requires_grad = False
            print("🔒 已冻结视觉主干网络")
        
        # 添加骨龄回归头
        self._add_bone_age_head()
        
        # 设置优化器
        trainable_params = filter(lambda p: p.requires_grad, self._model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        print(f"✅ 微调准备完成，学习率: {learning_rate}")
        return optimizer
    
    def _add_bone_age_head(self):
        """添加骨龄回归头"""
        # 获取特征维度
        feature_dim = self._model.config.projection_dim
        
        # 添加回归头
        self._model.bone_age_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)  # 输出骨龄值
        ).to(self.device)
        
        print("🔧 已添加骨龄回归头")
    
    def finetune_on_bone_age(
        self,
        train_data: list,
        epochs: int = 10,
        batch_size: int = 8
    ):
        """在骨龄数据上微调
        
        Args:
            train_data: 训练数据 [(image_path, bone_age), ...]
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if not hasattr(self._model, 'bone_age_head'):
            optimizer = self.prepare_for_finetuning()
        else:
            trainable_params = filter(lambda p: p.requires_grad, self._model.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        
        print(f"🎯 开始骨龄微调: {len(train_data)}个样本, {epochs}轮")
        
        criterion = torch.nn.MSELoss()
        self._model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 简单的批次处理
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # 处理批次数据
                images = []
                targets = []
                
                for image_path, bone_age in batch:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        images.append(image)
                        targets.append(bone_age)
                    except Exception as e:
                        logger.warning(f"跳过无效图像 {image_path}: {e}")
                        continue
                
                if not images:
                    continue
                
                # 预处理
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                
                # 提取特征
                image_features = self._model.get_image_features(**inputs)
                
                # 骨龄预测
                predictions = self._model.bone_age_head(image_features).squeeze()
                
                # 计算损失
                loss = criterion(predictions, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
        
        self._is_finetuned = True
        print("✅ 骨龄微调完成！")
        
        # 保存微调后的模型
        self.save_finetuned_model()
    
    def save_finetuned_model(self):
        """保存微调后的模型"""
        if not self._is_finetuned:
            print("⚠️ 模型尚未微调，跳过保存")
            return
        
        save_path = Path(self.cache_dir) / "finetuned_bone_age"
        save_path.mkdir(exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'is_finetuned': True,
            'model_name': self.model_name
        }, save_path / "model.pt")
        
        print(f"💾 微调模型已保存到: {save_path}")
    
    def load_finetuned_model(self):
        """加载微调后的模型"""
        save_path = Path(self.cache_dir) / "finetuned_bone_age" / "model.pt"
        
        if not save_path.exists():
            print("❌ 未找到微调模型")
            return False
        
        try:
            # 先加载基础模型
            self.load_model()
            
            # 添加回归头
            self._add_bone_age_head()
            
            # 加载微调权重
            checkpoint = torch.load(save_path, map_location=self.device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            
            self._is_finetuned = True
            print("✅ 微调模型加载成功！")
            return True
            
        except Exception as e:
            logger.error(f"微调模型加载失败: {e}")
            return False
    
    def predict_bone_age(self, image_path: str) -> float:
        """预测骨龄（需要微调后的模型）"""
        if not self._is_finetuned:
            raise ValueError("需要先微调模型才能预测骨龄")
        
        # 编码图像
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 提取特征
            image_features = self._model.get_image_features(**inputs)
            
            # 预测骨龄
            bone_age = self._model.bone_age_head(image_features).squeeze()
            
            return bone_age.item()


def setup_local_biomedclip():
    """设置本地BiomedCLIP环境"""
    print("🔧 设置本地BiomedCLIP环境...")
    
    # 创建本地模型管理器
    biomedclip = LocalBiomedCLIP()
    
    # 下载模型
    biomedclip.download_model()
    
    return biomedclip


if __name__ == "__main__":
    # 测试本地BiomedCLIP
    biomedclip = setup_local_biomedclip()
    
    print("🧪 测试图像编码...")
    # 这里需要真实的医学图像路径
    # features = biomedclip.encode_image("test_hand_xray.jpg")
    # print(f"特征维度: {features.shape}")
    
    print("✅ 本地BiomedCLIP设置完成！")