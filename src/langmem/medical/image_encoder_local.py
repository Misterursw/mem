"""
本地BiomedCLIP模型管理
处理模型下载、缓存和微调
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class LocalBiomedCLIP:
    """本地BiomedCLIP模型管理器"""

    def __init__(
        self,
        model_name: str = "/root/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
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
        # 使用transformers库能识别的统一缓存目录
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
        print(f"📥 检查并下载BiomedCLIP模型到: {self.cache_dir}")

        if force_download and Path(self.cache_dir).exists():
            print(f"🧹 强制重新下载，正在清空缓存目录: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            # 直接使用HuggingFace的下载和缓存机制
            print("🔄 正在下载/加载处理器...")
            AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            print("🔄 正在下载/加载模型...")
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            print("✅ BiomedCLIP模型下载/验证完成！")
            
            # 打印模型大小信息
            model_size_mb = self._get_model_size(model)
            print(f"📊 模型大小: {model_size_mb}MB")

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
            # 简化加载逻辑，直接从缓存目录加载
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)

            self._model.eval()

            print("✅ BiomedCLIP模型加载完成！")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            print("💡 尝试强制重新下载模型...")
            try:
                self.download_model(force_download=True)
                # 再次尝试加载
                self.load_model()
            except Exception as download_error:
                logger.error(f"强制重新下载失败: {download_error}")
                raise download_error

    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单张图像"""
        if self._model is None:
            self.load_model()

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                features = outputs.cpu().numpy().flatten()

            return features

        except Exception as e:
            logger.error(f"图像编码失败 {image_path}: {e}")
            raise
            
    # --- 微调相关函数 (保持不变) ---
    def prepare_for_finetuning(
        self,
        bone_age_dataset: Optional[Dict] = None,
        learning_rate: float = 1e-5,
        freeze_backbone: bool = True
    ):
        """准备骨龄微调"""
        if self._model is None:
            self.load_model()
        print("🎯 准备BiomedCLIP骨龄微调...")
        if freeze_backbone:
            for param in self._model.vision_model.parameters():
                param.requires_grad = False
            print("🔒 已冻结视觉主干网络")
        self._add_bone_age_head()
        trainable_params = filter(lambda p: p.requires_grad, self._model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        print(f"✅ 微调准备完成，学习率: {learning_rate}")
        return optimizer

    def _add_bone_age_head(self):
        """添加骨龄回归头"""
        feature_dim = self._model.config.projection_dim
        self._model.bone_age_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)
        ).to(self.device)
        print("🔧 已添加骨龄回归头")

    def finetune_on_bone_age(
        self,
        train_data: list,
        epochs: int = 10,
        batch_size: int = 8
    ):
        """在骨龄数据上微调"""
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
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                images = []
                targets = []
                for image_path, bone_age in batch:
                    try:
                        images.append(Image.open(image_path).convert("RGB"))
                        targets.append(bone_age)
                    except Exception as e:
                        logger.warning(f"跳过无效图像 {image_path}: {e}")
                        continue
                if not images: continue
                
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                image_features = self._model.get_image_features(**inputs)
                predictions = self._model.bone_age_head(image_features).squeeze()
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
        
        self._is_finetuned = True
        print("✅ 骨龄微调完成！")
        self.save_finetuned_model()

    def save_finetuned_model(self):
        """保存微调后的模型"""
        if not self._is_finetuned:
            print("⚠️ 模型尚未微调，跳过保存")
            return
        
        save_path = Path(self.cache_dir) / "finetuned_bone_age"
        save_path.mkdir(exist_ok=True)
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
            self.load_model()
            self._add_bone_age_head()
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
        
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            bone_age = self._model.bone_age_head(image_features).squeeze()
            return bone_age.item()

def setup_local_biomedclip():
    """设置本地BiomedCLIP环境"""
    print("🔧 设置本地BiomedCLIP环境...")
    biomedclip = LocalBiomedCLIP()
    # 第一次运行时会下载
    biomedclip.load_model()
    return biomedclip

if __name__ == "__main__":
    # 测试本地BiomedCLIP
    biomedclip = setup_local_biomedclip()
    
    print("\n🧪 测试图像编码...")
    print("   (请取消下面代码的注释，并提供一张真实图像的路径进行测试)")
    # try:
    #     # 创建一个假的空白图像用于测试
    #     dummy_image_path = "test_hand_xray.jpg"
    #     Image.new('RGB', (224, 224), color = 'red').save(dummy_image_path)
    #     features = biomedclip.encode_image(dummy_image_path)
    #     print(f"✅ 图像编码成功! 特征维度: {features.shape}")
    #     os.remove(dummy_image_path)
    # except Exception as e:
    #     print(f"❌ 图像编码测试失败: {e}")

    print("\n✅ 本地BiomedCLIP设置完成！")