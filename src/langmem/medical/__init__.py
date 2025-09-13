"""医学记忆管理模块

此模块为LangMem添加医学图像处理和诊断能力，特别针对医学影像诊断场景。

主要功能：
- 多模态记忆管理（图像+文本）
- 医学图像编码（支持biomedclip）
- 骨龄诊断专用功能
- 纠错学习机制
- 医学知识库集成
"""

from langmem.medical.manager import create_medical_memory_manager
from langmem.medical.schemas import MedicalImageMemory, BoneAgeMemory, DiagnosisCorrection

__all__ = [
    "create_medical_memory_manager",
    "MedicalImageMemory", 
    "BoneAgeMemory",
    "DiagnosisCorrection",
]