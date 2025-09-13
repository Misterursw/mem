"""骨龄诊断专用模块

提供骨龄诊断、纠错学习等专业功能。
"""

from langmem.medical.bone_age.classifier import BoneAgeClassifier
from langmem.medical.bone_age.knowledge import BoneAgeKnowledge

__all__ = [
    "BoneAgeClassifier",
    "BoneAgeKnowledge"
]