"""医学记忆相关的Schema定义

定义了医学图像记忆、骨龄诊断记忆等专用数据结构。
"""

import typing
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class MedicalImageMemory(BaseModel):
    """医学图像记忆结构
    
    存储医学图像的特征向量、诊断结果、元数据等信息。
    """
    
    content: str = Field(description="图像相关的文本描述或诊断结果")
    image_path: str = Field(description="图像文件路径")
    image_features: typing.Optional[list[float]] = Field(
        default=None, 
        description="图像特征向量（biomedclip编码结果）"
    )
    image_type: str = Field(default="unknown", description="图像类型（如X光、CT、MRI等）")
    body_part: str = Field(default="unknown", description="身体部位")
    patient_age: typing.Optional[float] = Field(default=None, description="患者年龄")
    patient_gender: typing.Optional[str] = Field(default=None, description="患者性别")
    diagnosis: typing.Optional[str] = Field(default=None, description="诊断结果")
    confidence: typing.Optional[float] = Field(
        default=None, 
        description="诊断置信度 (0.0-1.0)"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="记录时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }


class BoneAgeMemory(BaseModel):
    """骨龄诊断专用记忆结构
    
    专门用于骨龄诊断的记忆，包含详细的骨龄评估信息。
    """
    
    content: str = Field(description="骨龄诊断的详细描述")
    image_path: str = Field(description="手部X光片路径")
    image_features: typing.Optional[list[float]] = Field(
        default=None,
        description="图像特征向量"
    )
    chronological_age: float = Field(description="实际年龄（岁）")
    predicted_bone_age: float = Field(description="预测骨龄（岁）")
    actual_bone_age: typing.Optional[float] = Field(
        default=None, 
        description="实际骨龄（医生标注，岁）"
    )
    age_difference: typing.Optional[float] = Field(
        default=None,
        description="骨龄与实际年龄差异（岁）"
    )
    gender: str = Field(description="性别 (male/female)")
    assessment_method: str = Field(
        default="Greulich-Pyle", 
        description="评估方法 (Greulich-Pyle, TW3等)"
    )
    key_features: list[str] = Field(
        default_factory=list,
        description="关键诊断特征列表"
    )
    confidence: float = Field(description="诊断置信度 (0.0-1.0)")
    error_analysis: typing.Optional[str] = Field(
        default=None,
        description="错误分析（如果有纠错的话）"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="记录时间")
    
    def calculate_age_difference(self) -> float:
        """计算骨龄与实际年龄的差异"""
        if self.actual_bone_age is not None:
            return self.actual_bone_age - self.chronological_age
        return self.predicted_bone_age - self.chronological_age
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }


class DiagnosisCorrection(BaseModel):
    """诊断纠错记忆结构
    
    用于记录诊断错误和纠正信息，支持持续学习。
    """
    
    content: str = Field(description="纠错的详细说明")
    original_diagnosis: str = Field(description="原始诊断结果")
    corrected_diagnosis: str = Field(description="纠正后的诊断结果")
    image_path: str = Field(description="相关图像路径")
    error_type: str = Field(description="错误类型（如年龄估计偏差、特征识别错误等）")
    correction_reason: str = Field(description="纠正原因和依据")
    original_confidence: float = Field(description="原始诊断置信度")
    corrected_confidence: float = Field(description="纠正后置信度")
    learning_points: list[str] = Field(
        default_factory=list,
        description="学习要点列表"
    )
    corrector_type: str = Field(
        default="human_expert",
        description="纠错者类型 (human_expert, peer_review, literature等)"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="纠错时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }


class MedicalKnowledge(BaseModel):
    """医学知识库条目结构
    
    用于存储医学标准、指南等知识。
    """
    
    content: str = Field(description="知识内容")
    title: str = Field(description="知识标题")
    category: str = Field(description="知识分类（如诊断标准、解剖知识等）")
    subcategory: typing.Optional[str] = Field(default=None, description="子分类")
    source: str = Field(description="知识来源")
    evidence_level: str = Field(
        default="unknown",
        description="证据级别 (A, B, C等)"
    )
    applicable_age_range: typing.Optional[tuple[float, float]] = Field(
        default=None,
        description="适用年龄范围（岁）"
    )
    applicable_gender: typing.Optional[str] = Field(
        default=None,
        description="适用性别 (male/female/both)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="标签列表"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="录入时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }