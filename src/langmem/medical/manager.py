"""医学记忆管理器

扩展LangMem的核心功能，支持医学图像和多模态记忆管理。
"""

import typing
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from langmem import create_memory_store_manager
from langmem.medical.schemas import (
    MedicalImageMemory, 
    BoneAgeMemory, 
    DiagnosisCorrection,
    MedicalKnowledge
)
from langmem.medical.image_encoder import create_image_encoder, ImageEncoder


class MedicalMemoryManager:
    """医学记忆管理器
    
    结合了图像编码和医学专业知识的记忆管理系统。
    """
    
    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        image_encoder: str | ImageEncoder = "biomedclip",
        schemas: list[type] | None = None,
        namespace: tuple[str, ...] = ("medical_memories", "{langgraph_user_id}"),
        store: BaseStore | None = None,
        **kwargs
    ):
        """初始化医学记忆管理器
        
        Args:
            model: 语言模型
            image_encoder: 图像编码器类型或实例
            schemas: 记忆schema列表
            namespace: 存储命名空间
            store: 存储后端
            **kwargs: 传递给基础记忆管理器的其他参数
        """
        # 设置默认schema
        if schemas is None:
            schemas = [MedicalImageMemory, BoneAgeMemory, DiagnosisCorrection, MedicalKnowledge]
        
        # 创建图像编码器
        if isinstance(image_encoder, str):
            self.image_encoder = create_image_encoder(image_encoder)
        else:
            self.image_encoder = image_encoder
        
        # 创建基础记忆管理器
        self.base_manager = create_memory_store_manager(
            model,
            schemas=schemas,
            namespace=namespace,
            store=store,
            **kwargs
        )
        
        self.schemas = schemas
        self.namespace_template = self.base_manager.namespace
    
    async def add_image_memory(
        self,
        image_path: str | Path,
        content: str,
        *,
        image_type: str = "unknown",
        body_part: str = "unknown",
        patient_age: float | None = None,
        patient_gender: str | None = None,
        diagnosis: str | None = None,
        confidence: float | None = None,
        config: RunnableConfig | None = None
    ) -> str:
        """添加图像记忆
        
        Args:
            image_path: 图像路径
            content: 图像描述
            image_type: 图像类型
            body_part: 身体部位
            patient_age: 患者年龄
            patient_gender: 患者性别
            diagnosis: 诊断结果
            confidence: 置信度
            config: 运行配置
            
        Returns:
            记忆ID
        """
        # 编码图像
        features = self.image_encoder.encode(image_path)
        
        # 创建记忆对象
        memory = MedicalImageMemory(
            content=content,
            image_path=str(image_path),
            image_features=features,
            image_type=image_type,
            body_part=body_part,
            patient_age=patient_age,
            patient_gender=patient_gender,
            diagnosis=diagnosis,
            confidence=confidence
        )
        
        # 存储记忆
        namespace = self.namespace_template(config)
        await self.base_manager.aput(
            key=f"image_{hash(str(image_path))}",
            value={"kind": "MedicalImageMemory", "content": memory.model_dump()},
            config=config
        )
        
        return f"image_{hash(str(image_path))}"
    
    async def add_bone_age_memory(
        self,
        image_path: str | Path,
        content: str,
        chronological_age: float,
        predicted_bone_age: float,
        gender: str,
        *,
        actual_bone_age: float | None = None,
        assessment_method: str = "Greulich-Pyle",
        key_features: list[str] | None = None,
        confidence: float = 0.0,
        config: RunnableConfig | None = None
    ) -> str:
        """添加骨龄诊断记忆
        
        Args:
            image_path: 手部X光片路径
            content: 诊断描述
            chronological_age: 实际年龄
            predicted_bone_age: 预测骨龄
            gender: 性别
            actual_bone_age: 实际骨龄（医生标注）
            assessment_method: 评估方法
            key_features: 关键特征
            confidence: 置信度
            config: 运行配置
            
        Returns:
            记忆ID
        """
        # 编码图像
        features = self.image_encoder.encode(image_path)
        
        # 创建记忆对象
        memory = BoneAgeMemory(
            content=content,
            image_path=str(image_path),
            image_features=features,
            chronological_age=chronological_age,
            predicted_bone_age=predicted_bone_age,
            actual_bone_age=actual_bone_age,
            gender=gender,
            assessment_method=assessment_method,
            key_features=key_features or [],
            confidence=confidence
        )
        
        # 存储记忆
        namespace = self.namespace_template(config)
        memory_id = f"bone_age_{hash(str(image_path))}"
        await self.base_manager.aput(
            key=memory_id,
            value={"kind": "BoneAgeMemory", "content": memory.model_dump()},
            config=config
        )
        
        return memory_id
    
    async def add_correction_memory(
        self,
        image_path: str | Path,
        content: str,
        original_diagnosis: str,
        corrected_diagnosis: str,
        error_type: str,
        correction_reason: str,
        original_confidence: float,
        corrected_confidence: float,
        *,
        learning_points: list[str] | None = None,
        corrector_type: str = "human_expert",
        config: RunnableConfig | None = None
    ) -> str:
        """添加诊断纠错记忆
        
        Args:
            image_path: 相关图像路径
            content: 纠错说明
            original_diagnosis: 原始诊断
            corrected_diagnosis: 纠正后诊断
            error_type: 错误类型
            correction_reason: 纠正原因
            original_confidence: 原始置信度
            corrected_confidence: 纠正后置信度
            learning_points: 学习要点
            corrector_type: 纠错者类型
            config: 运行配置
            
        Returns:
            记忆ID
        """
        # 创建纠错记忆
        memory = DiagnosisCorrection(
            content=content,
            original_diagnosis=original_diagnosis,
            corrected_diagnosis=corrected_diagnosis,
            image_path=str(image_path),
            error_type=error_type,
            correction_reason=correction_reason,
            original_confidence=original_confidence,
            corrected_confidence=corrected_confidence,
            learning_points=learning_points or [],
            corrector_type=corrector_type
        )
        
        # 存储记忆
        namespace = self.namespace_template(config)
        memory_id = f"correction_{hash(f'{image_path}_{original_diagnosis}')}"
        await self.base_manager.aput(
            key=memory_id,
            value={"kind": "DiagnosisCorrection", "content": memory.model_dump()},
            config=config
        )
        
        return memory_id
    
    async def search_similar_images(
        self,
        image_path: str | Path,
        *,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        config: RunnableConfig | None = None
    ) -> list[dict]:
        """搜索相似图像记忆
        
        Args:
            image_path: 查询图像路径
            limit: 返回数量限制
            similarity_threshold: 相似度阈值
            config: 运行配置
            
        Returns:
            相似图像记忆列表
        """
        # 编码查询图像
        query_features = self.image_encoder.encode(image_path)
        
        # 搜索所有图像记忆
        all_memories = await self.base_manager.asearch(
            query="medical image",
            limit=100,  # 先获取更多结果
            config=config
        )
        
        # 计算相似度并过滤
        similar_memories = []
        for memory in all_memories:
            if memory.value.get("kind") in ["MedicalImageMemory", "BoneAgeMemory"]:
                content = memory.value.get("content", {})
                stored_features = content.get("image_features")
                
                if stored_features:
                    from langmem.medical.image_encoder import compute_image_similarity
                    similarity = compute_image_similarity(query_features, stored_features)
                    
                    if similarity >= similarity_threshold:
                        memory_dict = memory.dict()
                        memory_dict["similarity"] = similarity
                        similar_memories.append(memory_dict)
        
        # 按相似度排序并限制数量
        similar_memories.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_memories[:limit]
    
    async def search_corrections(
        self,
        error_type: str | None = None,
        *,
        limit: int = 10,
        config: RunnableConfig | None = None
    ) -> list[dict]:
        """搜索诊断纠错记忆
        
        Args:
            error_type: 错误类型过滤
            limit: 返回数量限制
            config: 运行配置
            
        Returns:
            纠错记忆列表
        """
        query = f"diagnosis correction {error_type}" if error_type else "diagnosis correction"
        
        memories = await self.base_manager.asearch(
            query=query,
            filter={"kind": "DiagnosisCorrection"} if not error_type else None,
            limit=limit,
            config=config
        )
        
        result = []
        for memory in memories:
            if error_type is None or memory.value.get("content", {}).get("error_type") == error_type:
                result.append(memory.dict())
        
        return result
    
    async def get_learning_insights(
        self,
        *,
        config: RunnableConfig | None = None
    ) -> dict:
        """获取学习洞察
        
        基于历史纠错记忆分析常见错误模式和改进建议。
        
        Args:
            config: 运行配置
            
        Returns:
            学习洞察报告
        """
        # 获取所有纠错记忆
        corrections = await self.search_corrections(config=config)
        
        if not corrections:
            return {"message": "暂无纠错记忆数据"}
        
        # 分析错误类型分布
        error_types = {}
        learning_points = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            error_type = content.get("error_type")
            points = content.get("learning_points", [])
            
            if error_type:
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            learning_points.extend(points)
        
        # 统计最常见的学习要点
        from collections import Counter
        common_points = Counter(learning_points).most_common(5)
        
        return {
            "total_corrections": len(corrections),
            "error_type_distribution": error_types,
            "common_learning_points": common_points,
            "suggestions": [
                "重点关注最常见的错误类型",
                "加强对关键学习要点的训练",
                "定期回顾纠错案例"
            ]
        }


def create_medical_memory_manager(
    model: str | BaseChatModel,
    *,
    image_encoder: str | ImageEncoder = "biomedclip",
    domain: str | None = None,
    namespace: tuple[str, ...] | None = None,
    store: BaseStore | None = None,
    **kwargs
) -> MedicalMemoryManager:
    """创建医学记忆管理器
    
    Args:
        model: 语言模型
        image_encoder: 图像编码器
        domain: 医学领域（如"bone_age"）
        namespace: 存储命名空间
        store: 存储后端
        **kwargs: 其他参数
        
    Returns:
        医学记忆管理器实例
    """
    # 根据领域调整命名空间
    if namespace is None:
        if domain:
            namespace = ("medical_memories", domain, "{langgraph_user_id}")
        else:
            namespace = ("medical_memories", "{langgraph_user_id}")
    
    return MedicalMemoryManager(
        model,
        image_encoder=image_encoder,
        namespace=namespace,
        store=store,
        **kwargs
    )