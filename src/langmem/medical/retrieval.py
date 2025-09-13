"""多模态检索和RAG增强模块

支持图像-文本混合检索，医学知识库检索等功能。
"""

import typing
from pathlib import Path
import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel

from langmem.medical.manager import MedicalMemoryManager
from langmem.medical.image_encoder import ImageEncoder, compute_image_similarity
from langmem.medical.schemas import MedicalKnowledge

logger = logging.getLogger(__name__)


class MultiModalRetriever:
    """多模态检索器
    
    支持基于图像、文本或图像+文本的混合检索。
    """
    
    def __init__(
        self,
        memory_manager: MedicalMemoryManager,
        *,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
        similarity_threshold: float = 0.3
    ):
        """初始化多模态检索器
        
        Args:
            memory_manager: 医学记忆管理器
            text_weight: 文本相似度权重
            image_weight: 图像相似度权重  
            similarity_threshold: 相似度阈值
        """
        self.memory_manager = memory_manager
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.similarity_threshold = similarity_threshold
    
    async def retrieve_similar_cases(
        self,
        query_text: str | None = None,
        query_image: str | Path | None = None,
        *,
        case_types: list[str] | None = None,
        limit: int = 10,
        config: RunnableConfig | None = None
    ) -> list[dict]:
        """检索相似病例
        
        Args:
            query_text: 查询文本
            query_image: 查询图像路径
            case_types: 病例类型过滤 (如 ["BoneAgeMemory", "MedicalImageMemory"])
            limit: 返回数量限制
            config: 运行配置
            
        Returns:
            相似病例列表，按相似度排序
        """
        if not query_text and not query_image:
            raise ValueError("必须提供查询文本或查询图像")
        
        # 获取图像特征（如果有图像查询）
        query_image_features = None
        if query_image:
            query_image_features = self.memory_manager.image_encoder.encode(query_image)
        
        # 基于文本搜索获取候选结果
        if query_text:
            text_candidates = await self.memory_manager.base_manager.asearch(
                query=query_text,
                limit=50,  # 获取更多候选结果
                config=config
            )
        else:
            # 如果没有文本查询，获取所有医学记忆
            text_candidates = await self.memory_manager.base_manager.asearch(
                query="medical",
                limit=100,
                config=config
            )
        
        # 过滤和计算相似度
        scored_results = []
        for candidate in text_candidates:
            memory_type = candidate.value.get("kind", "")
            
            # 类型过滤
            if case_types and memory_type not in case_types:
                continue
            
            # 计算混合相似度
            similarity_score = self._calculate_hybrid_similarity(
                candidate,
                query_text,
                query_image_features,
                memory_type
            )
            
            if similarity_score >= self.similarity_threshold:
                result = candidate.dict()
                result["similarity_score"] = similarity_score
                result["similarity_breakdown"] = self._get_similarity_breakdown(
                    candidate, query_text, query_image_features
                )
                scored_results.append(result)
        
        # 按相似度排序
        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:limit]
    
    def _calculate_hybrid_similarity(
        self,
        candidate: dict,
        query_text: str | None,
        query_image_features: list[float] | None,
        memory_type: str
    ) -> float:
        """计算混合相似度分数
        
        Args:
            candidate: 候选记忆
            query_text: 查询文本
            query_image_features: 查询图像特征
            memory_type: 记忆类型
            
        Returns:
            混合相似度分数
        """
        text_score = 0.0
        image_score = 0.0
        
        content = candidate.value.get("content", {})
        
        # 文本相似度（基于LangGraph的搜索分数）
        if query_text and hasattr(candidate, 'score') and candidate.score is not None:
            text_score = max(0, candidate.score)  # 确保非负
        
        # 图像相似度
        if query_image_features and memory_type in ["MedicalImageMemory", "BoneAgeMemory"]:
            stored_features = content.get("image_features")
            if stored_features:
                image_score = compute_image_similarity(query_image_features, stored_features)
                image_score = max(0, image_score)  # 确保非负，余弦相似度可能为负
        
        # 计算加权平均
        if query_text and query_image_features:
            # 双模态查询
            hybrid_score = self.text_weight * text_score + self.image_weight * image_score
        elif query_text:
            # 纯文本查询
            hybrid_score = text_score
        else:
            # 纯图像查询
            hybrid_score = image_score
        
        return hybrid_score
    
    def _get_similarity_breakdown(
        self,
        candidate: dict,
        query_text: str | None,
        query_image_features: list[float] | None
    ) -> dict:
        """获取相似度详细分解
        
        Args:
            candidate: 候选记忆
            query_text: 查询文本
            query_image_features: 查询图像特征
            
        Returns:
            相似度分解详情
        """
        breakdown = {}
        
        if query_text:
            breakdown["text_similarity"] = getattr(candidate, 'score', 0.0) or 0.0
        
        if query_image_features:
            content = candidate.value.get("content", {})
            stored_features = content.get("image_features")
            if stored_features:
                breakdown["image_similarity"] = compute_image_similarity(
                    query_image_features, stored_features
                )
            else:
                breakdown["image_similarity"] = 0.0
        
        return breakdown
    
    async def retrieve_knowledge(
        self,
        topic: str,
        *,
        category: str | None = None,
        evidence_level: str | None = None,
        age_range: tuple[float, float] | None = None,
        limit: int = 5,
        config: RunnableConfig | None = None
    ) -> list[dict]:
        """检索医学知识
        
        Args:
            topic: 知识主题
            category: 知识分类过滤
            evidence_level: 证据级别过滤
            age_range: 适用年龄范围过滤
            limit: 返回数量限制
            config: 运行配置
            
        Returns:
            相关医学知识列表
        """
        # 构建查询
        query = f"medical knowledge {topic}"
        if category:
            query += f" {category}"
        
        # 搜索知识条目
        knowledge_candidates = await self.memory_manager.base_manager.asearch(
            query=query,
            filter={"kind": "MedicalKnowledge"} if not category else None,
            limit=20,
            config=config
        )
        
        # 进一步过滤
        filtered_results = []
        for candidate in knowledge_candidates:
            content = candidate.value.get("content", {})
            
            # 分类过滤
            if category and content.get("category") != category:
                continue
            
            # 证据级别过滤
            if evidence_level and content.get("evidence_level") != evidence_level:
                continue
            
            # 年龄范围过滤
            if age_range:
                applicable_range = content.get("applicable_age_range")
                if applicable_range and not self._age_ranges_overlap(age_range, applicable_range):
                    continue
            
            filtered_results.append(candidate.dict())
        
        return filtered_results[:limit]
    
    def _age_ranges_overlap(
        self,
        range1: tuple[float, float],
        range2: tuple[float, float]
    ) -> bool:
        """检查两个年龄范围是否重叠
        
        Args:
            range1: 年龄范围1
            range2: 年龄范围2
            
        Returns:
            是否重叠
        """
        return not (range1[1] < range2[0] or range2[1] < range1[0])


class MedicalRAGSystem:
    """医学检索增强生成系统
    
    结合多模态检索和语言模型的医学问答系统。
    """
    
    def __init__(
        self,
        memory_manager: MedicalMemoryManager,
        llm: BaseChatModel,
        *,
        max_context_cases: int = 5,
        max_knowledge_items: int = 3
    ):
        """初始化医学RAG系统
        
        Args:
            memory_manager: 医学记忆管理器
            llm: 语言模型
            max_context_cases: 最大上下文病例数
            max_knowledge_items: 最大知识条目数
        """
        self.memory_manager = memory_manager
        self.llm = llm
        self.retriever = MultiModalRetriever(memory_manager)
        self.max_context_cases = max_context_cases
        self.max_knowledge_items = max_knowledge_items
    
    async def answer_medical_question(
        self,
        question: str,
        *,
        image_path: str | Path | None = None,
        patient_info: dict | None = None,
        config: RunnableConfig | None = None
    ) -> dict:
        """回答医学问题
        
        Args:
            question: 医学问题
            image_path: 相关图像路径
            patient_info: 患者信息 (age, gender等)
            config: 运行配置
            
        Returns:
            问答结果
        """
        try:
            # 1. 检索相关病例
            similar_cases = await self.retriever.retrieve_similar_cases(
                query_text=question,
                query_image=image_path,
                limit=self.max_context_cases,
                config=config
            )
            
            # 2. 检索相关医学知识
            # 简单提取关键词作为知识检索主题
            topic = self._extract_topic_from_question(question)
            knowledge_items = await self.retriever.retrieve_knowledge(
                topic=topic,
                limit=self.max_knowledge_items,
                config=config
            )
            
            # 3. 构建上下文
            context = self._build_context(similar_cases, knowledge_items, patient_info)
            
            # 4. 生成回答
            prompt = self._build_medical_prompt(question, context, image_path is not None)
            response = await self.llm.ainvoke(prompt)
            
            return {
                "question": question,
                "answer": response.content,
                "context_cases": len(similar_cases),
                "knowledge_items": len(knowledge_items),
                "similar_cases": similar_cases,
                "knowledge": knowledge_items,
                "patient_info": patient_info
            }
            
        except Exception as e:
            logger.error(f"回答医学问题失败: {e}")
            raise
    
    def _extract_topic_from_question(self, question: str) -> str:
        """从问题中提取主题关键词
        
        Args:
            question: 问题文本
            
        Returns:
            主题关键词
        """
        # 简单的关键词提取，实际应用中可以使用更复杂的NLP技术
        medical_keywords = [
            "骨龄", "bone age", "发育", "生长", "骨骺", "骨化",
            "X光", "诊断", "评估", "标准", "Greulich", "TW3"
        ]
        
        question_lower = question.lower()
        for keyword in medical_keywords:
            if keyword.lower() in question_lower:
                return keyword
        
        # 默认返回骨龄相关主题
        return "bone age diagnosis"
    
    def _build_context(
        self,
        similar_cases: list[dict],
        knowledge_items: list[dict],
        patient_info: dict | None
    ) -> str:
        """构建上下文信息
        
        Args:
            similar_cases: 相似病例
            knowledge_items: 知识条目
            patient_info: 患者信息
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        # 患者信息
        if patient_info:
            context_parts.append("患者信息:")
            for key, value in patient_info.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        # 相似病例
        if similar_cases:
            context_parts.append("相似病例:")
            for i, case in enumerate(similar_cases, 1):
                content = case.get("value", {}).get("content", {})
                case_description = content.get("content", "无描述")
                similarity = case.get("similarity_score", 0.0)
                context_parts.append(f"{i}. 相似度{similarity:.2f}: {case_description}")
            context_parts.append("")
        
        # 医学知识
        if knowledge_items:
            context_parts.append("相关医学知识:")
            for i, item in enumerate(knowledge_items, 1):
                content = item.get("value", {}).get("content", {})
                title = content.get("title", "未知标题")
                knowledge_content = content.get("content", "无内容")
                source = content.get("source", "未知来源")
                context_parts.append(f"{i}. {title} (来源: {source})")
                context_parts.append(f"   {knowledge_content}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_medical_prompt(
        self,
        question: str,
        context: str,
        has_image: bool
    ) -> str:
        """构建医学问答提示词
        
        Args:
            question: 问题
            context: 上下文
            has_image: 是否有图像
            
        Returns:
            提示词
        """
        image_note = "（注意：用户提供了医学图像，请结合图像信息）" if has_image else ""
        
        prompt = f"""你是一位经验丰富的医学专家，专长于医学影像诊断，特别是骨龄评估。请基于以下上下文信息回答问题。

上下文信息:
{context}

问题: {question} {image_note}

请提供专业、准确的回答，包括：
1. 直接回答问题
2. 基于上下文中相似病例的分析
3. 相关医学知识的应用
4. 必要的注意事项和建议
5. 如果信息不足，请明确指出

回答应该专业但易于理解，避免过度技术性的术语。"""

        return prompt
    
    async def get_diagnostic_support(
        self,
        image_path: str | Path,
        patient_age: float,
        patient_gender: str,
        *,
        clinical_question: str | None = None,
        config: RunnableConfig | None = None
    ) -> dict:
        """获取诊断支持
        
        专门用于医学影像诊断的支持功能。
        
        Args:
            image_path: 医学影像路径
            patient_age: 患者年龄
            patient_gender: 患者性别
            clinical_question: 临床问题
            config: 运行配置
            
        Returns:
            诊断支持结果
        """
        patient_info = {
            "age": patient_age,
            "gender": patient_gender
        }
        
        # 构建诊断问题
        if clinical_question:
            question = f"请分析这张医学影像并回答：{clinical_question}"
        else:
            question = f"请分析这张{patient_age}岁{patient_gender}患者的医学影像，提供诊断意见"
        
        # 获取诊断支持
        return await self.answer_medical_question(
            question=question,
            image_path=image_path,
            patient_info=patient_info,
            config=config
        )