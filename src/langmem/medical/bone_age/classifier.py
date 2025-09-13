"""骨龄诊断分类器

基于图像特征和医学记忆的骨龄诊断系统。
"""

import typing
from pathlib import Path
import logging
import math

from langchain_core.runnables import RunnableConfig

from langmem.medical.manager import MedicalMemoryManager
from langmem.medical.schemas import BoneAgeMemory

logger = logging.getLogger(__name__)


class BoneAgeClassifier:
    """骨龄诊断分类器
    
    结合图像分析和历史记忆的智能骨龄诊断系统。
    """
    
    def __init__(
        self,
        memory_manager: MedicalMemoryManager,
        *,
        default_confidence_threshold: float = 0.6,
        similarity_threshold: float = 0.75
    ):
        """初始化骨龄分类器
        
        Args:
            memory_manager: 医学记忆管理器
            default_confidence_threshold: 默认置信度阈值
            similarity_threshold: 相似度阈值
        """
        self.memory_manager = memory_manager
        self.confidence_threshold = default_confidence_threshold
        self.similarity_threshold = similarity_threshold
    
    async def diagnose(
        self,
        image_path: str | Path,
        chronological_age: float,
        gender: str,
        *,
        context: str | None = None,
        config: RunnableConfig | None = None
    ) -> dict:
        """进行骨龄诊断
        
        Args:
            image_path: 手部X光片路径
            chronological_age: 实际年龄（岁）
            gender: 性别 (male/female)
            context: 额外上下文信息
            config: 运行配置
            
        Returns:
            诊断结果字典
        """
        try:
            # 1. 搜索相似的历史案例
            similar_cases = await self.memory_manager.search_similar_images(
                image_path,
                limit=5,
                similarity_threshold=self.similarity_threshold,
                config=config
            )
            
            # 2. 基于相似案例进行初步预测
            predicted_age, confidence = self._predict_from_similar_cases(
                similar_cases, chronological_age, gender
            )
            
            # 3. 获取相关的纠错经验
            corrections = await self.memory_manager.search_corrections(
                error_type="bone_age_estimation",
                limit=10,
                config=config
            )
            
            # 4. 应用纠错经验调整预测
            adjusted_age, adjusted_confidence = self._apply_corrections(
                predicted_age, confidence, corrections, chronological_age, gender
            )
            
            # 5. 生成诊断报告
            diagnosis_result = {
                "predicted_bone_age": adjusted_age,
                "chronological_age": chronological_age,
                "age_difference": adjusted_age - chronological_age,
                "confidence": adjusted_confidence,
                "gender": gender,
                "similar_cases_found": len(similar_cases),
                "corrections_applied": len(corrections),
                "assessment_method": "AI-assisted with memory learning",
                "context": context,
                "recommendations": self._generate_recommendations(
                    adjusted_age, chronological_age, adjusted_confidence
                )
            }
            
            # 6. 保存诊断记忆
            content = self._generate_diagnosis_content(diagnosis_result)
            await self.memory_manager.add_bone_age_memory(
                image_path=image_path,
                content=content,
                chronological_age=chronological_age,
                predicted_bone_age=adjusted_age,
                gender=gender,
                confidence=adjusted_confidence,
                config=config
            )
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"骨龄诊断失败: {e}")
            raise
    
    def _predict_from_similar_cases(
        self,
        similar_cases: list[dict],
        chronological_age: float,
        gender: str
    ) -> tuple[float, float]:
        """基于相似案例预测骨龄
        
        Args:
            similar_cases: 相似案例列表
            chronological_age: 实际年龄
            gender: 性别
            
        Returns:
            (预测骨龄, 置信度)
        """
        if not similar_cases:
            # 没有相似案例，使用默认估计
            return chronological_age, 0.3
        
        # 过滤同性别案例
        same_gender_cases = []
        for case in similar_cases:
            content = case.get("value", {}).get("content", {})
            if content.get("gender") == gender:
                same_gender_cases.append(case)
        
        # 如果没有同性别案例，使用所有案例但降低置信度
        cases_to_use = same_gender_cases if same_gender_cases else similar_cases
        gender_penalty = 1.0 if same_gender_cases else 0.8
        
        # 计算加权平均骨龄
        total_weight = 0
        weighted_age_sum = 0
        
        for case in cases_to_use:
            content = case.get("value", {}).get("content", {})
            similarity = case.get("similarity", 0.5)
            
            # 优先使用实际骨龄，否则使用预测骨龄
            bone_age = content.get("actual_bone_age") or content.get("predicted_bone_age")
            if bone_age:
                weight = similarity * similarity  # 相似度平方作为权重
                weighted_age_sum += bone_age * weight
                total_weight += weight
        
        if total_weight > 0:
            predicted_age = weighted_age_sum / total_weight
            # 置信度基于案例数量和相似度
            base_confidence = min(0.9, 0.4 + 0.1 * len(cases_to_use))
            confidence = base_confidence * gender_penalty
        else:
            predicted_age = chronological_age
            confidence = 0.3
        
        return predicted_age, confidence
    
    def _apply_corrections(
        self,
        predicted_age: float,
        confidence: float,
        corrections: list[dict],
        chronological_age: float,
        gender: str
    ) -> tuple[float, float]:
        """应用历史纠错经验调整预测
        
        Args:
            predicted_age: 初始预测骨龄
            confidence: 初始置信度
            corrections: 纠错记忆列表
            chronological_age: 实际年龄
            gender: 性别
            
        Returns:
            (调整后骨龄, 调整后置信度)
        """
        if not corrections:
            return predicted_age, confidence
        
        # 分析纠错模式
        age_adjustments = []
        confidence_impacts = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            error_type = content.get("error_type", "")
            
            if "bone_age" in error_type.lower():
                original_conf = content.get("original_confidence", 0.5)
                corrected_conf = content.get("corrected_confidence", 0.5)
                
                # 如果当前置信度接近历史错误案例的置信度，应该更谨慎
                if abs(confidence - original_conf) < 0.2:
                    confidence_impacts.append(corrected_conf - original_conf)
                
                # 提取学习要点中的调整建议
                learning_points = content.get("learning_points", [])
                for point in learning_points:
                    if "年龄" in point and "调整" in point:
                        # 简单的启发式规则提取
                        age_adjustments.append(self._extract_age_adjustment(point))
        
        # 应用调整
        adjusted_age = predicted_age
        adjusted_confidence = confidence
        
        if age_adjustments:
            avg_adjustment = sum(age_adjustments) / len(age_adjustments)
            adjusted_age += avg_adjustment * 0.3  # 保守应用调整
        
        if confidence_impacts:
            avg_impact = sum(confidence_impacts) / len(confidence_impacts)
            adjusted_confidence = max(0.1, min(0.95, confidence + avg_impact * 0.5))
        
        return adjusted_age, adjusted_confidence
    
    def _extract_age_adjustment(self, learning_point: str) -> float:
        """从学习要点中提取年龄调整值
        
        Args:
            learning_point: 学习要点文本
            
        Returns:
            年龄调整值
        """
        # 简单的模式匹配，实际应用中可以使用更复杂的NLP技术
        try:
            if "偏高" in learning_point:
                return -0.2
            elif "偏低" in learning_point:
                return 0.2
            elif "准确" in learning_point:
                return 0.0
        except:
            pass
        
        return 0.0
    
    def _generate_recommendations(
        self,
        predicted_age: float,
        chronological_age: float,
        confidence: float
    ) -> list[str]:
        """生成诊断建议
        
        Args:
            predicted_age: 预测骨龄
            chronological_age: 实际年龄
            confidence: 置信度
            
        Returns:
            建议列表
        """
        recommendations = []
        age_diff = predicted_age - chronological_age
        
        # 基于年龄差异的建议
        if abs(age_diff) > 2.0:
            if age_diff > 0:
                recommendations.append("骨龄明显超前，建议进一步检查内分泌功能")
            else:
                recommendations.append("骨龄明显落后，建议评估生长激素水平")
        elif abs(age_diff) > 1.0:
            recommendations.append("骨龄与实际年龄存在轻度差异，建议定期随访")
        else:
            recommendations.append("骨龄发育正常，符合实际年龄")
        
        # 基于置信度的建议
        if confidence < 0.5:
            recommendations.append("诊断置信度较低，建议结合临床表现综合判断")
        elif confidence < 0.7:
            recommendations.append("建议由专业医师进一步确认诊断结果")
        
        # 通用建议
        recommendations.append("建议结合患儿身高、体重等生长发育指标综合评估")
        
        return recommendations
    
    def _generate_diagnosis_content(self, result: dict) -> str:
        """生成诊断内容描述
        
        Args:
            result: 诊断结果
            
        Returns:
            诊断内容文本
        """
        age_diff = result["age_difference"]
        confidence = result["confidence"]
        
        if age_diff > 1.0:
            status = "超前"
        elif age_diff < -1.0:
            status = "落后"
        else:
            status = "正常"
        
        content = (
            f"骨龄诊断: {result['predicted_bone_age']:.1f}岁 "
            f"(实际年龄{result['chronological_age']:.1f}岁), "
            f"骨龄发育{status} (差异{age_diff:.1f}岁), "
            f"诊断置信度{confidence:.2f}"
        )
        
        return content
    
    async def learn_from_correction(
        self,
        image_path: str | Path,
        predicted_age: float,
        actual_age: float,
        feedback: str,
        *,
        chronological_age: float | None = None,
        gender: str | None = None,
        config: RunnableConfig | None = None
    ) -> str:
        """从纠错中学习
        
        Args:
            image_path: 图像路径
            predicted_age: 原预测骨龄
            actual_age: 实际骨龄
            feedback: 纠错反馈
            chronological_age: 实际年龄
            gender: 性别
            config: 运行配置
            
        Returns:
            纠错记忆ID
        """
        # 分析错误类型
        age_diff = abs(predicted_age - actual_age)
        if age_diff > 2.0:
            error_type = "significant_bone_age_estimation_error"
        elif age_diff > 1.0:
            error_type = "moderate_bone_age_estimation_error"
        else:
            error_type = "minor_bone_age_estimation_error"
        
        # 生成学习要点
        learning_points = [feedback]
        if predicted_age > actual_age:
            learning_points.append("预测骨龄偏高，注意骨化中心成熟度评估")
        elif predicted_age < actual_age:
            learning_points.append("预测骨龄偏低，注意细微骨化特征")
        
        # 计算置信度
        original_confidence = max(0.1, 1.0 - age_diff / 5.0)
        corrected_confidence = 0.9  # 人工纠错的置信度较高
        
        # 创建纠错描述
        content = (
            f"骨龄诊断纠错: 原预测{predicted_age:.1f}岁，"
            f"实际{actual_age:.1f}岁，差异{age_diff:.1f}岁。"
            f"纠错反馈: {feedback}"
        )
        
        # 保存纠错记忆
        return await self.memory_manager.add_correction_memory(
            image_path=image_path,
            content=content,
            original_diagnosis=f"骨龄{predicted_age:.1f}岁",
            corrected_diagnosis=f"骨龄{actual_age:.1f}岁",
            error_type=error_type,
            correction_reason=feedback,
            original_confidence=original_confidence,
            corrected_confidence=corrected_confidence,
            learning_points=learning_points,
            config=config
        )
    
    async def get_diagnosis_history(
        self,
        *,
        limit: int = 20,
        config: RunnableConfig | None = None
    ) -> list[dict]:
        """获取诊断历史
        
        Args:
            limit: 返回数量限制
            config: 运行配置
            
        Returns:
            诊断历史列表
        """
        memories = await self.memory_manager.base_manager.asearch(
            query="bone age diagnosis",
            filter={"kind": "BoneAgeMemory"},
            limit=limit,
            config=config
        )
        
        return [memory.dict() for memory in memories]
    
    async def get_performance_metrics(
        self,
        *,
        config: RunnableConfig | None = None
    ) -> dict:
        """获取诊断性能指标
        
        Args:
            config: 运行配置
            
        Returns:
            性能指标字典
        """
        # 获取所有骨龄记忆
        memories = await self.get_diagnosis_history(limit=100, config=config)
        
        if not memories:
            return {"message": "暂无诊断历史数据"}
        
        # 计算性能指标
        total_cases = len(memories)
        accurate_cases = 0
        total_error = 0
        confidence_sum = 0
        
        for memory in memories:
            content = memory.get("value", {}).get("content", {})
            predicted = content.get("predicted_bone_age")
            actual = content.get("actual_bone_age")
            confidence = content.get("confidence", 0)
            
            confidence_sum += confidence
            
            if predicted is not None and actual is not None:
                error = abs(predicted - actual)
                total_error += error
                
                # 定义准确性标准（误差<1岁认为准确）
                if error < 1.0:
                    accurate_cases += 1
        
        accuracy_rate = accurate_cases / total_cases if total_cases > 0 else 0
        avg_confidence = confidence_sum / total_cases if total_cases > 0 else 0
        avg_error = total_error / total_cases if total_cases > 0 else 0
        
        return {
            "total_cases": total_cases,
            "accuracy_rate": accuracy_rate,
            "average_error": avg_error,
            "average_confidence": avg_confidence,
            "performance_grade": self._calculate_performance_grade(accuracy_rate, avg_error)
        }
    
    def _calculate_performance_grade(self, accuracy_rate: float, avg_error: float) -> str:
        """计算性能等级
        
        Args:
            accuracy_rate: 准确率
            avg_error: 平均误差
            
        Returns:
            性能等级
        """
        if accuracy_rate >= 0.9 and avg_error <= 0.5:
            return "优秀"
        elif accuracy_rate >= 0.8 and avg_error <= 1.0:
            return "良好"
        elif accuracy_rate >= 0.7 and avg_error <= 1.5:
            return "中等"
        else:
            return "需要改进"