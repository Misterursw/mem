"""纠错学习机制模块

实现从诊断纠错中持续学习的机制。
"""

import typing
from pathlib import Path
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from langchain_core.runnables import RunnableConfig

from langmem.medical.manager import MedicalMemoryManager

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """错误分析器
    
    分析诊断错误模式，提供学习洞察。
    """
    
    def __init__(self, memory_manager: MedicalMemoryManager):
        """初始化错误分析器
        
        Args:
            memory_manager: 医学记忆管理器
        """
        self.memory_manager = memory_manager
    
    async def analyze_error_patterns(
        self,
        *,
        time_window_days: int = 30,
        config: RunnableConfig | None = None
    ) -> dict:
        """分析错误模式
        
        Args:
            time_window_days: 分析时间窗口（天数）
            config: 运行配置
            
        Returns:
            错误模式分析结果
        """
        # 获取纠错记忆
        corrections = await self.memory_manager.search_corrections(config=config)
        
        if not corrections:
            return {"message": "暂无纠错数据"}
        
        # 过滤时间窗口内的数据
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_corrections = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            timestamp_str = content.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp >= cutoff_date:
                        recent_corrections.append(correction)
                except:
                    # 如果时间解析失败，包含在分析中
                    recent_corrections.append(correction)
        
        # 分析错误模式
        error_analysis = {
            "total_corrections": len(recent_corrections),
            "error_types": self._analyze_error_types(recent_corrections),
            "age_groups": self._analyze_age_group_errors(recent_corrections),
            "confidence_patterns": self._analyze_confidence_patterns(recent_corrections),
            "learning_trends": self._analyze_learning_trends(recent_corrections),
            "recommendations": self._generate_improvement_recommendations(recent_corrections)
        }
        
        return error_analysis
    
    def _analyze_error_types(self, corrections: list[dict]) -> dict:
        """分析错误类型分布
        
        Args:
            corrections: 纠错记忆列表
            
        Returns:
            错误类型分析
        """
        error_types = Counter()
        error_details = defaultdict(list)
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            error_type = content.get("error_type", "unknown")
            correction_reason = content.get("correction_reason", "")
            
            error_types[error_type] += 1
            error_details[error_type].append({
                "reason": correction_reason,
                "original_confidence": content.get("original_confidence", 0),
                "corrected_confidence": content.get("corrected_confidence", 0)
            })
        
        return {
            "distribution": dict(error_types),
            "most_common": error_types.most_common(3),
            "details": dict(error_details)
        }
    
    def _analyze_age_group_errors(self, corrections: list[dict]) -> dict:
        """分析不同年龄组的错误模式
        
        Args:
            corrections: 纠错记忆列表
            
        Returns:
            年龄组错误分析
        """
        age_groups = {
            "infant": (0, 2),      # 婴儿期
            "toddler": (2, 5),     # 幼儿期
            "child": (5, 12),      # 儿童期
            "adolescent": (12, 18) # 青少年期
        }
        
        age_group_errors = defaultdict(list)
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            # 尝试从纠错内容中提取年龄信息
            age = self._extract_age_from_correction(content)
            
            if age is not None:
                for group_name, (min_age, max_age) in age_groups.items():
                    if min_age <= age < max_age:
                        age_group_errors[group_name].append({
                            "age": age,
                            "error_type": content.get("error_type"),
                            "correction_reason": content.get("correction_reason")
                        })
                        break
        
        # 统计每个年龄组的错误频率
        age_group_stats = {}
        for group, errors in age_group_errors.items():
            age_group_stats[group] = {
                "error_count": len(errors),
                "common_error_types": Counter(e["error_type"] for e in errors if e["error_type"]).most_common(2)
            }
        
        return age_group_stats
    
    def _extract_age_from_correction(self, content: dict) -> float | None:
        """从纠错内容中提取年龄信息
        
        Args:
            content: 纠错内容
            
        Returns:
            提取的年龄，如果无法提取则返回None
        """
        # 简单的年龄提取逻辑，实际应用中可以更复杂
        correction_text = content.get("content", "") + " " + content.get("correction_reason", "")
        
        import re
        # 查找年龄模式
        age_patterns = [
            r"(\d+(?:\.\d+)?)\s*岁",
            r"(\d+(?:\.\d+)?)\s*years?\s*old",
            r"age\s*(\d+(?:\.\d+)?)"
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, correction_text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def _analyze_confidence_patterns(self, corrections: list[dict]) -> dict:
        """分析置信度模式
        
        Args:
            corrections: 纠错记忆列表
            
        Returns:
            置信度模式分析
        """
        confidence_data = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            original_conf = content.get("original_confidence")
            corrected_conf = content.get("corrected_confidence")
            
            if original_conf is not None and corrected_conf is not None:
                confidence_data.append({
                    "original": original_conf,
                    "corrected": corrected_conf,
                    "improvement": corrected_conf - original_conf,
                    "error_type": content.get("error_type")
                })
        
        if not confidence_data:
            return {"message": "无置信度数据"}
        
        # 分析置信度模式
        avg_original = sum(d["original"] for d in confidence_data) / len(confidence_data)
        avg_corrected = sum(d["corrected"] for d in confidence_data) / len(confidence_data)
        avg_improvement = sum(d["improvement"] for d in confidence_data) / len(confidence_data)
        
        # 分析过度自信的情况（原始置信度高但被纠错）
        overconfident_cases = [d for d in confidence_data if d["original"] > 0.8 and d["improvement"] < 0]
        
        return {
            "average_original_confidence": avg_original,
            "average_corrected_confidence": avg_corrected,
            "average_improvement": avg_improvement,
            "overconfident_cases": len(overconfident_cases),
            "confidence_calibration_needed": avg_original > avg_corrected + 0.1
        }
    
    def _analyze_learning_trends(self, corrections: list[dict]) -> dict:
        """分析学习趋势
        
        Args:
            corrections: 纠错记忆列表
            
        Returns:
            学习趋势分析
        """
        # 按时间排序纠错记忆
        dated_corrections = []
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            timestamp_str = content.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    dated_corrections.append((timestamp, content))
                except:
                    continue
        
        dated_corrections.sort(key=lambda x: x[0])
        
        if len(dated_corrections) < 2:
            return {"message": "数据不足以分析趋势"}
        
        # 分析错误频率趋势
        recent_half = dated_corrections[len(dated_corrections)//2:]
        early_half = dated_corrections[:len(dated_corrections)//2]
        
        recent_error_types = Counter(content["error_type"] for _, content in recent_half if content.get("error_type"))
        early_error_types = Counter(content["error_type"] for _, content in early_half if content.get("error_type"))
        
        # 分析改进情况
        improved_error_types = []
        worsened_error_types = []
        
        for error_type in set(list(recent_error_types.keys()) + list(early_error_types.keys())):
            recent_count = recent_error_types.get(error_type, 0)
            early_count = early_error_types.get(error_type, 0)
            
            # 考虑时间窗口长度差异
            recent_rate = recent_count / max(1, len(recent_half))
            early_rate = early_count / max(1, len(early_half))
            
            if recent_rate < early_rate * 0.8:  # 显著改善
                improved_error_types.append(error_type)
            elif recent_rate > early_rate * 1.2:  # 显著恶化
                worsened_error_types.append(error_type)
        
        return {
            "total_timespan_days": (dated_corrections[-1][0] - dated_corrections[0][0]).days,
            "improved_error_types": improved_error_types,
            "worsened_error_types": worsened_error_types,
            "recent_correction_rate": len(recent_half) / max(1, len(dated_corrections)),
            "learning_direction": "improving" if len(improved_error_types) > len(worsened_error_types) else "declining"
        }
    
    def _generate_improvement_recommendations(self, corrections: list[dict]) -> list[str]:
        """生成改进建议
        
        Args:
            corrections: 纠错记忆列表
            
        Returns:
            改进建议列表
        """
        recommendations = []
        
        # 基于错误类型分布的建议
        error_types = Counter()
        learning_points = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            error_type = content.get("error_type", "")
            points = content.get("learning_points", [])
            
            error_types[error_type] += 1
            learning_points.extend(points)
        
        # 针对最常见的错误类型给出建议
        if error_types:
            most_common_error = error_types.most_common(1)[0]
            error_type, count = most_common_error
            
            if "significant" in error_type:
                recommendations.append(f"需要加强基础诊断能力训练，{error_type}出现了{count}次")
            elif "moderate" in error_type:
                recommendations.append(f"建议提高诊断精度，关注细微特征识别")
            
            if count > len(corrections) * 0.3:  # 超过30%的错误都是同一类型
                recommendations.append("错误集中在特定类型，建议针对性专项训练")
        
        # 基于学习要点的建议
        if learning_points:
            point_frequency = Counter(learning_points)
            top_points = point_frequency.most_common(3)
            
            recommendations.append("重点关注以下学习要点:")
            for point, freq in top_points:
                recommendations.append(f"- {point} (出现{freq}次)")
        
        # 通用建议
        if len(corrections) > 10:
            recommendations.append("建议定期回顾纠错案例，总结经验教训")
            recommendations.append("考虑建立个人错误知识库，便于快速查询")
        
        return recommendations


class AdaptiveLearningSystem:
    """自适应学习系统
    
    基于纠错历史动态调整诊断策略。
    """
    
    def __init__(
        self,
        memory_manager: MedicalMemoryManager,
        *,
        learning_rate: float = 0.1,
        confidence_adjustment_factor: float = 0.05
    ):
        """初始化自适应学习系统
        
        Args:
            memory_manager: 医学记忆管理器
            learning_rate: 学习率
            confidence_adjustment_factor: 置信度调整因子
        """
        self.memory_manager = memory_manager
        self.learning_rate = learning_rate
        self.confidence_adjustment_factor = confidence_adjustment_factor
        self.error_analyzer = ErrorAnalyzer(memory_manager)
    
    async def adjust_diagnosis_confidence(
        self,
        predicted_age: float,
        initial_confidence: float,
        patient_age: float,
        gender: str,
        *,
        config: RunnableConfig | None = None
    ) -> tuple[float, str]:
        """基于历史错误调整诊断置信度
        
        Args:
            predicted_age: 预测骨龄
            initial_confidence: 初始置信度
            patient_age: 患者实际年龄
            gender: 性别
            config: 运行配置
            
        Returns:
            (调整后置信度, 调整说明)
        """
        # 获取相关的纠错历史
        corrections = await self.memory_manager.search_corrections(config=config)
        
        if not corrections:
            return initial_confidence, "无历史纠错数据，使用初始置信度"
        
        # 分析相似情况的历史错误
        age_diff = predicted_age - patient_age
        relevant_corrections = []
        
        for correction in corrections:
            content = correction.get("value", {}).get("content", {})
            
            # 提取历史案例的年龄差异信息
            historical_age_diff = self._extract_age_difference_from_correction(content)
            
            if (historical_age_diff is not None and 
                abs(historical_age_diff - age_diff) < 1.0 and  # 年龄差异相似
                content.get("original_confidence", 0) > 0):
                
                relevant_corrections.append(content)
        
        if not relevant_corrections:
            return initial_confidence, "无相似情况的纠错历史"
        
        # 计算置信度调整
        confidence_adjustments = []
        for correction in relevant_corrections:
            original_conf = correction.get("original_confidence", 0.5)
            corrected_conf = correction.get("corrected_confidence", 0.5)
            
            # 如果当前置信度与历史错误案例相似，应该降低置信度
            if abs(initial_confidence - original_conf) < 0.2:
                adjustment = (corrected_conf - original_conf) * self.confidence_adjustment_factor
                confidence_adjustments.append(adjustment)
        
        if confidence_adjustments:
            avg_adjustment = sum(confidence_adjustments) / len(confidence_adjustments)
            adjusted_confidence = max(0.1, min(0.95, initial_confidence + avg_adjustment))
            
            explanation = f"基于{len(confidence_adjustments)}个相似错误案例调整置信度 {avg_adjustment:+.3f}"
            return adjusted_confidence, explanation
        
        return initial_confidence, "相似情况的纠错历史无明显模式"
    
    def _extract_age_difference_from_correction(self, content: dict) -> float | None:
        """从纠错内容中提取年龄差异信息
        
        Args:
            content: 纠错内容
            
        Returns:
            年龄差异，如果无法提取则返回None
        """
        # 从纠错描述中提取预测年龄和实际年龄
        text = content.get("content", "") + " " + content.get("original_diagnosis", "") + " " + content.get("corrected_diagnosis", "")
        
        import re
        
        # 查找预测和实际年龄
        predicted_matches = re.findall(r"预测.{0,10}?(\d+(?:\.\d+)?)\s*岁", text)
        actual_matches = re.findall(r"实际.{0,10}?(\d+(?:\.\d+)?)\s*岁", text)
        
        if predicted_matches and actual_matches:
            try:
                predicted = float(predicted_matches[0])
                actual = float(actual_matches[0])
                return predicted - actual
            except ValueError:
                pass
        
        # 查找差异描述
        diff_matches = re.findall(r"差异.{0,5}?([+-]?\d+(?:\.\d+)?)\s*岁", text)
        if diff_matches:
            try:
                return float(diff_matches[0])
            except ValueError:
                pass
        
        return None
    
    async def get_learning_progress_report(
        self,
        *,
        config: RunnableConfig | None = None
    ) -> dict:
        """生成学习进度报告
        
        Args:
            config: 运行配置
            
        Returns:
            学习进度报告
        """
        # 分析不同时间段的错误模式
        error_analysis = await self.error_analyzer.analyze_error_patterns(
            time_window_days=30,
            config=config
        )
        
        # 获取总体性能指标
        from langmem.medical.bone_age.classifier import BoneAgeClassifier
        classifier = BoneAgeClassifier(self.memory_manager)
        performance_metrics = await classifier.get_performance_metrics(config=config)
        
        # 生成学习洞察
        insights = await self.memory_manager.get_learning_insights(config=config)
        
        return {
            "report_date": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "error_analysis": error_analysis,
            "learning_insights": insights,
            "recommendations": self._generate_learning_plan(error_analysis, performance_metrics)
        }
    
    def _generate_learning_plan(
        self,
        error_analysis: dict,
        performance_metrics: dict
    ) -> list[str]:
        """生成学习计划
        
        Args:
            error_analysis: 错误分析结果
            performance_metrics: 性能指标
            
        Returns:
            学习计划建议
        """
        plan = []
        
        # 基于性能等级的建议
        grade = performance_metrics.get("performance_grade", "需要改进")
        if grade == "需要改进":
            plan.append("优先级1: 基础诊断能力提升 - 系统学习骨龄评估标准")
            plan.append("优先级2: 加强图像识别训练 - 重点练习骨化中心识别")
        elif grade == "中等":
            plan.append("优先级1: 提高诊断精度 - 关注边界案例和细微特征")
            plan.append("优先级2: 减少系统性偏差 - 分析常见错误模式")
        elif grade in ["良好", "优秀"]:
            plan.append("优先级1: 维护当前水平 - 定期回顾复杂案例")
            plan.append("优先级2: 探索高级技术 - 多标准对比分析")
        
        # 基于错误分析的具体建议
        error_types = error_analysis.get("error_types", {})
        most_common_errors = error_types.get("most_common", [])
        
        if most_common_errors:
            error_type, count = most_common_errors[0]
            if "significant" in error_type:
                plan.append(f"专项训练: 解决{error_type}问题 - 建议每日练习10个相关案例")
            elif "moderate" in error_type:
                plan.append(f"精度提升: 针对{error_type}加强训练")
        
        # 基于学习趋势的建议
        trends = error_analysis.get("learning_trends", {})
        learning_direction = trends.get("learning_direction", "unknown")
        
        if learning_direction == "declining":
            plan.append("警告: 学习效果呈下降趋势 - 建议调整学习策略")
            plan.append("建议: 增加练习频率，寻求专家指导")
        elif learning_direction == "improving":
            plan.append("积极: 学习效果良好 - 保持当前学习节奏")
        
        return plan