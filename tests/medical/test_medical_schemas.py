"""
测试医学模块的Schema定义
"""

import pytest
from datetime import datetime
from pathlib import Path

from langmem.medical.schemas import (
    MedicalImageMemory,
    BoneAgeMemory,
    DiagnosisCorrection,
    MedicalKnowledge
)


class TestMedicalImageMemory:
    """测试医学图像记忆Schema"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        memory = MedicalImageMemory(
            content="8岁男孩手部X光片，骨龄发育正常",
            image_path="/path/to/image.jpg"
        )
        
        assert memory.content == "8岁男孩手部X光片，骨龄发育正常"
        assert memory.image_path == "/path/to/image.jpg"
        assert memory.image_type == "unknown"  # 默认值
        assert memory.body_part == "unknown"   # 默认值
        assert isinstance(memory.timestamp, datetime)
    
    def test_with_all_fields(self):
        """测试包含所有字段的创建"""
        features = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        memory = MedicalImageMemory(
            content="详细的医学图像描述",
            image_path="/medical/images/hand_xray.jpg",
            image_features=features,
            image_type="X-ray",
            body_part="hand",
            patient_age=8.5,
            patient_gender="male",
            diagnosis="骨龄8.2岁",
            confidence=0.85
        )
        
        assert memory.image_features == features
        assert memory.image_type == "X-ray"
        assert memory.body_part == "hand"
        assert memory.patient_age == 8.5
        assert memory.patient_gender == "male"
        assert memory.diagnosis == "骨龄8.2岁"
        assert memory.confidence == 0.85
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        memory = MedicalImageMemory(
            content="测试内容",
            image_path="/test/path.jpg",
            patient_age=10.0
        )
        
        # 测试model_dump
        data = memory.model_dump()
        assert "content" in data
        assert "image_path" in data
        assert "timestamp" in data
        
        # 测试从字典重建
        new_memory = MedicalImageMemory(**data)
        assert new_memory.content == memory.content
        assert new_memory.image_path == memory.image_path


class TestBoneAgeMemory:
    """测试骨龄记忆Schema"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        memory = BoneAgeMemory(
            content="8岁男孩骨龄诊断",
            image_path="/images/hand.jpg",
            chronological_age=8.0,
            predicted_bone_age=8.2,
            gender="male",
            confidence=0.75
        )
        
        assert memory.chronological_age == 8.0
        assert memory.predicted_bone_age == 8.2
        assert memory.gender == "male"
        assert memory.confidence == 0.75
        assert memory.assessment_method == "Greulich-Pyle"  # 默认值
    
    def test_age_difference_calculation(self):
        """测试年龄差异计算"""
        # 测试使用实际骨龄
        memory = BoneAgeMemory(
            content="测试",
            image_path="/test.jpg",
            chronological_age=8.0,
            predicted_bone_age=8.5,
            actual_bone_age=8.2,
            gender="male",
            confidence=0.8
        )
        
        # 应该使用actual_bone_age - chronological_age
        assert memory.calculate_age_difference() == 0.2
        
        # 测试没有实际骨龄的情况
        memory.actual_bone_age = None
        # 应该使用predicted_bone_age - chronological_age
        assert memory.calculate_age_difference() == 0.5
    
    def test_with_assessment_details(self):
        """测试包含评估详情的创建"""
        key_features = ["腕骨出现", "骨骺愈合", "掌骨发育"]
        
        memory = BoneAgeMemory(
            content="详细骨龄评估",
            image_path="/detailed/hand.jpg",
            chronological_age=12.0,
            predicted_bone_age=12.5,
            gender="female",
            assessment_method="TW3",
            key_features=key_features,
            confidence=0.9,
            error_analysis="预测稍微偏高"
        )
        
        assert memory.assessment_method == "TW3"
        assert memory.key_features == key_features
        assert memory.error_analysis == "预测稍微偏高"


class TestDiagnosisCorrection:
    """测试诊断纠错Schema"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        correction = DiagnosisCorrection(
            content="骨龄诊断纠错",
            original_diagnosis="骨龄8.5岁",
            corrected_diagnosis="骨龄8.2岁",
            image_path="/corrections/case1.jpg",
            error_type="moderate_estimation_error",
            correction_reason="掌骨骨骺发育评估偏高",
            original_confidence=0.8,
            corrected_confidence=0.9
        )
        
        assert correction.original_diagnosis == "骨龄8.5岁"
        assert correction.corrected_diagnosis == "骨龄8.2岁"
        assert correction.error_type == "moderate_estimation_error"
        assert correction.correction_reason == "掌骨骨骺发育评估偏高"
        assert correction.original_confidence == 0.8
        assert correction.corrected_confidence == 0.9
        assert correction.corrector_type == "human_expert"  # 默认值
    
    def test_with_learning_points(self):
        """测试包含学习要点的创建"""
        learning_points = [
            "注意骨骺成熟度细节",
            "参考多个标准",
            "考虑个体差异"
        ]
        
        correction = DiagnosisCorrection(
            content="学习型纠错",
            original_diagnosis="错误诊断",
            corrected_diagnosis="正确诊断",
            image_path="/learning/case.jpg",
            error_type="learning_error",
            correction_reason="学习过程",
            original_confidence=0.6,
            corrected_confidence=0.9,
            learning_points=learning_points,
            corrector_type="peer_review"
        )
        
        assert correction.learning_points == learning_points
        assert correction.corrector_type == "peer_review"


class TestMedicalKnowledge:
    """测试医学知识Schema"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        knowledge = MedicalKnowledge(
            content="Greulich-Pyle标准是基于手腕X光片的骨龄评估方法",
            title="Greulich-Pyle骨龄标准",
            category="诊断标准",
            source="医学文献"
        )
        
        assert knowledge.title == "Greulich-Pyle骨龄标准"
        assert knowledge.category == "诊断标准"
        assert knowledge.source == "医学文献"
        assert knowledge.evidence_level == "unknown"  # 默认值
    
    def test_with_detailed_info(self):
        """测试包含详细信息的创建"""
        tags = ["骨龄", "儿科", "放射学"]
        
        knowledge = MedicalKnowledge(
            content="详细的医学知识内容",
            title="骨龄评估指南",
            category="临床指南",
            subcategory="儿科放射学",
            source="权威医学期刊",
            evidence_level="A",
            applicable_age_range=(0.0, 18.0),
            applicable_gender="both",
            tags=tags
        )
        
        assert knowledge.subcategory == "儿科放射学"
        assert knowledge.evidence_level == "A"
        assert knowledge.applicable_age_range == (0.0, 18.0)
        assert knowledge.applicable_gender == "both"
        assert knowledge.tags == tags
    
    def test_age_range_validation(self):
        """测试年龄范围验证"""
        knowledge = MedicalKnowledge(
            content="测试内容",
            title="测试标题",
            category="测试分类",
            source="测试来源",
            applicable_age_range=(5.0, 15.0)
        )
        
        assert knowledge.applicable_age_range == (5.0, 15.0)


# 集成测试
class TestSchemaIntegration:
    """测试Schema之间的集成"""
    
    def test_memory_workflow(self):
        """测试完整的记忆工作流程"""
        # 1. 创建初始图像记忆
        image_memory = MedicalImageMemory(
            content="初始诊断",
            image_path="/workflow/test.jpg",
            diagnosis="骨龄8.5岁",
            confidence=0.7
        )
        
        # 2. 创建骨龄记忆
        bone_age_memory = BoneAgeMemory(
            content="骨龄详细评估",
            image_path=image_memory.image_path,  # 使用相同图像
            chronological_age=8.0,
            predicted_bone_age=8.5,
            gender="male",
            confidence=0.75
        )
        
        # 3. 创建纠错记忆
        correction_memory = DiagnosisCorrection(
            content="诊断纠错",
            original_diagnosis="骨龄8.5岁",
            corrected_diagnosis="骨龄8.2岁",
            image_path=image_memory.image_path,  # 使用相同图像
            error_type="moderate_error",
            correction_reason="重新评估后调整",
            original_confidence=0.75,
            corrected_confidence=0.9
        )
        
        # 验证数据一致性
        assert image_memory.image_path == bone_age_memory.image_path
        assert bone_age_memory.image_path == correction_memory.image_path
        assert bone_age_memory.confidence == correction_memory.original_confidence
    
    def test_json_roundtrip(self):
        """测试JSON序列化往返"""
        memories = [
            MedicalImageMemory(
                content="图像记忆",
                image_path="/test.jpg",
                patient_age=10.0
            ),
            BoneAgeMemory(
                content="骨龄记忆",
                image_path="/test.jpg",
                chronological_age=10.0,
                predicted_bone_age=10.2,
                gender="female",
                confidence=0.8
            ),
            DiagnosisCorrection(
                content="纠错记忆",
                original_diagnosis="原诊断",
                corrected_diagnosis="新诊断",
                image_path="/test.jpg",
                error_type="test_error",
                correction_reason="测试原因",
                original_confidence=0.6,
                corrected_confidence=0.9
            )
        ]
        
        # 测试每个schema的序列化往返
        for memory in memories:
            data = memory.model_dump()
            reconstructed = type(memory)(**data)
            
            # 验证关键字段
            assert reconstructed.content == memory.content
            assert reconstructed.image_path == memory.image_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])