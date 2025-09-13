"""骨龄诊断知识库

预置的骨龄诊断标准、指南和医学知识。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BoneAgeStandard:
    """骨龄标准定义"""
    name: str
    description: str
    age_range: Tuple[float, float]
    gender_specific: bool
    key_features: List[str]
    accuracy_notes: str


class BoneAgeKnowledge:
    """骨龄诊断知识库"""
    
    def __init__(self):
        self.standards = self._load_standards()
        self.developmental_markers = self._load_developmental_markers()
        self.diagnostic_tips = self._load_diagnostic_tips()
    
    def _load_standards(self) -> Dict[str, BoneAgeStandard]:
        """加载骨龄诊断标准"""
        return {
            "greulich_pyle": BoneAgeStandard(
                name="Greulich-Pyle标准",
                description="基于手腕X光片的骨龄评估标准，使用标准图谱进行比较",
                age_range=(0, 18),
                gender_specific=True,
                key_features=[
                    "桡骨远端骨化中心",
                    "尺骨远端骨化中心", 
                    "腕骨骨化中心数量和形态",
                    "掌骨骨骺出现和愈合",
                    "指骨骨骺发育程度"
                ],
                accuracy_notes="在2-16岁年龄段准确性较高，需要考虑种族差异"
            ),
            
            "tw3": BoneAgeStandard(
                name="TW3标准",
                description="Tanner-Whitehouse第3版标准，基于评分系统",
                age_range=(2, 16),
                gender_specific=True,
                key_features=[
                    "20个骨化中心的成熟度评分",
                    "桡骨、尺骨、掌骨、指骨的分别评估",
                    "腕骨的标准化评分",
                    "骨骺愈合程度量化"
                ],
                accuracy_notes="评分系统更客观，但需要专业训练"
            ),
            
            "chinese_standard": BoneAgeStandard(
                name="中国人骨龄标准",
                description="基于中国儿童青少年的骨龄发育标准",
                age_range=(0, 18),
                gender_specific=True,
                key_features=[
                    "适应中国人群的发育特点",
                    "考虑营养状况影响",
                    "地域差异的修正因子",
                    "现代儿童发育加速趋势"
                ],
                accuracy_notes="更适合中国儿童，需要考虑地区和社会经济因素"
            )
        }
    
    def _load_developmental_markers(self) -> Dict[str, Dict]:
        """加载骨发育标志点"""
        return {
            "carpal_bones": {
                "capitate": {
                    "appearance_age": {"male": 0.2, "female": 0.2},
                    "description": "头状骨，最早出现的腕骨"
                },
                "hamate": {
                    "appearance_age": {"male": 0.3, "female": 0.3},
                    "description": "钩骨，第二个出现的腕骨"
                },
                "triquetral": {
                    "appearance_age": {"male": 2.5, "female": 2.0},
                    "description": "三角骨"
                },
                "lunate": {
                    "appearance_age": {"male": 4.0, "female": 3.5},
                    "description": "月骨"
                },
                "scaphoid": {
                    "appearance_age": {"male": 5.5, "female": 4.5},
                    "description": "舟骨"
                },
                "trapezium": {
                    "appearance_age": {"male": 6.0, "female": 5.0},
                    "description": "大多角骨"
                },
                "trapezoid": {
                    "appearance_age": {"male": 6.5, "female": 5.5},
                    "description": "小多角骨"
                },
                "pisiform": {
                    "appearance_age": {"male": 10.0, "female": 8.5},
                    "description": "豌豆骨，最后出现的腕骨"
                }
            },
            
            "metacarpals": {
                "mc1_epiphysis": {
                    "appearance_age": {"male": 2.5, "female": 2.0},
                    "fusion_age": {"male": 16.0, "female": 14.5},
                    "description": "第1掌骨骨骺"
                },
                "mc2_epiphysis": {
                    "appearance_age": {"male": 3.0, "female": 2.5},
                    "fusion_age": {"male": 17.0, "female": 15.5},
                    "description": "第2掌骨骨骺"
                },
                "mc3_epiphysis": {
                    "appearance_age": {"male": 3.0, "female": 2.5},
                    "fusion_age": {"male": 17.0, "female": 15.5},
                    "description": "第3掌骨骨骺"
                },
                "mc4_epiphysis": {
                    "appearance_age": {"male": 3.5, "female": 3.0},
                    "fusion_age": {"male": 17.0, "female": 15.5},
                    "description": "第4掌骨骨骺"
                },
                "mc5_epiphysis": {
                    "appearance_age": {"male": 3.5, "female": 3.0},
                    "fusion_age": {"male": 16.5, "female": 15.0},
                    "description": "第5掌骨骨骺"
                }
            },
            
            "phalanges": {
                "proximal_phalanx": {
                    "appearance_age": {"male": 2.0, "female": 1.5},
                    "fusion_age": {"male": 16.5, "female": 15.0},
                    "description": "近节指骨骨骺"
                },
                "middle_phalanx": {
                    "appearance_age": {"male": 2.5, "female": 2.0},
                    "fusion_age": {"male": 16.0, "female": 14.5},
                    "description": "中节指骨骨骺"
                },
                "distal_phalanx": {
                    "appearance_age": {"male": 3.0, "female": 2.5},
                    "fusion_age": {"male": 15.5, "female": 14.0},
                    "description": "远节指骨骨骺"
                }
            },
            
            "radius_ulna": {
                "radius_distal": {
                    "appearance_age": {"male": 1.5, "female": 1.0},
                    "fusion_age": {"male": 17.5, "female": 16.0},
                    "description": "桡骨远端骨骺"
                },
                "ulna_distal": {
                    "appearance_age": {"male": 6.0, "female": 5.5},
                    "fusion_age": {"male": 17.5, "female": 16.0},
                    "description": "尺骨远端骨骺"
                }
            }
        }
    
    def _load_diagnostic_tips(self) -> List[Dict]:
        """加载诊断要点和常见错误"""
        return [
            {
                "category": "图像质量",
                "tips": [
                    "确保手部完全伸展，手指分开",
                    "手腕应该平放，不要弯曲",
                    "X光片对比度和清晰度要足够",
                    "避免重叠和遮挡"
                ],
                "common_errors": [
                    "手指弯曲导致骨骺重叠",
                    "手腕弯曲影响腕骨显示",
                    "曝光不足或过度"
                ]
            },
            
            {
                "category": "年龄评估",
                "tips": [
                    "优先观察腕骨的出现顺序",
                    "注意骨骺的成熟度而非仅仅是出现",
                    "考虑骨骺融合的时间点",
                    "综合评估所有骨化中心"
                ],
                "common_errors": [
                    "仅依赖单一骨化中心判断",
                    "忽视骨骺成熟度的细微变化",
                    "没有考虑性别差异",
                    "过度依赖单一标准"
                ]
            },
            
            {
                "category": "性别差异",
                "tips": [
                    "女性骨龄发育通常提前1-2年",
                    "青春期女性骨骺愈合更早",
                    "注意性激素对骨龄的影响",
                    "考虑青春期发育的个体差异"
                ],
                "common_errors": [
                    "没有应用性别特异性标准",
                    "忽视青春期性激素的影响",
                    "混用不同性别的参考标准"
                ]
            },
            
            {
                "category": "病理状态",
                "tips": [
                    "内分泌疾病会显著影响骨龄",
                    "营养不良可导致骨龄延迟",
                    "某些药物会影响骨发育",
                    "遗传因素需要考虑"
                ],
                "common_errors": [
                    "没有结合临床病史",
                    "忽视药物对骨发育的影响",
                    "单纯以骨龄诊断疾病"
                ]
            }
        ]
    
    def get_expected_bone_age(self, chronological_age: float, gender: str) -> Dict:
        """获取预期骨龄范围
        
        Args:
            chronological_age: 实际年龄
            gender: 性别
            
        Returns:
            预期骨龄信息
        """
        # 正常范围通常为实际年龄±1年
        normal_range = (chronological_age - 1.0, chronological_age + 1.0)
        
        # 性别调整
        if gender == "female":
            # 女性骨龄通常稍微提前
            adjustment = 0.2 if chronological_age > 8 else 0
        else:
            adjustment = 0
        
        expected_age = chronological_age + adjustment
        
        return {
            "expected_bone_age": expected_age,
            "normal_range": normal_range,
            "gender_adjustment": adjustment,
            "interpretation": self._interpret_bone_age_range(chronological_age, expected_age)
        }
    
    def _interpret_bone_age_range(self, chronological_age: float, bone_age: float) -> str:
        """解释骨龄与实际年龄的关系"""
        diff = bone_age - chronological_age
        
        if abs(diff) <= 1.0:
            return "骨龄发育正常，与实际年龄相符"
        elif diff > 2.0:
            return "骨龄明显超前，建议检查内分泌功能"
        elif diff > 1.0:
            return "骨龄轻度超前，建议定期观察"
        elif diff < -2.0:
            return "骨龄明显落后，建议评估生长激素水平"
        else:
            return "骨龄轻度落后，建议结合身高体重评估"
    
    def get_diagnostic_features_for_age(self, target_age: float, gender: str) -> List[str]:
        """获取特定年龄应该观察的诊断特征
        
        Args:
            target_age: 目标年龄
            gender: 性别
            
        Returns:
            应该观察的特征列表
        """
        features = []
        
        # 遍历所有发育标志点
        for bone_group, bones in self.developmental_markers.items():
            for bone_name, bone_info in bones.items():
                appearance_age = bone_info.get("appearance_age", {}).get(gender, 999)
                fusion_age = bone_info.get("fusion_age", {}).get(gender, 999)
                
                # 检查是否在相关年龄范围内
                if appearance_age <= target_age <= fusion_age:
                    description = bone_info.get("description", bone_name)
                    
                    if target_age - appearance_age < 2:
                        features.append(f"观察{description}的出现和早期发育")
                    elif fusion_age - target_age < 2:
                        features.append(f"评估{description}的成熟度和愈合程度")
                    else:
                        features.append(f"评估{description}的发育程度")
        
        return features[:10]  # 返回最相关的10个特征
    
    def get_common_errors_for_age(self, target_age: float) -> List[str]:
        """获取特定年龄段的常见诊断错误
        
        Args:
            target_age: 目标年龄
            
        Returns:
            常见错误列表
        """
        errors = []
        
        if target_age < 3:
            errors.extend([
                "忽视早期腕骨的微小变化",
                "过度依赖骨骺的有无而忽视形态",
                "没有考虑早产对骨龄的影响"
            ])
        elif target_age < 8:
            errors.extend([
                "腕骨识别困难导致遗漏",
                "骨骺形态变化评估不准确",
                "没有综合考虑所有骨化中心"
            ])
        elif target_age < 14:
            errors.extend([
                "青春期发育个体差异考虑不足",
                "性别差异应用错误",
                "骨骺愈合程度判断偏差"
            ])
        else:
            errors.extend([
                "接近成熟期的细微变化识别困难",
                "骨骺愈合完成时间判断不准",
                "成年骨骼与青少年骨骼区分困难"
            ])
        
        return errors
    
    def get_learning_recommendations(self, error_history: List[Dict]) -> List[str]:
        """基于错误历史给出学习建议
        
        Args:
            error_history: 错误历史列表
            
        Returns:
            学习建议列表
        """
        recommendations = []
        
        # 分析错误模式
        error_types = {}
        for error in error_history:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 基于最常见错误给出建议
        if error_types.get("significant_bone_age_estimation_error", 0) > 2:
            recommendations.append("建议加强骨化中心识别训练，特别是腕骨和骨骺的形态变化")
        
        if error_types.get("moderate_bone_age_estimation_error", 0) > 3:
            recommendations.append("建议多参考不同标准（GP, TW3）的对比结果")
        
        # 通用建议
        recommendations.extend([
            "定期回顾经典骨龄图谱",
            "加强对性别差异的认识",
            "重视临床病史的结合",
            "注意个体发育差异"
        ])
        
        return recommendations[:5]  # 返回最重要的5个建议