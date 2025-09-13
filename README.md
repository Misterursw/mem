# LangMem 医学模块

LangMem医学模块为AI代理提供专业的医学图像记忆管理和诊断支持功能，特别针对骨龄诊断等医学影像分析场景。

## 🏥 核心功能

### 多模态记忆管理
- **图像编码**: 支持BiomedCLIP等医学专用图像编码器
- **结构化记忆**: 医学图像记忆、骨龄诊断记忆、纠错记忆等专用Schema
- **智能检索**: 图像-文本混合相似性搜索

### 骨龄诊断系统
- **智能诊断**: 基于历史案例和医学知识的骨龄评估
- **多标准支持**: Greulich-Pyle、TW3、中国人标准等
- **置信度评估**: 基于相似案例数量和质量的置信度计算

### 纠错学习机制
- **错误分析**: 自动分析诊断错误模式和趋势
- **持续学习**: 从专家纠错中提取学习要点并改进诊断
- **性能跟踪**: 诊断准确率、置信度校准等指标监控

### 医学RAG系统
- **智能问答**: 结合病例检索和医学知识的专业问答
- **诊断支持**: 为医学影像提供基于证据的诊断建议
- **知识整合**: 自动整合相似病例和标准医学知识

## 🚀 快速开始

### 安装

```bash
# 安装基础LangMem
pip install langmem

# 安装医学模块依赖
pip install langmem[medical]

# 或使用uv
uv sync --group medical
```

### 基本使用

```python
import asyncio
from langmem.medical import create_medical_memory_manager
from langmem.medical.bone_age import BoneAgeClassifier
from langgraph.store.memory import InMemoryStore

async def main():
    # 创建医学记忆管理器
    manager = create_medical_memory_manager(
        model="anthropic:claude-3-5-sonnet-latest",
        image_encoder="biomedclip",
        domain="bone_age",
        store=InMemoryStore(
            index={
                "dims": 512,
                "embed": "openai:text-embedding-3-small"
            }
        )
    )
    
    # 创建骨龄诊断器
    classifier = BoneAgeClassifier(manager)
    
    # 进行骨龄诊断
    result = await classifier.diagnose(
        image_path="/path/to/hand_xray.jpg",
        chronological_age=8.5,
        gender="male",
        context="常规体检骨龄评估"
    )
    
    print(f"预测骨龄: {result['predicted_bone_age']:.1f}岁")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"建议: {result['recommendations'][0]}")

asyncio.run(main())
```

### 纠错学习

```python
# 从专家纠错中学习
await classifier.learn_from_correction(
    image_path="/path/to/hand_xray.jpg",
    predicted_age=8.5,
    actual_age=8.2,
    feedback="注意掌骨骨骺的成熟度，该患儿发育稍慢",
    chronological_age=8.0,
    gender="male"
)

# 查看学习进度
performance = await classifier.get_performance_metrics()
print(f"诊断准确率: {performance['accuracy_rate']:.1%}")
print(f"平均误差: {performance['average_error']:.1f}岁")
```

### 医学RAG问答

```python
from langmem.medical.retrieval import MedicalRAGSystem
from langchain_anthropic import ChatAnthropic

# 创建RAG系统
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
rag_system = MedicalRAGSystem(manager, llm)

# 医学问答
answer = await rag_system.answer_medical_question(
    question="8岁男孩骨龄超前1年意味着什么？",
    patient_info={"age": 8, "gender": "male"}
)

print(answer["answer"])
```

## 🏗️ 架构设计

### 核心组件

```
langmem/medical/
├── __init__.py              # 模块入口
├── schemas.py               # 数据结构定义
├── manager.py               # 医学记忆管理器
├── image_encoder.py         # 图像编码器
├── retrieval.py            # 多模态检索和RAG
├── learning.py             # 纠错学习机制
└── bone_age/               # 骨龄诊断专用模块
    ├── __init__.py
    ├── classifier.py       # 骨龄分类器
    └── knowledge.py        # 骨龄知识库
```

### 数据Schema

```python
# 医学图像记忆
class MedicalImageMemory(BaseModel):
    content: str                        # 图像描述
    image_path: str                     # 图像路径
    image_features: list[float] | None  # 图像特征向量
    image_type: str                     # 图像类型
    body_part: str                      # 身体部位
    patient_age: float | None           # 患者年龄
    diagnosis: str | None               # 诊断结果
    confidence: float | None            # 置信度

# 骨龄诊断记忆
class BoneAgeMemory(BaseModel):
    content: str                        # 诊断描述
    chronological_age: float            # 实际年龄
    predicted_bone_age: float           # 预测骨龄
    actual_bone_age: float | None       # 实际骨龄
    gender: str                         # 性别
    assessment_method: str              # 评估方法
    key_features: list[str]             # 关键特征
    confidence: float                   # 置信度

# 诊断纠错记忆
class DiagnosisCorrection(BaseModel):
    content: str                        # 纠错说明
    original_diagnosis: str             # 原始诊断
    corrected_diagnosis: str            # 纠正诊断
    error_type: str                     # 错误类型
    correction_reason: str              # 纠正原因
    learning_points: list[str]          # 学习要点
```

### 图像编码器

支持多种医学图像编码器：

```python
from langmem.medical.image_encoder import create_image_encoder

# BiomedCLIP编码器（推荐）
encoder = create_image_encoder("biomedclip")

# ResNet编码器（备选）
encoder = create_image_encoder("resnet", model_name="resnet50")

# 编码图像
features = encoder.encode("/path/to/medical_image.jpg")
```

## 📚 医学知识库

### 预置诊断标准

- **Greulich-Pyle标准**: 基于标准图谱的骨龄评估
- **TW3标准**: 基于评分系统的客观评估方法
- **中国人标准**: 适合中国儿童的骨龄发育标准

### 发育标志点

自动提供不同年龄段的关键观察特征：

```python
from langmem.medical.bone_age.knowledge import BoneAgeKnowledge

knowledge = BoneAgeKnowledge()

# 获取8岁男孩应观察的特征
features = knowledge.get_diagnostic_features_for_age(8.0, "male")
print(features)
# ['观察桡骨远端骨化中心的出现和早期发育', 
#  '评估腕骨骨化中心数量和形态', ...]

# 获取常见错误
errors = knowledge.get_common_errors_for_age(8.0)
```

## 🔬 高级功能

### 自适应学习

系统会根据历史错误自动调整诊断置信度：

```python
from langmem.medical.learning import AdaptiveLearningSystem

learning_system = AdaptiveLearningSystem(manager)

# 基于历史错误调整置信度
adjusted_confidence, explanation = await learning_system.adjust_diagnosis_confidence(
    predicted_age=8.5,
    initial_confidence=0.8,
    patient_age=8.0,
    gender="male"
)

print(f"调整后置信度: {adjusted_confidence:.2f}")
print(f"调整说明: {explanation}")
```

### 性能分析

```python
# 获取详细的学习进度报告
report = await learning_system.get_learning_progress_report()

print(f"总体性能等级: {report['performance_metrics']['performance_grade']}")
print(f"最常见错误: {report['error_analysis']['error_types']['most_common']}")
print(f"学习趋势: {report['error_analysis']['learning_trends']['learning_direction']}")
```

### 多模态检索

```python
from langmem.medical.retrieval import MultiModalRetriever

retriever = MultiModalRetriever(manager)

# 图像+文本混合检索
similar_cases = await retriever.retrieve_similar_cases(
    query_text="8岁男孩骨龄评估",
    query_image="/path/to/query_image.jpg",
    limit=5
)

# 医学知识检索
knowledge = await retriever.retrieve_knowledge(
    topic="bone age assessment",
    category="diagnostic_standard",
    age_range=(6.0, 10.0)
)
```

## ⚙️ 配置选项

### 环境变量

```bash
# 必需的API密钥
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"  # 用于文本嵌入

# 可选配置
export CUDA_VISIBLE_DEVICES="0"  # GPU配置（如果使用）
```

### 模型配置

```python
# 自定义配置
manager = create_medical_memory_manager(
    model="anthropic:claude-3-5-sonnet-latest",
    image_encoder="biomedclip",  # 或 "resnet"
    domain="bone_age",
    namespace=("medical", "bone_age", "{user_id}"),
    query_model="anthropic:claude-3-5-haiku-latest",  # 更快的查询模型
    query_limit=10,
    store=your_store
)
```

## 🧪 测试

```bash
# 运行医学模块测试
pytest tests/medical/ -v

# 运行特定测试
pytest tests/medical/test_medical_schemas.py -v

# 运行完整示例
python examples/medical_bone_age_example.py
```

## 📋 数据格式要求

### 医学图像

- **支持格式**: DICOM (.dcm), JPEG (.jpg), PNG (.png)
- **推荐分辨率**: 512x512或更高
- **图像质量**: 清晰度好，对比度适中
- **手部X光**: 手部完全伸展，手指分开，无重叠

### 诊断数据

```python
# 标准诊断数据格式
diagnosis_data = {
    "image_path": "/path/to/hand_xray.jpg",
    "chronological_age": 8.5,           # 实际年龄（岁）
    "gender": "male",                    # "male" 或 "female"
    "predicted_bone_age": 8.7,          # 预测骨龄（岁）
    "actual_bone_age": 8.2,             # 实际骨龄（可选，用于纠错）
    "assessment_method": "Greulich-Pyle",
    "confidence": 0.85
}
```

## 🤝 贡献指南

1. Fork项目仓库
2. 创建功能分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add your feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 提交Pull Request

## 📄 许可证

本项目采用与LangMem主项目相同的许可证。

## 🆘 支持

- **文档**: [LangMem文档](https://docs.anthropic.com/langmem)
- **问题反馈**: [GitHub Issues](https://github.com/anthropics/langmem/issues)
- **讨论**: [GitHub Discussions](https://github.com/anthropics/langmem/discussions)

## ⚠️ 免责声明

本医学模块仅供研究和教育目的使用，不应作为临床诊断的唯一依据。任何医学诊断都应由合格的医疗专业人员进行确认。