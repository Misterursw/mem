#!/usr/bin/env python3
"""
LangMem医学模块使用示例 - 骨龄诊断

这个示例展示了如何使用LangMem的医学模块进行骨龄诊断，
包括图像分析、记忆管理和纠错学习等功能。

使用前请安装医学依赖：
uv sync --group medical
"""

import asyncio
import logging
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

# 医学模块导入
from langmem.medical import create_medical_memory_manager
from langmem.medical.bone_age import BoneAgeClassifier
from langmem.medical.retrieval import MedicalRAGSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数 - 演示医学模块的完整工作流程"""
    
    # 1. 初始化组件
    print("🔧 初始化医学记忆管理系统...")
    
    # 配置存储（支持向量搜索）
    store = InMemoryStore(
        index={
            "dims": 512,  # biomedclip特征维度
            "embed": "openai:text-embedding-3-small",  # 文本嵌入
        }
    )
    
    # 创建医学记忆管理器
    manager = create_medical_memory_manager(
        model="anthropic:claude-3-5-sonnet-latest",
        image_encoder="biomedclip",  # 使用biomedclip进行图像编码
        domain="bone_age",
        store=store
    )
    
    # 创建骨龄分类器
    classifier = BoneAgeClassifier(manager)
    
    # 创建医学RAG系统
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    rag_system = MedicalRAGSystem(manager, llm)
    
    print("✅ 初始化完成！\n")
    
    # 2. 演示骨龄诊断功能
    await demonstrate_bone_age_diagnosis(classifier)
    
    # 3. 演示纠错学习
    await demonstrate_error_learning(classifier)
    
    # 4. 演示医学RAG问答
    await demonstrate_medical_rag(rag_system)
    
    # 5. 显示学习洞察
    await show_learning_insights(manager, classifier)


async def demonstrate_bone_age_diagnosis(classifier: BoneAgeClassifier):
    """演示骨龄诊断功能"""
    print("🦴 演示骨龄诊断功能")
    print("=" * 50)
    
    # 模拟诊断案例
    test_cases = [
        {
            "image_path": "/path/to/hand_xray_1.jpg",  # 实际使用时替换为真实路径
            "chronological_age": 8.5,
            "gender": "male",
            "description": "8岁半男孩，常规骨龄检查"
        },
        {
            "image_path": "/path/to/hand_xray_2.jpg",
            "chronological_age": 12.0,
            "gender": "female", 
            "description": "12岁女孩，身材矮小评估"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 案例 {i}: {case['description']}")
        
        try:
            # 注意：这里使用模拟路径，实际使用时需要真实的X光片
            # 为了演示，我们假设有这些图像文件
            print(f"   图像路径: {case['image_path']}")
            print(f"   实际年龄: {case['chronological_age']}岁")
            print(f"   性别: {case['gender']}")
            
            # 在实际环境中，这里会进行真实的诊断
            # result = await classifier.diagnose(
            #     image_path=case["image_path"],
            #     chronological_age=case["chronological_age"],
            #     gender=case["gender"],
            #     context=case["description"]
            # )
            
            # 模拟诊断结果
            simulated_result = {
                "predicted_bone_age": case["chronological_age"] + 0.5,
                "chronological_age": case["chronological_age"],
                "age_difference": 0.5,
                "confidence": 0.75,
                "gender": case["gender"],
                "similar_cases_found": 2,
                "corrections_applied": 1,
                "recommendations": [
                    "骨龄发育正常，符合实际年龄",
                    "建议结合患儿身高、体重等生长发育指标综合评估"
                ]
            }
            
            print(f"   🎯 预测骨龄: {simulated_result['predicted_bone_age']:.1f}岁")
            print(f"   📊 置信度: {simulated_result['confidence']:.2f}")
            print(f"   📈 年龄差异: {simulated_result['age_difference']:+.1f}岁")
            print(f"   🔍 参考案例: {simulated_result['similar_cases_found']}个")
            print(f"   🎓 应用纠错: {simulated_result['corrections_applied']}个")
            print(f"   💡 建议: {simulated_result['recommendations'][0]}")
            
        except Exception as e:
            print(f"   ❌ 诊断失败 (模拟环境): {e}")
            print(f"   💭 实际使用时需要真实的X光片图像")


async def demonstrate_error_learning(classifier: BoneAgeClassifier):
    """演示纠错学习功能"""
    print("\n🎓 演示纠错学习功能")
    print("=" * 50)
    
    # 模拟纠错场景
    correction_cases = [
        {
            "image_path": "/path/to/hand_xray_1.jpg",
            "predicted_age": 9.0,
            "actual_age": 8.2,
            "feedback": "注意掌骨骨骺的成熟度评估，该患儿发育稍慢",
            "chronological_age": 8.5,
            "gender": "male"
        },
        {
            "image_path": "/path/to/hand_xray_2.jpg", 
            "predicted_age": 11.5,
            "actual_age": 12.3,
            "feedback": "女孩青春期前期，骨龄发育较快，需要注意腕骨愈合程度",
            "chronological_age": 12.0,
            "gender": "female"
        }
    ]
    
    for i, case in enumerate(correction_cases, 1):
        print(f"\n📝 纠错案例 {i}:")
        print(f"   原预测: {case['predicted_age']:.1f}岁")
        print(f"   实际骨龄: {case['actual_age']:.1f}岁")
        print(f"   误差: {abs(case['predicted_age'] - case['actual_age']):.1f}岁")
        print(f"   专家反馈: {case['feedback']}")
        
        try:
            # 在实际环境中进行纠错学习
            # correction_id = await classifier.learn_from_correction(
            #     image_path=case["image_path"],
            #     predicted_age=case["predicted_age"],
            #     actual_age=case["actual_age"],
            #     feedback=case["feedback"],
            #     chronological_age=case["chronological_age"],
            #     gender=case["gender"]
            # )
            
            # 模拟结果
            print(f"   ✅ 纠错记忆已保存")
            print(f"   📚 学习要点已提取和存储")
            
        except Exception as e:
            print(f"   ❌ 纠错学习失败 (模拟环境): {e}")


async def demonstrate_medical_rag(rag_system: MedicalRAGSystem):
    """演示医学RAG问答功能"""
    print("\n🤖 演示医学RAG问答功能")
    print("=" * 50)
    
    # 医学问题示例
    questions = [
        {
            "question": "8岁男孩的骨龄比实际年龄大1.5岁，这意味着什么？",
            "patient_info": {"age": 8.0, "gender": "male"}
        },
        {
            "question": "Greulich-Pyle标准和TW3标准在骨龄评估中有什么区别？",
            "patient_info": None
        },
        {
            "question": "女孩在青春期前期的骨龄发育有什么特点？",
            "patient_info": {"age": 11.0, "gender": "female"}
        }
    ]
    
    for i, item in enumerate(questions, 1):
        print(f"\n❓ 问题 {i}: {item['question']}")
        
        if item['patient_info']:
            print(f"   患者信息: {item['patient_info']}")
        
        try:
            # 在实际环境中获取RAG回答
            # answer = await rag_system.answer_medical_question(
            #     question=item["question"],
            #     patient_info=item["patient_info"]
            # )
            
            # 模拟回答
            simulated_answers = [
                "骨龄超前1.5岁提示该儿童骨骼发育较快，可能与营养状况良好、内分泌功能活跃等因素有关。建议结合身高生长曲线和临床表现综合评估，必要时检查相关激素水平。",
                
                "Greulich-Pyle标准基于标准图谱比较，操作相对简单但主观性较强；TW3标准采用评分系统，更加客观和精确，但需要专业训练。两种方法各有优缺点，临床应用中可以结合使用。",
                
                "女孩在青春期前期（9-11岁）骨龄发育通常比男孩提前6-12个月，腕骨出现顺序和骨骺愈合时间都相对较早。此时期要特别注意观察月经初潮的预测指标。"
            ]
            
            print(f"   🤖 回答: {simulated_answers[i-1]}")
            print(f"   📊 参考病例: 模拟环境 - 3个相似案例")
            print(f"   📚 医学知识: 模拟环境 - 2个相关知识条目")
            
        except Exception as e:
            print(f"   ❌ RAG问答失败 (模拟环境): {e}")


async def show_learning_insights(manager, classifier: BoneAgeClassifier):
    """显示学习洞察"""
    print("\n📈 学习洞察和性能分析")
    print("=" * 50)
    
    try:
        # 获取学习洞察
        # insights = await manager.get_learning_insights()
        # performance = await classifier.get_performance_metrics()
        
        # 模拟学习洞察
        simulated_insights = {
            "total_corrections": 5,
            "error_type_distribution": {
                "moderate_bone_age_estimation_error": 3,
                "minor_bone_age_estimation_error": 2
            },
            "common_learning_points": [
                ("注意掌骨骨骺发育程度", 2),
                ("重视腕骨愈合时间", 2),
                ("考虑性别差异", 1)
            ],
            "suggestions": [
                "重点关注最常见的错误类型",
                "加强对关键学习要点的训练"
            ]
        }
        
        simulated_performance = {
            "total_cases": 15,
            "accuracy_rate": 0.73,
            "average_error": 0.8,
            "average_confidence": 0.72,
            "performance_grade": "良好"
        }
        
        print(f"📊 总体性能:")
        print(f"   诊断案例数: {simulated_performance['total_cases']}")
        print(f"   准确率: {simulated_performance['accuracy_rate']:.1%}")
        print(f"   平均误差: {simulated_performance['average_error']:.1f}岁")
        print(f"   平均置信度: {simulated_performance['average_confidence']:.2f}")
        print(f"   性能等级: {simulated_performance['performance_grade']}")
        
        print(f"\n🎓 学习统计:")
        print(f"   总纠错次数: {simulated_insights['total_corrections']}")
        print(f"   错误类型分布:")
        for error_type, count in simulated_insights['error_type_distribution'].items():
            print(f"     - {error_type}: {count}次")
        
        print(f"\n💡 常见学习要点:")
        for point, freq in simulated_insights['common_learning_points']:
            print(f"   - {point} (出现{freq}次)")
        
        print(f"\n🚀 改进建议:")
        for suggestion in simulated_insights['suggestions']:
            print(f"   - {suggestion}")
            
    except Exception as e:
        print(f"❌ 获取学习洞察失败 (模拟环境): {e}")


def print_installation_guide():
    """打印安装指南"""
    print("📦 LangMem医学模块安装指南")
    print("=" * 50)
    print()
    print("1. 安装基础LangMem:")
    print("   pip install langmem")
    print()
    print("2. 安装医学模块依赖:")
    print("   pip install langmem[medical]")
    print("   # 或者使用uv:")
    print("   uv sync --group medical")
    print()
    print("3. 配置模型访问:")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("   export OPENAI_API_KEY='your-key'  # 用于文本嵌入")
    print()
    print("4. 准备医学图像数据:")
    print("   - 手部X光片 (DICOM或常见图像格式)")
    print("   - 建议分辨率: 512x512 或更高")
    print("   - 确保图像质量清晰，手部完全展开")
    print()


if __name__ == "__main__":
    print("🏥 LangMem医学模块示例")
    print("=" * 50)
    print()
    
    # 检查是否在模拟环境中运行
    print("⚠️  注意: 这是一个演示示例，使用模拟数据")
    print("   在实际使用中，请提供真实的医学图像文件")
    print()
    
    # 显示安装指南
    print_installation_guide()
    
    # 运行主程序
    try:
        asyncio.run(main())
        print("\n✅ 示例运行完成！")
        print("\n📚 更多信息:")
        print("   - 文档: https://docs.anthropic.com/langmem")
        print("   - GitHub: https://github.com/anthropics/langmem")
        
    except KeyboardInterrupt:
        print("\n👋 示例已终止")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        print("\n💡 这通常是因为缺少必要的依赖或API密钥")
        print("   请参考上面的安装指南进行配置")