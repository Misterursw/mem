#!/usr/bin/env python3
"""
LangMem医学模块 - 快速演示
一键运行，体验骨龄诊断的完整交互流程

运行方式：
python quick_demo.py
"""

import asyncio
import os
from pathlib import Path

# 检查依赖
try:
    from langchain_anthropic import ChatAnthropic
    from langgraph.store.memory import InMemoryStore
    from langmem.medical import create_medical_memory_manager
    from langmem.medical.bone_age import BoneAgeClassifier
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请先安装: pip install langmem[medical]")
    exit(1)

# 检查API密钥
if not os.getenv('ANTHROPIC_API_KEY'):
    print("❌ 请设置环境变量: export ANTHROPIC_API_KEY='your-key'")
    exit(1)

if not os.getenv('OPENAI_API_KEY'):
    print("❌ 请设置环境变量: export OPENAI_API_KEY='your-key'")
    exit(1)


class BoneAgeDemo:
    """骨龄诊断演示系统"""
    
    def __init__(self):
        self.manager = None
        self.classifier = None
        self.conversation_history = []
    
    async def initialize(self):
        """初始化系统"""
        print("🔧 正在初始化医学记忆系统...")
        
        # 创建存储（支持图像向量搜索）
        store = InMemoryStore(
            index={
                "dims": 512,  # BiomedCLIP特征维度
                "embed": "openai:text-embedding-3-small"  # 文本嵌入
            }
        )
        
        # 创建医学记忆管理器
        self.manager = create_medical_memory_manager(
            model="anthropic:claude-3-5-sonnet-latest",  # 使用Claude作为base模型
            image_encoder="biomedclip",  # 医学图像编码器
            domain="bone_age",
            store=store
        )
        
        # 创建骨龄诊断器
        self.classifier = BoneAgeClassifier(self.manager)
        
        print("✅ 系统初始化完成！")
        print()
    
    async def simulate_conversation(self):
        """模拟理想的对话交互"""
        print("🦴 开始骨龄诊断对话演示")
        print("=" * 50)
        
        # 模拟用户上传图片并请求诊断
        print("👤 用户: [上传手部X光片] 请判断这个8岁男孩的骨龄")
        print()
        
        # 系统进行诊断
        await self.ai_diagnose("hand_xray_demo.jpg", 8.0, "male")
        
        # 用户纠错
        print("👤 用户: 你说的不对，按照掌骨第三掌骨骨骺的发育程度来看，应该是7.8岁")
        print()
        
        # 系统学习纠错
        await self.ai_learn_correction(7.8, "掌骨第三掌骨骨骺发育程度显示实际骨龄较预测偏小")
        
        # 再次类似诊断，展示学习效果
        print("👤 用户: [上传另一张类似的X光片] 这个也是8岁男孩，骨龄多少？")
        print()
        
        await self.ai_diagnose_improved("hand_xray_demo2.jpg", 8.0, "male")
    
    async def ai_diagnose(self, image_path, age, gender):
        """AI诊断"""
        print("🤖 AI: 正在分析图像...")
        
        # 模拟诊断过程
        print("   📊 正在使用BiomedCLIP提取图像特征...")
        print("   🔍 正在搜索相似历史案例...")
        print("   📚 正在查询医学知识库...")
        
        # 模拟诊断结果
        predicted_age = 8.3
        confidence = 0.75
        
        print(f"🎯 AI: 根据分析，这个{age}岁{gender}孩的骨龄是 **{predicted_age}岁**")
        print(f"   📈 置信度: {confidence:.1%}")
        print(f"   📋 依据: 基于腕骨发育程度和骨骺愈合情况")
        print(f"   🔍 参考了3个相似历史案例")
        print()
        
        # 保存诊断记忆
        self.conversation_history.append({
            "type": "diagnosis",
            "predicted_age": predicted_age,
            "actual_age": age,
            "confidence": confidence
        })
    
    async def ai_learn_correction(self, corrected_age, feedback):
        """AI学习纠错"""
        print("🤖 AI: 感谢您的纠正！让我学习一下...")
        print()
        
        # 模拟纠错学习过程
        print("   📝 正在分析诊断误差...")
        print("   🎓 正在提取学习要点...")
        print("   💾 正在保存纠错记忆...")
        
        # 更新记忆
        last_diagnosis = self.conversation_history[-1]
        error = abs(last_diagnosis["predicted_age"] - corrected_age)
        
        print(f"✅ AI: 我明白了！")
        print(f"   ❌ 我的预测偏差: {error:.1f}岁")
        print(f"   📚 学习要点: {feedback}")
        print(f"   🧠 已记录: 需要更仔细观察掌骨骨骺发育细节")
        print(f"   💡 下次遇到类似情况会更加准确")
        print()
        
        # 保存纠错记忆
        self.conversation_history.append({
            "type": "correction",
            "corrected_age": corrected_age,
            "feedback": feedback,
            "learning_points": ["注意掌骨骨骺发育细节", "重视专家反馈"]
        })
    
    async def ai_diagnose_improved(self, image_path, age, gender):
        """AI改进后的诊断"""
        print("🤖 AI: 好的，让我再次分析...")
        print()
        print("   📊 正在使用BiomedCLIP提取图像特征...")
        print("   🧠 正在应用之前学到的经验...")
        print("   🔍 特别关注掌骨骨骺发育程度...")
        
        # 模拟改进后的诊断
        predicted_age = 7.9  # 更准确了
        confidence = 0.85    # 置信度提高
        
        print(f"🎯 AI: 基于之前的学习经验，这个{age}岁{gender}孩的骨龄是 **{predicted_age}岁**")
        print(f"   📈 置信度: {confidence:.1%} (比之前提高了)")
        print(f"   📋 改进依据: 特别关注了掌骨第三掌骨骨骺发育程度")
        print(f"   🎓 应用了之前的学习要点")
        print(f"   ✨ 诊断准确性显著提升！")
        print()
    
    async def show_technical_details(self):
        """显示技术详情"""
        print("🔧 技术架构详情")
        print("=" * 50)
        print("📊 Base模型: Anthropic Claude-3.5-Sonnet (主要推理)")
        print("🖼️  图像编码: Microsoft BiomedCLIP (医学图像特征)")
        print("📝 文本嵌入: OpenAI text-embedding-3-small (语义搜索)")
        print("💾 存储方式: LangGraph InMemoryStore (支持向量搜索)")
        print("🧠 记忆类型: 图像记忆 + 骨龄记忆 + 纠错记忆")
        print("🔍 检索方式: 图像相似度 + 文本语义 混合检索")
        print()
    
    async def show_rag_knowledge(self):
        """显示RAG知识库"""
        print("📚 医学知识库 (RAG)")
        print("=" * 50)
        print("🦴 预置标准:")
        print("   - Greulich-Pyle 标准 (0-18岁)")
        print("   - TW3标准 (2-16岁)")  
        print("   - 中国人骨龄标准 (0-18岁)")
        print()
        print("📖 发育标志:")
        print("   - 腕骨出现时间 (8个腕骨)")
        print("   - 掌骨骨骺发育")
        print("   - 指骨骨骺愈合")
        print("   - 桡骨尺骨发育")
        print()
        print("💡 诊断要点:")
        print("   - 不同年龄段关键特征")
        print("   - 性别差异注意事项")
        print("   - 常见错误和避免方法")
        print()
        print("📍 添加自定义知识:")
        print("   1. 编辑 langmem/medical/bone_age/knowledge.py")
        print("   2. 使用 manager.base_manager.aput() 动态添加")
        print("   3. 通过 MedicalKnowledge Schema 结构化存储")
        print()
    
    def show_file_structure(self):
        """显示文件结构"""
        print("📁 关键文件说明")
        print("=" * 50)
        print("🚀 quick_demo.py           # 快速演示脚本 (本文件)")
        print("📖 examples/medical_bone_age_example.py  # 完整使用示例")
        print("📚 MEDICAL_README.md       # 详细文档")
        print()
        print("核心代码:")
        print("🏥 src/langmem/medical/")
        print("   ├── manager.py          # 医学记忆管理器")  
        print("   ├── schemas.py          # 数据结构定义")
        print("   ├── image_encoder.py    # BiomedCLIP图像编码")
        print("   ├── retrieval.py        # 多模态检索+RAG")
        print("   ├── learning.py         # 纠错学习机制")
        print("   └── bone_age/")
        print("       ├── classifier.py   # 骨龄诊断器")
        print("       └── knowledge.py    # 医学知识库 📍")
        print()


async def main():
    """主函数"""
    print("🏥 LangMem医学模块 - 快速演示")
    print("=" * 50)
    print()
    
    demo = BoneAgeDemo()
    
    try:
        # 1. 初始化系统
        await demo.initialize()
        
        # 2. 演示理想的对话交互
        await demo.simulate_conversation()
        
        # 3. 显示技术详情
        await demo.show_technical_details()
        
        # 4. 显示RAG知识库
        await demo.show_rag_knowledge()
        
        # 5. 显示文件结构
        demo.show_file_structure()
        
        print("✅ 演示完成！")
        print()
        print("📋 下一步:")
        print("1. 准备真实的手部X光片图像")
        print("2. 参考 examples/medical_bone_age_example.py")
        print("3. 阅读 MEDICAL_README.md 了解详细用法")
        print("4. 根据需要在 knowledge.py 中添加自定义医学知识")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print()
        print("💡 常见问题:")
        print("- 确保已安装依赖: pip install langmem[medical]")
        print("- 确保已设置API密钥: ANTHROPIC_API_KEY 和 OPENAI_API_KEY")
        print("- 检查网络连接是否正常")


if __name__ == "__main__":
    asyncio.run(main())