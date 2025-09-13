#!/usr/bin/env python3
"""
使用Gemini API的LangMem医学模块配置示例
"""

import asyncio
import os
from pathlib import Path

# Gemini集成
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.store.memory import InMemoryStore
    from langmem.medical import create_medical_memory_manager
    from langmem.medical.bone_age import BoneAgeClassifier
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请先安装: pip install langchain-google-genai")
    exit(1)


async def setup_with_gemini():
    """使用Gemini API配置医学模块"""
    
    # 配置Gemini API
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
    
    print("🔧 使用Gemini API初始化系统...")
    
    # 创建Gemini模型实例
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # 或 gemini-1.5-flash
        temperature=0.1,  # 医学诊断需要稳定输出
        google_api_key="AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
    )
    
    # 创建存储
    store = InMemoryStore(
        index={
            "dims": 512,  # BiomedCLIP特征维度
            "embed": "google:text-embedding-004"  # 使用Google的嵌入模型
        }
    )
    
    # 创建医学记忆管理器（使用Gemini）
    manager = create_medical_memory_manager(
        model=gemini_model,  # 传入Gemini模型实例
        image_encoder="biomedclip",
        domain="bone_age", 
        store=store
    )
    
    # 创建骨龄诊断器
    classifier = BoneAgeClassifier(manager)
    
    print("✅ Gemini配置完成！")
    
    return manager, classifier


async def test_gemini_integration():
    """测试Gemini集成"""
    try:
        manager, classifier = await setup_with_gemini()
        
        print("\n🧪 测试Gemini医学问答...")
        
        # 测试基本医学问答
        from langmem.medical.retrieval import MedicalRAGSystem
        gemini_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key="AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
        )
        
        rag_system = MedicalRAGSystem(manager, gemini_model)
        
        # 模拟医学问答
        response = await rag_system.answer_medical_question(
            question="8岁男孩骨龄超前1岁意味着什么？",
            patient_info={"age": 8, "gender": "male"}
        )
        
        print(f"🤖 Gemini回答: {response['answer'][:200]}...")
        print("✅ Gemini集成测试成功！")
        
    except Exception as e:
        print(f"❌ Gemini集成测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_gemini_integration())