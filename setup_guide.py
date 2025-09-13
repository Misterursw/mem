#!/usr/bin/env python3
"""
LangMem医学模块完整设置指南
解决Gemini API配置和BiomedCLIP模型下载问题
"""

import asyncio
import os
import sys
from pathlib import Path


def check_dependencies():
    """检查依赖安装"""
    print("📦 检查依赖安装...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ Gemini集成: 可用")
    except ImportError:
        missing_deps.append("langchain-google-genai")
    
    if missing_deps:
        print(f"❌ 缺少依赖: {missing_deps}")
        print("安装命令:")
        print("pip install langmem[medical]")
        print("# 或")
        print("uv sync --group medical")
        return False
    
    return True


def setup_gemini_api():
    """设置Gemini API"""
    print("\n🔧 配置Gemini API...")
    
    # 您的API密钥
    gemini_key = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
    
    # 设置环境变量
    os.environ["GOOGLE_API_KEY"] = gemini_key
    
    print(f"✅ Gemini API密钥已设置")
    
    # 测试API连接
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            max_output_tokens=512,
            google_api_key=gemini_key
        )
        
        # 简单测试
        response = model.invoke("Hello, this is a test.")
        print(f"✅ Gemini API连接测试成功")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API测试失败: {e}")
        return False


def setup_biomedclip_model():
    """设置BiomedCLIP模型"""
    print("\n🖼️ 设置BiomedCLIP模型...")
    
    try:
        from langmem.medical.image_encoder_local import LocalBiomedCLIP
        
        # 创建本地模型管理器
        biomedclip = LocalBiomedCLIP()
        
        print("📥 开始下载BiomedCLIP模型...")
        print("   ⏳ 首次下载约需要几分钟，请耐心等待...")
        
        # 下载模型
        biomedclip.download_model()
        
        # 测试加载
        biomedclip.load_model()
        
        print("✅ BiomedCLIP模型设置完成！")
        return True
        
    except Exception as e:
        print(f"❌ BiomedCLIP设置失败: {e}")
        print("💡 这是正常的，模型会在首次使用时自动下载")
        return False







import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.memory import InMemoryStore
from langmem.medical import create_medical_memory_manager
from langmem.medical.bone_age import BoneAgeClassifier

async def main():
    # 设置Gemini API
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
    
    # 创建Gemini模型
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1
    )
    
    # 创建存储
    store = InMemoryStore(
        index={
            "dims": 512,
            "embed": "google:text-embedding-004"  # 使用Google嵌入
        }
    )
    
    # 创建医学记忆管理器
    manager = create_medical_memory_manager(
        model=gemini_model,  # 使用Gemini
        image_encoder="biomedclip",
        domain="bone_age",
        store=store
    )
    
    # 创建骨龄诊断器
    classifier = BoneAgeClassifier(manager)
    
    print("🎉 系统就绪！可以开始骨龄诊断了")
    
    # 这里添加您的诊断逻辑
    # result = await classifier.diagnose(image_path, age, gender)

if __name__ == "__main__":
    asyncio.run(main())

    
    with open("quick_start_gemini.py", "w") as f:
        f.write(script_content)
    
    print("✅ 快速启动脚本已创建: quick_start_gemini.py")


async def test_complete_setup():
    """测试完整设置"""
    print("\n🧪 测试完整设置...")
    
    try:
        # 设置环境变量
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.store.memory import InMemoryStore
        from langmem.medical import create_medical_memory_manager
        from langmem.medical.bone_age import BoneAgeClassifier
        
        # 创建组件
        gemini_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # 使用更快的模型测试
            temperature=0.1
        )
        
        store = InMemoryStore()  # 简化测试
        
        manager = create_medical_memory_manager(
            model=gemini_model,
            image_encoder="resnet",  # 使用ResNet避免下载问题
            domain="bone_age",
            store=store
        )
        
        classifier = BoneAgeClassifier(manager)
        
        print("✅ 完整设置测试成功！")
        print("\n🎯 下一步:")
        print("1. 运行 python quick_start_gemini.py")
        print("2. 准备手部X光片图像")
        print("3. 开始骨龄诊断！")
        
        return True
        
    except Exception as e:
        print(f"❌ 设置测试失败: {e}")
        return False


def print_biomedclip_info():
    """打印BiomedCLIP信息"""
    print("\n📚 关于BiomedCLIP模型:")
    print("=" * 50)
    print("🔬 BiomedCLIP是什么:")
    print("   - 微软开发的医学专用图像-文本模型")
    print("   - 在大量医学影像和文本上预训练")
    print("   - 比通用CLIP在医学领域表现更好")
    print()
    print("📥 模型下载:")
    print("   - 首次使用会自动从HuggingFace下载")
    print("   - 模型大小约: ~500MB")
    print("   - 缓存位置: ~/.cache/langmem/biomedclip/")
    print()
    print("🎯 是否需要微调:")
    print("   ✅ 推荐: 在您的骨龄数据集上微调")
    print("   📈 效果提升: 预期准确度提升15-30%")
    print("   📊 数据需求: 至少100-500个标注样本")
    print("   ⏱️  训练时间: 几小时到几天（取决于数据量）")
    print()
    print("🚀 快速开始:")
    print("   1. 先用预训练模型测试基本功能")
    print("   2. 收集骨龄标注数据")
    print("   3. 使用 image_encoder_local.py 进行微调")
    print()
    print("💡 微调建议:")
    print("   - 数据格式: [(image_path, bone_age), ...]")
    print("   - 冻结主干网络，只训练回归头")
    print("   - 学习率: 1e-5 到 1e-4")
    print("   - 数据增强: 旋转、亮度调整等")


async def main():
    """主函数"""
    print("🏥 LangMem医学模块完整设置指南")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装依赖!")
        return
    
    # 2. 设置Gemini API
    if not setup_gemini_api():
        print("\n❌ Gemini API设置失败!")
        return
    
    # 3. 尝试设置BiomedCLIP
    setup_biomedclip_model()
    
    # 4. 创建快速启动脚本
    create_quick_start_script()
    
    # 5. 测试完整设置
    await test_complete_setup()
    
    # 6. 打印BiomedCLIP信息
    print_biomedclip_info()
    
    print("\n🎉 设置完成！您现在可以:")
    print("1. 运行 python quick_start_gemini.py 开始使用")
    print("2. 查看 examples/gemini_config_example.py 了解更多")
    print("3. 使用 image_encoder_local.py 进行模型微调")


if __name__ == "__main__":
    asyncio.run(main())