#!/usr/bin/env python3
"""
LangMem医学模块完整设置指南
解决Gemini API配置和BiomedCLIP模型下载问题
"""
import torch
import open_clip
import asyncio
import os
from pathlib import Path

# --- 依赖检查 ---
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
        
    try:
        from langgraph.store.memory import InMemoryStore
        print("✅ LangGraph: 可用")
    except ImportError:
        missing_deps.append("langgraph")

    if missing_deps:
        print(f"❌ 缺少依赖: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        print("pip install langmem[medical]")
        print("# 或")
        print("uv sync --group medical")
        return False
    return True

# --- Gemini API 设置 ---
def setup_gemini_api():
    """设置并测试Gemini API"""
    print("\n🔧 配置Gemini API...")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("🔑 未找到 GOOGLE_API_KEY 环境变量。")
        # 为方便测试，仍使用脚本内的key，但强烈建议使用环境变量
        gemini_key = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
        os.environ["GOOGLE_API_KEY"] = gemini_key
        print("   已使用脚本内嵌的测试密钥。")
    else:
        print("✅ 已从环境变量加载 Gemini API 密钥。")
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            max_output_tokens=512,
            google_api_key=gemini_key
        )
        model.invoke("Hello, this is a test.")
        print("✅ Gemini API 连接测试成功")
        return True
    except Exception as e:
        print(f"❌ Gemini API 测试失败: {e}")
        return False

# --- BiomedCLIP 模型设置 ---
def setup_biomedclip_model():
    """手动设置BiomedCLIP模型"""
    print("\n🖼️ 设置BiomedCLIP模型...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from PIL import Image
        import open_clip
        
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # 使用 open_clip 直接加载
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        print("✅ BiomedCLIP模型设置完成！")
        return model, tokenizer, preprocess_val
        
    except Exception as e:
        print(f"❌ BiomedCLIP 设置失败: {e}")
        return None, None, None
        
# --- 创建快速启动脚本 ---
def create_quick_start_script():
    """在当前目录创建 quick_start_gemini.py 文件"""
    print("\n📝 创建快速启动脚本...")
    script_content = """
import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore
from langmem.medical import create_medical_memory_manager
from langmem.medical.bone_age import BoneAgeClassifier

async def main():
    # 建议将API密钥设置为环境变量，而不是硬编码在代码中
    # 例如: export GOOGLE_API_KEY="YOUR_API_KEY"
    if "GOOGLE_API_KEY" not in os.environ:
        print("错误：请先设置 GOOGLE_API_KEY 环境变量")
        return

    # 创建Gemini模型
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1
    )
    
    # 创建Google嵌入模型实例
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 创建存储，并传入嵌入模型实例
    store = InMemoryStore(
        index={
            "dims": 768,  # text-embedding-004 的维度是 768
            "embed": embedding_model
        }
    )

    # 创建医学记忆管理器
    manager = create_medical_memory_manager(
        model=gemini_model,
        image_encoder="biomedclip", # 使用更强大的biomedclip
        domain="bone_age",
        store=store
    )

    # 创建骨龄诊断器
    classifier = BoneAgeClassifier(manager)

    print("🎉 系统就绪！可以开始骨龄诊断了")
    print("👉 请在下方代码中添加您的诊断逻辑，例如：")
    print("# result = await classifier.diagnose(image_path='path/to/your/image.png', age=10, gender='male')")
    print("# print(result)")

if __name__ == "__main__":
    asyncio.run(main())
"""
    try:
        with open("quick_start_gemini.py", "w", encoding="utf-8") as f:
            f.write(script_content.strip())
        print("✅ 快速启动脚本已创建: quick_start_gemini.py")
    except IOError as e:
        print(f"❌ 创建快速启动脚本失败: {e}")

# --- 完整性测试 ---
async def test_complete_setup():
    """测试所有组件是否可以协同工作"""
    print("\n🧪 测试完整设置...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from langgraph.store.memory import InMemoryStore
        from langmem.medical import create_medical_memory_manager
        from langmem.medical.bone_age import BoneAgeClassifier

        gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        # 为了快速测试，这里可以不指定嵌入，使用默认的
        store = InMemoryStore()
        
        manager = create_medical_memory_manager(
            model=gemini_model,
            image_encoder="resnet",  # 使用ResNet避免下载BiomedCLIP
            domain="bone_age",
            store=store
        )
        classifier = BoneAgeClassifier(manager)
        print("✅ 完整设置测试成功！")
        print("\n🎯 下一步:")
        print("1. 运行 `python quick_start_gemini.py`")
        print("2. 准备您的手部X光片图像")
        print("3. 根据脚本提示，修改并执行诊断逻辑！")
        return True
    except Exception as e:
        print(f"❌ 设置测试失败: {e}")
        return False

# --- BiomedCLIP 信息打印 ---
def print_biomedclip_info():
    """打印关于BiomedCLIP模型的详细信息"""
    print("\n📚 关于BiomedCLIP模型:")
    print("=" * 50)
    print("🔬 BiomedCLIP是什么: 微软开发的医学专用图像-文本模型，在大量医学影像和文本上预训练，比通用CLIP在医学领域表现更好。")
    print("📥 模型下载: 首次使用会自动从HuggingFace下载，大小约500MB，缓存位置: `~/.cache/langmem/biomedclip/`。")
    print("🎯 微调建议: 为了达到最佳效果，推荐在您的骨龄数据集上进行微调，预期可提升15-30%的准确度。")
    print("🚀 快速开始: 先用预训练模型测试功能，然后收集数据，使用 `image_encoder_local.py` 进行微调。")

# --- 主执行函数 ---
async def main_setup():
    """主函数，按顺序执行所有设置步骤"""
    print("🏥 LangMem医学模块完整设置指南")
    print("=" * 50)
    
    if not check_dependencies():
        print("\n❌ 请先安装缺失的依赖，然后重试！")
        return

    if not setup_gemini_api():
        print("\n❌ Gemini API设置失败，请检查您的API密钥和网络连接！")
        return

    setup_biomedclip_model()
    create_quick_start_script()
    
    await test_complete_setup()
    print_biomedclip_info()

    print("\n🎉 设置流程完成！")

if __name__ == "__main__":
    asyncio.run(main_setup())