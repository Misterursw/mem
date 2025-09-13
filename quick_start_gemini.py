import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore
from langmem.medical import create_medical_memory_manager
from langmem.medical.bone_age import BoneAgeClassifier

async def main():
    # 建议将API密钥设置为环境变量，而不是硬编码在代码中
    # 例如: export GOOGLE_API_KEY="YOUR_API_KEY"
    gemini_key = "AIzaSyBBtS2L2eKWAjzvfEAirB9faF_KVN8YeJ8"
    os.environ["GOOGLE_API_KEY"] = gemini_key
    if "GOOGLE_API_KEY" not in os.environ:
        print("错误：请先设置 GOOGLE_API_KEY 环境变量")
        return

    # 创建Gemini模型
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
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